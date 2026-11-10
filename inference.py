"""Inference-only ResNet fusion, adapting the T-unrolled LIF idea from spformers.

Reference: putshua/spformers, f5bfa1d9fd6e13c61802929c175397f57746ca39.
This implementation preserves SpikingJelly's decay_input update and >= threshold.
Each call starts at zero membrane potential and returns time-averaged logits.
"""
import copy

import torch
import torch.nn.functional as F
from torch import nn
import triton
import triton.language as tl


@triton.jit
def _lif_kernel(X, Y, MASK, SCALE, OFFSET, SKIP, M: tl.constexpr,
                CHW: tl.constexpr, HW: tl.constexpr, XT: tl.constexpr,
                T: tl.constexpr, TAU: tl.constexpr, THRESHOLD: tl.constexpr,
                DECAY_INPUT: tl.constexpr, HAS_MASK: tl.constexpr,
                AFFINE: tl.constexpr, PRE_ADD: tl.constexpr, POST_ADD: tl.constexpr,
                BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = i < M
    active = valid
    if HAS_MASK:
        active = valid & tl.load(MASK + i % CHW, valid, other=0)
    if AFFINE:
        c = i % CHW // HW
        a = tl.load(SCALE + c, valid, other=0)
        b = tl.load(OFFSET + c, valid, other=0)
    v = tl.full((BLOCK,), 0, tl.float32)
    for t in tl.static_range(T):
        z = tl.load(X + t * XT + i, active, other=0).to(tl.float32)
        if AFFINE:
            z = z * a + b
        if PRE_ADD:
            z = z + tl.load(SKIP + t * M + i, valid, other=0)
        if DECAY_INPUT:
            v = v + (z - v) / TAU
        else:
            v = v * (1 - 1 / TAU) + z
        fired = v >= THRESHOLD
        v = tl.where(fired, 0., v)
        y = (fired & active).to(tl.float32)
        if POST_ADD:
            y = y + tl.load(SKIP + t * M + i, valid, other=0)
        tl.store(Y + t * M + i, y, valid)


def lif_sequence(x, *, T=4, tau=2., threshold=1., decay_input=True,
                 mask=None, affine=None, residual=None, pre_add=False, static=False):
    if not x.is_cuda or x.dtype != torch.float32:
        raise ValueError('Fused inference requires CUDA float32 inputs')
    if static:
        n, c, h, w = x.shape
    else:
        t, n, c, h, w = x.shape
        if t != T:
            raise ValueError('Unexpected time dimension')
    x = x.contiguous()
    if mask is not None and mask.numel() != c*h*w:
        raise ValueError('Spatial mask does not match this input resolution')
    y = torch.empty((T, n, c, h, w), device=x.device, dtype=x.dtype)
    m = n*c*h*w
    scale, offset = affine if affine is not None else (x, x)
    _lif_kernel[(triton.cdiv(m, 256),)](
        x, y, mask if mask is not None else x, scale, offset,
        residual if residual is not None else x,
        M=m, CHW=c*h*w, HW=h*w, XT=0 if static else m, T=T,
        TAU=tau, THRESHOLD=threshold, DECAY_INPUT=decay_input,
        HAS_MASK=mask is not None, AFFINE=affine is not None,
        PRE_ADD=residual is not None and pre_add,
        POST_ADD=residual is not None and not pre_add,
        BLOCK=256, enable_fp_fusion=False)
    return y



@triton.jit
def _sparse_conv_lif_kernel(X,W,BIAS,POINTS,SCALE,OFFSET,Y,SKIP,
                            COUNT:tl.constexpr,CI:tl.constexpr,HI:tl.constexpr,WI:tl.constexpr,
                            CO:tl.constexpr,HO:tl.constexpr,WO:tl.constexpr,
                            KH:tl.constexpr,KW:tl.constexpr,SH:tl.constexpr,SW:tl.constexpr,
                            PH:tl.constexpr,PW:tl.constexpr,XT:tl.constexpr,T:tl.constexpr,
                            TAU:tl.constexpr,THRESHOLD:tl.constexpr,DECAY_INPUT:tl.constexpr,
                            HAS_BIAS:tl.constexpr,AFFINE:tl.constexpr,POST_ADD:tl.constexpr,
                            BM:tl.constexpr,BK:tl.constexpr):
    # Evaluate only frozen mask coordinates; no im2col or dense convolution output.
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    valid=row<COUNT
    point=tl.load(POINTS+row,valid,other=0)
    oc=point//(HO*WO)
    oh=point//WO%HO
    ow=point%WO
    k0=tl.arange(0,BK)
    if AFFINE:
        scale=tl.load(SCALE+oc,valid,other=0)
        offset=tl.load(OFFSET+oc,valid,other=0)
    v=tl.zeros((BM,),tl.float32)
    for t in tl.static_range(T):
        acc=tl.zeros((BM,),tl.float32)
        for start in range(0,CI*KH*KW,BK):
            k=start+k0
            ic=k//(KH*KW)
            ih=oh[:,None]*SH-PH+k[None,:]//KW%KH
            iw=ow[:,None]*SW-PW+k[None,:]%KW
            x=tl.load(X+t*XT+ic[None,:]*HI*WI+ih*WI+iw,
                       valid[:,None]&(k[None,:]<CI*KH*KW)&(ih>=0)&(ih<HI)&(iw>=0)&(iw<WI),other=0)
            weight=tl.load(W+oc[:,None]*(CI*KH*KW)+k[None,:],
                            valid[:,None]&(k[None,:]<CI*KH*KW),other=0)
            acc+=tl.sum(x.to(tl.float32)*weight.to(tl.float32),1)
        if HAS_BIAS:
            acc+=tl.load(BIAS+oc,valid,other=0)
        if AFFINE:
            acc=acc*scale+offset
        if DECAY_INPUT:
            v=v+(acc-v)/TAU
        else:
            v=v*(1-1/TAU)+acc
        fired=v>=THRESHOLD
        v=tl.where(fired,0.,v)
        y=fired.to(tl.float32)
        if POST_ADD:
            y+=tl.load(SKIP+t*CO*HO*WO+point,valid,other=0)
        tl.store(Y+t*CO*HO*WO+point,y,valid)


def _conv_copy(conv, out_index=None, in_index=None):
    if conv.groups != 1:
        raise ValueError('Only groups=1 is supported')
    weight = conv.weight.detach()
    bias = conv.bias.detach() if conv.bias is not None else None
    if out_index is not None:
        weight = weight[out_index]
        if bias is not None:
            bias = bias[out_index]
    if in_index is not None:
        weight = weight[:, in_index]
    result = nn.Conv2d(weight.shape[1], weight.shape[0], conv.kernel_size,
                        stride=conv.stride, padding=conv.padding,
                        dilation=conv.dilation, groups=1, bias=bias is not None,
                        device=weight.device, dtype=weight.dtype)
    result.weight = nn.Parameter(weight.contiguous().clone(), requires_grad=False)
    if bias is not None:
        result.bias = nn.Parameter(bias.clone(), requires_grad=False)
    return result.eval()


def _norm_copy(norm, index=None):
    if norm is None:
        return None
    result = nn.BatchNorm2d(len(index) if index is not None else norm.num_features,
                            eps=norm.eps, device=norm.weight.device, dtype=norm.weight.dtype)
    for key in ('weight','bias','running_mean','running_var'):
        value = getattr(norm, key).detach()
        if index is not None:
            value = value[index]
        getattr(result, key).data.copy_(value)
    return result.eval()


class FusedConv(nn.Module):
    def __init__(self, conv, norm, node=None, mask=None, *, T=4, static=False,
                 mode='triton', out_index=None, in_index=None):
        super().__init__()
        self.conv = _conv_copy(conv, out_index, in_index)
        self.norm = _norm_copy(norm, out_index)
        self.T, self.static = T, static
        self.has_node = node is not None
        self.mode = mode
        if node is not None:
            if node.v_reset != 0. or node.step_mode != 'm':
                raise ValueError('Only multistep hard-reset-to-zero LIF is supported')
            self.tau, self.threshold, self.decay_input = node.tau, node.v_threshold, node.decay_input
        if mask is not None:
            mask = (mask.detach() > 0).squeeze(0).squeeze(0)
            if out_index is not None:
                mask = mask[out_index]
            mask = mask.contiguous()
        self.register_buffer('mask', mask)
        self.register_buffer('points', torch.nonzero(mask.flatten()).flatten().to(torch.int32)
                             if mask is not None and mode == 'sparse' else None)
        self.register_buffer('scale', None)
        self.register_buffer('offset', None)
        if mode == 'fold_bn' and self.norm is not None:
            self.conv = torch.nn.utils.fusion.fuse_conv_bn_eval(self.conv, self.norm)
            self.norm = None
        elif mode in ('bn_lif','compact','sparse') and self.norm is not None and self.has_node:
            scale = self.norm.weight.detach() * torch.rsqrt(self.norm.running_var + self.norm.eps)
            self.scale = scale.contiguous()
            self.offset = (self.norm.bias.detach() - self.norm.running_mean*scale).contiguous()
            self.norm = None

    def sparse_forward(self,x,residual=None):
        x=x.contiguous()
        if self.static:
            n,ci,hi,wi=x.shape
        else:
            _,n,ci,hi,wi=x.shape
        sh,sw=self.conv.stride;ph,pw=self.conv.padding;kh,kw=self.conv.kernel_size
        ho=(hi+2*ph-kh)//sh+1;wo=(wi+2*pw-kw)//sw+1
        co=self.conv.out_channels
        if tuple(self.mask.shape)!=(co,ho,wo):
            raise ValueError('Spatial mask does not match this input resolution')
        y=residual.clone() if residual is not None else torch.zeros(
            (self.T,n,co,ho,wo),device=x.device,dtype=x.dtype)
        if self.points.numel():
            _sparse_conv_lif_kernel[(triton.cdiv(self.points.numel(),8),)](
                x,self.conv.weight,self.conv.bias if self.conv.bias is not None else x,
                self.points,self.scale if self.scale is not None else x,
                self.offset if self.offset is not None else x,y,
                residual if residual is not None else x,
                COUNT=self.points.numel(),CI=ci,HI=hi,WI=wi,CO=co,HO=ho,WO=wo,
                KH=kh,KW=kw,SH=sh,SW=sw,PH=ph,PW=pw,XT=0 if self.static else ci*hi*wi,
                T=self.T,TAU=self.tau,THRESHOLD=self.threshold,DECAY_INPUT=self.decay_input,
                HAS_BIAS=self.conv.bias is not None,AFFINE=self.scale is not None,
                POST_ADD=residual is not None,BM=8,BK=128,enable_fp_fusion=False)
        return y

    def forward(self, x, residual=None, pre_add=False):
        n=x.shape[0] if self.static else x.shape[1]
        if self.points is not None and n == 1 and not pre_add and self.conv.dilation == (1,1):
            return self.sparse_forward(x,residual)
        if self.static:
            flat = x
        else:
            t, n = x.shape[:2]
            flat = x.flatten(0, 1)
        y = F.conv2d(flat, self.conv.weight, self.conv.bias, self.conv.stride,
                     self.conv.padding, self.conv.dilation, self.conv.groups)
        if self.norm is not None:
            y = self.norm(y)
        if not self.static:
            y = y.view(t, n, *y.shape[1:])
        if not self.has_node:
            return y
        return lif_sequence(y, T=self.T, tau=self.tau, threshold=self.threshold,
                             decay_input=self.decay_input, mask=self.mask,
                             affine=(self.scale,self.offset) if self.scale is not None else None,
                             residual=residual, pre_add=pre_add, static=self.static)


def _from_sew(block, *, T, mode, out_index=None, in_index=None):
    return FusedConv(block.conv, block.norm, block.node, block.neuron_mask.mask_value,
                      T=T, static=block.static, mode=mode,
                      out_index=out_index, in_index=in_index)


class FusedBlock(nn.Module):
    def __init__(self, block, *, sew, T, mode):
        super().__init__()
        self.sew = sew
        self.constant = None
        self.shortcut = None
        if sew:
            if block.downsample is not None:
                self.shortcut = _from_sew(block.downsample, T=T, mode=mode)
            mask = block.conv1.neuron_mask.mask_value
            keep = None
            if mode in ('compact','sparse') and mask is not None:
                keep = torch.nonzero((mask > 0).flatten(3).any(-1).flatten()).flatten()
            if keep is not None and len(keep) == 0:
                # A zero intermediate activation can still produce spikes after
                # the next BN's offset. Cache that whole branch, rather than drop it.
                shape = (T,1,*mask.shape[2:])
                with torch.inference_mode():
                    y = block.conv2(torch.zeros(shape,device=mask.device,dtype=torch.float32))
                    from spikingjelly.activation_based import functional
                    functional.reset_net(block.conv2)
                self.register_buffer('constant_output',y.detach().clone())
                self.constant = True
                return
            self.conv1 = _from_sew(block.conv1,T=T,mode=mode,out_index=keep)
            self.conv2 = _from_sew(block.conv2,T=T,mode=mode,in_index=keep)
        else:
            if block.downsample is not None:
                self.shortcut = FusedConv(block.downsample[0],block.downsample[1],T=T,mode=mode)
            self.conv1 = FusedConv(block.conv1,block.bn1,block.sn1,T=T,mode=mode)
            self.conv2 = FusedConv(block.conv2,block.bn2,block.sn2,T=T,mode=mode)

    def forward(self,x):
        identity = self.shortcut(x) if self.shortcut is not None else x
        if self.constant:
            return identity + self.constant_output
        return self.conv2(self.conv1(x), residual=identity, pre_add=not self.sew)


class FusedResNet18(nn.Module):
    def __init__(self, model, *, sew, mode='triton'):
        super().__init__()
        self.T = model.T if sew else model.time_steps
        self.sew = sew
        source = model if sew else model.backbone
        if sew:
            self.stem = _from_sew(source.conv1,T=self.T,mode=mode)
        else:
            # Keep the original [T*N,...] convolution shape in the post-add stem.
            self.stem = FusedConv(source.conv1,source.bn1,source.sn1,T=self.T,mode=mode)
        self.blocks = nn.ModuleList([FusedBlock(b,sew=sew,T=self.T,mode=mode)
                                    for stage in (source.layer1,source.layer2,source.layer3,source.layer4)
                                    for b in stage])
        self.fc = source.fc
        self.boost = source.boost if sew else None
        self.eval()

    def forward(self,x):
        if self.training:
            raise RuntimeError('This graph is inference-only; call eval()')
        if not x.is_cuda or x.dtype != torch.float32:
            raise ValueError('Fused inference requires CUDA float32 inputs')
        if not self.sew:
            x = x.unsqueeze(0).repeat(self.T,1,1,1,1)
        x = self.stem(x)
        t,n = x.shape[:2]
        x = F.max_pool2d(x.flatten(0,1),3,2,1)
        x = x.view(t,n,*x.shape[1:])
        for block in self.blocks:
            x = block(x)
        x = F.adaptive_avg_pool2d(x.flatten(0,1),1).flatten(1)
        x = F.linear(x,self.fc.weight,self.fc.bias)
        if self.boost is not None:
            x = F.avg_pool1d(x.unsqueeze(1),10,10).squeeze(1)
        return x.view(t,n,-1).mean(0)


def optimize(model, *, sew=False, mode='triton'):
    """Return an inference graph; use it under torch.inference_mode().

    triton: LIF/mask/residual fusion, BN unchanged.
    bn_lif: also apply frozen BN affine inside the neuron kernel.
    fold_bn: fold BN into convolution weights (different FP32 rounding).
    compact: bn_lif plus removal of dead intermediate channels/constant branches.
    sparse: compact plus BS=1 active-output Conv/BN/LIF kernels; larger batches
            fall back to the compact dense-convolution graph.
    """
    if model.training:
        raise ValueError('Load checkpoint and call eval() before graph conversion')
    if mode not in ('triton','bn_lif','fold_bn','compact','sparse'):
        raise ValueError(mode)
    if not sew and not hasattr(model,'backbone'):
        result = copy.deepcopy(model)
        result.conv1 = torch.nn.utils.fusion.fuse_conv_bn_eval(result.conv1,result.bn1)
        result.bn1 = nn.Identity()
        for stage in (result.layer1,result.layer2,result.layer3,result.layer4):
            for block in stage:
                for i in (1,2):
                    setattr(block,f'conv{i}',torch.nn.utils.fusion.fuse_conv_bn_eval(
                        getattr(block,f'conv{i}'),getattr(block,f'bn{i}')))
                    setattr(block,f'bn{i}',nn.Identity())
                if block.downsample is not None:
                    block.downsample[0] = torch.nn.utils.fusion.fuse_conv_bn_eval(
                        block.downsample[0],block.downsample[1])
                    block.downsample[1] = nn.Identity()
        return result.eval()
    return FusedResNet18(model,sew=sew,mode=mode)
