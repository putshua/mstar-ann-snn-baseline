"""Classification metrics and a five-run CUDA Event inference timer."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from spikingjelly.activation_based import functional, neuron

from train import build_model
from mstar_data import transforms_for
from pruning.models import sew_resnet
from pruning.models.submodules.sparse import ConvBlock
from inference import optimize

ROOT=Path(__file__).resolve().parent


def load_model(path):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    cfg=checkpoint['config']
    if 'arch' in cfg:
        if cfg['arch'] not in ('resnet18','resnet34'):
            raise ValueError('This benchmark accepts ANN ResNet-18/34 or SEW weights')
        model=build_model(cfg['arch'],False,cfg['stem'])
        meta={key:cfg[key] for key in ('mean','std','classes')};sew=False
    else:
        architecture=cfg.get('model','sew_resnet18').removesuffix('_imagenet')
        if architecture not in ('sew_resnet18','sew_resnet34'):
            raise ValueError('This benchmark accepts SEW-ResNet-18/34')
        model=getattr(sew_resnet,architecture)(T=cfg['T'],zero_init_residual=cfg.get('zero_init_residual',False))
        for name,module in model.named_modules():
            if isinstance(module,ConvBlock):
                module.sparse_neurons=not cfg['not_prune_neuron']
                key=name+'.neuron_mask.mask_value'
                if key in checkpoint['model']:
                    module.neuron_mask.mask_value=torch.nn.Parameter(torch.empty_like(checkpoint['model'][key]))
                module._pruning(False)
        meta=checkpoint['mstar_metadata'];sew=True
    model.load_state_dict(checkpoint['model'],strict=True)
    return model.cuda().eval(),meta,sew,cfg


def reference(model,x,sew):
    output=model(x)
    if sew:output=output.mean(0)
    functional.reset_net(model)
    return output


def neuron_counts(model,x,sew):
    counts=[]
    def count(module,inputs,output):
        counts.append(output[0,0].numel() if sew else output[0].numel())
    nodes=[m for m in model.modules() if isinstance(m,neuron.LIFNode if sew else torch.nn.ReLU)]
    hooks=[m.register_forward_hook(count) for m in nodes]
    try:reference(model,x,sew)
    finally:
        for hook in hooks:hook.remove()
    masks=[m.neuron_mask.mask_value for m in model.modules()
           if isinstance(m,ConvBlock) and m.neuron_mask.mask_value is not None]
    total=sum(counts);remaining=sum(int((m>0).sum()) for m in masks) if masks else total
    mask_parameters=sum(m.numel() for m in masks)
    return {'type':'LIF' if sew else 'ReLU','activation_sites':len(counts),
            'total':total,'remaining':remaining,'remaining_fraction':remaining/total,
            'network_parameters':sum(p.numel() for p in model.parameters())-mask_parameters,
            'mask_parameters':mask_parameters}


def classification(logits,labels):
    from sklearn import metrics
    p=torch.from_numpy(logits).double().softmax(1).numpy()
    y=labels;pred=p.argmax(1);onehot=np.eye(10,dtype=int)[y]
    cm=metrics.confusion_matrix(y,pred,labels=np.arange(10))
    precision,recall,f1,support=metrics.precision_recall_fscore_support(y,pred,labels=np.arange(10),zero_division=0)
    auc=np.array([metrics.roc_auc_score(onehot[:,i],p[:,i]) for i in range(10)])
    ap=metrics.average_precision_score(onehot,p,average=None)
    confidence=p.max(1);correct=pred==y
    bins=np.clip(np.ceil(confidence*15).astype(int)-1,0,14)
    ece=sum(np.mean(bins==b)*abs(correct[bins==b].mean()-confidence[bins==b].mean()) for b in range(15) if np.any(bins==b))
    summary={'accuracy':float(np.mean(correct)),
             'top3_accuracy':float(metrics.top_k_accuracy_score(y,p,k=3,labels=np.arange(10))),
             'top5_accuracy':float(metrics.top_k_accuracy_score(y,p,k=5,labels=np.arange(10))),
             'balanced_accuracy':float(metrics.balanced_accuracy_score(y,pred)),
             'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),'macro_f1':float(f1.mean()),
             'weighted_f1':float(np.average(f1,weights=support)),
             'roc_auc_macro_ovr':float(auc.mean()),'roc_auc_micro_ovr':float(metrics.roc_auc_score(onehot.ravel(),p.ravel())),
             'map_macro':float(ap.mean()),'ap_micro':float(metrics.average_precision_score(onehot,p,average='micro')),
             'cohen_kappa':float(metrics.cohen_kappa_score(y,pred)),'mcc':float(metrics.matthews_corrcoef(y,pred)),
             'log_loss':float(metrics.log_loss(y,p,labels=np.arange(10))),
             'brier_multiclass':float(np.mean(np.sum((p-onehot)**2,axis=1))),'ece_15_bins':float(ece)}
    tp=cm.diagonal();fp=cm.sum(0)-tp;fn=cm.sum(1)-tp;tn=len(y)-tp-fp-fn
    per_class=[{'class_index':i,'support':int(support[i]),'precision':float(precision[i]),
                'recall':float(recall[i]),'f1':float(f1[i]),'specificity':float(tn[i]/(tn[i]+fp[i])),
                'roc_auc_ovr':float(auc[i]),'ap_ovr':float(ap[i])} for i in range(10)]
    return {'metrics':summary,'per_class':per_class,'confusion_matrix':cm.tolist()}


def measure(run,x,warmup,runs,iterations):
    for _ in range(warmup):run(x)
    torch.cuda.synchronize()
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
    values=[]
    for _ in range(runs):
        start.record()
        for _ in range(iterations):run(x)
        end.record();end.synchronize()
        values.append(start.elapsed_time(end)/iterations)
    return {'mean_ms':float(np.mean(values)),'std_ms':float(np.std(values,ddof=1)) if len(values)>1 else 0.,'run_mean_ms':values,
            'timer':'CUDA Event','iterations_per_run':iterations}


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='SOC test directory')
    p.add_argument('--uploaded-dir',type=Path)
    p.add_argument('--ann-checkpoint',type=Path,help='Evaluate a supplied ANN checkpoint')
    p.add_argument('--sew-checkpoint',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results')
    p.add_argument('--evaluate',action='store_true',help='Recompute full test metrics once; otherwise reuse saved metrics')
    p.add_argument('--runs',type=int,default=5)
    p.add_argument('--iterations',type=int,default=100)
    p.add_argument('--warmup',type=int,default=20)
    args=p.parse_args()
    if min(args.runs,args.iterations,args.warmup)<1:p.error('Counts must be positive')
    paired=args.ann_checkpoint is not None or args.sew_checkpoint is not None
    if not paired and args.uploaded_dir is None:p.error('--uploaded-dir is required unless a checkpoint pair is supplied')
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    path=args.output/'metrics.json'
    cache=json.loads(path.read_text()).get('models',{}) if path.exists() else {}
    if paired:
        specs=[]
        if args.ann_checkpoint:specs.append(('ann',args.ann_checkpoint,()))
        if args.sew_checkpoint:specs.append(('sew',args.sew_checkpoint,()))
    else:
        specs=[('ann',ROOT/'checkpoints/ann_resnet18.pt',('fold_bn',)),
               ('sew_dense',args.uploaded_dir/'sew_resnet18_mstar_dense/checkpoint_final.pth',('bn_lif',))]
        for tag in ('5e-6','1e-5','2e-5'):
            specs.append((f'pruned_{tag}',args.uploaded_dir/f'sew_resnet18_mstar_{tag}/checkpoint_final.pth',('bn_lif','sparse')))
    results={};images=None;shared=None;ds=None
    for group,checkpoint,modes in specs:
        original,meta,sew,cfg=load_model(checkpoint)
        if paired:group=cfg.get('arch',cfg.get('model')).removesuffix('_imagenet')
        signature=(meta['mean'],meta['std'],meta['classes'])
        if shared is None:
            shared=signature;ds=datasets.ImageFolder(args.data,transforms_for(meta['mean'],meta['std']))
            if ds.classes!=meta['classes']:raise ValueError('Dataset class order differs from checkpoint')
            x=ds[0][0].unsqueeze(0).cuda()
            if args.evaluate:images=torch.stack([ds[i][0] for i in range(len(ds))]).cuda()
        elif shared!=signature:raise ValueError('Models have different preprocessing')
        neurons=neuron_counts(original,x,sew)
        cases=[('baseline',original,lambda x,m=original,s=sew:reference(m,x,s))]
        for mode in modes:
            model=optimize(original,sew=sew,mode=mode)
            cases.append((mode,model,lambda x,m=model:m(x)))
        for mode,model,run in cases:
            name=group+'/'+mode
            if args.evaluate:
                logits=torch.cat([run(images[i:i+1]).cpu() for i in range(len(images))]).numpy()
                result=classification(logits,np.asarray(ds.targets))
                for row in result['per_class']:row['class']=ds.classes[row.pop('class_index')]
            elif name in cache:
                result={key:cache[name][key] for key in ('metrics','per_class','confusion_matrix')}
            else:p.error(f'No saved metrics for {name}; run once with --evaluate')
            result['parameters']=sum(v.numel() for v in model.parameters())
            result['neurons']=neurons
            result['inference']=measure(run,x,args.warmup,args.runs,args.iterations)
            results[name]=result
            print(name,'accuracy',result['metrics']['accuracy'],'ms',result['inference']['mean_ms'],flush=True)
        del cases,model,original
    report={'gpu':torch.cuda.get_device_name(0),'dtype':'float32','batch_size':1,'snn_time_steps':4,
            'test_images':len(ds),'runs':args.runs,
            'std_definition':'Sample standard deviation (ddof=1) across run means.','models':results}
    args.output.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    rows=[{'model':name,**r['metrics'],'parameters':r['parameters'],
           'neurons_total':r['neurons']['total'],'neurons_remaining':r['neurons']['remaining'],
           'neurons_remaining_fraction':r['neurons']['remaining_fraction'],
           'inference_time_ms':r['inference']['mean_ms'],'inference_time_std_ms':r['inference']['std_ms']} for name,r in results.items()]
    with (args.output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


if __name__=='__main__':main()
