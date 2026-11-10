"""MSTAR SOC classification metrics and synchronized batch-one CUDA latency."""
import argparse
import csv
import importlib.metadata
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets
from spikingjelly.activation_based import functional, neuron

from analyze import calculate, save_json, save_figure
from mstar_data import transforms_for
from train import build_model, loader
from pruning.models.sew_resnet import sew_resnet18
from pruning.models.submodules.sparse import ConvBlock
from pruning.utils import SOPMonitor

ROOT = Path(__file__).resolve().parent


def load_model(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    if 'arch' in cfg:
        model = build_model(cfg['arch'], False, cfg['stem'], cfg.get('time_steps', 4),
                            cfg.get('lif_tau', 2.0))
        meta = {key: cfg[key] for key in ('mean', 'std', 'classes')}
        sew = False
    else:
        model = sew_resnet18(T=cfg['T'], num_classes=10, in_channels=1,
                             zero_init_residual=cfg.get('zero_init_residual', False))
        for name, module in model.named_modules():
            if isinstance(module, ConvBlock):
                module.sparse_neurons = not cfg['not_prune_neuron']
                key = name + '.neuron_mask.mask_value'
                if key in ckpt['model']:
                    module.neuron_mask.mask_value = torch.nn.Parameter(
                        torch.empty_like(ckpt['model'][key]))
                module._pruning(False)
        meta = ckpt['mstar_metadata']
        sew = True
    model.load_state_dict(ckpt['model'], strict=True)
    config_keys = ('seed', 'arch', 'model', 'epochs', 'epoch_search', 'epoch_finetune',
                   'time_steps', 'T', 'lif_tau', 'penalty_lmbda', 'not_prune_neuron',
                   'optimizer', 'lr', 'search_lr', 'finetune_lr', 'weight_decay',
                   'batch_size', 'augment', 'no_augment')
    config = {key: cfg[key] for key in config_keys if key in cfg}
    return model.cuda().eval(), meta, sew, config


def forward(model, x, sew):
    output = model(x)
    if sew:
        output = output.mean(0)
    functional.reset_net(model)
    return output


def model_size(model, sample, sew):
    units = []
    def capture(module, inputs, output):
        units.append(output[0, 0].numel())
    nodes = [m for m in model.modules() if isinstance(m, neuron.LIFNode)]
    hooks = [m.register_forward_hook(capture) for m in nodes]
    forward(model, sample, sew)
    for hook in hooks:
        hook.remove()
    total = sum(units)
    masks = [m.neuron_mask.mask_value for m in model.modules()
             if isinstance(m, ConvBlock) and m.neuron_mask.mask_value is not None]
    mask_count = sum(p.numel() for p in masks)
    remaining = sum(int((p > 0).sum()) for p in masks) if masks else total
    count = sum(p.numel() for p in model.parameters())
    return {'parameters_network': count - mask_count, 'parameters_masks': mask_count,
            'parameters_total': count, 'lif_modules': len(nodes),
            'lif_units_total': total, 'lif_units_remaining': remaining,
            'lif_units_remaining_fraction': remaining / total if total else None}


@torch.inference_mode()
def latency_all(entries, data, args):
    models = []
    for key, _, path in entries:
        model, meta, sew, _ = load_model(path)
        models.append((key, model, sew))
    ds = datasets.ImageFolder(data, transforms_for(meta['mean'], meta['std']))
    targets = np.asarray(ds.targets)
    rng = np.random.default_rng(0)
    groups = [rng.permutation(np.flatnonzero(targets == i)) for i in range(10)]
    indices = [int(groups[i][j]) for j in range(min(map(len, groups))) for i in range(10)][:args.timing_images]
    images = torch.stack([ds[i][0] for i in indices]).cuda()
    for i in range(args.warmup):
        x = images[i % len(images):i % len(images) + 1]
        for index in rng.permutation(len(models)):
            _, model, sew = models[index]
            forward(model, x, sew)
    torch.cuda.synchronize()
    times = {key: [] for key, _, _ in models}
    gpu_times = {key: [] for key, _, _ in models}
    trials = {key: [] for key, _, _ in models}
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for trial in range(args.repeats):
        current = {key: [] for key, _, _ in models}
        for image in images:
            x = image.unsqueeze(0)
            for index in rng.permutation(len(models)):
                key, model, sew = models[index]
                torch.cuda.synchronize()
                start = time.perf_counter()
                start_event.record()
                forward(model, x, sew)
                end_event.record()
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
                times[key].append(elapsed)
                current[key].append(elapsed)
                gpu_times[key].append(start_event.elapsed_time(end_event))
        for key in times:
            trials[key].append(float(np.mean(current[key])))
        print('Latency trial', trial + 1, {key: round(trials[key][-1], 4) for key in times}, flush=True)
    results = {}
    for key, _, _ in models:
        values = times[key]
        results[key] = {'batch_size': 1, 'mean_ms': float(np.mean(values)),
                        'median_ms': float(np.median(values)),
                        'p95_ms': float(np.percentile(values, 95)),
                        'std_ms': float(np.std(values)), 'trial_mean_ms': trials[key],
                        'cuda_event_mean_ms': float(np.mean(gpu_times[key])),
                        'cuda_event_median_ms': float(np.median(gpu_times[key])),
                        'cuda_event_p95_ms': float(np.percentile(gpu_times[key], 95)),
                        'images_per_trial': len(images), 'repeats': args.repeats,
                        'warmup_iterations_per_model': args.warmup,
                        'model_order': 'Seed-0 shuffled order for each image; all models interleaved.'}
    return results


def plots(results, scores, output):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    ncols = min(4, len(results)); nrows = (len(results) + ncols - 1) // ncols
    figs = [plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows),
                          squeeze=False, layout='constrained') for _ in range(3)]
    colors = plt.get_cmap('tab10').colors
    for pos, (key, result) in enumerate(results.items()):
        axes = [a.flat[pos] for _, a in figs]
        cm = np.asarray(result['confusion_matrix'])
        ax = axes[0]
        ax.imshow(cm/cm.sum(1, keepdims=True), cmap='Blues', vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(cm):
            ax.text(j, i, str(value), ha='center', va='center', fontsize=7,
                    color='white' if i == j else '#24344a')
        ax.set(xticks=range(10), yticks=range(10), xlabel='Predicted', ylabel='True')
        ax.set_xticklabels(result['classes'], rotation=60, ha='right')
        ax.set_yticklabels(result['classes'])
        y, p = scores[key]['labels'], scores[key]['probabilities']
        onehot = np.eye(10, dtype=int)[y]
        for i, name in enumerate(result['classes']):
            fpr,tpr,_ = metrics.roc_curve(onehot[:,i],p[:,i])
            axes[1].plot(fpr,tpr,color=colors[i],lw=1,label=name)
            precision,recall,_ = metrics.precision_recall_curve(onehot[:,i],p[:,i],drop_intermediate=True)
            axes[2].step(recall,precision,where='post',color=colors[i],lw=1,label=name)
        fpr,tpr,_=metrics.roc_curve(onehot.ravel(),p.ravel())
        axes[1].plot(fpr,tpr,'k--',lw=1.5,label='micro')
        precision,recall,_=metrics.precision_recall_curve(onehot.ravel(),p.ravel(),drop_intermediate=True)
        axes[2].step(recall,precision,where='post',color='black',ls='--',lw=1.5,label='micro')
        for ax in axes:
            ax.set_title(key)
        for ax in axes[1:]:
            ax.set(xlim=(0,1),ylim=(0,1.005));ax.grid(alpha=.2)
            ax.legend(fontsize=6,ncol=2,loc='lower right' if ax is axes[1] else 'lower left')
        axes[1].set(xlabel='False positive rate',ylabel='True positive rate')
        axes[2].set(xlabel='Recall',ylabel='Precision')
    for (fig,axes),name in zip(figs,('confusion_matrices','roc_curves','precision_recall_curves')):
        for ax in axes.flat[len(results):]:
            ax.set_visible(False)
        save_figure(fig,output,name)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True,help='SOC test directory')
    parser.add_argument('--checkpoints',type=Path,default=ROOT/'checkpoints')
    parser.add_argument('--uploaded-dir',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'results/analysis')
    parser.add_argument('--warmup',type=int,default=100)
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--timing-images',type=int,default=256)
    parser.add_argument('--cpu-core',type=int,help='Pin the benchmark process to one available CPU')
    parser.add_argument('--latency-only',action='store_true',help='Update timing in an existing metrics.json')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        parser.error('A CUDA GPU is required for this latency protocol')
    if min(args.warmup,args.repeats,args.timing_images)<1:
        parser.error('warmup, repeats and timing-images must be positive')
    if args.cpu_core is not None:
        os.sched_setaffinity(0,{args.cpu_core})
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    entries=[('ann','ANN ResNet-18',args.checkpoints/'ann_resnet18.pt'),
             ('snn','Post-add LIF ResNet-18',args.checkpoints/'snn_resnet18_lif_t4.pt'),
             ('sew_dense','SEW ResNet-18, local',args.checkpoints/'sew_resnet18_dense.pt'),
             ('sew_pruned','SEW pruned 1e-5, local',args.checkpoints/'sew_resnet18_pruned.pt')]
    if args.uploaded_dir:
        for tag in ('dense','5e-6','1e-5','2e-5'):
            entries.append((f'uploaded_{tag}',f'SEW {tag}, uploaded',
                            args.uploaded_dir/f'sew_resnet18_mstar_{tag}'/'checkpoint_final.pth'))
    args.output.mkdir(parents=True,exist_ok=True)
    environment={key:importlib.metadata.version(key) for key in
                 ('torch','torchvision','spikingjelly','numpy','scikit-learn','matplotlib','cupy-cuda13x')}
    environment.update({'gpu':torch.cuda.get_device_name(0),'cpu_threads':1,
                        'cpu_affinity':sorted(os.sched_getaffinity(0))})
    protocol={'dataset':'MSTAR SOC, 10 classes, 2425 test images, 15 degrees',
              'input':[1,128,128],'dtype':'float32','classification_batch_size':64,
              'latency_batch_size':1,'snn_time_steps':4,
              'timing':'Synchronized wall-clock model forward, time readout and neuron reset; excludes image IO, preprocessing and host-to-device transfer.',
              'cuda_event_timing':'GPU timeline interval for the same forward and reset; may include gaps between kernel launches.',
              'latency_inputs':'Fixed seed-0 class-interleaved test subset, preloaded on GPU; models run in shuffled order for each image.',
              'backends':{'ann':'torch',
                          'snn':'SpikingJelly eval TorchScript',
                          'SEW':'SpikingJelly eval TorchScript; CuPy configured for training'}}
    results=(json.loads((args.output/'metrics.json').read_text())['models']
             if args.latency_only else {})
    scores={}
    for key,name,path in ([] if args.latency_only else entries):
        print('Evaluating',key,flush=True)
        model,meta,sew,config=load_model(path)
        ds=datasets.ImageFolder(args.data,transforms_for(meta['mean'],meta['std']))
        if ds.classes!=meta['classes']:
            raise ValueError('Dataset class order differs from checkpoint')
        targets=np.asarray(ds.targets)
        rng=np.random.default_rng(0)
        groups=[rng.permutation(np.flatnonzero(targets==i)) for i in range(10)]
        indices=[int(groups[i][j]) for j in range(min(map(len,groups))) for i in range(10)][:args.timing_images]
        images=torch.stack([ds[i][0] for i in indices]).cuda()
        size=model_size(model,images[:1],sew)
        monitor=SOPMonitor(model) if sew else None
        if monitor is not None:
            monitor.enable()
        logits=[];labels=[]
        for x,y in loader(ds,64,4,cuda=True):
            logits.append(forward(model,x.cuda(non_blocking=True),sew).cpu().numpy());labels.append(y.numpy())
        logits=np.concatenate(logits);labels=np.concatenate(labels)
        probabilities=torch.from_numpy(logits).double().softmax(1).numpy()
        score={'labels':labels,'probabilities':probabilities,'classes':np.asarray(ds.classes),
               'metadata':json.dumps({'configuration':config,'preprocessing':meta})}
        record=calculate(key,score);record['model']=name
        record['summary']['top5_accuracy']=float(np.mean(np.any(np.argsort(probabilities,axis=1)[:,-5:]==labels[:,None],axis=1)))
        record['model_size']=size
        if monitor is not None:
            record['sops_g_per_image']=sum(v.item() for v in monitor.records)/1e9/len(ds)
            record['sops_scope']='Spike-driven convolutions; static stem and fc excluded.'
            monitor.remove_hooks()
        record['configuration']=config;record['preprocessing']=meta;record.pop('provenance',None)
        results[key]=record;scores[key]=score
        save_json(args.output/'metrics.json',{'environment':environment,'protocol':protocol,'models':results})
        print(key,'accuracy',record['summary']['accuracy'],flush=True)
        del model,images;torch.cuda.empty_cache()
    timings = latency_all(entries, args.data, args)
    for key, timing in timings.items():
        results[key]['latency_bs1'] = timing
    save_json(args.output/'metrics.json', {'environment':environment, 'protocol':protocol, 'models':results})
    rows=[{'model':key,**r['summary'],**r['model_size'],'sops_g_per_image':r.get('sops_g_per_image'),
           **{f'latency_bs1_{k}':v for k,v in r['latency_bs1'].items()
              if k in ('mean_ms','median_ms','p95_ms','std_ms','cuda_event_mean_ms',
                        'cuda_event_median_ms','cuda_event_p95_ms')}}
          for key,r in results.items()]
    for filename,data in [('summary.csv',rows),('per_class.csv',[{'model':key,**row} for key,r in results.items() for row in r['per_class']])]:
        with (args.output/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');writer.writeheader();writer.writerows(data)
    if not args.latency_only:
        plots(results,scores,args.output)
    print('Results:',args.output,flush=True)


if __name__=='__main__':
    main()
