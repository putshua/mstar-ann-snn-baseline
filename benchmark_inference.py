"""Full SOC accuracy and BS=1 inference ablations for fused ResNet graphs."""
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

from analyze import calculate, save_json, save_figure
from benchmark import load_model, forward
from inference import optimize
from mstar_data import transforms_for

ROOT=Path(__file__).resolve().parent


def tensor_storage(model):
    parameters=list(model.parameters());buffers=list(model.buffers())
    return {'parameter_elements':sum(t.numel() for t in parameters),
            'buffer_elements':sum(t.numel() for t in buffers),
            'model_tensor_bytes':sum(t.numel()*t.element_size() for t in parameters+buffers)}


def plot_results(results,output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    groups=list(dict.fromkeys(r['checkpoint_group'] for r in results.values()))
    markers={'baseline':'x','triton':'^','bn_lif':'o','fold_bn':'s','compact':'D','sparse':'P'}
    colors=plt.get_cmap('tab10').colors
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharey=True,layout='constrained')
    for index,key in enumerate(groups):
        for result in results.values():
            if result['checkpoint_group']!=key:
                continue
            for ax in axes:
                ax.scatter(result['latency_bs1']['mean_ms'],100*result['summary']['accuracy'],
                           color=colors[index],marker=markers[result['variant']],s=55,alpha=.9)
    axes[0].set(xlabel='BS=1 mean inference time (ms)',ylabel='SOC accuracy (%)',
                title='Original and fused inference')
    axes[1].set(xlabel='BS=1 mean inference time (ms)',xlim=(.85,2.55),
                title='Fused variants (zoom)')
    for ax in axes:
        ax.grid(alpha=.2)
    handles=[Line2D([],[],color=colors[i],marker='o',ls='',label=key) for i,key in enumerate(groups)]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.07),ncol=6,fontsize=9)
    handles=[Line2D([],[],color='#444444',marker=mark,ls='',label=name) for name,mark in markers.items()]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,-.055),ncol=6,fontsize=9)
    save_figure(fig,output,'accuracy_latency')


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--uploaded-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/inference')
    p.add_argument('--cpu-core',type=int)
    p.add_argument('--warmup',type=int,default=100)
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--timing-images',type=int,default=256)
    args=p.parse_args()
    if args.cpu_core is not None:os.sched_setaffinity(0,{args.cpu_core})
    torch.set_num_threads(1);torch.manual_seed(0)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    args.output.mkdir(parents=True,exist_ok=True)
    specs=[('ann',ROOT/'checkpoints/ann_resnet18.pt',('fold_bn',)),
           ('snn',ROOT/'checkpoints/snn_resnet18_lif_t4.pt',('triton','bn_lif','fold_bn')),
           ('sew_dense',args.uploaded_dir/'sew_resnet18_mstar_dense/checkpoint_final.pth',('triton','bn_lif','fold_bn'))]
    for tag in ('5e-6','1e-5','2e-5'):
        modes=('triton','bn_lif','fold_bn','compact','sparse') if tag=='5e-6' else ('bn_lif','compact','sparse')
        specs.append(('pruned_'+tag,args.uploaded_dir/f'sew_resnet18_mstar_{tag}/checkpoint_final.pth',modes))
    models=[];results={};images=None;labels=None;classes=None
    report={'environment':{'gpu':torch.cuda.get_device_name(0),'torch':str(torch.__version__),
                          'triton':importlib.metadata.version('triton'),'spikingjelly':importlib.metadata.version('spikingjelly'),
                          'cpu_threads':1,'cpu_affinity':sorted(os.sched_getaffinity(0)),
                          'cudnn_deterministic':True,'cudnn_benchmark':False,
                          'tf32_cudnn':torch.backends.cudnn.allow_tf32,
                          'tf32_matmul':torch.backends.cuda.matmul.allow_tf32},
            'source':{'repository':'https://github.com/putshua/spformers','revision':'f5bfa1d9fd6e13c61802929c175397f57746ca39'},
            'protocol':{'dtype':'float32','classification_batch_size':1,'latency_batch_size':1,'T':4,
                        'timing':'Synchronized wall clock, full forward plus temporal readout and state reset. Excludes IO, preprocessing and H2D.',
                        'order':'Seed-0 shuffled model order per image; all models interleaved.',
                        'equations':'v += (x-v)/tau, threshold >= 1, hard reset 0, initial v=0 per image.',
                        'runtime_storage':'Stored parameter/buffer elements in the inference graph; distinct from trained parameter count.'},
            'models':results}
    for key,path,modes in specs:
        original,meta,sew,cfg=load_model(path)
        if images is None:
            ds=datasets.ImageFolder(args.data,transforms_for(meta['mean'],meta['std']))
            images=torch.stack([ds[i][0] for i in range(len(ds))]).cuda()
            labels=np.asarray(ds.targets);classes=np.asarray(ds.classes)
            reference_meta=(meta['mean'],meta['std'],meta['classes'])
        elif (meta['mean'],meta['std'],meta['classes'])!=reference_meta:
            raise ValueError('These models require different input preprocessing')
        cases=[('baseline',original,lambda x,m=original,s=sew:forward(m,x,s))]
        for mode in modes:
            candidate=optimize(original,sew=sew,mode=mode)
            cases.append((mode,candidate,lambda x,m=candidate:m(x)))
        for mode,model,run in cases:
            name=key+'/'+mode
            print('Evaluating',name,flush=True)
            # Compile the exact BS=1 graph before accuracy collection and timing.
            run(images[:1]);torch.cuda.synchronize()
            logits=[]
            for index in range(len(images)):
                logits.append(run(images[index:index+1]).cpu())
            logits=torch.cat(logits).numpy()
            score={'labels':labels,'classes':classes,
                   'probabilities':torch.from_numpy(logits).double().softmax(1).numpy(),
                   'metadata':'{}'}
            result=calculate(name,score);result.pop('provenance',None)
            prob=score['probabilities']
            result['summary']['top5_accuracy']=float(np.mean(np.any(np.argsort(prob,axis=1)[:,-5:]==labels[:,None],axis=1)))
            result.update({'checkpoint_group':key,'variant':mode,'configuration':cfg,
                           'runtime_storage':tensor_storage(model)})
            results[name]=result
            models.append((name,run))
            save_json(args.output/'metrics.json',report)
            print(name,'accuracy',result['summary']['accuracy'],'parameters',result['runtime_storage']['parameter_elements'],flush=True)
    rng=np.random.default_rng(0)
    groups=[rng.permutation(np.flatnonzero(labels==i)) for i in range(10)]
    indices=[int(groups[i][j]) for j in range(min(map(len,groups))) for i in range(10)][:args.timing_images]
    samples=images[indices]
    for i in range(args.warmup):
        x=samples[i%len(samples):i%len(samples)+1]
        for j in rng.permutation(len(models)):models[j][1](x)
    torch.cuda.synchronize()
    times={name:[] for name,_ in models};gpu={name:[] for name,_ in models};trials={name:[] for name,_ in models}
    start_event=torch.cuda.Event(enable_timing=True);end_event=torch.cuda.Event(enable_timing=True)
    for trial in range(args.repeats):
        values={name:[] for name,_ in models}
        for image in samples:
            x=image.unsqueeze(0)
            for j in rng.permutation(len(models)):
                name,run=models[j]
                torch.cuda.synchronize();start=time.perf_counter();start_event.record()
                run(x)
                end_event.record();torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000
                times[name].append(ms);values[name].append(ms);gpu[name].append(start_event.elapsed_time(end_event))
        for name in trials:trials[name].append(float(np.mean(values[name])))
        print('Latency trial',trial+1,{k:round(trials[k][-1],3) for k in trials},flush=True)
    for name,result in results.items():
        result['latency_bs1']={'mean_ms':float(np.mean(times[name])),
                               'median_ms':float(np.median(times[name])),
                               'p95_ms':float(np.percentile(times[name],95)),
                               'std_ms':float(np.std(times[name])),
                               'cuda_event_mean_ms':float(np.mean(gpu[name])),
                               'trial_mean_ms':trials[name],
                               'images_per_trial':len(samples),'repeats':args.repeats,'warmup':args.warmup}
    for name,result in results.items():
        baseline=results[result['checkpoint_group']+'/baseline']
        result['latency_bs1']['speedup_mean']=baseline['latency_bs1']['mean_ms']/result['latency_bs1']['mean_ms']
    save_json(args.output/'metrics.json',report)
    rows=[{'model':name,**r['summary'],**r['runtime_storage'],
           **{f'latency_bs1_{k}':v for k,v in r['latency_bs1'].items() if k in ('mean_ms','median_ms','p95_ms','cuda_event_mean_ms','speedup_mean')}}
          for name,r in results.items()]
    for filename,data in [('summary.csv',rows),('per_class.csv',[{'model':name,**row} for name,r in results.items() for row in r['per_class']])]:
        with (args.output/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
    plot_results(results,args.output)
    print('Saved',args.output,flush=True)


if __name__=='__main__':main()
