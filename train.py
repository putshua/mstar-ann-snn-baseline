"""Single-device ResNet baseline; test data is evaluated only after training."""
import argparse
import csv
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models

from mstar_data import normalization_stats, split_indices, transforms_for


def build_model(arch='resnet18', pretrained=False, stem='standard', time_steps=4, lif_tau=2.0):
    if arch == 'spiking_resnet18':
        if pretrained or stem != 'standard':
            raise ValueError('SNN baseline uses scratch initialization and the standard stem')
        from snn import LIFResNet18
        return LIFResNet18(build_model('resnet18'), time_steps=time_steps, tau=lif_tau)
    weights = None
    if pretrained:
        weights = models.ResNet18_Weights.DEFAULT if arch == 'resnet18' else models.ResNet34_Weights.DEFAULT
    model = getattr(models, arch)(weights=weights)
    old = model.conv1
    if stem == 'standard':
        model.conv1 = nn.Conv2d(1, 64, 7, stride=2, padding=3, bias=False)
        if pretrained:
            with torch.no_grad():
                model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
    else:
        if pretrained:
            raise ValueError('Small stem is a scratch-only ablation; use standard for ImageNet weights')
        model.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    if not pretrained:
        nn.init.kaiming_normal_(model.conv1.weight, mode='fan_out', nonlinearity='relu')
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def loader(ds, batch_size, workers, shuffle=False, cuda=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=cuda, persistent_workers=workers > 0)


def reset_model_state(model):
    reset = getattr(model, 'reset_state', None)
    if reset is not None:
        reset()


@torch.inference_mode()
def evaluate(model, batches, device):
    model.eval()
    loss, total = 0.0, 0
    cm = torch.zeros(10, 10, dtype=torch.int64)
    predictions = []
    for x, y in batches:
        logits = model(x.to(device, non_blocking=True))
        loss += nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        pred = logits.argmax(1).cpu()
        cm += torch.bincount(y * 10 + pred, minlength=100).reshape(10, 10)
        predictions.extend(zip(y.tolist(), pred.tolist()))
        total += len(y)
        reset_model_state(model)
    diagonal = cm.diag().double()
    recall = diagonal / cm.sum(1).clamp_min(1)
    precision = diagonal / cm.sum(0).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    return {'loss': loss / total, 'accuracy': diagonal.sum().item() / total,
            'balanced_accuracy': recall.mean().item(), 'macro_f1': f1.mean().item(),
            'per_class_recall': recall.tolist(), 'confusion_matrix': cm.tolist(), 'n': total}, predictions


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def save_checkpoint(path, model, config, epoch):
    tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'config': config, 'epoch': epoch}, tmp)
    os.replace(tmp, path)


def write_predictions(path, ds, predictions):
    with Path(path).open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_class', 'predicted_class'])
        for (name, _), (target, pred) in zip(ds.samples, predictions):
            writer.writerow([name, ds.classes[target], ds.classes[pred]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/MSTAR/soc'))
    parser.add_argument('--output', type=Path, default=Path('runs/resnet18_seed0'))
    parser.add_argument('--arch', choices=['resnet18', 'resnet34', 'spiking_resnet18'], default='resnet18')
    parser.add_argument('--time-steps', type=int, default=4)
    parser.add_argument('--lif-tau', type=float, default=2.0)
    parser.add_argument('--stem', choices=['standard', 'small'], default='standard')
    parser.add_argument('--pretrained', choices=['none', 'imagenet'], default='none')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-fraction', type=float, default=0,
                        help='0 = fixed epochs, all training data, final checkpoint; >0 = select on train-only holdout')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cpu-threads', type=int, default=8)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--allow-nonstandard', action='store_true')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.weight_decay < 0 or not 0 <= args.val_fraction < 1:
        parser.error('epochs>=1, batch-size>=2, lr>0, weight-decay>=0, 0<=val-fraction<1 required')
    if args.workers < 0 or args.cpu_threads < 1:
        parser.error('workers>=0, cpu-threads>=1 required')
    if args.stem == 'small' and args.pretrained != 'none':
        parser.error('small stem is supported only for scratch training')
    if args.arch == 'spiking_resnet18' and (args.stem != 'standard' or args.pretrained != 'none'):
        parser.error('SNN baseline requires --stem standard --pretrained none')
    if args.time_steps < 1 or args.lif_tau <= 1:
        parser.error('time-steps >= 1 and lif-tau > 1 required')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Expose a host GPU first; --device cpu is only an explicit CPU fallback.')
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    base = datasets.ImageFolder(args.data / 'train')
    train_ids, val_ids = split_indices(base.targets, args.val_fraction, args.seed)
    mean, std = normalization_stats(base, train_ids)
    train_ds = Subset(datasets.ImageFolder(args.data / 'train', transforms_for(mean, std, not args.no_augment)), train_ids)
    val_ds = Subset(datasets.ImageFolder(args.data / 'train', transforms_for(mean, std)), val_ids)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    source_metadata = args.data.parent / f'{args.data.name}_source.json'
    if source_metadata.exists():
        config['dataset_source'] = json.loads(source_metadata.read_text())
    config.update({'mean': mean, 'std': std, 'classes': base.classes,
                   'n_train': len(train_ids), 'n_val': len(val_ids),
                   'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                   'python': platform.python_version(),
                   'gpu': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
    save_json(args.output / 'config.json', config)
    model = build_model(args.arch, args.pretrained == 'imagenet', args.stem,
                        args.time_steps, args.lif_tau).to(device)
    if hasattr(model, 'spec'):
        config['snn'] = model.spec
    config['parameters'] = sum(p.numel() for p in model.parameters())
    save_json(args.output / 'config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    amp = args.device == 'cuda' and not args.no_amp
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    train_batches = loader(train_ds, args.batch_size, args.workers, True, args.device == 'cuda')
    val_batches = loader(val_ds, args.batch_size, args.workers, False, args.device == 'cuda') if val_ids else None
    best = (-1.0, float('-inf'))
    print(json.dumps(config, indent=2), flush=True)
    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        total = correct = 0
        loss_sum = 0.0
        lr = optimizer.param_groups[0]['lr']
        for x, y in train_batches:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite training loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            reset_model_state(model)
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()
        row = {'epoch': epoch, 'lr': lr, 'train_loss': loss_sum / total, 'train_accuracy': correct / total}
        if val_batches is not None:
            val, _ = evaluate(model, val_batches, device)
            row.update({'val_loss': val['loss'], 'val_accuracy': val['accuracy']})
            score = (val['accuracy'], -val['loss'])
            if score > best:
                best = score
                save_checkpoint(args.output / 'best.pt', model, config, epoch)
        row['seconds'] = time.perf_counter() - epoch_start
        with (args.output / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    save_checkpoint(args.output / 'last.pt', model, config, args.epochs)
    selected = args.output / ('best.pt' if val_ids else 'last.pt')
    checkpoint = torch.load(selected, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    test_ds = datasets.ImageFolder(args.data / 'test', transforms_for(mean, std))
    metrics, predictions = evaluate(model, loader(test_ds, args.batch_size, args.workers, cuda=args.device == 'cuda'), device)
    metrics.update({'selected_checkpoint': selected.name, 'selected_epoch': checkpoint['epoch'],
                    'classes': base.classes, 'elapsed_seconds': time.perf_counter() - start,
                    'peak_gpu_memory_mb': torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else None,
                    'nonstandard_data': args.allow_nonstandard})
    save_json(args.output / 'test_metrics.json', metrics)
    write_predictions(args.output / 'test_predictions.csv', test_ds, predictions)
    print('FINAL_TEST ' + json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
