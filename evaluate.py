"""Evaluate a local baseline checkpoint, using its saved preprocessing."""
import argparse
from pathlib import Path

import torch
from torchvision import datasets

from mstar_data import transforms_for
from train import build_model, evaluate, loader, save_json, write_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/MSTAR/soc/test'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(8)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    model = build_model(cfg['arch'], False, cfg['stem'], cfg.get('time_steps', 4),
                        cfg.get('lif_tau', 2.0)).to(args.device)
    model.load_state_dict(ckpt['model'])
    ds = datasets.ImageFolder(args.data, transforms_for(cfg['mean'], cfg['std']))
    if ds.classes != cfg['classes']:
        raise ValueError('Dataset classes differ from checkpoint')
    metrics, predictions = evaluate(model, loader(ds, 64, 0, cuda=args.device == 'cuda'), args.device)
    metrics.update({'checkpoint': str(args.checkpoint), 'classes': cfg['classes'], 'selected_epoch': ckpt['epoch']})
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / 'test_metrics.json', metrics)
    write_predictions(args.output / 'test_predictions.csv', ds, predictions)
    print(metrics)


if __name__ == '__main__':
    main()
