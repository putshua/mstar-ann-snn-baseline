"""Shared grayscale preprocessing and train-only normalization for MSTAR."""
import numpy as np
from PIL import Image
from torchvision import transforms


def transforms_for(mean, std, augment=False):
    ops = [transforms.Grayscale(1), transforms.CenterCrop(128)]
    if augment:
        ops.append(transforms.RandomCrop(128, padding=4, padding_mode='reflect'))
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize([mean], [std])])


def split_indices(targets, val_fraction, seed):
    rng = np.random.default_rng(seed)
    train, val = [], []
    targets = np.asarray(targets)
    for label in range(10):
        indices = np.flatnonzero(targets == label)
        rng.shuffle(indices)
        count = max(1, round(len(indices) * val_fraction)) if val_fraction else 0
        if count >= len(indices):
            raise ValueError('Each class must retain at least one training image')
        val.extend(indices[:count].tolist())
        train.extend(indices[count:].tolist())
    return sorted(train), sorted(val)


def normalization_stats(ds, indices):
    total = square = 0.0
    pixels = 0
    tfm = transforms.Compose([transforms.Grayscale(1), transforms.CenterCrop(128), transforms.ToTensor()])
    for i in indices:
        with Image.open(ds.samples[i][0]) as image:
            x = tfm(image).double()
        total += x.sum().item()
        square += x.square().sum().item()
        pixels += x.numel()
    mean = total / pixels
    std = max((max(square / pixels - mean * mean, 0)) ** 0.5, 1e-6)
    return mean, std
