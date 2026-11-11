"""MSTAR SOC preprocessing adapted from the local ANN/SNN baseline."""
import math
from pathlib import Path

from PIL import Image
from torchvision import datasets, transforms


def transforms_for(mean, std, augment=False, input_size=128, input_mode='center_crop'):
    if input_mode == 'center_crop':
        spatial = transforms.CenterCrop(input_size)
    elif input_mode == 'resize_full':
        spatial = transforms.Resize((input_size, input_size), antialias=True)
    else:
        raise ValueError(f'Unsupported input mode: {input_mode}')
    ops = [transforms.Grayscale(1), spatial]
    if augment:
        ops.append(transforms.RandomCrop(input_size, padding=4, padding_mode='reflect'))
    return transforms.Compose(ops + [transforms.ToTensor(),
                                     transforms.Normalize([mean], [std])])


def normalization_stats(dataset, input_size=128, input_mode='center_crop'):
    total = square = 0.0
    pixels = 0
    transform = transforms_for(0., 1., input_size=input_size, input_mode=input_mode)
    for path, _ in dataset.samples:
        with Image.open(path) as image:
            x = transform(image).double()
        total += x.sum().item()
        square += x.square().sum().item()
        pixels += x.numel()
    mean = total / pixels
    std = max(max(square / pixels - mean * mean, 0) ** 0.5, 1e-6)
    return mean, std


def load_mstar(root, augment=True, metadata=None, test_only=False,
               input_size=128, input_mode='center_crop'):
    """Return datasets and preprocessing; evaluation can use just root/test."""
    root = Path(root)
    train = None if test_only else datasets.ImageFolder(root / 'train')
    test = datasets.ImageFolder(root / 'test')
    if len(test.classes) != 10:
        raise ValueError('MSTAR SOC requires exactly ten classes')
    if train is not None and train.class_to_idx != test.class_to_idx:
        raise ValueError('MSTAR train/test class mappings differ')
    if metadata is None:
        if train is None:
            raise ValueError('MSTAR test-only evaluation requires checkpoint preprocessing')
        mean, std = normalization_stats(train, input_size, input_mode)
        metadata = {'mean': mean, 'std': std, 'classes': train.classes,
                    'class_to_idx': train.class_to_idx, 'input_size': [1, input_size, input_size],
                    'input_mode': input_mode}
    else:
        if (metadata['classes'] != test.classes or
                metadata['class_to_idx'] != test.class_to_idx):
            raise ValueError('MSTAR dataset classes differ from checkpoint')
        shape = metadata['input_size']
        if len(shape) != 3 or shape[0] != 1 or shape[1] != shape[2] or shape[1] < 32 or shape[1] % 32:
            raise ValueError('Invalid MSTAR checkpoint input size')
        input_size = shape[1]
        input_mode = metadata.get('input_mode', 'center_crop')
        mean, std = metadata['mean'], metadata['std']
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError('Invalid MSTAR checkpoint normalization')
    if train is not None:
        train.transform = transforms_for(mean, std, augment, input_size, input_mode)
    test.transform = transforms_for(mean, std, False, input_size, input_mode)
    return train, test, metadata

