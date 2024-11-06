import random

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose


def crop_signal(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Randomly crop a signal to a fixed length."""
    offset = 1 + x.size(-1) - seq_len
    start = torch.randint(0, offset, (1,)).item()
    return x[..., start : start + seq_len]


class RandomCrop:
    def __init__(self, seq_len=1_000, p=1.0):
        self.seq_len = seq_len
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = crop_signal(x, self.seq_len)
        return x


class RandomResizeCrop:
    def __init__(self, seq_len=1_000, p=1.0):
        self.seq_len = seq_len
        self.p = p

    def __call__(self, x):
        size = x.shape[-1]  # Original size
        if random.random() < self.p:
            x = crop_signal(x, self.seq_len)
            # Resize back to original size
            x = F.interpolate(
                x.unsqueeze(1), size=size, mode="linear", align_corners=True
            )

        return x


class RandomRescale:
    def __init__(self, scale_range=(0.9, 1.1), p=1.0):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            scale = random.uniform(*self.scale_range)
            x = x * scale
        return x


class RandomShift:
    def __init__(self, shift_range=(-0.1, 0.1), p=1.0):
        self.shift_range = shift_range
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            shift = random.uniform(*self.shift_range)
            x = x + shift
        return x


class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = torch.flip(x, [-1])
        return x


class RandomInvert:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = -x
        return x


class RandomGaussian:
    def __init__(self, mean=0, std=0.1, p=1.0):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            noise = torch.randn_like(x) * self.std + self.mean
            x = x + noise
        return x


class RandomShuffle:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            x = x[torch.randperm(x.size(0))]
        return x


class Normalize:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        if self.mean is None:
            self.mean = x.median(dim=1).unsqueeze(1)
        if self.std is None:
            self.std = x.std(dim=1).unsqueeze(1)

        return (x - self.mean) / self.std


def get_transform(
    seq_len=1_000,
    normalize=True,
    shuffle=True,
    flip=True,
    invert=True,
    rescale=True,
    shift=True,
    resize_crop=False,
):
    transforms = [RandomCrop(seq_len=seq_len)]
    if normalize:
        transforms.append(Normalize())
    if shuffle:
        transforms.append(RandomShuffle(p=1.0))
    if flip:
        transforms.append(RandomFlip(p=0.5))
    if invert:
        transforms.append(RandomInvert(p=0.5))
    if rescale:
        transforms.append(RandomRescale(p=0.5))
    if shift:
        transforms.append(RandomShift(p=0.5))
    if resize_crop:
        transforms.append(RandomResizeCrop(seq_len=seq_len))

    return Compose(transforms)
