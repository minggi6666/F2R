"""Noise models used by F2R RGB experiments.

The training scripts mainly use fixed Gaussian noise (``AddNoiseTorch``) and
blind Gaussian noise (``AddNoiseBlind``). Additional synthetic noise operators
are kept here for ablation or extension, but they are implemented without
project-specific side effects so the module can be safely imported.
"""

from __future__ import annotations

import math
import random
import threading
from typing import Iterable, Sequence

import torch


class LockedIterator:
    """Thread-safe wrapper around an iterator."""

    def __init__(self, iterator: Iterable):
        self.iterator = iter(iterator)
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


class AddNoiseTorch:
    """Add zero-mean Gaussian noise with a fixed sigma.

    Args:
        sigma: Noise standard deviation in the usual 8-bit image scale.

    Input tensors can be ``[B, C, H, W]`` or ``[B, T, C, H, W]`` in ``[0, 1]``.
    """

    def __init__(self, sigma: float):
        self.sigma_ratio = float(sigma) / 255.0

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return img + torch.randn_like(img) * self.sigma_ratio


class AddNoiseBlind:
    """Add Gaussian noise whose sigma is uniformly sampled for each call."""

    def __init__(self, min_sigma: float, max_sigma: float):
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        sigma = torch.empty((), device=img.device).uniform_(self.min_sigma, self.max_sigma)
        return img + torch.randn_like(img) * sigma / 255.0


class AddNoiseNoniidv2Torch:
    """Add non-IID Gaussian noise with a different sigma per batch item."""

    def __init__(self, min_sigma: float, max_sigma: float):
        self.min_sigma = float(min_sigma)
        self.max_sigma = float(max_sigma)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        batch = img.shape[0]
        sigma = torch.empty(batch, device=img.device).uniform_(self.min_sigma, self.max_sigma) / 255.0
        view_shape = [batch] + [1] * (img.ndim - 1)
        return img + torch.randn_like(img) * sigma.view(*view_shape)


class AddPoissonNoiseRandom:
    """Add Poisson noise with a random peak value per batch/channel."""

    def __init__(self, min_peak: float, max_peak: float):
        self.min_peak = float(min_peak)
        self.max_peak = float(max_peak)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim < 4:
            raise ValueError("Expected at least [B, C, H, W].")
        batch, channels = img.shape[:2]
        peak = torch.empty(batch, channels, device=img.device).uniform_(self.min_peak, self.max_peak)
        view_shape = [batch, channels] + [1] * (img.ndim - 2)
        peak = peak.view(*view_shape).clamp_min(1e-6)
        return torch.poisson(img.clamp_min(0) * peak).float() / peak


class AddPoissonNoise:
    """Add Poisson noise with a fixed peak value."""

    def __init__(self, peak: float):
        self.peak = float(peak)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        peak = max(self.peak, 1e-6)
        return torch.poisson(img.clamp_min(0) * peak).float() / peak


class AddNoiseImpulseV2Torch:
    """Add salt-and-pepper impulse noise to selected channels."""

    def __init__(self, min_amount: float, max_amount: float, s_vs_p: float = 0.5):
        if not (0 <= min_amount <= max_amount <= 1):
            raise ValueError("Impulse noise amount must be in [0, 1].")
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)
        self.s_vs_p = float(s_vs_p)

    def __call__(self, img: torch.Tensor, bands: Sequence[int] | None = None) -> torch.Tensor:
        out = img.clone()
        channels = out.shape[-3]
        bands = range(channels) if bands is None else bands
        for band in bands:
            amount = torch.empty((), device=out.device).uniform_(self.min_amount, self.max_amount)
            target = out[..., band, :, :]
            flip_mask = torch.rand_like(target) < amount
            salt_mask = torch.rand_like(target) < self.s_vs_p
            target[flip_mask & salt_mask] = 1.0
            target[flip_mask & ~salt_mask] = 0.0
        return out


class AddNoiseStripeTorch:
    """Add vertical stripe noise to selected channels."""

    def __init__(self, min_amount: float, max_amount: float):
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)

    def __call__(self, img: torch.Tensor, bands: Sequence[int] | None = None) -> torch.Tensor:
        out = img.clone()
        channels = out.shape[-3]
        width = out.shape[-1]
        bands = range(channels) if bands is None else bands
        min_num = int(math.floor(self.min_amount * width))
        max_num = int(math.floor(self.max_amount * width))
        for band in bands:
            num = 0 if max_num <= min_num else int(torch.randint(min_num, max_num + 1, (), device=out.device))
            if num <= 0:
                continue
            loc = torch.randperm(width, device=out.device)[:num]
            stripe = torch.empty(num, device=out.device).uniform_(-0.25, 0.25)
            out[..., band, :, loc] = out[..., band, :, loc] - stripe.view(*([1] * (out.ndim - 2)), num)
        return out


class AddNoiseDeadlineTorch:
    """Set random vertical lines to zero."""

    def __init__(self, min_amount: float, max_amount: float):
        self.min_amount = float(min_amount)
        self.max_amount = float(max_amount)

    def __call__(self, img: torch.Tensor, bands: Sequence[int] | None = None) -> torch.Tensor:
        out = img.clone()
        channels = out.shape[-3]
        width = out.shape[-1]
        bands = range(channels) if bands is None else bands
        min_num = int(math.ceil(self.min_amount * width))
        max_num = int(math.ceil(self.max_amount * width))
        for band in bands:
            num = 0 if max_num <= min_num else int(torch.randint(min_num, max_num + 1, (), device=out.device))
            if num <= 0:
                continue
            loc = torch.randperm(width, device=out.device)[:num]
            out[..., band, :, loc] = 0.0
        return out


class AddNoiseMixed:
    """Randomly choose one noise operator from a weighted list."""

    def __init__(self, noise_ops: Sequence, weights: Sequence[float] | None = None):
        if not noise_ops:
            raise ValueError("noise_ops must not be empty.")
        self.noise_ops = list(noise_ops)
        self.weights = None if weights is None else list(weights)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        op = random.choices(self.noise_ops, weights=self.weights, k=1)[0]
        return op(img)


class AddNoiseComplex(AddNoiseMixed):
    """Mixture of stripe, dead-line, and impulse noise for stress tests."""

    def __init__(self):
        super().__init__(
            noise_ops=[
                AddNoiseStripeTorch(0.05, 0.15),
                AddNoiseDeadlineTorch(0.05, 0.15),
                AddNoiseImpulseV2Torch(min_amount=0.1, max_amount=0.7),
            ],
            weights=[1 / 3, 1 / 3, 1 / 3],
        )
