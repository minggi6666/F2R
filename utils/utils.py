"""Core utility functions for F2R.

Only project-relevant helpers are kept here: logging, RAW packing/demosaic, and
PyTorch PSNR/SSIM metrics. This avoids carrying unrelated utilities from older
experiments into the open-source release.
"""

from __future__ import annotations

import logging
import os
import random
from datetime import datetime
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def get_timestamp() -> str:
    """Return a compact timestamp for log/checkpoint filenames."""
    return datetime.now().strftime("%y%m%d-%H%M%S")


def mkdir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def mkdirs(paths: str | Iterable[str]) -> None:
    """Create one or multiple directories."""
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def set_random_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name: str, root: str, phase: str, level=logging.INFO, screen=False, tofile=False) -> None:
    """Set up a logger for training/validation.

    Repeated calls with the same ``logger_name`` first clear existing handlers to
    avoid duplicated log lines when scripts are relaunched in interactive shells.
    """
    os.makedirs(root, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.handlers.clear()
    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    if tofile:
        log_file = os.path.join(root, f"{phase}_{get_timestamp()}.log")
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def pack_gbrg_raw(raw: np.ndarray) -> np.ndarray:
    """Pack a single-channel GBRG Bayer RAW image into four half-resolution planes.

    Output channel order follows the original training code:
    ``[R, G_on_R_row, G_on_B_row, B]`` for a GBRG mosaic.
    """
    height = raw.shape[0] - raw.shape[0] % 2
    width = raw.shape[1] - raw.shape[1] % 2
    raw = raw[:height, :width]
    return np.stack(
        (
            raw[1:height:2, 0:width:2],
            raw[1:height:2, 1:width:2],
            raw[0:height:2, 1:width:2],
            raw[0:height:2, 0:width:2],
        ),
        axis=0,
    )


def depack_gbrg_raw(raw: np.ndarray) -> np.ndarray:
    """Invert :func:`pack_gbrg_raw` for visualization/debugging."""
    _, height, width = raw.shape
    out = np.zeros((height * 2, width * 2), dtype=raw.dtype)
    out[1::2, 0::2] = raw[0]
    out[1::2, 1::2] = raw[1]
    out[0::2, 1::2] = raw[2]
    out[0::2, 0::2] = raw[3]
    return out


def demosaic(raw_seq: torch.Tensor) -> torch.Tensor:
    """Convert packed RAW tensors to a lightweight RGB proxy.

    Args:
        raw_seq: Tensor with shape ``[B, T, 4, H, W]``.

    Returns:
        Tensor with shape ``[B, T, 3, H, W]`` where the two green channels are
        averaged. This fast proxy is used for optical-flow estimation, not for
        final ISP-quality rendering.
    """
    if raw_seq.ndim != 5 or raw_seq.shape[2] != 4:
        raise ValueError(f"Expected RAW tensor [B, T, 4, H, W], got {tuple(raw_seq.shape)}")
    rgb_seq = torch.empty(
        raw_seq.shape[0], raw_seq.shape[1], 3, raw_seq.shape[3], raw_seq.shape[4],
        dtype=raw_seq.dtype,
        device=raw_seq.device,
    )
    rgb_seq[:, :, 0] = raw_seq[:, :, 0]
    rgb_seq[:, :, 1] = (raw_seq[:, :, 1] + raw_seq[:, :, 2]) * 0.5
    rgb_seq[:, :, 2] = raw_seq[:, :, 3]
    return rgb_seq


def _crop_border(img: torch.Tensor, crop_border: int) -> torch.Tensor:
    if crop_border <= 0:
        return img
    return img[..., crop_border:-crop_border, crop_border:-crop_border]


def calculate_psnr_pt(img: torch.Tensor, img2: torch.Tensor, crop_border: int = 0) -> torch.Tensor:
    """Calculate PSNR for tensors in ``[0, 1]``.

    Args:
        img: Tensor with shape ``[B, C, H, W]``.
        img2: Tensor with the same shape as ``img``.
        crop_border: Optional border crop before metric computation.
    """
    if img.shape != img2.shape:
        raise ValueError(f"Image shapes are different: {tuple(img.shape)} vs {tuple(img2.shape)}")
    img = _crop_border(img, crop_border).to(torch.float64)
    img2 = _crop_border(img2, crop_border).to(torch.float64)
    mse = torch.mean((img - img2) ** 2, dim=(-3, -2, -1))
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))


def _ssim_pth(img: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Core SSIM implementation using a Gaussian window."""
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11)
    window = window.expand(img.size(1), 1, 11, 11).to(dtype=img.dtype, device=img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean(dim=(-3, -2, -1))


def calculate_ssim_pt(img: torch.Tensor, img2: torch.Tensor, crop_border: int = 0) -> torch.Tensor:
    """Calculate SSIM for tensors in ``[0, 1]``.

    The computation follows the same convention as the original code: tensors
    are scaled to the 8-bit range before SSIM is evaluated.
    """
    if img.shape != img2.shape:
        raise ValueError(f"Image shapes are different: {tuple(img.shape)} vs {tuple(img2.shape)}")
    img = _crop_border(img, crop_border).to(torch.float64) * 255.0
    img2 = _crop_border(img2, crop_border).to(torch.float64) * 255.0
    return _ssim_pth(img, img2)
