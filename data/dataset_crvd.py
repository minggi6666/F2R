"""Dataset loaders and RAW noise simulation for CRVD experiments."""

import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import utils.utils as util


def add_raw_noise(clean_tensor):
    """Add random CRVD Poisson-Gaussian RAW noise.

    The input is expected to be normalized RAW in [0, 1]. Both a single frame
    batch ``[B, C, H, W]`` and a video batch ``[B, T, C, H, W]`` are supported.
    A CRVD ISO noise profile is randomly sampled for each batch item, so callers
    do not need to pass a noise type or ISO level.
    """
    if clean_tensor.dim() == 4:
        squeeze_time = True
        clean_tensor = clean_tensor.unsqueeze(1)
    elif clean_tensor.dim() == 5:
        squeeze_time = False
    else:
        raise ValueError(
            f"Expected clean_tensor as [B, C, H, W] or [B, T, C, H, W], got {tuple(clean_tensor.shape)}"
        )

    device = clean_tensor.device
    dtype = clean_tensor.dtype
    B, T, C, H, W = clean_tensor.shape

    # CRVD calibrated noise parameters from ISO1600/3200/6400/12800/25600.
    a_list = torch.tensor([3.513262, 6.955588, 13.486051, 26.585953, 52.032536], device=device, dtype=dtype)
    b_list = torch.tensor([11.917691, 38.117816, 130.818508, 484.539790, 1819.818657], device=device, dtype=dtype)

    indices = torch.randint(0, len(a_list), (B,), device=device)
    a = a_list[indices].view(B, 1, 1, 1, 1)
    b = b_list[indices].view(B, 1, 1, 1, 1)

    black_level = 240.0
    white_level = 2 ** 12 - 1  # 4095
    raw_scale = white_level - black_level  # 3855

    simulated_raw = clean_tensor * raw_scale + black_level
    gt_raw = torch.clamp(simulated_raw, min=black_level, max=white_level)

    signal = torch.clamp((gt_raw - black_level) / a, min=0.0)
    shot_noise = torch.poisson(signal) * a
    read_noise = torch.randn_like(gt_raw) * torch.sqrt(b)
    noisy_raw = shot_noise + read_noise + black_level

    noisy_raw = torch.clamp(noisy_raw, 0.0, white_level)
    noisy_tensor = torch.clamp(noisy_raw - black_level, min=0.0) / raw_scale

    return noisy_tensor.squeeze(1) if squeeze_time else noisy_tensor


class CRVDTDataset(Dataset):
    """Training Dataset for CRVD (Continuous RAW Video Denoising)"""
    TOTAL_FRAMES_PER_VIDEO = 7
    NUM_NOISE_GROUPS = 10
    NOISY_TEMPLATE = 'frame{}_noisy{}.tiff'
    CLEAN_TEMPLATE = 'frame{}_clean.tiff'

    def __init__(self, crvd_path, n_frames, patch_size):
        super().__init__()
        self.crvd_path = crvd_path
        self.n_frames = n_frames
        self.patch_size = patch_size
        self.samples = []
        self.scene_ids = range(7, 12)
        self.iso_list = [1600, 3200, 6400, 12800, 25600]

        min_frame_idx, max_frame_idx = 1, self.TOTAL_FRAMES_PER_VIDEO

        for scene_id in self.scene_ids:
            for iso_val in self.iso_list:
                data_folder = os.path.join(self.crvd_path, f"scene{scene_id}", f"ISO{iso_val}")
                for noise_idx in range(self.NUM_NOISE_GROUPS):
                    for center_frame in range(min_frame_idx, max_frame_idx + 1):
                        self.samples.append({
                            'data_folder': data_folder,
                            'noise_idx': noise_idx,
                            'center_frame': center_frame,
                            'min_idx': min_frame_idx,
                            'max_idx': max_frame_idx
                        })
        print(f"CRVD Train Dataset: Fetched {len(self.samples)} samples (Mixed ISO).")

    def __len__(self):
        return len(self.samples)

    def _crop_coords(self, H, W):
        xx = random.randint(0, max(0, H - self.patch_size)) if H > self.patch_size else 0
        yy = random.randint(0, max(0, W - self.patch_size)) if W > self.patch_size else 0
        return xx, yy

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_indices = [i for i in range(sample['min_idx'], sample['max_idx'] + 1) if i != sample['center_frame']]

        noisy_frames_list = []
        # Apply the same RAW-safe augmentation to all temporal frames so that
        # the Bayer phase remains aligned across the sequence.
        aug_mode = np.random.randint(0, 4)
        for frame_idx in frame_indices:
            noisy_path = os.path.join(sample['data_folder'], self.NOISY_TEMPLATE.format(frame_idx, sample['noise_idx']))
            noisy_frame = cv2.imread(noisy_path, -1).astype(np.float32)
            noisy_frame = self.bayer_preserving_augmentation(noisy_frame, aug_mode)
            noisy_frames_list.append(util.pack_gbrg_raw(noisy_frame))

        center_path = os.path.join(sample['data_folder'],
                                   self.NOISY_TEMPLATE.format(sample['center_frame'], sample['noise_idx']))
        center_raw = cv2.imread(center_path, -1).astype(np.float32)
        center_raw = self.bayer_preserving_augmentation(center_raw, aug_mode)
        center_raw = util.pack_gbrg_raw(center_raw)

        noisy_seq_stacked = np.stack(noisy_frames_list, axis=0)
        h, w = noisy_seq_stacked.shape[-2:]
        xx, yy = self._crop_coords(h, w)

        noisy_patch = noisy_seq_stacked[..., xx:xx + self.patch_size, yy:yy + self.patch_size]
        center_patch = center_raw[..., xx:xx + self.patch_size, yy:yy + self.patch_size]

        norm_scale = 2 ** 12 - 1 - 240
        noisy_patch = np.maximum(noisy_patch - 240, 0) / norm_scale
        center_patch = np.maximum(center_patch - 240, 0) / norm_scale

        return torch.from_numpy(noisy_patch), torch.from_numpy(center_patch)

    @staticmethod
    def bayer_preserving_augmentation(raw, aug_mode):
        """Apply simple augmentation while preserving the GBRG Bayer phase.

        Horizontal/vertical flips remove one row/column to keep the phase
        compatible with ``pack_gbrg_raw`` after flipping.
        """
        if aug_mode == 0:
            return raw
        if aug_mode == 1:
            return np.flip(raw, axis=1)[:, 1:-1]
        if aug_mode == 2:
            return np.flip(raw, axis=0)[1:-1, :]
        return np.transpose(raw, (1, 0))


class CRVDVDataset(Dataset):
    """Validation Dataset for CRVD"""
    TOTAL_FRAMES_PER_VIDEO = 7
    CLEAN_TEMPLATE = 'frame{}_clean_and_slightly_denoised.tiff'
    NOISY_TEMPLATE = 'frame{}_noisy{}.tiff'

    def __init__(self, crvd_path, fixed_iso, n_frames):
        super().__init__()
        self.samples = []
        iso_str = f"ISO{fixed_iso}"
        for scene_id in range(7, 12):
            data_folder = os.path.join(crvd_path, f"scene{scene_id}", iso_str)
            for center_frame in range(1, self.TOTAL_FRAMES_PER_VIDEO + 1):
                self.samples.append({
                    'data_folder': data_folder, 'noise_idx': 0,
                    'center_frame': center_frame, 'min_idx': 1, 'max_idx': self.TOTAL_FRAMES_PER_VIDEO
                })
        print(f"CRVD Val Dataset (ISO {fixed_iso}): Fetched {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_indices = [i for i in range(sample['min_idx'], sample['max_idx'] + 1) if i != sample['center_frame']]

        noisy_frames_list = []
        for frame_idx in frame_indices:
            noisy_path = os.path.join(sample['data_folder'], self.NOISY_TEMPLATE.format(frame_idx, sample['noise_idx']))
            noisy_frames_list.append(util.pack_gbrg_raw(cv2.imread(noisy_path, -1).astype(np.float32)))

        noisy_seq_stacked = np.stack(noisy_frames_list, axis=0)

        noisy_center_path = os.path.join(sample['data_folder'],
                                         self.NOISY_TEMPLATE.format(sample['center_frame'], sample['noise_idx']))
        noisy_center_frame = util.pack_gbrg_raw(cv2.imread(noisy_center_path, -1).astype(np.float32))

        clean_path = os.path.join(sample['data_folder'], self.CLEAN_TEMPLATE.format(sample['center_frame']))
        clean_frame = util.pack_gbrg_raw(cv2.imread(clean_path, -1).astype(np.float32))

        norm_scale = 2 ** 12 - 1 - 240
        noisy_seq_stacked = np.maximum(noisy_seq_stacked - 240, 0) / norm_scale
        frames_stacked = np.maximum(clean_frame - 240, 0) / norm_scale
        noisy_center_frame = np.maximum(noisy_center_frame - 240, 0) / norm_scale

        return torch.from_numpy(noisy_seq_stacked), torch.from_numpy(frames_stacked), torch.from_numpy(
            noisy_center_frame)