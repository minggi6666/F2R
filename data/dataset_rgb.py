"""Dataset loaders for RGB video denoising experiments."""

import os
import glob
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def augment_img(img, mode=0):
    """
    Apply common spatial augmentations (rotation/flip).
    Modified from Kai Zhang (https://github.com/cszn).
    """
    if mode == 0:
        return img
    elif mode == 1:
        return np.flipud(np.rot90(img)).copy()
    elif mode == 2:
        return np.flipud(img).copy()
    elif mode == 3:
        return np.rot90(img, k=3).copy()
    elif mode == 4:
        return np.flipud(np.rot90(img, k=2)).copy()
    elif mode == 5:
        return np.rot90(img).copy()
    elif mode == 6:
        return np.rot90(img, k=2).copy()
    elif mode == 7:
        return np.flipud(np.rot90(img, k=3)).copy()


class DataLoader_RGB(Dataset):
    """
    Training Dataset loader for RGB Video Frames.
    Loads sequences of frames (len=seq_len) and crops them into patches.
    """
    def __init__(self, data_dir, patch, seq_len=7, img_aug=True):
        super().__init__()
        self.patch = patch
        self.seq_len = seq_len
        self.img_aug = img_aug
        self.mid = seq_len // 2
        self.samples = []

        video_dirs = sorted([p for p in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(p)])

        for vdir in video_dirs:
            frames = sorted(glob.glob(os.path.join(vdir, '*')))
            n_frames = len(frames)
            for i in range(n_frames):
                self.samples.append((frames, i))

        print(f'DataLoader_RGB: Fetched {len(self.samples)} samples from {len(video_dirs)} videos.')

    def _crop_coords(self, H, W):
        xx = random.randint(0, max(0, H - self.patch)) if H > self.patch else 0
        yy = random.randint(0, max(0, W - self.patch)) if W > self.patch else 0
        return xx, yy

    def __getitem__(self, index):
        frames_all, target_idx = self.samples[index]
        n_frames = len(frames_all)

        # Handle boundary conditions for sequences
        if target_idx < self.mid:
            start_idx = 0
        elif target_idx >= n_frames - self.mid:
            start_idx = n_frames - self.seq_len
        else:
            start_idx = target_idx - self.mid

        window_indices = list(range(start_idx, start_idx + self.seq_len))
        current_pos_in_window = window_indices.index(target_idx)

        # Ensure the target frame is strictly in the middle of the window
        if current_pos_in_window != self.mid:
            window_indices.pop(current_pos_in_window)
            window_indices.insert(self.mid, target_idx)

        imgs = []
        aug_mode = np.random.randint(0, 8) if self.img_aug else 0

        for idx in window_indices:
            fp = frames_all[idx]
            im = cv2.imread(fp, cv2.IMREAD_COLOR).astype(np.float32) / 255.0  # (H,W,C)
            im = augment_img(im, aug_mode)
            im = im.transpose(2, 0, 1)  # (C,H,W)
            im = torch.from_numpy(im)
            imgs.append(im)

        imgs = torch.stack(imgs, dim=0)  # (T, C, H, W)
        _, _, H, W = imgs.shape
        xx, yy = self._crop_coords(H, W)

        # Spatial crop across the sequence
        imgs = imgs[..., xx:xx + self.patch, yy:yy + self.patch]
        return imgs

    def __len__(self):
        return len(self.samples)

class DataLoader_val(Dataset):
    """
    Validation Dataset loader for RGB Video Frames.
    Loads sequences, handles resizing depending on the dataset type (Davis/Set8),
    and avoids patching for full-frame validation.
    """

    def __init__(self, data_dir, seq_len=3, dataset='davis'):
        super().__init__()
        self.seq_len = seq_len
        self.mid = seq_len // 2
        self.resize_h, self.resize_w = (480, 854) if dataset == 'davis' else (540, 960)
        self.samples = []

        video_dirs = sorted([p for p in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(p)])
        for vdir in video_dirs:
            frames = sorted(glob.glob(os.path.join(vdir, '*')))
            n_frames = len(frames)
            for i in range(n_frames):
                self.samples.append((frames, i))

        print(f'DataLoader_val ({dataset}): Fetched {len(self.samples)} samples from {len(video_dirs)} videos.')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frames_all, target_idx = self.samples[idx]
        n_frames = len(frames_all)

        # Boundary checks
        if target_idx < self.mid:
            start_idx = 0
        elif target_idx >= n_frames - self.mid:
            start_idx = n_frames - self.seq_len
            if start_idx < 0: start_idx = 0
        else:
            start_idx = target_idx - self.mid

        end_idx = min(start_idx + self.seq_len, n_frames)
        window_indices = list(range(start_idx, end_idx))

        # Ensure target is in the middle
        if target_idx in window_indices:
            current_pos = window_indices.index(target_idx)
            if current_pos != self.mid:
                window_indices.pop(current_pos)
                window_indices.insert(self.mid, target_idx)

        imgs = []
        for i in window_indices:
            fp = frames_all[i]
            im = cv2.imread(fp, cv2.IMREAD_COLOR).astype(np.float32)
            h, w, _ = im.shape
            # Resize if it differs from the expected dimensions
            if h != self.resize_h or w != self.resize_w:
                im = cv2.resize(im, (self.resize_w, self.resize_h))
            im = im.transpose(2, 0, 1) / 255.0
            imgs.append(torch.from_numpy(im))

        imgs = torch.stack(imgs, dim=0)  # (T, C, H, W)
        return imgs