"""Evaluate the Stage-2 RGB F2R model on DAVIS/Set8-style videos."""

import os
import glob
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from einops import rearrange

# === F2R Module Imports ===
from models.fusion_s2 import Fusion
from models.nafnet_rgb import Baseline as NAFNetDenoiser
from models.pytorch_pwc.pwc import PWCNet as PWCNet
from models.pytorch_pwc.extract_flow import get_flow_2frames_train
from data.add_noise import AddNoiseTorch, AddNoiseBlind
from utils.utils import calculate_psnr_pt, calculate_ssim_pt
from utils.log import tcolor, Color
from utils.checkpoint import load_model_weights, load_pwcnet_weights


parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./models/rgb_sigma30.pth', help='Path to Stage-2 RGB Fusion checkpoint')
parser.add_argument('--nafnet_path', type=str, default='./models/nafnet_rgb.pth', help='Path to RGB NAFNet checkpoint')
parser.add_argument('--val_dir_davis', type=str, default='./datasets/DAVIS/test', help='DAVIS validation root')
parser.add_argument('--val_dir_set8', type=str, default='./datasets/Set8/test', help='Optional Set8 validation root')
parser.add_argument('--save_dir', type=str, default='./results', help='Directory to save results')
parser.add_argument("--noisetype", type=str, default="gauss30",
                    choices=['gauss10', 'gauss20', 'gauss30', 'gauss40', 'gauss50', 'blind'])
parser.add_argument('--gpu_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--pwc_path', type=str, default='./models/pwc-default.pth', help='Optional local PWC-Net checkpoint; downloaded if missing')

parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--len', type=int, default=9, help='Sequence length (T)')
parser.add_argument('--n_features', type=int, default=72)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def get_noise_func(img, noisetype: str):
    if noisetype.startswith('gauss'):
        sigma = int(noisetype.replace('gauss', ''))
        return AddNoiseTorch(sigma)(img)
    return AddNoiseBlind(5, 50)(img)


def flow_to_ref(coarse, estimator, ref_idx):
    B, T, C, H, W = coarse.shape
    ref_frame = coarse[:, ref_idx]
    flow_list = []
    for i in range(T):
        if i == ref_idx: continue
        if i < ref_idx: _, flow_ir = get_flow_2frames_train(estimator, torch.cat(
            [coarse[:, i].unsqueeze(1), ref_frame.unsqueeze(1)], dim=1))
        if i > ref_idx: flow_ir, _ = get_flow_2frames_train(estimator, torch.cat(
            [ref_frame.unsqueeze(1), coarse[:, i].unsqueeze(1)], dim=1))
        flow_list.append(flow_ir)
    flow_stack = torch.stack(flow_list, dim=1).squeeze(2)
    return rearrange(flow_stack, 'b t xy h w -> (b t) xy h w')


def save_result_frame(rec_tensor, key_frame_path, psnr, ssim, save_root):
    img_np = rec_tensor.cpu().detach().numpy().transpose(1, 2, 0)
    img_uint8 = (np.clip(img_np, 0, 1) * 255.0).astype(np.uint8)

    path_parts = key_frame_path.replace('\\', '/').split('/')
    seq_name = path_parts[-2]
    file_name_raw = os.path.splitext(path_parts[-1])[0]

    seq_save_dir = os.path.join(save_root, seq_name)
    os.makedirs(seq_save_dir, exist_ok=True)
    save_full_path = os.path.join(seq_save_dir, f"{file_name_raw}_P{psnr:.2f}_S{ssim:.4f}.jpg")
    cv2.imwrite(save_full_path, img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


class TestDataset_RGB(Dataset):
    def __init__(self, data_dir, seq_len=9):
        super().__init__()
        self.seq_len = seq_len
        self.samples = []
        if os.path.exists(data_dir):
            video_dirs = sorted([p for p in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(p)])
            for vdir in video_dirs:
                frames = sorted(glob.glob(os.path.join(vdir, '*')))
                for i in range(len(frames) - seq_len + 1):
                    self.samples.append(frames[i:i + seq_len])
            print(tcolor(f'Fetched {len(self.samples)} clips from {data_dir}', c=Color.Green))
        else:
            print(tcolor(f'Warning: Directory {data_dir} does not exist.', c=Color.Red))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        imgs = []
        for fp in frame_paths:
            im = cv2.imread(fp, cv2.IMREAD_COLOR).astype(np.float32).transpose(2, 0, 1) / 255.0
            imgs.append(torch.from_numpy(im))
        return torch.stack(imgs, dim=0), frame_paths


def test_dataset(loader, dataset_name, models):
    Denoiser, Estimator, FusionNet = models
    print(f"\n--- Testing {dataset_name} ---")
    save_root_fusion = os.path.join(args.save_dir, dataset_name, 'Fusion')
    save_root_coarse = os.path.join(args.save_dir, dataset_name, 'Coarse')

    with torch.no_grad():
        for idx, (clean, paths) in enumerate(loader):
            clean = clean.cuda()
            B, T, C, H, W = clean.shape

            # Padding Logic (Divisible by 8)
            factor = 8
            h_pad = (factor - H % factor) % factor
            w_pad = (factor - W % factor) % factor
            clean_padded = F.pad(clean.view(-1, C, H, W), (0, w_pad, 0, h_pad), mode='reflect').view(B, T, C, H + h_pad,
                                                                                                     W + w_pad) if (
                        h_pad or w_pad) else clean
            H_padded, W_padded = clean_padded.shape[-2:]

            ref_frame = T // 2
            key_frame_path = paths[ref_frame][0]

            noisy_padded = get_noise_func(clean_padded, args.noisetype).cuda()

            # Inference
            coarse_padded = Denoiser(noisy_padded.reshape(-1, C, H_padded, W_padded)).reshape(B, T, C, H_padded,
                                                                                              W_padded)
            f_padded = flow_to_ref(coarse_padded, Estimator, ref_frame)

            input_fusion_padded = torch.cat([coarse_padded, noisy_padded - coarse_padded], dim=2)
            output_padded = FusionNet(input_fusion_padded.reshape(B * T, C * 2, H_padded, W_padded),
                                      coarse_padded[:, ref_frame], f_padded)

            # Unpad & Clamp
            output = torch.clamp(output_padded[..., :H, :W], 0, 1)
            coarse = torch.clamp(coarse_padded[..., :H, :W], 0, 1)

            # Metrics Calculation
            psnr_fusion = calculate_psnr_pt(output, clean[:, ref_frame], crop_border=0).item()
            ssim_fusion = calculate_ssim_pt(output, clean[:, ref_frame], crop_border=0).item()
            psnr_coarse = calculate_psnr_pt(coarse[:, ref_frame], clean[:, ref_frame], crop_border=0).item()
            ssim_coarse = calculate_ssim_pt(coarse[:, ref_frame], clean[:, ref_frame], crop_border=0).item()

            print(
                f"[{dataset_name}] Frame {idx}: Fusion {psnr_fusion:.2f}/{ssim_fusion:.4f} | Coarse {psnr_coarse:.2f}/{ssim_coarse:.4f}")

            save_result_frame(output[0], key_frame_path, psnr_fusion, ssim_fusion, save_root_fusion)
            save_result_frame(coarse[0, ref_frame], key_frame_path, psnr_coarse, ssim_coarse, save_root_coarse)


if __name__ == '__main__':
    Denoiser = NAFNetDenoiser().cuda()
    Estimator = PWCNet().cuda()
    FusionNet = Fusion(t=int(args.len - 1), dim=args.n_features).cuda()

    load_model_weights(Denoiser, args.nafnet_path, map_location='cuda')
    load_pwcnet_weights(Estimator, args.pwc_path, map_location='cuda')

    load_model_weights(FusionNet, args.checkpoint, map_location='cuda')

    Denoiser.eval()
    Estimator.eval()
    FusionNet.eval()

    davisLoader = DataLoader(TestDataset_RGB(args.val_dir_davis, args.len), batch_size=args.val_batch_size,
                             shuffle=False)
    test_dataset(davisLoader, 'davis', (Denoiser, Estimator, FusionNet))

    if args.val_dir_set8:
        set8Loader = DataLoader(TestDataset_RGB(args.val_dir_set8, args.len), batch_size=args.val_batch_size,
                                shuffle=False)
        test_dataset(set8Loader, 'set8', (Denoiser, Estimator, FusionNet))

    print("\nTesting Finished.")