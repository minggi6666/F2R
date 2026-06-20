"""Evaluate the Stage-2 RAW F2R model on CRVD and save ISP-rendered images."""

import os
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from einops import rearrange

# === F2R Module Imports ===
from models.fusion_raw_s2 import Fusion
from models.nafnet_raw import Baseline as NAFNetDenoiser
from models.pytorch_pwc.pwc import PWCNet as PWCNet
from models.pytorch_pwc.extract_flow import extract_flow_torch
from models.network_isp import ISP  # ISP 网络
from utils.utils import calculate_psnr_pt, calculate_ssim_pt, pack_gbrg_raw, demosaic
from utils.checkpoint import load_model_weights, load_pwcnet_weights

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='./models/raw.pth', help='Path to Stage-2 RAW Fusion checkpoint')
parser.add_argument('--nafnet_path', type=str, default='./models/nafnet_raw.pth', help='Path to RAW NAFNet checkpoint')
parser.add_argument('--isp_path', type=str, default='./models/ISP_CNN.pth', help='Path to ISP checkpoint')
parser.add_argument('--crvd_root', type=str, default='./datasets/CRVD', help='CRVD dataset root')
parser.add_argument('--save_dir', type=str, default='./results/raw', help='Directory to save visual results')
parser.add_argument('--test_iso', type=int, default=25600)
parser.add_argument('--len', type=int, default=7, help='Sequence length (T)')
parser.add_argument('--n_features', type=int, default=72)
parser.add_argument('--gpu_devices', type=str, default='0')
parser.add_argument('--pwc_path', type=str, default='./models/pwc-default.pth', help='Optional local PWC-Net checkpoint; downloaded if missing')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
torch.backends.cudnn.enabled = True

def flow_to_ref(coarse, estimator, ref_frame):
    B, T, C, H, W = coarse.shape
    flow_list = [extract_flow_torch(estimator, ref_frame, coarse[:, i]) for i in range(T)]
    flow_stack = torch.stack(flow_list, dim=1).squeeze(2)
    return rearrange(flow_stack, 'b t xy h w -> (b t) xy h w')


def tensor_to_img(tensor):
    img_np = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(img_np * 255, 0, 255).astype(np.uint8)

class CRVDVDataset_Test(Dataset):
    TOTAL_FRAMES_PER_VIDEO = 7
    CLEAN_TEMPLATE = 'frame{}_clean_and_slightly_denoised.tiff'
    NOISY_TEMPLATE = 'frame{}_noisy0.tiff'  # 验证集固定使用 noisy0

    def __init__(self, crvd_path, fixed_iso, n_frames):
        super().__init__()
        self.samples = []
        iso_str = f"ISO{fixed_iso}"
        for scene_id in range(7, 12):
            data_folder = os.path.join(crvd_path, f"scene{scene_id}", iso_str)
            for center_frame in range(1, self.TOTAL_FRAMES_PER_VIDEO + 1):
                self.samples.append({
                    'data_folder': data_folder,
                    'center_frame': center_frame,
                    'scene_id': scene_id
                })
        print(f"CRVD Test Dataset (ISO {fixed_iso}) initialized: {len(self.samples)} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frame_indices = [i for i in range(1, self.TOTAL_FRAMES_PER_VIDEO + 1) if i != sample['center_frame']]

        noisy_frames_list = []
        for f_idx in frame_indices:
            noisy_path = os.path.join(sample['data_folder'], self.NOISY_TEMPLATE.format(f_idx))
            noisy_frames_list.append(pack_gbrg_raw(cv2.imread(noisy_path, -1).astype(np.float32)))

        noisy_seq_stacked = np.stack(noisy_frames_list, axis=0)
        noisy_center_frame = pack_gbrg_raw(
            cv2.imread(os.path.join(sample['data_folder'], self.NOISY_TEMPLATE.format(sample['center_frame'])),
                       -1).astype(np.float32))
        clean_frame = pack_gbrg_raw(
            cv2.imread(os.path.join(sample['data_folder'], self.CLEAN_TEMPLATE.format(sample['center_frame'])),
                       -1).astype(np.float32))

        norm_scale = 2 ** 12 - 1 - 240
        noisy_seq_stacked = np.maximum(noisy_seq_stacked - 240, 0) / norm_scale
        noisy_center_frame = np.maximum(noisy_center_frame - 240, 0) / norm_scale
        clean_frame = np.maximum(clean_frame - 240, 0) / norm_scale

        return torch.from_numpy(noisy_seq_stacked), torch.from_numpy(clean_frame), torch.from_numpy(noisy_center_frame), \
        sample['center_frame'], sample['scene_id']


def test_outdoor_crvd():
    print("Loading Models...")
    Denoiser = NAFNetDenoiser().cuda()
    Estimator = PWCNet().cuda()
    FusionNet = Fusion(t=int(args.len - 1), dim=args.n_features).cuda()
    Net_isp = ISP().cuda()

    load_model_weights(Denoiser, args.nafnet_path, map_location='cuda')
    load_pwcnet_weights(Estimator, args.pwc_path, map_location='cuda')
    load_model_weights(Net_isp, args.isp_path, map_location='cuda')
    load_model_weights(FusionNet, args.checkpoint, map_location='cuda')

    Denoiser.eval()
    Estimator.eval()
    FusionNet.eval()
    Net_isp.eval()

    val_loader = DataLoader(CRVDVDataset_Test(args.crvd_root, args.test_iso, args.len), batch_size=1, shuffle=False)

    save_dirs = {name: os.path.join(args.save_dir, f'sceneX_iso{args.test_iso}', name) for name in
                 ['clean', 'noisy', 'rec', 'coarse']}
    for d in save_dirs.values(): os.makedirs(d, exist_ok=True)

    print(f"Start Testing... Results will be saved to {args.save_dir}/sceneX_iso{args.test_iso}")

    with torch.no_grad():
        for idx, (noisy, clean, ref_frame, frame_idx, scene_id_tensor) in enumerate(val_loader):
            noisy, clean, ref_frame = noisy.cuda(), clean.cuda(), ref_frame.cuda()
            frame_id, scene_id = frame_idx.item(), scene_id_tensor.item()
            B, T, C, H, W = noisy.shape

            # Padding Logic
            factor = 8
            h_pad = (factor - H % factor) % factor
            w_pad = (factor - W % factor) % factor

            ref_padded = F.pad(ref_frame, (0, w_pad, 0, h_pad), mode='constant') if (h_pad or w_pad) else ref_frame
            noisy_padded = F.pad(noisy.view(-1, C, H, W), (0, w_pad, 0, h_pad), mode='constant').view(B, T, C,
                                                                                                      H + h_pad,
                                                                                                      W + w_pad) if (
                        h_pad or w_pad) else noisy
            H_padded, W_padded = noisy_padded.shape[-2:]

            # Inference
            coarse_padded = Denoiser(noisy_padded.reshape(-1, C, H_padded, W_padded)).reshape(B, T, C, H_padded,
                                                                                              W_padded)
            deref_padded = Denoiser(ref_padded)

            f = flow_to_ref(demosaic(coarse_padded), Estimator, demosaic(deref_padded.unsqueeze(1)).squeeze(1))

            input_fusion_padded = torch.cat([
                torch.cat([deref_padded.unsqueeze(1), coarse_padded], dim=1),
                torch.cat([ref_padded.unsqueeze(1), noisy_padded], dim=1) - torch.cat(
                    [deref_padded.unsqueeze(1), coarse_padded], dim=1)
            ], dim=2)

            output_padded = FusionNet(input_fusion_padded.reshape(B * (T + 1), C * 2, H_padded, W_padded), deref_padded,
                                      f)

            # ISP Conversion & Clamping
            rec_rgb = torch.clamp(Net_isp(torch.clamp(output_padded[..., :H, :W], 0, 1)), 0, 1)
            coarse_rgb = torch.clamp(Net_isp(torch.clamp(deref_padded[..., :H, :W], 0, 1)), 0, 1)
            clean_rgb = torch.clamp(Net_isp(torch.clamp(clean, 0, 1)), 0, 1)
            noisy_rgb = torch.clamp(Net_isp(torch.clamp(ref_frame, 0, 1)), 0, 1)

            # Metrics
            psnr_rec, ssim_rec = calculate_psnr_pt(rec_rgb, clean_rgb, crop_border=0).item(), calculate_ssim_pt(rec_rgb,
                                                                                                                clean_rgb,
                                                                                                                crop_border=0).item()
            psnr_coarse, ssim_coarse = calculate_psnr_pt(coarse_rgb, clean_rgb,
                                                         crop_border=0).item(), calculate_ssim_pt(coarse_rgb, clean_rgb,
                                                                                                  crop_border=0).item()
            psnr_noisy, ssim_noisy = calculate_psnr_pt(noisy_rgb, clean_rgb, crop_border=0).item(), calculate_ssim_pt(
                noisy_rgb, clean_rgb, crop_border=0).item()

            print(f"Scene {scene_id} Frame {frame_id:04d}: Fusion PSNR {psnr_rec:.2f} | Coarse PSNR {psnr_coarse:.2f}")

            # Save Images
            cv2.imwrite(os.path.join(save_dirs['clean'], f"scene{scene_id}_frame{frame_id:04d}_clean.jpg"),
                        tensor_to_img(clean_rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(
                os.path.join(save_dirs['noisy'], f"scene{scene_id}_frame{frame_id:04d}_noisy_psnr{psnr_noisy:.2f}.jpg"),
                tensor_to_img(noisy_rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(save_dirs['rec'],
                                     f"scene{scene_id}_frame{frame_id:04d}_rec_psnr{psnr_rec:.2f}_ssim{ssim_rec:.4f}.jpg"),
                        tensor_to_img(rec_rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            cv2.imwrite(os.path.join(save_dirs['coarse'],
                                     f"scene{scene_id}_frame{frame_id:04d}_coarse_psnr{psnr_coarse:.2f}_ssim{ssim_coarse:.4f}.jpg"),
                        tensor_to_img(coarse_rgb), [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    print("Testing finished.")


if __name__ == '__main__':
    test_outdoor_crvd()