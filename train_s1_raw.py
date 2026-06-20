"""Train Stage-1 RAW F2R: blind temporal fusion from neighboring CRVD RAW frames."""

import os
import gc
import random
import logging
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange

# === F2R Module Imports ===
from models.fusion_raw_s1 import Fusion as Fusions1
from models.nafnet_raw import Baseline as Denoiser_super
from models.pytorch_pwc.pwc import PWCNet as Estimator
from models.pytorch_pwc.extract_flow import extract_flow_torch
from data.dataset_crvd import CRVDTDataset, CRVDVDataset
from utils.utils import setup_logger, calculate_psnr_pt, calculate_ssim_pt, demosaic
from utils.checkpoint import load_model_weights, load_pwcnet_weights

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets/CRVD', help='CRVD dataset root')
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--lr_F', type=float, default=3e-4)
parser.add_argument('--lr_final', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--len', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--n_features', type=int, default=72)
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--nafnet_path', type=str, default='./models/nafnet_raw.pth', help='Frozen RAW NAFNet checkpoint')
parser.add_argument('--pwc_path', type=str, default='./models/pwc-default.pth', help='Optional local PWC-Net checkpoint; downloaded if missing')
parser.add_argument('--n_snapshot', type=int, default=5)
parser.add_argument('--log_name', type=str, default='f2r_raw_s1')
parser.add_argument('--save_model_path', type=str, default='./experiments')

# Config Init
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def setup_distributed():
    is_ddp, main_proc, local_rank, world_size = False, True, 0, 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if "LOCAL_RANK" in os.environ:
        is_ddp = True
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=30))
        world_size = dist.get_world_size()
        main_proc = (dist.get_rank() == 0)
    return is_ddp, main_proc, local_rank, world_size, device


is_ddp, main_proc, local_rank, world_size, device = setup_distributed()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
args.save_path = os.path.join(args.save_model_path, args.log_name, systime)
os.makedirs(args.save_path, exist_ok=True)
setup_logger("train", args.save_path, "train_" + args.log_name, level=logging.INFO, screen=True, tofile=True)


def flow_to_ref(coarse, estimator, ref_frame):
    B, T, C, H, W = coarse.shape
    flow_list = [extract_flow_torch(estimator, ref_frame, coarse[:, i]) for i in range(T)]
    flow_stack = torch.stack(flow_list, dim=1).squeeze(2)
    return rearrange(flow_stack, 'b t xy h w -> (b t) xy h w')


# --- Datasets ---
Traincrvd = DataLoader(CRVDTDataset(args.data_dir, args.len, args.patch_size), batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
val_iso_list = [1600, 3200, 6400, 12800, 25600]
val_loaders = {
    f"ISO{iso}": DataLoader(CRVDVDataset(args.data_dir, iso, args.len), batch_size=1, shuffle=False, pin_memory=True)
    for iso in val_iso_list}

best_metrics_all = {f"ISO{iso}": {'epoch': 0, 'psnr': 0.0, 'ssim': 0.0, 'coarse_psnr': 0.0, 'coarse_ssim': 0.0} for iso
                    in val_iso_list}

# --- Models ---
Denoiser = Denoiser_super().to(device)
Estimator = Estimator().to(device)
FusionNet = Fusions1(t=int(args.len - 1), dim=args.n_features).to(device)

load_model_weights(Denoiser, args.nafnet_path, map_location=device)
load_pwcnet_weights(Estimator, args.pwc_path, map_location=device)

if is_ddp:
    Denoiser = DDP(Denoiser, device_ids=[local_rank], output_device=local_rank)
    Estimator = DDP(Estimator, device_ids=[local_rank], output_device=local_rank)
    FusionNet = DDP(FusionNet, device_ids=[local_rank], output_device=local_rank)

# Optimization
steps_per_epoch = len(Traincrvd)
optimizer_F = optim.Adam(FusionNet.parameters(), lr=args.lr_F)
scheduler_F = CosineAnnealingLR(optimizer_F, T_max=args.epochs * steps_per_epoch, eta_min=args.lr_final)

if main_proc:
    loss_csv_path = os.path.join(args.save_path, "train_loss.csv")
    with open(loss_csv_path, "w") as f: f.write("epoch,loss\n")
    print("Initialization finished. Training started...")

for epoch in range(1, args.epochs + 1):
    Denoiser.eval();
    Estimator.eval();
    FusionNet.train()
    epoch_loss_sum, epoch_batch_cnt = torch.zeros(1).to(device), torch.zeros(1).to(device)

    for noisy, ref_frame in Traincrvd:
        B, T, C, H, W = noisy.shape
        noisy, ref_frame = noisy.to(device), ref_frame.to(device)

        with torch.no_grad():
            coarse = Denoiser(noisy.reshape(-1, C, H, W)).reshape(noisy.shape)
            deref = Denoiser(ref_frame).reshape(ref_frame.shape)
            f = flow_to_ref(demosaic(coarse), Estimator, demosaic(deref.unsqueeze(1)).squeeze(1))

        input_feats = torch.cat([coarse, noisy - coarse], dim=2)
        output = FusionNet(input_feats.reshape(B * T, C * 2, H, W), deref, f)

        loss = torch.mean((output - ref_frame) ** 2)
        optimizer_F.zero_grad()
        loss.backward()
        optimizer_F.step()
        scheduler_F.step()

        epoch_loss_sum += loss.item()
        epoch_batch_cnt += 1

    if is_ddp:
        dist.barrier()
        dist.all_reduce(epoch_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(epoch_batch_cnt, op=dist.ReduceOp.SUM)

    if main_proc:
        avg_loss = epoch_loss_sum.item() / max(epoch_batch_cnt.item(), 1.0)
        with open(loss_csv_path, "a") as f: f.write(f"{epoch},{avg_loss:.6f}\n")
        print(f"Epoch {epoch} Training Loss: {avg_loss:.6f}")

    if epoch % args.n_snapshot == 0:
        FusionNet.eval()
        if main_proc:
            os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
            torch.save(FusionNet.state_dict(), os.path.join(args.save_path, 'models', f'epoch_Fusion_{epoch:03d}.pth'))


        @torch.inference_mode()
        def run_val(loader, dataset_name, best_metrics):
            psnr_sum, ssim_sum = 0.0, 0.0
            cnt = 0
            for noisy, clean, ref_frame in loader:
                noisy, clean, ref_frame = noisy.to(device), clean.to(device), ref_frame.to(device)
                B, T, C, H, W = noisy.shape

                # Stage-1 RAW inference: frozen NAFNet gives coarse frames; F2R fuses neighbors.
                coarse_padded = Denoiser(noisy.reshape(-1, C, H, W)).reshape(B, T, C, H, W)
                deref_padded = Denoiser(ref_frame)
                f = flow_to_ref(demosaic(coarse_padded), Estimator, demosaic(deref_padded.unsqueeze(1)).squeeze(1))

                input_fusion = torch.cat([coarse_padded, noisy - coarse_padded], dim=2)
                output = FusionNet(input_fusion.reshape(B * T, C * 2, H, W), deref_padded, f)

                output = torch.clamp(output, 0, 1)
                output = (torch.round(output * (2 ** 12 - 1 - 240) + 240) - 240) / (2 ** 12 - 1 - 240)

                psnr_sum += calculate_psnr_pt(output, clean, crop_border=0).sum().item()
                ssim_sum += calculate_ssim_pt(output, clean, crop_border=0).sum().item()
                cnt += B

            if main_proc:
                psnr, ssim = psnr_sum / cnt, ssim_sum / cnt
                if psnr > best_metrics['psnr']:
                    best_metrics.update({'epoch': epoch, 'psnr': psnr, 'ssim': ssim})
                print(f"[{dataset_name}] Epoch {epoch}: PSNR {psnr:.4f} / SSIM {ssim:.4f}")


        for iso_name, loader in val_loaders.items():
            run_val(loader, iso_name, best_metrics_all[iso_name])

        gc.collect();
        torch.cuda.empty_cache()

# if dist.is_initialized(): dist.destroy_process_group()