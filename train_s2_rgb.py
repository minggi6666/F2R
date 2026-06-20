"""Train Stage-2 RGB F2R with center-frame residual refinement."""

import os
import gc
import random
import logging
import argparse
import datetime
from collections import OrderedDict

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange

# === F2R Module Imports ===
from models.fusion_s1 import Fusion as Fusions1  # Stage 1 Architecture
from models.fusion_s2 import Fusion as Fusions2  # Stage 2 Architecture (Deform Conv)
from models.nafnet_rgb import Baseline as Denoiser_super
from models.pytorch_pwc.pwc import PWCNet
from models.pytorch_pwc.extract_flow import get_flow_2frames_train
from data.dataset_rgb import DataLoader_RGB, DataLoader_val
from data.add_noise import *
from utils.utils import setup_logger, calculate_psnr_pt, calculate_ssim_pt
from utils.log import tcolor, Color, gradient_num_color
from utils.checkpoint import load_model_weights, load_pwcnet_weights, torch_load, clean_state_dict, get_state_dict

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets/train', help='Training RGB video root')
parser.add_argument('--val_dir_davis', type=str, default='./datasets/DAVIS/test', help='DAVIS validation root')
parser.add_argument('--val_dir_set8', type=str, default='./datasets/Set8/test', help='Optional Set8 validation root')
parser.add_argument("--noisetype", type=str, default="gauss20",
                    choices=['gauss10', 'gauss20', 'gauss30', 'gauss40', 'gauss50', 'gauss5_50'])

parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr_F', type=float, default=3e-4)
parser.add_argument('--lr_final', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--val_batch_size', type=int, default=1)
parser.add_argument('--len', type=int, default=9)
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--n_features', type=int, default=72)
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--nafnet_path', type=str, default='./models/nafnet_rgb.pth', help='Frozen RGB NAFNet checkpoint')
parser.add_argument('--stage1_path', type=str, default='./models/xxx.pth', help='Trained Stage-1 RGB Fusion checkpoint')
parser.add_argument('--pwc_path', type=str, default='./models/pwc-default.pth', help='Optional local PWC-Net checkpoint; downloaded if missing')
parser.add_argument('--n_snapshot', type=int, default=1)

parser.add_argument('--log_name', type=str, default='f2r_rgb_s2')
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
torch.set_num_threads(10)


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
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    return is_ddp, main_proc, local_rank, world_size, device


@torch.no_grad()
def dist_all_reduce_sum_(t: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


is_ddp, main_proc, local_rank, world_size, device = setup_distributed()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
args.save_path = os.path.join(args.save_model_path, args.log_name, systime)
os.makedirs(args.save_path, exist_ok=True)
setup_logger("train", args.save_path, "train_" + args.log_name, level=logging.INFO, screen=True, tofile=True)


# Flow Helper
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


def get_noise_func(img, noisetype: str):
    if noisetype == 'gauss10':
        return AddNoiseTorch(10)(img)
    elif noisetype == 'gauss20':
        return AddNoiseTorch(20)(img)
    elif noisetype == 'gauss30':
        return AddNoiseTorch(30)(img)
    elif noisetype == 'gauss40':
        return AddNoiseTorch(40)(img)
    elif noisetype == 'gauss50':
        return AddNoiseTorch(50)(img)
    else:
        return AddNoiseBlind(5, 50)(img)

# --- Datasets ---
TrainingDataset_RGB = DataLoader_RGB(args.data_dir, args.patch_size, args.len, img_aug=True)
train_sampler = DistributedSampler(TrainingDataset_RGB, shuffle=True, drop_last=True) if is_ddp else None
RGBLoader = DataLoader(
    TrainingDataset_RGB,
    batch_size=args.batch_size,
    sampler=train_sampler,
    shuffle=(train_sampler is None),
    drop_last=True,
    pin_memory=True,
    num_workers=8,
    prefetch_factor=4,
)

davisLoader = DataLoader(DataLoader_val(args.val_dir_davis, args.len, 'davis'), batch_size=args.val_batch_size,
                         shuffle=False)
set8Loader = DataLoader(DataLoader_val(args.val_dir_set8, args.len, 'set8'), batch_size=args.val_batch_size,
                        shuffle=False)

# --- Models ---
Denoiser = Denoiser_super().to(device)
Estimator = PWCNet().to(device)
FusionNet_S1 = Fusions1(t=int(args.len - 1), dim=args.n_features).to(device)
FusionNet_S2 = Fusions2(t=int(args.len - 1), dim=args.n_features).to(device)

# Load Checkpoints
load_model_weights(Denoiser, args.nafnet_path, map_location=device)
load_pwcnet_weights(Estimator, args.pwc_path, map_location=device)

# Load Pre-trained Stage 1 Fusion
load_model_weights(FusionNet_S1, args.stage1_path, map_location=device)

# Stage-2 FusionNet is trained from scratch unless you manually load a resume checkpoint.

if is_ddp:
    Denoiser = DDP(Denoiser, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
                   broadcast_buffers=True)
    Estimator = DDP(Estimator, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
                    broadcast_buffers=True)
    FusionNet_S1 = DDP(FusionNet_S1, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
                       broadcast_buffers=True)
    FusionNet_S2 = DDP(FusionNet_S2, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False,
                       broadcast_buffers=True)

# Optimization
steps_per_epoch = len(RGBLoader)
total_iterations = args.epochs * steps_per_epoch
optimizer_F = optim.Adam(FusionNet_S2.parameters(), lr=args.lr_F)
scheduler_F = CosineAnnealingLR(optimizer_F, T_max=total_iterations, eta_min=args.lr_final)

if main_proc:
    log_dir_path = os.path.join(args.save_model_path, args.log_name, systime)
    loss_csv_path = os.path.join(log_dir_path, "train_loss.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f: f.write("epoch,loss\n")
    print("Stage 2 Initialization finished. Training started...")

epoch_init = 1

for epoch in range(epoch_init, args.epochs + 1):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    if main_proc:
        print(tcolor(f"Epoch {epoch}, LearningRate of Fusion = {optimizer_F.param_groups[0]['lr']}", c=Color.Yellow))

    Denoiser.eval()
    Estimator.eval()
    FusionNet_S1.eval()
    FusionNet_S2.train()

    epoch_loss_sum = torch.zeros(1).to(device)
    epoch_batch_cnt = torch.zeros(1).to(device)

    for clean in RGBLoader:
        B, T, C, H, W = clean.shape
        clean = clean.to(device)
        noisy = get_noise_func(clean, args.noisetype)
        ref_frame = T // 2
        not_ref = [i for i in range(T) if i != ref_frame]

        with torch.no_grad():
            coarse = Denoiser(noisy.reshape(-1, C, H, W)).reshape(clean.shape)
            f = flow_to_ref(coarse, Estimator, ref_frame)
            input_s1 = torch.cat([coarse, (noisy - coarse)], dim=2)

            # Forward pass through Stage 1
            output_s1 = FusionNet_S1(input_s1[:, not_ref].reshape(-1, C * 2, H, W), coarse[:, ref_frame], f)

            # Pseudo data generation for Stage 2
            pseudo_n = get_noise_func(output_s1.detach(), args.noisetype)
            noisy[:, ref_frame] = pseudo_n.detach()
            pseudo_coarse = Denoiser(noisy.reshape(-1, C, H, W)).reshape(noisy.shape)
            pseudo_input = torch.cat([pseudo_coarse, (noisy - pseudo_coarse)], dim=2)
            pseudo_f = flow_to_ref(pseudo_coarse, Estimator, ref_frame)

        # Forward pass through Stage 2 (Trainable)
        output = FusionNet_S2(pseudo_input.reshape(-1, C * 2, H, W), pseudo_coarse[:, ref_frame], pseudo_f)
        loss = torch.mean(abs(output - output_s1))

        # Backprop
        optimizer_F.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(FusionNet_S2.parameters(), 0.1)
        optimizer_F.step()
        scheduler_F.step()

        epoch_loss_sum += loss.item()
        epoch_batch_cnt += 1

    if epoch % args.n_snapshot == 0:
        Denoiser.eval()
        Estimator.eval()
        FusionNet_S1.eval()
        FusionNet_S2.eval()

        if main_proc:
            save_path = os.path.join(args.save_path, 'models')
            os.makedirs(save_path, exist_ok=True)
            torch.save(FusionNet_S2.state_dict(), os.path.join(save_path, f'epoch_Fusion_{epoch:03d}.pth'))

            validation_path = os.path.join(args.save_path, "validation")
            os.makedirs(validation_path, exist_ok=True)

        @torch.inference_mode()
        def run_validation_s2(loader, dataset_name, best_metrics):
            psnr1_sum, ssim1_sum = torch.zeros(1, device=device), torch.zeros(1, device=device)
            psnr_co_sum, ssim_co_sum = torch.zeros(1, device=device), torch.zeros(1, device=device)
            cnt_sum = torch.zeros(1, device=device)

            for clean in loader:
                clean = clean.to(device, non_blocking=True)
                B, T, C, H, W = clean.shape

                factor = 8
                h_pad = (factor - H % factor) % factor
                w_pad = (factor - W % factor) % factor

                if h_pad != 0 or w_pad != 0:
                    clean_padded = F.pad(clean.view(-1, C, H, W), (0, w_pad, 0, h_pad), mode='constant')
                    clean_padded = clean_padded.view(B, T, C, H + h_pad, W + w_pad)
                else:
                    clean_padded = clean

                H_padded, W_padded = clean_padded.shape[-2:]
                ref_frame = T // 2

                noisy_padded = get_noise_func(clean_padded, args.noisetype).to(device)

                coarse_padded = Denoiser(noisy_padded.reshape(-1, C, H_padded, W_padded)).reshape(B, T, C, H_padded, W_padded)
                input_fusion_padded = torch.cat([coarse_padded, noisy_padded - coarse_padded], dim=2)
                f_padded = flow_to_ref(coarse_padded, Estimator, ref_frame)

                output_padded = FusionNet_S2(
                    input_fusion_padded.reshape(B * T, C * 2, H_padded, W_padded),
                    coarse_padded[:, ref_frame],
                    f_padded)

                output = output_padded[..., :H, :W]
                coarse = coarse_padded[..., :H, :W]

                output_1 = torch.clamp(output, 0, 1)
                coarse_cr = torch.clamp(coarse, 0, 1)

                psnr1_sum += calculate_psnr_pt(output_1, clean[:, ref_frame]).sum()
                ssim1_sum += calculate_ssim_pt(output_1, clean[:, ref_frame]).sum()
                psnr_co_sum += calculate_psnr_pt(coarse_cr[:, ref_frame], clean[:, ref_frame]).sum()
                ssim_co_sum += calculate_ssim_pt(coarse_cr[:, ref_frame], clean[:, ref_frame]).sum()
                cnt_sum += B

            if is_ddp:
                dist.all_reduce(psnr1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(ssim1_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(psnr_co_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(ssim_co_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(cnt_sum, op=dist.ReduceOp.SUM)

            if main_proc:
                cnt = max(cnt_sum.item(), 1.0)
                psnr_1, ssim_1 = (psnr1_sum / cnt).item(), (ssim1_sum / cnt).item()
                co_psnr, co_ssim = (psnr_co_sum / cnt).item(), (ssim_co_sum / cnt).item()

                if psnr_1 > best_metrics['psnr']:
                    best_metrics.update({'epoch': epoch, 'psnr': psnr_1, 'ssim': ssim_1, 'coarse_psnr': co_psnr,
                                         'coarse_ssim': co_ssim})

                with open(os.path.join(validation_path, f"validate_log_{dataset_name}.csv"), "a") as f:
                    f.write(
                        f"{dataset_name}epoch:{epoch},psnr1:{psnr_1:.6f},ssim1:{ssim_1:.6f},copsnr:{co_psnr:.6f},cossim:{co_ssim:.6f}\n")

                print(
                    f"[{dataset_name}]: PSNR_1 {psnr_1:.4f}, SSIM_1 {ssim_1:.4f}, coarsePSNR {co_psnr:.4f}, coarseSSIM {co_ssim:.4f}")

                color_v_min, color_v_max = (38.6, 39.5) if dataset_name == 'Davis' else (37.3, 38.0)
                print(
                    f"[{dataset_name} Best] @ Epoch {best_metrics['epoch']:03d}: PSNR1 {gradient_num_color(best_metrics['psnr'], v_min=color_v_min, v_max=color_v_max)}, SSIM1 {best_metrics['ssim']:.4f}")

        if epoch == args.epochs:
            run_validation_s2(davisLoader, "Davis", best_metrics_davis)
            run_validation_s2(set8Loader, "Set8", best_metrics_set8)

        if is_ddp: dist.barrier()
        gc.collect()
        torch.cuda.empty_cache()