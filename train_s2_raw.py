"""Train Stage-2 RAW F2R: center-frame residual refinement with a frozen Stage-1 model."""

import os
import gc
import random
import logging
import argparse
import datetime
import time
from collections import OrderedDict

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
from models.fusion_raw_s2 import Fusion as Fusions2
from models.nafnet_raw import Baseline as Denoiser_super
from models.pytorch_pwc.pwc import PWCNet as Estimator
from models.pytorch_pwc.extract_flow import extract_flow_torch
from data.dataset_crvd import CRVDTDataset, CRVDVDataset, add_raw_noise
from utils.utils import setup_logger, calculate_psnr_pt, calculate_ssim_pt, demosaic
from utils.log import tcolor, Color, gradient_num_color
from utils.checkpoint import load_model_weights, load_pwcnet_weights

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='./datasets/CRVD', help='CRVD dataset root')
parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs')
parser.add_argument('--lr_F', type=float, default=3e-4, help='Learning rate for training')
parser.add_argument('--lr_final', type=float, default=0, help='Final learning rate')
parser.add_argument('--batch_size', type=int, default=8, help='Size of each training batch')
parser.add_argument('--len', type=int, default=7, help='Sequence length (T)')
parser.add_argument('--patch_size', type=int, default=256)
parser.add_argument('--n_channel', type=int, default=4)
parser.add_argument('--n_features', type=int, default=72)
parser.add_argument('--gpu_devices', default='0', type=str)
parser.add_argument('--nafnet_path', type=str, default='./models/nafnet_raw.pth', help='Frozen RAW NAFNet checkpoint')
parser.add_argument('--stage1_path', type=str, default='./models/xxx.pth', help='Trained Stage-1 RAW Fusion checkpoint')
parser.add_argument('--stage2_init_path', type=str, default=None, help='Optional Stage-2 checkpoint for warm start/resume-style training')
parser.add_argument('--start_epoch', type=int, default=1, help='First epoch index. Use >1 when continuing Stage-2 training.')
parser.add_argument('--pwc_path', type=str, default='./models/pwc-default.pth', help='Optional local PWC-Net checkpoint; downloaded if missing')
parser.add_argument('--n_snapshot', type=int, default=5)

parser.add_argument('--log_name', type=str, default='f2r_raw_s2')
parser.add_argument('--save_model_path', type=str, default='./experiments')

# Config Init
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
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
        os.environ.setdefault("OMP_NUM_THREADS", "1")
    return is_ddp, main_proc, local_rank, world_size, device


@torch.no_grad()
def dist_all_reduce_sum_(t: torch.Tensor):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t

is_ddp, main_proc, local_rank, world_size, device = setup_distributed()
systime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
torch.set_num_threads(10)

args.save_path = os.path.join(args.save_model_path, args.log_name, systime)
os.makedirs(args.save_path, exist_ok=True)
setup_logger("train", args.save_path, "train_" + args.log_name, level=logging.INFO, screen=True, tofile=True)

def flow_to_ref(coarse, estimator, ref_frame):
    B, T, C, H, W = coarse.shape
    flow_list = []
    for i in range(T):
        flow_ir = extract_flow_torch(estimator, ref_frame, coarse[:, i])
        flow_list.append(flow_ir)
    flow_stack = torch.stack(flow_list, dim=1).squeeze(2)
    return rearrange(flow_stack, 'b t xy h w -> (b t) xy h w')


def add_random_raw_noise_to_center(center_clean: torch.Tensor) -> torch.Tensor:
    """Add random CRVD-style RAW noise only to the pseudo center frame.

    Stage-2 training first uses the frozen Stage-1 fusion model to predict a
    pseudo-clean center frame. Instead of passing a fixed noise type or ISO
    argument, this function randomly samples one of the CRVD Poisson-Gaussian
    noise profiles inside ``add_raw_noise`` for each batch item.

    Args:
        center_clean: Tensor with shape ``[B, C, H, W]`` and value range [0, 1].

    Returns:
        Noisy pseudo center frame with shape ``[B, C, H, W]``.
    """
    if center_clean.dim() != 4:
        raise ValueError(f"Expected center_clean as [B, C, H, W], got {tuple(center_clean.shape)}")
    return add_raw_noise(center_clean.unsqueeze(1)).squeeze(1)


# --- Datasets ---
Traincrvd = DataLoader(CRVDTDataset(args.data_dir, args.len, args.patch_size),
                       batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

val_iso_list = [1600, 3200, 6400, 12800, 25600]
val_loaders = {}
for iso in val_iso_list:
    val_loaders[f"ISO{iso}"] = DataLoader(CRVDVDataset(args.data_dir, iso, args.len),
                                          batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

if main_proc:
    print(f"验证集准备完毕: 包含 {len(val_loaders)} 个 ISO 等级 ({val_iso_list})")

best_metrics_all = {}
for iso in val_iso_list:
    best_metrics_all[f"ISO{iso}"] = {'epoch': 0, 'psnr': 0.0, 'ssim': 0.0, 'coarse_psnr': 0.0, 'coarse_ssim': 0.0}

# --- Models ---
Denoiser = Denoiser_super().to(device)
Estimator = Estimator().to(device)
FusionNet_S1 = Fusions1(t=int(args.len - 1), dim=args.n_features).to(device)
FusionNet_S2 = Fusions2(t=int(args.len - 1), dim=args.n_features).to(device)

# Load Checkpoints
load_model_weights(Denoiser, args.nafnet_path, map_location=device)
load_pwcnet_weights(Estimator, args.pwc_path, map_location=device)

# Load Stage-1 pretrained model used to generate pseudo-clean center targets.
load_model_weights(FusionNet_S1, args.stage1_path, map_location=device)

# Optional: warm-start Stage-2 from a previous checkpoint. This replaces the
# previous hard-coded ``epoch_Fusion_125.pth`` style loading and keeps the
# open-source script path-agnostic.
if args.stage2_init_path is not None:
    load_model_weights(FusionNet_S2, args.stage2_init_path, map_location=device)

if is_ddp:
    Denoiser = DDP(Denoiser, device_ids=[local_rank], output_device=local_rank)
    Estimator = DDP(Estimator, device_ids=[local_rank], output_device=local_rank)
    FusionNet_S1 = DDP(FusionNet_S1, device_ids=[local_rank], output_device=local_rank)
    FusionNet_S2 = DDP(FusionNet_S2, device_ids=[local_rank], output_device=local_rank)

# Optimization
steps_per_epoch = len(Traincrvd)
total_iterations = args.epochs * steps_per_epoch
optimizer_F = optim.Adam(FusionNet_S2.parameters(), lr=args.lr_F)

# Keep scheduler position consistent when ``--start_epoch`` is used together
# with ``--stage2_init_path``. For fresh training, last_epoch remains -1.
for param_group in optimizer_F.param_groups:
    param_group.setdefault('initial_lr', args.lr_F)
scheduler_last_epoch = max(args.start_epoch - 1, 0) * steps_per_epoch - 1
scheduler_F = CosineAnnealingLR(optimizer_F, T_max=total_iterations, eta_min=args.lr_final, last_epoch=scheduler_last_epoch)

epoch_init = args.start_epoch

if main_proc:
    log_dir_path = os.path.join(args.save_model_path, args.log_name, systime)
    loss_csv_path = os.path.join(log_dir_path, "train_loss.csv")
    if not os.path.exists(loss_csv_path):
        with open(loss_csv_path, "w") as f: f.write("epoch,loss\n")
    print(f"Batchsize={args.batch_size}, number of epoch={args.epochs}")
    print('Initialization finished. Training started...')

for epoch in range(epoch_init, args.epochs + 1):
    if main_proc:
        current_lr_F = optimizer_F.param_groups[0]['lr']
        print(tcolor(f"Epoch {epoch}, LearningRate of Fusion = {current_lr_F}", c=Color.Yellow))

    Denoiser.eval()
    Estimator.eval()
    FusionNet_S1.eval()
    FusionNet_S2.train()

    epoch_loss_sum = torch.zeros(1).to(device)
    epoch_batch_cnt = torch.zeros(1).to(device)

    for noisy, ref_frame in Traincrvd:
        B, T, C, H, W = noisy.shape
        noisy = noisy.to(device)
        ref_frame = ref_frame.to(device)

        with torch.no_grad():
            coarse = Denoiser(noisy.reshape(-1, C, H, W)).reshape(noisy.shape)
            deref = Denoiser(ref_frame).reshape(ref_frame.shape)

            # --- Stage 1 Inference ---
            f1 = flow_to_ref(demosaic(coarse), Estimator, demosaic(deref.unsqueeze(1)).squeeze(1))
            input_s1 = torch.cat([coarse, noisy - coarse], dim=2)
            output_s1 = FusionNet_S1(input_s1.reshape(B * T, C * 2, H, W), deref, f1)

            # --- Pseudo Data Generation ---
            pseudo_center_noisy = add_random_raw_noise_to_center(output_s1.detach())
            pseudo_noisy = torch.cat([pseudo_center_noisy.unsqueeze(1), noisy], dim=1)

            pseudo_coarse = Denoiser(pseudo_noisy.reshape(-1, C, H, W)).reshape(B, T + 1, C, H, W)
            pseudo_input = torch.cat([pseudo_coarse, (pseudo_noisy - pseudo_coarse)], dim=2)

            # Estimate flow from each neighboring frame to the noisy pseudo
            # center frame after coarse denoising.
            pseudo_f = flow_to_ref(demosaic(pseudo_coarse[:, 1:]), Estimator, demosaic(pseudo_coarse[:, 0].unsqueeze(1)).squeeze(1))

        # --- Stage 2 Training ---
        output = FusionNet_S2(pseudo_input.reshape(-1, C * 2, H, W), pseudo_coarse[:, 0], pseudo_f)

        # Loss using pseudo clean target from S1
        loss = torch.mean(abs(output - output_s1))

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
        with open(loss_csv_path, "a") as f:
            f.write(f"{epoch},{avg_loss:.6f}\n")
        print(f"Epoch {epoch} Training Loss: {avg_loss:.6f} (Saved to CSV)")

    if epoch % args.n_snapshot == 0:
        Denoiser.eval()
        Estimator.eval()
        FusionNet_S1.eval()
        FusionNet_S2.eval()

        torch.cuda.synchronize()
        if is_ddp: dist.barrier()

        if main_proc:
            save_path = os.path.join(args.save_path, 'models')
            os.makedirs(save_path, exist_ok=True)
            torch.save(FusionNet_S2.state_dict(), os.path.join(save_path, f'epoch_Fusion_{epoch:03d}.pth'))

            validation_path = os.path.join(args.save_path, "validation")
            os.makedirs(validation_path, exist_ok=True)


        def clear_cache():
            gc.collect()
            torch.cuda.empty_cache()


        @torch.inference_mode()
        def run_validation_over_loader(loader, dataset_name: str, device, best_metrics: dict):
            psnr1_sum = torch.zeros(1, device=device)
            ssim1_sum = torch.zeros(1, device=device)
            psnr_co_sum = torch.zeros(1, device=device)
            ssim_co_sum = torch.zeros(1, device=device)
            cnt_sum = torch.zeros(1, device=device)

            for _, valid_image in enumerate(loader):
                noisy, clean, ref_frame = valid_image
                clean = clean.to(device)
                noisy = noisy.to(device)
                ref_frame = ref_frame.to(device)
                B, T, C, H, W = noisy.shape

                # Padding logic
                factor = 8
                h_pad = (factor - H % factor) % factor
                w_pad = (factor - W % factor) % factor

                if h_pad != 0 or w_pad != 0:
                    ref_frame_padded = F.pad(ref_frame, (0, w_pad, 0, h_pad), mode='constant')
                    noisy_padded = F.pad(noisy.view(-1, C, H, W), (0, w_pad, 0, h_pad), mode='constant').view(B, T, C, H + h_pad, W + w_pad)
                else:
                    noisy_padded = noisy
                    ref_frame_padded = ref_frame

                H_padded, W_padded = noisy_padded.shape[-2:]

                # Inference via FusionNet_S2
                coarse_padded = Denoiser(noisy_padded.reshape(-1, C, H_padded, W_padded)).reshape(B, T, C, H_padded, W_padded)
                deref_padded = Denoiser(ref_frame_padded)

                f = flow_to_ref(demosaic(coarse_padded), Estimator, demosaic(deref_padded.unsqueeze(1)).squeeze(1))

                coarse_padded = torch.cat([deref_padded.unsqueeze(1), coarse_padded], dim=1)
                noisy_padded = torch.cat([ref_frame_padded.unsqueeze(1), noisy_padded], dim=1)

                input_fusion_padded = torch.cat([coarse_padded, noisy_padded - coarse_padded], dim=2)
                output_padded = FusionNet_S2(input_fusion_padded.reshape(B * (T + 1), C * 2, H_padded, W_padded),
                                             deref_padded, f)

                # Unpad & Clamp
                output = output_padded[..., :H, :W]
                deref = deref_padded[..., :H, :W]

                def clamp_round(x):
                    x = torch.clamp(x, 0, 1)
                    return (torch.round(x * (2 ** 12 - 1 - 240) + 240) - 240) / (2 ** 12 - 1 - 240)

                output = clamp_round(output)
                deref = clamp_round(deref)

                # Metrics calculation
                psnr = calculate_psnr_pt(output, clean, crop_border=0)
                ssim = calculate_ssim_pt(output, clean, crop_border=0)
                psnr1_sum += psnr.sum()
                ssim1_sum += ssim.sum()

                psnr_c = calculate_psnr_pt(deref, clean, crop_border=0)
                ssim_c = calculate_ssim_pt(deref, clean, crop_border=0)
                psnr_co_sum += psnr_c.sum()
                ssim_co_sum += ssim_c.sum()

                cnt_sum += torch.tensor([B], device=device, dtype=psnr1_sum.dtype)

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

                log_path = os.path.join(validation_path, f"validate_log_{dataset_name}.csv")
                with open(log_path, "a") as f:
                    f.write(
                        f"epoch:{epoch},psnr1:{psnr_1:.4f},ssim1:{ssim_1:.4f},copsnr:{co_psnr:.4f},cossim:{co_ssim:.4f}\n")

                print(f"[{dataset_name}] Epoch {epoch}: PSNR {psnr_1:.4f} / SSIM {ssim_1:.4f} "
                      f"(Best: {best_metrics['psnr']:.4f} @ Ep{best_metrics['epoch']}), "
                      f"coarsePSNR {best_metrics['coarse_psnr']:.4f}, coarseSSIM {best_metrics['coarse_ssim']:.4f}")


        # Iterate over all ISO validation loaders
        if main_proc: print(f"--- Starting Validation @ Epoch {epoch} ---")
        for iso_name, loader in val_loaders.items():
            clear_cache()
            run_validation_over_loader(loader, iso_name, device, best_metrics_all[iso_name])
            clear_cache()
            torch.cuda.synchronize()

        if is_ddp: dist.barrier()

# if dist.is_initialized(): dist.destroy_process_group()