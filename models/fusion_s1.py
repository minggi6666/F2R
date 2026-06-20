"""Stage-1 RGB F2R fusion network."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def flow_warp(x, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
    """
    Warp an image or feature map based on optical flow.
    """
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing='ij'
    )
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    vgrid = grid + flow.permute(0, 2, 3, 1)

    if mode == 'nearest4':
        vgrid_x_floor = 2.0 * torch.floor(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_x_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 0]) / max(w - 1, 1) - 1.0
        vgrid_y_floor = 2.0 * torch.floor(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0
        vgrid_y_ceil = 2.0 * torch.ceil(vgrid[:, :, :, 1]) / max(h - 1, 1) - 1.0

        output00 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output01 = F.grid_sample(x, torch.stack((vgrid_x_floor, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output10 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_floor), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        output11 = F.grid_sample(x, torch.stack((vgrid_x_ceil, vgrid_y_ceil), dim=3), mode='nearest',
                                 padding_mode=padding_mode, align_corners=align_corners)
        return torch.cat([output00, output01, output10, output11], 1)
    else:
        vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
        vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
        vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
        output = F.grid_sample(x, vgrid_scaled, mode=mode, padding_mode=padding_mode, align_corners=align_corners)
        return output


class FlowGuidedBlock(nn.Module):
    """
    Stage 1: Flow-guided feature alignment block using standard convolutions
    with Spatial and Channel Attention (SE & SA).
    """

    def __init__(self, c, t):
        super().__init__()
        self.T = t
        self.conv1 = nn.Conv2d(in_channels=c * t, out_channels=c * t, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c * t, out_channels=c * t, kernel_size=3, padding=1, stride=1, groups=c * t,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=c * t, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        # Channel Attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c * t, out_channels=c * t // 2, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=c * t // 2, out_channels=c * t, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=True),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=True),
            nn.Sigmoid()
        )
        self.gelu = nn.GELU()
        self.norm = nn.InstanceNorm3d(c, affine=True)

    def forward(self, inp, flow):
        x_warp = flow_warp(inp, flow, mode='bilinear')
        x_warp = rearrange(x_warp, '(b t) c h w -> b c t h w', t=self.T)
        x = self.norm(x_warp)
        x = rearrange(x, 'b c t h w -> b (c t) h w')

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x * self.se(x)

        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        x = x * self.sa(torch.cat([max_out, avg_out], dim=1))
        x = self.conv3(x)
        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(n_feat * 4, n_feat, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Fusion(nn.Module):
    """
    Stage 1 Fusion Network based on U-Net architecture.
    Aligns and fuses temporal features to reconstruct the reference frame.
    """

    def __init__(self, in_ch=3, t=4, out_ch=3, dim=32, bias=True):
        super(Fusion, self).__init__()
        self.T = t
        self.embed = nn.Conv2d(in_ch * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # ---------- Encoder ----------
        self.c1 = int(dim)
        self.encoder_level1 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, bias=bias)
        self.down1_2 = Downsample(self.c1)

        self.c2 = int(dim)
        self.encoder_level2 = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, bias=bias)
        self.down2_3 = Downsample(self.c2)

        self.c3 = int(dim)
        self.encoder_level3 = nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=1, bias=bias)
        self.down3_4 = Downsample(self.c3)

        self.c4 = int(dim)
        self.encoder_level4 = nn.Conv2d(self.c4, self.c4, kernel_size=3, padding=1, bias=bias)

        # ---------- Latent & Decoder ----------
        self.warp_latent = FlowGuidedBlock(self.c4, self.T)

        self.up4_3 = Upsample(self.c4)
        self.warp_level3 = FlowGuidedBlock(self.c3, self.T)
        self.reduce_chan_level3 = nn.Conv2d(self.c3 * 2, self.c3, kernel_size=3, padding=1, bias=False)
        self.decoder_level3 = nn.Conv2d(self.c3, self.c3, kernel_size=3, padding=1, bias=bias)

        self.up3_2 = Upsample(self.c3)
        self.warp_level2 = FlowGuidedBlock(self.c2, self.T)
        self.reduce_chan_level2 = nn.Conv2d(self.c2 * 2, self.c2, kernel_size=3, padding=1, bias=False)
        self.decoder_level2 = nn.Conv2d(self.c2, self.c2, kernel_size=3, padding=1, bias=bias)

        self.up2_1 = Upsample(self.c2)
        self.warp_level1 = FlowGuidedBlock(self.c1, self.T)
        self.reduce_chan_level1 = nn.Conv2d(self.c1 * 2, self.c1, kernel_size=3, padding=1, bias=False)
        self.decoder_level1 = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, bias=bias)

        # ---------- Refinement & Output ----------
        self.warp_refinement = FlowGuidedBlock(self.c1, self.T)
        self.reduce_refinement_level = nn.Conv2d(self.c1 * 2, self.c1, kernel_size=3, padding=1, bias=False)
        self.refinement = nn.Conv2d(self.c1, self.c1, kernel_size=3, padding=1, bias=bias)
        self.output = nn.Conv2d(dim, out_ch, kernel_size=3, padding=1, bias=False)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, inp, coarse_img, flow):
        _, _, h, w = inp.shape
        flow1 = F.interpolate(flow, scale_factor=0.5, mode='bilinear', align_corners=True) / 2.0
        flow2 = F.interpolate(flow1, scale_factor=0.5, mode='bilinear', align_corners=True) / 2.0
        flow3 = F.interpolate(flow2, scale_factor=0.5, mode='bilinear', align_corners=True) / 2.0

        # Encoder
        inp_enc_level1 = self.embed(inp)
        out_enc_level1 = self.lrelu(self.encoder_level1(inp_enc_level1))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.lrelu(self.encoder_level2(inp_enc_level2))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.lrelu(self.encoder_level3(inp_enc_level3))

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.lrelu(self.encoder_level4(inp_enc_level4))

        # Latent Warping
        latent = self.lrelu(self.warp_latent(out_enc_level4, flow3))

        # Decoder
        inp_dec_level3 = self.up4_3(latent)
        out_enc_level3 = self.lrelu(self.warp_level3(out_enc_level3, flow2))
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)
        inp_dec_level3 = self.lrelu(self.reduce_chan_level3(inp_dec_level3))
        out_dec_level3 = self.lrelu(self.decoder_level3(inp_dec_level3))

        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_enc_level2 = self.lrelu(self.warp_level2(out_enc_level2, flow1))
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)
        inp_dec_level2 = self.lrelu(self.reduce_chan_level2(inp_dec_level2))
        out_dec_level2 = self.lrelu(self.decoder_level2(inp_dec_level2))

        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_enc_level1 = self.lrelu(self.warp_level1(out_enc_level1, flow))
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)
        inp_dec_level1 = self.lrelu(self.reduce_chan_level1(inp_dec_level1))
        out_dec_level1 = self.lrelu(self.decoder_level1(inp_dec_level1))

        # Refinement & Residual Connection
        inp_enc_level1 = self.lrelu(self.warp_refinement(inp_enc_level1, flow))
        inp_ref_level1 = torch.cat([inp_enc_level1, out_dec_level1], dim=1)
        inp_ref_level1 = self.lrelu(self.reduce_refinement_level(inp_ref_level1))
        out_dec_level1 = self.lrelu(self.refinement(inp_ref_level1))

        out = self.output(out_dec_level1)
        return out + coarse_img