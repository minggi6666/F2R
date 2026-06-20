"""Stage-2 RAW F2R fusion network with deformable alignment."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import deform_conv2d


def flow_warp(x, flow, mode='bilinear', padding_mode='zeros', align_corners=True):
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, device=x.device, dtype=x.dtype),
                                    torch.arange(0, w, device=x.device, dtype=x.dtype), indexing='ij')
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid = grid.unsqueeze(0).expand(b, -1, -1, -1)
    vgrid = grid + flow.permute(0, 2, 3, 1)

    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(x, vgrid_scaled, mode=mode, padding_mode=padding_mode, align_corners=align_corners)


class FlowGuidedBlock(nn.Module):
    """
    Deformable Convolution-based alignment block for RAW video frames.
    """

    def __init__(self, channels, t, kernel_size=3, padding=1, deformable_groups=8):
        super(FlowGuidedBlock, self).__init__()
        self.T = t
        self.in_channels = channels
        self.out_channels = channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.deformable_groups = deformable_groups

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.in_channels * 3 + 2, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 3 * kernel_size * kernel_size * self.deformable_groups, 3, 1, 1),
        )
        self.beta = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1), requires_grad=True)

        self.weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(self.out_channels))
        self.conv_fusion = nn.Conv2d(self.out_channels * (self.T + 1), self.out_channels, 3, 1, 1)
        self.init_weights()

    def init_weights(self):
        torch.nn.init.constant_(self.conv_offset[-1].weight, 0.)
        torch.nn.init.constant_(self.conv_offset[-1].bias, 0.)
        torch.nn.init.normal_(self.weight, 0., 0.01)
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.)
        torch.nn.init.kaiming_normal_(self.conv_fusion.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, inp, flow):
        x = rearrange(inp, '(b t) c h w -> b t c h w', t=self.T + 1)
        ref = x[:, 0]
        x = rearrange(x[:, 1:], 'b t c h w -> (b t) c h w')

        x_warp = flow_warp(x, flow, mode='bilinear')
        offset = torch.cat([x, x_warp, ref.repeat_interleave(self.T, dim=0), flow], dim=1)
        offset = self.conv_offset(offset)
        o1, o2, mask = torch.chunk(offset, 3, dim=1)

        offset = torch.cat((o1, o2), dim=1)
        offset = offset + flow.flip(1).repeat(1, self.kernel_size * self.kernel_size * self.deformable_groups, 1, 1)
        mask = torch.sigmoid(mask)

        aligned_feats = rearrange(
            deform_conv2d(input=x, offset=offset, weight=self.weight, bias=self.bias, padding=self.padding, mask=mask),
            '(b t) c h w -> b (t c) h w', t=self.T)
        aligned_feats = torch.cat([ref, aligned_feats], dim=1)
        aligned_feat = self.conv_fusion(aligned_feats)

        return aligned_feat * self.beta + ref


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(nn.PixelUnshuffle(2),
                                  nn.Conv2d(n_feat * 4, n_feat, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, x): return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x): return self.body(x)


class Fusion(nn.Module):
    """
    Fusion Network for RAW format images.
    Expects 4-channel Bayer input (in_ch=4, out_ch=4).
    """

    def __init__(self, in_ch=4, t=4, out_ch=4, dim=32, bias=True):
        super(Fusion, self).__init__()
        self.T = t
        self.embed = nn.Conv2d(in_ch * 2, dim, kernel_size=3, stride=1, padding=1, bias=bias)

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

        inp_enc_level1 = self.embed(inp)
        out_enc_level1 = self.lrelu(self.encoder_level1(inp_enc_level1))

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.lrelu(self.encoder_level2(inp_enc_level2))

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.lrelu(self.encoder_level3(inp_enc_level3))

        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.lrelu(self.encoder_level4(inp_enc_level4))

        latent = self.lrelu(self.warp_latent(out_enc_level4, flow3))

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

        inp_enc_level1 = self.lrelu(self.warp_refinement(inp_enc_level1, flow))
        inp_ref_level1 = torch.cat([inp_enc_level1, out_dec_level1], dim=1)
        inp_ref_level1 = self.lrelu(self.reduce_refinement_level(inp_ref_level1))
        out_dec_level1 = self.lrelu(self.refinement(inp_ref_level1))

        out = self.output(out_dec_level1)
        return out + coarse_img