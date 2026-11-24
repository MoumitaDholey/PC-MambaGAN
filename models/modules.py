# coding=utf-8
"""
PC-MambaGAN generator building blocks.

Paper term                              -> code
---------------------------------------    ---------------------------------------------
Generator (encoder / bottleneck / decoder) -> I2IMamba
PC-Mamba block                             -> prMambaWithCNN
  local branch (BN + MobileNetV3)          ->   ParallelConvLayer
  bidirectional square-scan SSM            ->   MambaLayerOnlyLBP
  FACF (Frequency-Aware Channel Fusion)    ->   ChannelCompression (uses EinFFT)
  SFMAM (Selective Feature Modulation)     ->   SFMAM
MobileNetV3 refinement stage               -> BottleneckCNN

Attribute names inside the modules are kept unchanged so that checkpoints
trained with the original code load into this version.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from mamba_ssm import Mamba

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# MobileNetV3 block (local refinement)
# ---------------------------------------------------------------------------
class MobileNetV3Block(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, expansion_factor=6, use_se=True,
                 activation="h-swish"):
        super(MobileNetV3Block, self).__init__()
        self.use_residual = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion_factor

        # Activation function
        self.activation = nn.Hardswish(inplace=True) if activation == "h-swish" else nn.ReLU(inplace=True)

        # Expansion: Pointwise convolution
        self.expansion = nn.Sequential(
            nn.Conv2d(in_channels, expanded_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            self.activation,
        )

        # Depthwise convolution
        self.depthwise = nn.Sequential(
            nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3, stride=stride, padding=1,
                      groups=expanded_channels, bias=False),
            nn.BatchNorm2d(expanded_channels),
            self.activation,
        )

        # Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation(expanded_channels)

        # Projection: Pointwise convolution
        self.projection = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        identity = x
        x = self.expansion(x)
        x = self.depthwise(x)
        if self.use_se:
            x = self.se(x)
        x = self.projection(x)
        if self.use_residual:
            return x + identity  # Residual connection
        else:
            return x


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExcitation, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.activation = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        # Squeeze
        scale = torch.mean(x, dim=(2, 3), keepdim=True)
        scale = self.activation(self.fc2(torch.relu(self.fc1(scale))))
        # Excitation
        return x * scale


class BottleneckCNN(nn.Module):
    """MobileNetV3 refinement stage of the interleaved bottleneck."""

    def __init__(self):
        super(BottleneckCNN, self).__init__()
        setattr(self, "bottleneck_cnn", nn.Sequential(MobileNetV3Block()))

    def forward(self, x):
        x = self.bottleneck_cnn(x)
        return x


# ---------------------------------------------------------------------------
# FACF: Frequency-Aware Channel Fusion
# ---------------------------------------------------------------------------
# SiMBA: Simplified Mamba-based Architecture for Vision and Multivariate Time series
class EinFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden_size = dim
        self.num_blocks = 4
        self.block_size = self.hidden_size // self.num_blocks
        assert self.hidden_size % self.num_blocks == 0
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.complex_weight_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_weight_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_1 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)
        self.complex_bias_2 = nn.Parameter(
            torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * self.scale)

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, N, self.num_blocks, self.block_size)

        x = torch.fft.fft2(x, dim=(1, 2), norm='ortho')  # FFT on N dimension

        x_real_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[0]) - self.multiply(x.imag, self.complex_weight_1[1]) +
            self.complex_bias_1[0])
        x_imag_1 = F.relu(
            self.multiply(x.real, self.complex_weight_1[1]) + self.multiply(x.imag, self.complex_weight_1[0]) +
            self.complex_bias_1[1])
        x_real_2 = self.multiply(x_real_1, self.complex_weight_2[0]) - self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[1]) + \
                   self.complex_bias_2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_2[1]) + self.multiply(x_imag_1,
                                                                                     self.complex_weight_2[0]) + \
                   self.complex_bias_2[1]

        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = F.softshrink(x, lambd=self.sparsity_threshold) if self.sparsity_threshold else x
        x = torch.view_as_complex(x)

        x = torch.fft.ifft2(x, dim=(1, 2), norm="ortho")

        # keep the real part ("fused_dropout" is not implemented for ComplexFloat)
        x = x.to(torch.float32)
        x = x.reshape(B, N, C)
        return x


class ChannelCompression(nn.Module):
    """FACF: spectral path (LayerNorm -> EinFFT -> 1x1 conv) + spatial path (two 3x3 convs)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(ChannelCompression, self).__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.einfft = EinFFT(in_channels)
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        out = self.block(x)
        fft_out = x.flatten(2).permute(0, 2, 1)  # Reshape to (B, N, C)
        fft_out = self.einfft(self.norm(fft_out))
        fft_out = fft_out.permute(0, 2, 1).view(B, int(C / 1), H, W)
        fft_out = self.conv(fft_out)
        out += fft_out
        out = F.relu(out)
        return out


# ---------------------------------------------------------------------------
# Bidirectional square-scan state-space operator
# ---------------------------------------------------------------------------
class MambaLayerOnlyLBP(nn.Module):
    """Bidirectional square-scan Mamba layer.

    The scan order is stored as permutation matrices in models/lbp_*.npy,
    produced by models/lbp_path_generate.py.

    Args:
        dim (int): Model dimension.
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
    """

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba1 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.mamba2 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.conv1d = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        self.spiral_eye = torch.tensor(np.load(os.path.join(_MODEL_DIR, "lbp_spiral_eye.npy")),
                                       dtype=torch.float)
        self.despiral_eye = torch.tensor(np.load(os.path.join(_MODEL_DIR, "lbp_despiral_eye.npy")),
                                         dtype=torch.float)
        self.despiral_r_eye = torch.tensor(np.load(os.path.join(_MODEL_DIR, "lbp_despiral_r_eye.npy")),
                                           dtype=torch.float)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim

        x1 = x.view(B, C, -1)
        device = next(self.parameters()).device
        x1 = x1.to(device)
        self.spiral_eye = self.spiral_eye.to(device)
        x1 = torch.einsum('ij,klj->kli', self.spiral_eye, x1).permute(0, 2, 1)   # square-scan order
        x2 = torch.flip(x1, dims=[1])                                               # reverse order

        # forward and reverse scans through two separate Mamba SSMs
        norm_out1 = self.norm(x1)
        mamba_out1 = self.mamba1(norm_out1)
        norm_out2 = self.norm(x2)
        mamba_out2 = self.mamba2(norm_out2)

        # back to the spatial grid
        self.despiral_eye = self.despiral_eye.to(device)
        self.despiral_r_eye = self.despiral_r_eye.to(device)
        out1 = torch.einsum('ij,klj->kli', self.despiral_eye, mamba_out1.permute(0, 2, 1)).view(B, C, H, W)
        out2 = torch.einsum('ij,klj->kli', self.despiral_r_eye, mamba_out2.permute(0, 2, 1)).view(B, C, H, W)

        concatenated = torch.cat((out1, out2), dim=1)
        output = self.conv1d(concatenated)
        mamba_out = output.view(-1, 256, 64, 64)

        return mamba_out


# ---------------------------------------------------------------------------
# Local branch of the PC-Mamba block
# ---------------------------------------------------------------------------
class ParallelConvLayer(nn.Module):
    """Local branch: BatchNorm followed by a MobileNetV3 block."""

    def __init__(self, in_channels=256, out_channels=256):
        super().__init__()
        self.dim = in_channels
        self.norm = nn.BatchNorm2d(self.dim)
        setattr(self, "bottleneck_conv_cnn", nn.Sequential(MobileNetV3Block()))

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.dim
        device = next(self.parameters()).device
        x = x.to(device)
        out = self.norm(x)
        out = self.bottleneck_conv_cnn(out)
        return out


# ---------------------------------------------------------------------------
# SFMAM: Selective Feature Modulation Aggregation Module
# ---------------------------------------------------------------------------
def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    """Channel shuffle for an input of shape [B, H, W, C]."""
    batch_size, height, width, num_channels = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, height, width, groups, channels_per_group)
    x = torch.transpose(x, 3, 4).contiguous()
    x = x.view(batch_size, height, width, -1)
    return x


class eca_layer(nn.Module):
    """Efficient Channel Attention (ECA) with a per-channel 1D convolution."""

    def __init__(self, channel, k_size):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.k_size = k_size
        self.conv = nn.Conv1d(channel, channel, kernel_size=k_size, bias=False, groups=channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)  # [B, C, 1, 1]
        y = nn.functional.unfold(y.transpose(-1, -3),
                                 kernel_size=(1, self.k_size),
                                 padding=(0, (self.k_size - 1) // 2))
        y = self.conv(y.transpose(-1, -2)).unsqueeze(-1)
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        return x


class SFMAM(nn.Module):
    """
    Selective Feature Modulation Aggregation Module (SFMAM)
      - complementary cross-gating of the local and global features
      - concat + NHWC channel shuffle
      - ECA attention

    Input :  F_L, F_G : [B, C, H, W]
    Output:  Y        : [B, 2C, H, W]
    """

    def __init__(self, ch_1: int, shuffle_groups: int = None, eca_k: int = 1):
        super().__init__()
        C = ch_1
        self.C = C

        # Depthwise conv shared by both gates
        self.DWConv = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=3,
            padding=1,
            groups=C,
            bias=True
        )

        self.relu = nn.ReLU(inplace=True)
        self.gelu = nn.GELU()
        self.sigmoid = nn.Sigmoid()

        # Shuffle groups must divide 2C (default C // 2)
        self.shuffle_groups = shuffle_groups if shuffle_groups is not None else (C // 2)

        # ECA attention on concatenated channels
        self.attention = eca_layer(channel=2 * C, k_size=eca_k)

    def forward(self, F_L: Tensor, F_G: Tensor) -> Tensor:
        assert F_L.shape == F_G.shape, \
            f"F_L and F_G must have same shape, got {F_L.shape} vs {F_G.shape}"
        B, C, H, W = F_L.shape
        assert C == self.C, f"Expected channel={self.C}, got {C}"

        # ---- Step 1: modulation gates ----
        g_L = self.sigmoid(self.relu(self.DWConv(F_L)))  # local gate
        g_G = self.sigmoid(self.gelu(self.DWConv(F_G)))  # global gate

        # ---- Step 2: complementary modulation ----
        R_L = g_L * F_G
        R_G = g_G * F_L

        S_L = (1.0 - g_G) * F_L
        S_G = (1.0 - g_L) * F_G

        F_L_tilde = R_L + S_L
        F_G_tilde = R_G + S_G

        # ---- Step 3: aggregation ----
        out = torch.cat([F_L_tilde, F_G_tilde], dim=1)  # [B,2C,H,W]

        # ---- Step 4: shuffle + ECA ----
        out = out.permute(0, 2, 3, 1).contiguous()  # NHWC
        assert out.shape[-1] % self.shuffle_groups == 0, \
            f"Shuffle requires channels(={out.shape[-1]}) % groups(={self.shuffle_groups}) == 0"
        out = channel_shuffle(out, groups=self.shuffle_groups)
        out = out.permute(0, 3, 1, 2).contiguous()  # NCHW

        out = self.attention(out)
        return out


# ---------------------------------------------------------------------------
# PC-Mamba block
# ---------------------------------------------------------------------------
class prMambaWithCNN(nn.Module):
    """PC-Mamba block: parallel local branch and square-scan SSM, FACF, SFMAM, 1x1 projection.

    Args:
        in_channels (int): Number of input channels (256).
        d_state (int): SSM state expansion factor.
        d_conv (int): Local convolution width.
        expand (int): Block expansion factor.
    """

    def __init__(self, in_channels, d_state=16, d_conv=4, expand=2):
        super().__init__()
        ngf = 64

        # Local branch
        self.conv_layer = ParallelConvLayer(in_channels=in_channels)

        # Global branch: bidirectional square-scan SSM
        self.mamba_layer = MambaLayerOnlyLBP(dim=in_channels, d_state=d_state, d_conv=d_conv, expand=expand)

        # FACF: Concat[input, SSM output] (2C) -> C
        self.cc = ChannelCompression(ngf * 8, ngf * 4)

        # 1x1 projection after SFMAM (2C -> C)
        self.conv1d = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)

        # SFMAM (attribute name "mifa" kept for checkpoint compatibility)
        self.mifa = SFMAM(ch_1=256)

    def forward(self, x):
        conv_out = self.conv_layer(x)                  # F_loc

        mamba_out = self.mamba_layer(x)                # F_glob
        x2 = torch.cat([x, mamba_out], dim=1)
        x2 = self.cc(x2)                               # F_spec

        MAP = self.mifa(conv_out, x2)                  # SFMAM
        MAP = self.conv1d(MAP)
        output = MAP.view(-1, 256, 64, 64)
        return output


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class I2IMamba(nn.Module):
    """PC-MambaGAN generator: U-Net encoder, 10-stage interleaved bottleneck, decoder.

    Bottleneck schedule: PC-Mamba blocks at stages 1, 4, 7, 10; MobileNetV3 blocks elsewhere.
    """

    def __init__(self, input_dim, output_dim=1):
        super(I2IMamba, self).__init__()
        ngf = 64
        use_bias = False
        norm_layer = nn.BatchNorm2d

        # Encoder 1
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_dim, ngf, kernel_size=7, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        setattr(self, "encoder_1", nn.Sequential(*model))

        # Encoder 2
        mult = 1
        model = [
            nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
        setattr(self, "encoder_2", nn.Sequential(*model))

        # Encoder 3
        mult = 2
        model = [
            nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True),
        ]
        setattr(self, "encoder_3", nn.Sequential(*model))

        # Interleaved bottleneck (64 x 64 x 256)
        channels = 256
        self.bottleneck_1 = prMambaWithCNN(channels)
        self.bottleneck_2 = BottleneckCNN()
        self.bottleneck_3 = BottleneckCNN()
        self.bottleneck_4 = prMambaWithCNN(channels)
        self.bottleneck_5 = BottleneckCNN()
        self.bottleneck_6 = BottleneckCNN()
        self.bottleneck_7 = prMambaWithCNN(channels)
        self.bottleneck_8 = BottleneckCNN()
        self.bottleneck_9 = BottleneckCNN()
        self.bottleneck_10 = prMambaWithCNN(channels)

        # Decoder 1
        mult = 4
        model = [
            nn.ConvTranspose2d(2 * ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        setattr(self, "decoder_1", nn.Sequential(*model))

        # Decoder 2
        mult = 2
        model = [
            nn.ConvTranspose2d(2 * ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(ngf * mult / 2)),
            nn.ReLU(True),
        ]
        setattr(self, "decoder_2", nn.Sequential(*model))

        # Decoder 3
        model = [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(2 * ngf, output_dim, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        setattr(self, "decoder_3", nn.Sequential(*model))

    def forward(self, x):
        # Encoder
        x1 = self.encoder_1(x)
        x2 = self.encoder_2(x1)
        x3 = self.encoder_3(x2)

        # Interleaved bottleneck (10 stages)
        x = self.bottleneck_1(x3)
        x = self.bottleneck_2(x)
        x = self.bottleneck_3(x)
        x = self.bottleneck_4(x)
        x = self.bottleneck_5(x)
        x = self.bottleneck_6(x)
        x = self.bottleneck_7(x)
        x = self.bottleneck_8(x)
        x = self.bottleneck_9(x)
        x = self.bottleneck_10(x)

        # Decoder with skip connections
        x = self.decoder_1(torch.cat([x, x3], dim=1))
        x = self.decoder_2(torch.cat([x, x2], dim=1))
        x = self.decoder_3(torch.cat([x, x1], dim=1))
        return x
