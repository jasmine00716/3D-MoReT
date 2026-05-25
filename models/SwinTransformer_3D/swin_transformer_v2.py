# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
from einops import rearrange


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1,1,1), stride=1, padding=(0, 0, 0), bias=False),
        # nn.BatchNorm3d(oup),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1,3,3), stride=1, padding=(0, 1, 1), bias=False),
        # nn.BatchNorm3d(oup),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.gn = nn.GroupNorm(1, int(n_channels)).to(torch.device("cuda:5"))
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False).to(torch.device("cuda:5"))
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 4),
                               padding=(1, 1, 1), bias=False).to(torch.device("cuda:5"))
        self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False).to(torch.device("cuda:5"))

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        out = self.conv3(F.relu(self.gn(out)))
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_channels, n_output_channels):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.conv1x1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return x

# Dual up-sample
class UpSample(nn.Module):
    def __init__(self, input_resolution, in_channels, out_channels, scale_factor):
        super(UpSample, self).__init__()
        self.input_resolution = input_resolution
        self.factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.up_b = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(torch.device("cuda:5"))

        # if self.factor == 2:
        #     self.up_b1 = nn.ConvTranspose3d(768, 384, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(torch.device("cuda:5"))
        #     self.resblock1 = ResidualBlock(768)
        #     self.up_b2 = nn.ConvTranspose3d(384, 192, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(torch.device("cuda:5"))
        #     self.resblock2 = ResidualBlock(384).to(dev_0)
        #     self.up_b3 = nn.ConvTranspose3d(192, 96, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(torch.device("cuda:5"))
        #     self.resblock3 = ResidualBlock(192).to(dev_0)
        #     self.up_b4 = nn.ConvTranspose3d(96, 40, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(torch.device("cuda:5"))
        #     self.resblock4 = ResidualBlock(96).to(dev_0)
            # self.conv = nn.Conv3d(in_channels, in_channels//2, (1, 1 , 1), 1, (0, 0, 0), bias=False)
            # self.up_p = nn.Sequential(nn.Conv3d(in_channels, 2*in_channels, 1, 1, 0, bias=False),
            #                           nn.PReLU(),
            #                           nn.PixelShuffle(scale_factor),
            #                           nn.Conv3d(in_channels//2, in_channels//2, (1, 1, 1), stride=1, padding=(0, 0, 0), bias=False))
            #
            # self.up_b = nn.Sequential(nn.Conv3d(in_channels, in_channels, 1, 1, 0),
            #                           nn.PReLU(),
            #                           nn.Upsample(scale_factor=(1, 2, 2), mode='nearest', align_corners=False),
            #                           nn.Conv3d(in_channels, in_channels // 2, 1, stride=1, padding=0, bias=False))
        # elif self.factor == 4:
        #     self.conv = nn.Conv3d(2*in_channels, in_channels, 1, 1, 0, bias=False)
        #     self.up_p = nn.Sequential(nn.Conv3d(in_channels, 16 * in_channels, (1, 1, 1), 1, 0, bias=False),
        #                               nn.PReLU(),
        #                               nn.PixelShuffle(scale_factor),
        #                               nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))
        #
        #     self.up_b = nn.Sequential(nn.Conv3d(in_channels, in_channels, 1, 1, 0),
        #                               nn.PReLU(),
        #                               nn.Upsample(scale_factor=(1, 4, 4), mode='nearest', align_corners=False),
        #                               nn.Conv3d(in_channels, in_channels, 1, stride=1, padding=0, bias=False))
    def forward(self, x):
        """
        x: B, L = S*H*W, C
        """
        S, H, W = self.input_resolution
        B, L, C = x.shape

        x = x.view(B, S, H, W, C)  # B, S, H, W, C
        x = x.permute(0, 4, 1, 2, 3)  # B, C, S, H, W
        out = self.up_b(x)

        # x_p = self.up_p(x)  # pixel shuffle
        # x_b = self.up_b(x)  # bilinear
        # out = self.conv(torch.cat([x_p, x_b], dim=1))
        # up_b1 = self.up_b1(x)
        # x = torch.cat((up_b1, mvit_0), dim=1)

        # x = self.up_b2(x)
        # print(x.shape)
        # x = self.up_b3(x)
        # print(x.shape)
        # out = self.up_b4(x)
        # print(out.shape)

        return out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features).to(torch.device("cuda:5"))
        self.act = act_layer().to(torch.device("cuda:5"))
        self.fc2 = nn.Linear(hidden_features, out_features).to(torch.device("cuda:5"))
        self.drop = nn.Dropout(drop).to(torch.device("cuda:5"))

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, S, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, S, H, W, C = x.shape
    x = x.view(B, S, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, S, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, S, H, W, C)
    """
    B = int(windows.shape[0] / (S * H * W / window_size / window_size))
    x = windows.view(B, S, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, S, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.,
                 pretrained_window_size=[0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to(torch.device("cuda:5"))

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False)).to(torch.device("cuda:5"))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=False).to(torch.device("cuda:5"))
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim)).to(torch.device("cuda:5"))
            self.v_bias = nn.Parameter(torch.zeros(dim)).to(torch.device("cuda:5"))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop).to(torch.device("cuda:5"))
        self.proj = nn.Linear(dim, dim).to(torch.device("cuda:5"))
        self.proj_drop = nn.Dropout(proj_drop).to(torch.device("cuda:5"))
        self.softmax = nn.Softmax(dim=-1).to(torch.device("cuda:5"))

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv_bias = qkv_bias.to(torch.device("cuda:5"))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias).to(torch.device("cuda:5"))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4).to(torch.device("cuda:5"))
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)).to(torch.device("cuda:5"))
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(torch.device("cuda:5"))).exp().to(torch.device("cuda:5"))
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.GroupNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.GroupNorm, pretrained_window_size=0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # self.norm1 = norm_layer(dim).to(torch.device("cuda:5"))
        self.norm1 = norm_layer(1, dim).to(torch.device("cuda:5"))
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size)).to(torch.device("cuda:5"))

        self.drop_path = DropPath(drop_path).to(torch.device("cuda:5")) if drop_path > 0. else nn.Identity().to(torch.device("cuda:5"))
        # self.norm2 = norm_layer(dim).to(torch.device("cuda:5"))
        self.norm2 = norm_layer(1, dim).to(torch.device("cuda:5"))
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop).to(torch.device("cuda:5"))

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            S, H, W = self.input_resolution
            img_mask = torch.zeros((1, S, H, W, 1))  # 1 S H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size).to(torch.device("cuda:5"))  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # print(x.shape)
        S, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == S * H * W, "input feature has wrong size"
        # print(B, S, H, W, C)

        shortcut = x
        x = x.contiguous().view(B, S, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, S, H, W)  # B S' H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, C, S * H * W)
        x = self.norm1(x)
        x = x.view(B, S * H * W, C)
        x = shortcut + self.drop_path(x)
        # x = x.permute(0, 2, 1)

        # FFN(Feed Forward Network, MLP)
        x = self.mlp(x)
        # x = x.permute(0, 2, 1)
        x = x.view(B, C, -1)
        x = self.norm2(x)
        x = x.view(B, -1, C)
        x = x + self.drop_path(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        S, H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.GroupNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.GroupNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        # self.norm = norm_layer(2 * dim)
        self.norm = norm_layer(1, 2 * dim)

    def forward(self, x):
        """
        x: B, S*H*W, C
        """
        S, H, W = self.input_resolution
        B, L, C = x.shape
        assert L == S * H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.contiguous().view(B, S, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x).to(torch.device("cuda:5"))
        x = x.permute(0, 2, 1)
        x = self.norm(x).to(torch.device("cuda:5"))
        x = x.permute(0, 2, 1)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        S, H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.GroupNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.GroupNorm, downsample=None, upsample=None, use_checkpoint=False,
                 pretrained_window_size=0):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size)
            for i in range(depth)]).to(torch.device("cuda:5"))

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer).to(torch.device("cuda:5"))
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(20, 224, 224), patch_size=(1, 2, 2), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = img_size
        patch_size = patch_size
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(40, embed_dim, kernel_size=patch_size, stride=patch_size).to("cuda:5")
        if norm_layer is not None:
            self.norm = norm_layer(1, embed_dim).to("cuda:5")
        else:
            self.norm = None

    def forward(self, x):
        B, C, S, H, W = x.shape  # (batch size, channel(time), slice, height, width)
        # FIXME look at relaxing size constraints
        assert S == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({S}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        # x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)
        # x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        x = self.proj(x).flatten(2)

        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.GroupNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=(20, 224, 224), patch_size=(1, 2, 2), in_chans=3, out_chans=5,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.GroupNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], **kwargs):
        super().__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None).to(torch.device("cuda:5"))
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim)).to(torch.device("cuda:5"))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate).to(torch.device("cuda:5"))

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList().to(torch.device("cuda:5"))
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0],
                                                 patches_resolution[1] // (2 ** i_layer),
                                                 patches_resolution[2] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer]).to(torch.device("cuda:5"))
            self.layers.append(layer)

        self.norm = norm_layer(1, self.num_features).to(torch.device("cuda:5"))
        # self.avgpool = nn.AdaptiveAvgPool1d(1).to(torch.device("cuda:5"))
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.up = UpSample(input_resolution=(patches_resolution[0],
        #                                     patches_resolution[1] // (2 ** i_layer),
        #                                     patches_resolution[2] // (2 ** i_layer)),
        #                        in_channels=embed_dim, scale_factor=2)
        self.up_b1 = UpSample(input_resolution=(patches_resolution[0],
                                            patches_resolution[1] // (2 ** i_layer),
                                            patches_resolution[2] // (2 ** i_layer)),
                               in_channels=768, out_channels=192, scale_factor=2).to(torch.device("cuda:5"))
        self.up_b2 = UpSample(input_resolution=(20, 28, 28),
                              in_channels=1152, out_channels=384, scale_factor=2).to(torch.device("cuda:5"))
        self.up_b3 = UpSample(input_resolution=(20, 56, 56),
                                       in_channels=1152, out_channels=96, scale_factor=2).to(torch.device("cuda:5"))
        self.up_b4 = UpSample(input_resolution=(20, 112, 112),
                                       in_channels=384, out_channels=48, scale_factor=2).to(torch.device("cuda:5"))

        self.resblock1 = ResidualBlock(576).to(torch.device("cuda:5"))
        self.resblock2 = ResidualBlock(576).to(torch.device("cuda:5"))
        self.resblock3 = ResidualBlock(192).to(torch.device("cuda:5"))
        self.resblock4 = ResidualBlock(96).to(torch.device("cuda:5"))

        self.conv1x1 = nn.Conv3d(in_channels=768, out_channels=384, kernel_size=(1, 1, 1)).to(torch.device("cuda:5"))

        self.last_conv = nn.Conv3d(in_channels=48, out_channels=self.out_chans, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False).to(torch.device("cuda:5"))


        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    # def forward_features(self, x):
    #     x = self.patch_embed(x)
    #     if self.ape:  # absolute position embedding
    #         x = x + self.absolute_pos_embed
    #     x = self.pos_drop(x)
    #
    #     for layer in self.layers:
    #         # print("layer:", x.shape)
    #         x = layer(x)
    #     x = self.norm(x)  # B L C
    #
    #     return x

    def forward(self, x):
        # forward_features
        x = self.patch_embed(x)

        if self.ape:  # absolute position embedding
            x = x + self.absolute_pos_embed
        x0 = self.pos_drop(x)
        # print(x0.shape)  # 1, 250880, 96 (112, 112)

        layers_out = []
        x = x0
        for i, layer in enumerate(self.layers):
            # print("layer:", x.shape)
            x = layer(x)
            layers_out.append(x)
        x = x.permute(0, 2, 1)
        x = self.norm(x)  # B L C
        x = x.permute(0, 2, 1)
        # print(x.shape)  # 1, 3920, 768

        # upsampling
        x = self.up_b1(x)
        # print(x.shape) # 1, 192, 20, 28, 28

        layers_out[1] = layers_out[1].contiguous().view(1, -1, 20, 28, 28)
        # print(layers_out[1].shape)  # 1, 384, 20, 28, 28
        x = torch.cat((x, layers_out[1]), dim=1)
        # print(x.shape)  # 1, 576, 20, 28, 28
        res1 = self.resblock1(x)
        # print(res1.shape)  # 1, 576, 20, 28, 28
        x = torch.cat((x, res1), dim=1)
        # print(x.shape)  # 1, 1152, 20, 28, 28

        x = x.contiguous().view(1, -1, 1152)
        # print(x.shape)  # 1, 15680, 1152
        x = self.up_b2(x)  # 1, 384, 20, 56, 56
        # print(x.shape)
        layers_out[0] = layers_out[0].contiguous().view(1, -1, 20, 56, 56)
        # print(layers_out[0].shape)  # 1, 192, 20, 56, 56
        x = torch.cat((x, layers_out[0]), dim=1)  # 1, 576, 20, 56, 56
        # print(x.shape)
        res2 = self.resblock2(x)  # 1, 576, 20, 56, 56
        # print(res2.shape)
        x = torch.cat((x, res2), dim=1) # 1, 1152, 20, 56, 56
        # print(x.shape)


        x = x.contiguous().view(1, -1, 1152)  # 1, 62720, 1152
        # print(x.shape)
        x = self.up_b3(x) # 1, 96, 20, 112, 112
        # print(x.shape)
        x0 = x0.contiguous().view(1, -1, 20, 112, 112) # 1, 96, 20, 112, 112
        # print(x0.shape)
        x = torch.cat((x, x0), dim=1) # 1, 192, 20, 112, 112
        # print(x.shape)
        res3 = self.resblock3(x)  # 1, 192, 20, 112, 112
        # print(res3.shape)
        x = torch.cat((x, res3), dim=1)  # 1, 384, 20, 112, 112
        # print(x.shape)

        x = x.contiguous().view(1, -1, 384)
        # print(x.shape)
        x = self.up_b4(x)
        # print(x.shape)

        x = self.last_conv(x)
        # print(x.shape)
        # breakpoint()
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops


if __name__ == '__main__':
    # python models/SwinTransformer_3D/swin_transformer_v2.py

    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:5"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> swin transformer")

    device = torch.device("cuda:0")

    net = SwinTransformerV2(img)
    summary(vit, (40, 20, 224, 224), batch_size=1, device="cuda")

    # out = vit(img)