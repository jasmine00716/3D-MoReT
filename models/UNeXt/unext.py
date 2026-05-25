import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
import pdb


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2
        # self.pad = 0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    #     def shift(x, dim):
    #         x = F.pad(x, "constant", 0)
    #         x = torch.chunk(x, shift_size, 1)
    #         x = [ torch.roll(x_c, shift, dim) for x_s, shift in zip(x, range(-pad, pad+1))]
    #         x = torch.cat(x, 1)
    #         return x[:, :, pad:-pad, pad:-pad]

    def forward(self, x, S, H, W):
        # pdb.set_trace()
        B, N, C = x.shape
        # # print("B, N, C:", x.shape)

        xn = x.transpose(1, 2).view(B, C, S, H, W).contiguous()
        # # print("xn(BCHW):", xn.shape)
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0) ###
        # # print("xn(pad):", xn.shape)
        xs = torch.chunk(xn, self.shift_size, 1)
        # # print("xs:", len(xs))
        # # print(self.pad)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        # # print("x_cat", x_cat.shape)
        x_s = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_s, 4, self.pad, W)

        # # print("x_s", x_s.shape)

        x_s = x_s.reshape(B, C, S * H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        x = self.dwconv(x, S, H, W)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, S, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 3, self.pad, H)
        x_s = torch.narrow(x_cat, 4, self.pad, W)
        x_s = x_s.reshape(B, C, S * H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, S, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), S, H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, (1,3,3), 1, (0,1,1), bias=True, groups=dim)

    def forward(self, x, S, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, S, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=(1, patch_size[0], patch_size[1]), stride=stride,
                              padding=(0, patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, S, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, S, H, W


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], device=torch.device("cuda:4"), **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv3d(40, 16, (1, 3, 3), stride=(1,2,2), padding=(0,1,1)).to(device)
        self.encoder2 = nn.Conv3d(16, 32, (1, 3, 3), stride=(1,1,1), padding=(0,1,1)).to(device)
        self.encoder3 = nn.Conv3d(32, 128, (1, 3, 3), stride=(1,1,1), padding=(0,1,1)).to(device)

        self.ebn1 = nn.GroupNorm(1,16).to(device)
        self.ebn2 = nn.GroupNorm(1,32).to(device)
        self.ebn3 = nn.GroupNorm(1,128).to(device)

        self.norm3 = norm_layer(embed_dims[1]).to(device)
        self.norm4 = norm_layer(embed_dims[2]).to(device)

        self.dnorm3 = norm_layer(160).to(device)
        self.dnorm4 = norm_layer(128).to(device)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1]).to(device)
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2]).to(device)

        self.decoder1 = nn.Conv3d(256, 160, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder2 = nn.Conv3d(160, 128, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder3 = nn.Conv3d(128, 32, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder4 = nn.Conv3d(32, 16, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder5 = nn.Conv3d(16, 5, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder6 = nn.Conv3d(5, 5, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)

        self.dbn1 = nn.GroupNorm(1,160).to(device)
        self.dbn2 = nn.GroupNorm(1,128).to(device)
        self.dbn3 = nn.GroupNorm(1,32).to(device)
        self.dbn4 = nn.GroupNorm(1,16).to(device)
        self.dbn5 = nn.GroupNorm(1, 5).to(device)

        self.final = nn.Conv3d(5, num_classes, kernel_size=1).to(device)

        self.soft = nn.Softmax(dim=1).to(device)

    def forward(self, x):

        B = x.shape[0]
        # print(x.shape)
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)),(1, 3, 3), (1, 2, 2), (0, 1, 1)))
        # print(out.shape)
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), (1, 3, 3), (1, 2, 2), (0, 1, 1)))
        # print(out.shape)
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), (1, 3, 3), (1, 2, 2), (0, 1, 1)))
        t3 = out
        # print(out.shape)

        ### Tokenized MLP Stage
        ### Stage 4

        out, S, H, W = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, S, H, W)
        out = self.norm3(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out
        # print(out.shape)

        ### Bottleneck

        out, S, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, S, H, W)
        out = self.norm4(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        # print(out.shape)

        ### Stage 4

        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), size=(10, 7, 7), mode='trilinear'))
        # print(out.shape)

        out = torch.add(out, t4)

        _, _, S, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, S, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), size=(20, 14, 14), mode='trilinear'))
        out = torch.add(out, t3)
        _, _, S, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, S, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), size=(20, 28, 28), mode='trilinear'))
        out = torch.add(out, t2)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), size=(20, 56, 56), mode='trilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)), size=(20, 112, 112), mode='trilinear'))

        out = F.relu(F.interpolate(self.decoder6(out), size=(20, 224, 224), mode='trilinear'))

        return self.final(out)


class UNext_S(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP w less parameters

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[32, 64, 128, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], device=torch.device("cuda:4"),**kwargs):
        super().__init__()

        self.encoder1 = nn.Conv3d(40, 8, (1,3,3), stride=(1,2,2), padding=(0,1,1)).to(device)
        self.encoder2 = nn.Conv3d(8, 16, (1,3,3), stride=(1,1,1), padding=(0,1,1)).to(device)
        self.encoder3 = nn.Conv3d(16, 32, (1,3,3), stride=(1,1,1), padding=(0,1,1)).to(device)

        self.ebn1 = nn.GroupNorm(1,8).to(device)
        self.ebn2 = nn.GroupNorm(1,16).to(device)
        self.ebn3 = nn.GroupNorm(1,32).to(device)

        self.norm3 = norm_layer(embed_dims[1]).to(device)
        self.norm4 = norm_layer(embed_dims[2]).to(device)

        self.dnorm3 = norm_layer(64).to(device)
        self.dnorm4 = norm_layer(32).to(device)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.block2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.dblock1 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.dblock2 = nn.ModuleList([shiftedBlock(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])]).to(device)

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1]).to(device)
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2]).to(device)

        self.decoder1 = nn.Conv3d(128, 64, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder2 = nn.Conv3d(64, 32, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder3 = nn.Conv3d(32, 16, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder4 = nn.Conv3d(16, 8, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder5 = nn.Conv3d(8, 5, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)
        self.decoder6 = nn.Conv3d(5, 5, 3, stride=(1, 3, 3), padding=(1, 2, 2)).to(device)

        self.dbn1 = nn.GroupNorm(1,64).to(device)
        self.dbn2 = nn.GroupNorm(1,32).to(device)
        self.dbn3 = nn.GroupNorm(1,16).to(device)
        self.dbn4 = nn.GroupNorm(1,8).to(device)
        self.dbn5 = nn.GroupNorm(1,5).to(device)

        self.final = nn.Conv3d(5, num_classes, kernel_size=1).to(device)

        self.soft = nn.Softmax(dim=1).to(device)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = F.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), (1, 3, 3), (1, 2, 2), (0, 1, 1)))
        # # print(out.shape)
        t1 = out
        ### Stage 2
        out = F.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), (1, 3, 3), (1, 2, 2), (0, 1, 1)))
        # # print(out.shape)
        t2 = out
        ### Stage 3
        out = F.relu(F.max_pool3d(self.ebn3(self.encoder3(out)),  (1, 3, 3), (1, 2, 2), (0, 1, 1)))
        # # print(out.shape)
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, S, H, W = self.patch_embed3(out) # (1, 490, 64) 10 7 7
        # # print(out.shape, S, H, W)
        for i, blk in enumerate(self.block1):
            out = blk(out, S, H, W)
        out = self.norm3(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck

        out, S, H, W = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, S, H, W)
        out = self.norm4(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### Stage 4

        # print(">", out.shape)
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(out)), size=(10, 7, 7), mode='trilinear'))

        # # print(out.shape)
        # # print(t4.shape)

        out = torch.add(out, t4)

        _, _, S, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, S, H, W)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        # print(out.shape)
        # print(t3.shape)
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), size=(20, 14, 14), mode='trilinear'))
        # print(out.shape)
        out = torch.add(out, t3)
        _, _, S, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, S, H, W)

        out = self.dnorm4(out)
        out = out.reshape(B, S, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), size=(20, 28, 28), mode='trilinear'))
        # print(out.shape)
        out = torch.add(out, t2)
        # print(out.shape)
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), size=(20, 56, 56), mode='trilinear'))
        # print(out.shape)
        out = torch.add(out, t1)
        # print(out.shape)
        out = F.relu(F.interpolate(self.dbn5(self.decoder5(out)), size=(20, 112, 112), mode='trilinear'))
        # print(out.shape)
        out = F.relu(F.interpolate(self.decoder6(out), size=(20, 224, 224), mode='trilinear'))

        return self.final(out)


if __name__ == '__main__':
    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:4"))

    # network = UNext_S(num_classes=5)
    network = UNext(num_classes=5)
    # output = network(img)
    # print(output.shape)

    # totalparams = 0
    # for name, param in network.named_parameters():
    #     if param.requires_grad:
    #         # print(f"Parameter name: {name}, {param.numel()}")
    #         totalparams += param.numel()
    # print(f">> the total parameters: {totalparams}")
    #
    # buffer_size = 0
    # for buffer in network.buffers():
    #     buffer_size += buffer.nelement() * buffer.element_size()
    #
    # size_all_mb = (totalparams + buffer_size) / 1024 ** 2
    # print('>> model size: {:.3f}MB\n'.format(size_all_mb))
    #
    # from thop import profile
    #
    # # output = network(img)
    # flops, params, ret_layer_info = profile(network.to(torch.device("cuda:6")), inputs=(img.to(torch.device("cuda:6")),), ret_layer_info=True)
    # print('params', params)
    # print('FLOPs:', flops)

    print(">> unext")
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 10

    import numpy as np

    timings = np.zeros((repetitions, 1))

    _ = network(img)

    # MEASURE PERFORMANCE
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = network(img)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
            total_time += curr_time
            # print(curr_time)
    print(total_time / repetitions)