import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


# Swin Transformer 3D Components
class SwinTransformer3D(nn.Module):
    def __init__(self, dim, heads, depth, hidden_dim):
        super(SwinTransformer3D, self).__init__()
        self.layers = nn.ModuleList([])
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SwinTransformerBlock3D(dim, heads=heads, hidden_dim=hidden_dim)
            ]))

    def forward(self, x):
        for norm, block in self.layers:
            # x = x.view(-1, self.dim)
            x = block(norm(x) + x)
        return x


class SwinTransformerBlock3D(nn.Module):
    def __init__(self, dim, heads, hidden_dim):
        super(SwinTransformerBlock3D, self).__init__()
        self.dim = dim
        self.attention = SwinAttention3D(dim, heads=heads)
        self.mlp = SwinMLP3D(dim, hidden_dim=hidden_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return x


class SwinAttention3D(nn.Module):
    def __init__(self, dim, heads):
        super(SwinAttention3D, self).__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads

        self.to_qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, stride=1, padding=0)
        self.to_out = nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, channels, depth, height, width = x.size()

        qkv = self.to_qkv(x).view(batch_size, 3, self.heads, self.head_dim, depth, height, width)
        queries, keys, values = qkv.chunk(3, dim=1)

        queries = queries.view(batch_size * self.heads, self.head_dim, depth, height * width).transpose(2, 3)
        keys = keys.view(batch_size * self.heads, self.head_dim, depth, height * width)
        values = values.view(batch_size * self.heads, self.head_dim, depth, height * width)

        attention = torch.bmm(queries, keys.transpose(2, 3))
        attention = self.softmax(attention / (self.head_dim ** 0.5))

        out = torch.bmm(attention, values)
        out = out.transpose(2, 3).view(batch_size, self.heads, self.head_dim, depth, height, width).transpose(1, 2)
        out = out.contiguous().view(batch_size, self.dim, depth, height, width)

        out = self.to_out(out)
        return out


class SwinMLP3D(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(SwinMLP3D, self).__init__()
        layers = []
        for _ in range(hidden_dim):
            layers.extend([
                nn.Conv3d(dim, dim * 2, kernel_size=1),
                nn.GELU(),
                nn.Conv3d(dim * 2, dim, kernel_size=1),
                nn.GELU()
            ])
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


# Swin-UNet 3D Model
class SwinUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SwinUNet3D, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            SwinTransformer3D(dim=64, heads=4, depth=2, hidden_dim=2),
            # ... Add more SwinTransformer3D blocks as needed for encoding
        )

        self.decoder = nn.Sequential(
            # ... Add Transpose Convolutional layers and Swin Transformer blocks for decoding
            # Example:
            nn.ConvTranspose3d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            SwinTransformer3D(dim=32, heads=4, depth=2, hidden_dim=2),  # Example decoder block
            # ...
        )

        self.final_conv = nn.Conv3d(32, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_conv(x)
        return x


# Example usage
# Assuming input is a 5D tensor: (batch_size, channels, depth, height, width)
model = SwinUNet3D(in_channels=40, out_channels=1)
input_data = torch.randn(1, 40, 20, 224, 224)  # Example input data
output = model(input_data)
print(output.shape)  # Print output shape

# import torch
# from torch import nn, einsum
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange
# from typing import Union, List
# import numpy as np
# from timm.models.layers import trunc_normal_
#
#
# class CyclicShift3D(nn.Module):
#     def __init__(self, displacement):
#         super().__init__()
#
#         assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
#         if type(displacement) is int:
#             displacement = np.array([displacement, displacement, displacement])
#         self.displacement = displacement
#
#     def forward(self, x):
#         return torch.roll(x, shifts=(self.displacement[0], self.displacement[1], self.displacement[2]), dims=(1, 2, 3))
#
#
# class Residual3D(nn.Module):
#     def __init__(self, fn):
#         super().__init__()
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(x, **kwargs) + x
#
#
# class PreNorm3D(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.norm = nn.LayerNorm(dim)
#         self.fn = fn
#
#     def forward(self, x, **kwargs):
#         return self.fn(self.norm(x), **kwargs)
#
#
# class FeedForward3D(nn.Module):
#     def __init__(self, dim, hidden_dim, dropout: float = 0.0):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#         self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#
#     def forward(self, x):
#         x = self.net(x)
#         x = self.drop(x)
#         return x
#
#
# def create_mask3D(window_size: Union[int, List[int]], displacement: Union[int, List[int]],
#                   x_shift: bool, y_shift: bool, z_shift: bool):
#     assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#     if type(window_size) is int:
#         window_size = np.array([window_size, window_size, window_size])
#
#     assert type(displacement) is int or len(displacement) == 3, f'displacement must be 1 or 3 dimension'
#     if type(displacement) is int:
#         displacement = np.array([displacement, displacement, displacement])
#
#     assert len(window_size) == len(displacement)
#     for i in range(len(window_size)):
#         assert 0 < displacement[i] < window_size[i], \
#             f'Incorrect displacement on axis {i}, dimensions include X (i=0), Y (i=1), and Z (i=2)'
#
#     mask = torch.zeros(window_size[0] * window_size[1] * window_size[2],
#                        window_size[0] * window_size[1] * window_size[2])  # (wx*wy*wz, wx*wy*wz)
#     mask = rearrange(mask, '(x1 y1 z1) (x2 y2 z2) -> x1 y1 z1 x2 y2 z2',
#                      x1=window_size[0], y1=window_size[1], x2=window_size[0], y2=window_size[1])
#
#     x_dist, y_dist, z_dist = displacement[0], displacement[1], displacement[2]
#
#     if x_shift:
#         #      x1     y1 z1     x2     y2 z2
#         mask[-x_dist:, :, :, :-x_dist, :, :] = float('-inf')
#         mask[:-x_dist, :, :, -x_dist:, :, :] = float('-inf')
#
#     if y_shift:
#         #   x1   y1       z1 x2  y2       z2
#         mask[:, -y_dist:, :, :, :-y_dist, :] = float('-inf')
#         mask[:, :-y_dist, :, :, -y_dist:, :] = float('-inf')
#
#     if z_shift:
#         #   x1  y1  z1       x2 y2  z2
#         mask[:, :, -z_dist:, :, :, :-z_dist] = float('-inf')
#         mask[:, :, :-z_dist, :, :, -z_dist:] = float('-inf')
#
#     mask = rearrange(mask, 'x1 y1 z1 x2 y2 z2 -> (x1 y1 z1) (x2 y2 z2)')
#     return mask
#
#
# # Reference from video_swin_transformer:
# # #https://github.com/MohammadRezaQaderi/Video-Swin-Transformer/blob/c3cd8639decf19a25303615db0b6c1195495f5bb/mmaction/models/backbones/swin_transformer.py#L119
# def get_relative_distances(window_size):
#     assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#     if type(window_size) is int:
#         window_size = np.array([window_size, window_size, window_size])
#     indices = torch.tensor(
#         np.array(
#             [[x, y, z] for x in range(window_size[0]) for y in range(window_size[1]) for z in range(window_size[2])]))
#
#     distances = indices[None, :, :] - indices[:, None, :]
#     # distance:(n,n,3) n =window_size[0]*window_size[1]*window_size[2]
#     return distances
#
#
# class WindowAttention3D(nn.Module):
#     def __init__(self, dim: int, heads: int, head_dim: int, shifted: bool, window_size: Union[int, List[int]],
#                  relative_pos_embedding: bool = True):
#         super().__init__()
#
#         assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#         if type(window_size) is int:
#             window_size = np.array([window_size, window_size, window_size])
#         else:
#             window_size = np.array(window_size)
#
#         inner_dim = head_dim * heads
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#         self.window_size = window_size
#         # self.relative_pos_embedding = relative_pos_embedding
#         self.shifted = shifted
#
#         if self.shifted:
#             displacement = window_size // 2
#             self.cyclic_shift = CyclicShift3D(-displacement)
#             self.cyclic_back_shift = CyclicShift3D(displacement)
#             self.x_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=True, y_shift=False, z_shift=False), requires_grad=False)
#             self.y_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=False, y_shift=True, z_shift=False), requires_grad=False)
#             self.z_mask = nn.Parameter(create_mask3D(window_size=window_size, displacement=displacement,
#                                                      x_shift=False, y_shift=False, z_shift=True), requires_grad=False)
#
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)  # QKV three parts
#
#         # if self.relative_pos_embedding:
#         #     self.relative_indices = get_relative_distances(window_size)
#         #     # relative_indices的形状为 (n,n,3) n=window_size[0]*window_size[1]*window_size[2],
#         #
#         #     for i in range(len(window_size)):  # 在每个维度上进行偏移
#         #         self.relative_indices[:, :, i] += window_size[i] - 1
#         #
#         #     self.pos_embedding = nn.Parameter(
#         #         torch.randn(2 * window_size[0] - 1, 2 * window_size[1] - 1, 2 * window_size[2] - 1)
#         #     )
#         # else:
#         # self.pos_embedding = nn.Parameter(torch.randn(window_size[0] * window_size[1] * window_size[2],
#         #                                               window_size[0] * window_size[1] * window_size[2]))
#
#         self.softmax = nn.Softmax(dim=-1)
#         self.to_out = nn.Linear(inner_dim, dim)
#
#     def forward(self, x):
#         if self.shifted:
#             x = self.cyclic_shift(x)
#
#         b, n_x, n_y, n_z, _, h = *x.shape, self.heads
#
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         nw_x = n_x // self.window_size[0]
#         nw_y = n_y // self.window_size[1]
#         nw_z = n_z // self.window_size[2]
#
#         q, k, v = map(
#             lambda t: rearrange(t, 'b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d) -> b h (nw_x nw_y nw_z) (w_x w_y w_z) d',
#                                 h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2]), qkv)
#
#         dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale # matrix multiplication of q and k
#
#         # if self.relative_pos_embedding:
#         #     dots += self.pos_embedding[self.relative_indices[:, :, 0].long(), self.relative_indices[:, :, 1].long(),
#         #                                self.relative_indices[:, :, 2].long()]
#         # else:
#         #   dots += self.pos_embedding  # Broadcasting occurs
#
#         if self.shifted:
#             # Move the number of windows on the x-axis to the end for overlay with the mask on the x-axis, and so on
#             dots = rearrange(dots, 'b h (n_x n_y n_z) i j -> b h n_y n_z n_x i j',
#                              n_x=nw_x, n_y=nw_y)
#             #   b   h n_y n_z n_x i j
#             dots[:, :, :, :, -1] += self.x_mask
#
#             dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h n_x n_z n_y i j')
#             dots[:, :, :, :, -1] += self.y_mask
#
#             dots = rearrange(dots, 'b h n_x n_z n_y i j -> b h n_x n_y n_z i j')
#             dots[:, :, :, :, -1] += self.z_mask
#
#             dots = rearrange(dots, 'b h n_y n_z n_x i j -> b h (n_x n_y n_z) i j')
#
#         # attn = dots.softmax(dim=-1)
#         attn = self.softmax(dots)
#         out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)   # matrix multiplication of attn and v
#
#         # nw_x represents the number of windows on the x-axis, nw_y on the y-axis, and nw_Z on the z-axis
#         # w_x represents x_window_size, w_y represents y_window_size, w_z represents z_window_size
#         #                     b 3  (8,8,8)         （7,  7,  7） 96 -> b  56          56          56        288
#         out = rearrange(out, 'b h (nw_x nw_y nw_z) (w_x w_y w_z) d -> b (nw_x w_x) (nw_y w_y) (nw_z w_z) (h d)',
#                         h=h, w_x=self.window_size[0], w_y=self.window_size[1], w_z=self.window_size[2],
#                         nw_x=nw_x, nw_y=nw_y, nw_z=nw_z)
#         out = self.to_out(out)
#
#         if self.shifted:
#             out = self.cyclic_back_shift(out)
#         return out
#
#
# class SwinBlock3D(nn.Module):  # Will not change the input spatial resolution
#     def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size: Union[int, List[int]],
#                  relative_pos_embedding: bool = True, dropout: float = 0.0):
#         super().__init__()
#         self.attention_block = Residual3D(PreNorm3D(dim, WindowAttention3D(dim=dim,
#                                                                            heads=heads,
#                                                                            head_dim=head_dim,
#                                                                            shifted=shifted,
#                                                                            window_size=window_size,
#                                                                            relative_pos_embedding=relative_pos_embedding)))
#         self.mlp_block = Residual3D(PreNorm3D(dim, FeedForward3D(dim=dim, hidden_dim=mlp_dim, dropout=dropout)))
#
#     def forward(self, x):
#         x = self.attention_block(x)
#         x = self.mlp_block(x)
#         return x
#
#
# class Norm(nn.Module):
#     def __init__(self, dim, channel_first: bool = True):
#         super(Norm, self).__init__()
#         if channel_first:
#             self.net = nn.Sequential(
#                 Rearrange('b c s h w -> b s h w c'),
#                 nn.LayerNorm(dim),
#                 Rearrange('b s h w c -> b c s h w')
#             )
#
#             # self.net = nn.InstanceNorm3d(dim, eps=1e-5, momentum=0.1, affine=False)
#         else:
#             self.net = nn.LayerNorm(dim)
#
#     def forward(self, x):
#         x = self.net(x)
#         return x
#
#
# class PatchMerging3D(nn.Module):
#     def __init__(self, in_dim, out_dim, downscaling_factor):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv3d(in_dim, out_dim, kernel_size=downscaling_factor, stride=downscaling_factor),
#             Norm(dim=out_dim),
#         )
#
#     def forward(self, x):
#         # x: B, C, S, H, W
#         x = self.net(x)
#         return x  # B,  H //down_scaling, W//down_scaling, D//down_scaling, out_dim
#
#
# class PatchExpanding3D(nn.Module):
#     def __init__(self, in_dim, out_dim, up_scaling_factor):
#         super(PatchExpanding3D, self).__init__()
#
#         stride = up_scaling_factor
#         kernel_size = up_scaling_factor
#         padding = (kernel_size - stride) // 2
#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#             Norm(out_dim),
#         )
#
#     def forward(self, x):
#         '''X: B,C,X,Y,Z'''
#         x = self.net(x)
#         return x
#
#
# class FinalExpand3D(nn.Module):  # Used for voxel final classification
#     def __init__(self, in_dim, out_dim, up_scaling_factor):  # stl is an abbreviation for second_to_last
#         super(FinalExpand3D, self).__init__()
#
#         stride = up_scaling_factor
#         kernel_size = up_scaling_factor
#         padding = (kernel_size - stride) // 2
#         self.net = nn.Sequential(
#             nn.ConvTranspose3d(in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding),
#             Norm(out_dim),
#             nn.PReLU()
#         )
#
#     def forward(self, x):
#         '''X: B,C,H,W,D'''
#         x = self.net(x)
#         return x
#
#
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(ConvBlock, self).__init__()
#         groups = min(in_ch, out_ch)
#         self.net = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
#             Norm(dim=out_ch),
#             nn.PReLU(),
#
#             nn.Conv3d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, groups=groups),
#             Norm(dim=out_ch),
#             nn.PReLU(),
#         )
#
#     def forward(self, x):
#         x2 = x.clone()
#         x = self.net(x) * x2
#         return x
#
#
# class Encoder(nn.Module):
#     def __init__(self, in_dims, hidden_dimension, layers, downscaling_factor, num_heads, head_dim,
#                  window_size: Union[int, List[int]], relative_pos_embedding: bool = True, dropout: float = 0.0):
#         super().__init__()
#         assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
#
#         self.patch_partition = PatchMerging3D(in_dim=in_dims, out_dim=hidden_dimension,
#                                               downscaling_factor=downscaling_factor)
#         self.conv_block = ConvBlock(in_ch=hidden_dimension, out_ch=hidden_dimension)
#
#         self.re1 = Rearrange('b c h w d -> b h w d c')
#         self.swin_layers = nn.ModuleList([])
#         for _ in range(layers // 2):
#             self.swin_layers.append(nn.ModuleList([
#                 SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
#                             shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#                 SwinBlock3D(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
#                             shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#             ]))
#         self.re2 = Rearrange('b h w d c -> b c h w d')
#
#     def forward(self, x):
#         x = self.patch_partition(x)
#         x2 = self.conv_block(x)  # Convolutional block learns short-range dependencies
#
#         x = self.re1(x)
#         for regular_block, shifted_block in self.swin_layers:   # Blocks in swin_layers learn long-range dependencies
#             x = regular_block(x)
#             x = shifted_block(x)
#         x = self.re2(x)
#
#         x = x + x2  # Fusion of long and short-range dependency information
#         return x
#
#
# class Decoder(nn.Module):
#     def __init__(self, in_dims, out_dims, layers, up_scaling_factor, num_heads, head_dim,
#                  window_size: Union[int, List[int]], relative_pos_embedding, dropout: float = 0.0):
#         super().__init__()
#         assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
#
#         self.patch_expand = PatchExpanding3D(in_dim=in_dims, out_dim=out_dims,
#                                              up_scaling_factor=up_scaling_factor)
#
#         self.conv_block = ConvBlock(in_ch=out_dims, out_ch=out_dims)
#         self.re1 = Rearrange('b c h w d -> b h w d c')
#         self.swin_layers = nn.ModuleList([])
#         for _ in range(layers // 2):
#             self.swin_layers.append(nn.ModuleList([
#                 SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
#                             shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#                 SwinBlock3D(dim=out_dims, heads=num_heads, head_dim=head_dim, mlp_dim=out_dims * 4,
#                             shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
#                             dropout=dropout),
#             ]))
#         self.re2 = Rearrange('b h w d c -> b c h w d')
#
#     def forward(self, x):
#         x = self.patch_expand(x)
#
#         x2 = self.conv_block(x)
#
#         x = self.re1(x)
#         for regular_block, shifted_block in self.swin_layers:
#             x = regular_block(x)
#             x = shifted_block(x)
#         x = self.re2(x)
#
#         x = x + x2
#         return x
#
#
# class SwinUnet3D(nn.Module):
#     def __init__(self, *, hidden_dim, layers, heads, in_channel=1, num_classes=2, head_dim=32,
#                  window_size: Union[int, List[int]] = 7, downscaling_factors=(2, 2, 2, 2),
#                  relative_pos_embedding=True, dropout: float = 0.0, skip_style='stack',
#                  stl_channels: int = 32):  # second_to_last_channels
#         super().__init__()
#
#         self.dsf = downscaling_factors
#         self.window_size = window_size
#
#         self.enc12 = Encoder(in_dims=in_channel, hidden_dimension=hidden_dim, layers=layers[0],
#                              downscaling_factor=downscaling_factors[0], num_heads=heads[0],
#                              head_dim=head_dim, window_size=window_size, dropout=dropout,
#                              relative_pos_embedding=relative_pos_embedding)
#         self.enc3 = Encoder(in_dims=hidden_dim, hidden_dimension=hidden_dim * 2,
#                             layers=layers[1],
#                             downscaling_factor=downscaling_factors[1], num_heads=heads[1],
#                             head_dim=head_dim, window_size=window_size, dropout=dropout,
#                             relative_pos_embedding=relative_pos_embedding)
#         self.enc4 = Encoder(in_dims=hidden_dim * 2, hidden_dimension=hidden_dim * 4,
#                             layers=layers[2],
#                             downscaling_factor=downscaling_factors[2], num_heads=heads[2],
#                             head_dim=head_dim, window_size=window_size, dropout=dropout,
#                             relative_pos_embedding=relative_pos_embedding)
#         self.enc5 = Encoder(in_dims=hidden_dim * 4, hidden_dimension=hidden_dim * 8,
#                             layers=layers[3],
#                             downscaling_factor=downscaling_factors[3], num_heads=heads[3],
#                             head_dim=head_dim, window_size=window_size, dropout=dropout,
#                             relative_pos_embedding=relative_pos_embedding)
#
#         self.dec4 = Decoder(in_dims=hidden_dim * 8, out_dims=hidden_dim * 4,
#                             layers=layers[2],
#                             up_scaling_factor=downscaling_factors[3], num_heads=heads[2],
#                             head_dim=head_dim, window_size=window_size, dropout=dropout,
#                             relative_pos_embedding=relative_pos_embedding)
#
#         self.dec3 = Decoder(in_dims=hidden_dim * 4, out_dims=hidden_dim * 2,
#                             layers=layers[1],
#                             up_scaling_factor=downscaling_factors[2], num_heads=heads[1],
#                             head_dim=head_dim, window_size=window_size, dropout=dropout,
#                             relative_pos_embedding=relative_pos_embedding)
#
#         self.dec12 = Decoder(in_dims=hidden_dim * 2, out_dims=hidden_dim,
#                              layers=layers[0],
#                              up_scaling_factor=downscaling_factors[1], num_heads=heads[0],
#                              head_dim=head_dim, window_size=window_size, dropout=dropout,
#                              relative_pos_embedding=relative_pos_embedding)
#
#         self.converge4 = Converge(hidden_dim * 4)
#         self.converge3 = Converge(hidden_dim * 2)
#         self.converge12 = Converge(hidden_dim)
#
#         self.final = FinalExpand3D(in_dim=hidden_dim, out_dim=stl_channels,
#                                    up_scaling_factor=downscaling_factors[0])
#         self.out = nn.Sequential(
#             # nn.Linear(stl_channels, num_classes),
#             # Rearrange('b h w d c -> b c h w d'),
#             nn.Conv3d(stl_channels, num_classes, kernel_size=1)
#         )
#         # 参数初始化
#         self.init_weight()
#
#     def forward(self, img):
#         window_size = self.window_size
#         assert type(window_size) is int or len(window_size) == 3, f'window_size must be 1 or 3 dimension'
#         if type(window_size) is int:
#             window_size = np.array([window_size, window_size, window_size])
#         _, _, x_s, y_s, z_s = img.shape
#         x_ws, y_ws, z_ws = window_size  # 7 7 7
#         x_ws = 2
#         # print(x_ws, y_ws, z_ws)
#
#         assert x_s % (x_ws * 10) == 0, f'The size on the x-axis must be divisible by x_window_size*10'
#         assert y_s % (y_ws * 32) == 0, f'The size on the y-axis must be divisible by y_window_size*32'
#         assert z_s % (z_ws * 32) == 0, f'The size on the z-axis must be divisible by z_window_size*32'
#
#         down12_1 = self.enc12(img)  # (B,C, X//4, Y//4, Z//4)
#         down3 = self.enc3(down12_1)  # (B, 2C,X//8, Y//8, Z//8)
#         down4 = self.enc4(down3)  # (B, 4C,X//16, Y//16, Z//16)
#         features = self.enc5(down4)  # (B, 8C,X//32, Y//32, Z//32)
#
#         up4 = self.dec4(features)  # (B, 8C, X//16, Y//16, Z//16 )
#         # Merge up1 and down3
#         up4 = self.converge4(up4, down4)  # (B, 4C, X//16, Y//16, Z//16)
#
#         up3 = self.dec3(up4)  # ((B, 2C,X//8, Y//8, Z//8)
#         # Merge up2 and down2
#         up3 = self.converge3(up3, down3)  # (B,2C, X//8, Y//8)
#
#         up12 = self.dec12(up3)  # (B,C, X//4, Y//4, Z// 4)
#         # Merge up3 and down1
#         up12 = self.converge12(up12, down12_1)  # (B,C, X//4, Y//4, Z//4)
#
#         out = self.final(up12)  # (B,num_classes, X, Y, Z)
#         out = self.out(out)
#         return out
#
#     def init_weight(self):
#         for m in self.modules():
#             if isinstance(m, (nn.Linear, nn.Conv3d, nn.ConvTranspose3d)):
#                 trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             if isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.weight, 1.0)
#                 nn.init.constant_(m.bias, 0.0)
#
#
# class Converge(nn.Module):
#     def __init__(self, dim: int):
#         '''
#         stack: Fusion is achieved through stacking + linear transformation.
#         add: Skip connection is achieved by direct addition.
#         '''
#         super(Converge, self).__init__()
#         self.norm = Norm(dim=dim)
#
#     def forward(self, x, enc_x):
#         '''
#          x: B,C,X,Y,Z
#         enc_x:B,C,X,Y,Z
#         '''
#         assert x.shape == enc_x.shape
#         x = x + enc_x
#         x = self.norm(x)
#         return x
#
#
# # In the original paper, layers=[2,2,6,2]
# def swinUnet_t_3D(hidden_dim=96, layers=(2, 2, 4, 2), heads=(3, 6, 9, 12), num_classes: int = 2, **kwargs):
#     return SwinUnet3D(hidden_dim=hidden_dim, layers=layers, heads=heads, num_classes=num_classes, **kwargs)
#
#
# if __name__ == '__main__':
#     img = torch.rand(1, 40, 20, 224, 224)
#
#     network = swinUnet_t_3D()
#     network(img)