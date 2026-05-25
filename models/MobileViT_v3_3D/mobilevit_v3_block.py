import torch
import torch.nn as nn
from einops import rearrange


def conv_1x1_gn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )


def conv_nxn_gn(inp, oup, kernel_size=3, stride=1, padding=(0, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=padding, bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

def dw_conv_nxn_gn(inp, oup, kernel_size=3, stride=1, padding=(0, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=padding, bias=False, groups=inp),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b p h n d -> b p n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, in_channel, kernel_size, patch_size, mlp_dim, n_transformer_blocks=2, dim_head=32, no_fusion=False, dilation=1 , dropout=0.):
        super(MobileViTBlock, self).__init__()

        self.ps, self.ph, self.pw = patch_size

        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        self.conv_3x3_in = dw_conv_nxn_gn(in_channel, in_channel, kernel_size)
        self.conv_1x1_gn_in = conv_1x1_gn(in_channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv_1x1_gn_out = conv_1x1_gn(dim, in_channel)
        self.fusion = None

        # For MobileViTv3: input+global --> local+global
        self.no_fusion = no_fusion
        if not no_fusion:
            self.fusion = conv_1x1_gn(dim + in_channel, in_channel)

    def forward(self, x):
        res = x.clone()

        # Local representations
        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        x = self.conv_3x3_in(x)
        x = self.conv_1x1_gn_in(x)

        y = x.clone()

        # Global representations
        _, t, s, h, w = x.shape
        if h % 2 == 0 and w % 2 == 0:
            # unfold
            x = rearrange(x, 'b t (s ps) (h ph) (w pw) -> b (ps ph pw) (s h w) t', ps=self.ps, ph=self.ph, pw=self.pw)
            # print(x.shape)
            x = self.transformer(x)
            # print(x.shape)

            #fold
            x = rearrange(x, 'b (ps ph pw) (s h w) t -> b t (s ps) (h ph) (w pw)', s=s//self.ps, h=h//self.ph, w=w//self.pw, ps=self.ps, ph=self.ph, pw=self.pw)
            # print(x.shape)
        else:
            # unfold
            x = rearrange(x, 'b t (s 1) (h 1) (w 1) -> b (1 1 1) (s h w) t')
            x = self.transformer(x)

            # fold
            x = rearrange(x, 'b (1 1 1) (s h w) t -> b t (s 1) (h 1) (w 1)', s=s, h=h, w=w)

        # Fusion
        x = self.conv_1x1_gn_out(x)

        x = torch.cat((x, y), 1)

        if not self.no_fusion:
            x = self.fusion(x)

        x += res
        return x
