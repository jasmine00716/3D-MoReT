import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1, padding=(0, 1, 1)):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1, kernel_size, kernel_size), stride=(1, stride, stride), padding=padding, bias=False),
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
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ps, self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()
        # Local representations
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)

        # Global representations
        _, t, s, h, w = x.shape
        if s % 2 == 0 and h % 2 == 0 and w % 2 == 0:
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
        x = self.conv3(x)
        # print(x.shape)

        x = torch.cat((x, y), 1)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        # breakpoint()
        return x