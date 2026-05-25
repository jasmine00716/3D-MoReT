import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

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

def UnfoldingOrFolding(img_size, patch_size, unfold=True):
    # print(img_size.shape)
    t, s, h, w = img_size
    ps, ph, pw = patch_size
    if unfold:
        return Rearrange('b t (s ps) (h ph) (w pw) -> b (ps ph pw) (s h w) t', ps=ps, ph=ph, pw=pw)
    else:
        return Rearrange('b (ps ph pw) (s h w) t -> b t (s ps) (h ph) (w pw)', s=s//ps, h=h//ph, w=w//pw, ps=ps, ph=ph, pw=pw)

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
            # print(x.shape)
            x = attn(x) + x
            x = ff(x) + x
            # attn = attn(x)
            # print(attn.shape)
            # x = attn + x
            # print(x.shape)
            # ff = ff(x)
            # print(ff.shape)
            # x = ff + x
            # print(x.shape)
            # print()

        return x

class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, in_channel, kernel_size, patch_size, mlp_dim, n_transformer_blocks=2, dim_head=32, no_fusion=False, dilation=1 , dropout=0.):
        super(MobileViTBlock, self).__init__()

        self.patch_size = patch_size

        # For MobileViTv3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        self.conv_3x3_in = dw_conv_nxn_gn(in_channel, in_channel, kernel_size)
        self.conv_1x1_gn_in = conv_1x1_gn(in_channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv_1x1_gn_out = conv_1x1_gn(dim*3, in_channel)
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

        # unfold
        x1 = UnfoldingOrFolding((t,s,h,w), patch_size=self.patch_size[0], unfold=True)(x)
        x1 = self.transformer(x1)
        #fold
        x1 = UnfoldingOrFolding((t,s,h,w), patch_size=self.patch_size[0], unfold=False)(x1)
        # print(x1.shape)

        x2 = UnfoldingOrFolding((t, s, h, w), patch_size=self.patch_size[1], unfold=True)(x)
        x2 = self.transformer(x2)
        # fold
        x2 = UnfoldingOrFolding((t, s, h, w), patch_size=self.patch_size[1], unfold=False)(x2)
        # print(x2.shape)

        x3 = UnfoldingOrFolding((t, s, h, w), patch_size=self.patch_size[2], unfold=True)(x)
        x3 = self.transformer(x3)
        # fold
        x3 = UnfoldingOrFolding((t, s, h, w), patch_size=self.patch_size[2], unfold=False)(x3)
        # print(x3.shape)

        x = torch.cat((x1, x2, x3), dim=1)
        # print(x.shape)

        # Fusion
        x = self.conv_1x1_gn_out(x)
        # print(x.shape)

        x = torch.cat((x, y), 1)
        # print(x.shape)

        if not self.no_fusion:
            x = self.fusion(x)
            # print(x.shape)

        x += res

        # breakpoint()
        return x


if __name__ == '__main__':
    img = torch.rand(1, 128, 20, 6, 6)  # (batch_size, time(==channel), slice, height, width)

    network_architecture = {
        "parameters": {
            "in_channel": 128,
            "kernel_size": 3,
            "patch_size": [(20, 2, 2), (2, 2, 6), (2, 6, 2)],
            "dim": 32,  # 96
            "depth": 3,
            "mlp_dim": 128,
        }
    }

    params = network_architecture['parameters']
    network = MobileViTBlock(**params)
    out = network(img)