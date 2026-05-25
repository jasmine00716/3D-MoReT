import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device), indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)  # b s h w t

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head),
                FeedForward(dim, mlp_dim)
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
        return self.norm(x)


class SimpleViT(nn.Module):
    def __init__(self, dim, depth, mlp_dim, patch_size, heads=8, dim_head=64):
        super().__init__()
        ps, ph, pw = patch_size
        patch_dim = mlp_dim * ps * ph * pw
        # print("mlp dim", mlp_dim)
        # print("patch dim", patch_dim)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b t (s ps) (h ph) (w pw) -> b s h w (ph pw ps t)', ph=ph, pw=pw, ps=ps),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, patch_dim),
            nn.LayerNorm(patch_dim),
            Rearrange('b s h w (ph pw ps t) -> b t (s ps) (h ph) (w pw)', ph=ph, pw=pw, ps=ps),
        )

        # self.to_latent = nn.Identity()
        # self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        _, t, s, h, w = x.shape

        x = self.to_patch_embedding(x)
        pe = posemb_sincos_3d(x)

        before_shape = x.shape
        x = rearrange(x, 'b s h w c -> b (s h w) c') + pe

        x = self.transformer(x)
        # print(x.shape)

        x = rearrange(x, 'b (s h w) c -> b s h w c', s=before_shape[1], h=before_shape[2], w=before_shape[3])
        x = self.to_out(x)

        return x


if __name__ == '__main__':
    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:6"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> vit 3d")

    network_architecture = {
        "parameters": {
            "dims": [64, 80, 96],
            "channels": [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 512, 512, 1024],
            "kernel_size": 3,
            "patch_size": (2, 2, 2),
            "num_classes": 7,
            "expansion": 2,
            "device": torch.device("cuda:4"),
        }
    }

    params = network_architecture['parameters']
    network = SimpleViT(**params)
