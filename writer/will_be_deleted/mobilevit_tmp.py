import torch
import torch.nn as nn

from einops import rearrange
import torch.nn.functional as F

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
    )

def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm3d(oup),
        nn.SiLU()
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


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobilevitBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ps, self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)

    def forward(self, x):
        y = x.clone()  # (1, 48, 3, 28, 28)

        # Local representations
        x = self.conv1(x)  # (1, 48, 3, 28, 28)
        x = self.conv2(x)  # (1, 64, 3, 28, 28)

        # Global representations
        _, t, s, h, w = x.shape  # (1, 64, 3, 28, 28)

        # unfold
        x = rearrange(x, 'b t (s ps) (h ph) (w pw) -> b (ps ph pw) (s h w) t', ps=self.ps, ph=self.ph, pw=self.pw)  # (1, 4, 588, 64)

        x = self.transformer(x)  # (1, 4, 588, 64)

        #fold
        x = rearrange(x, 'b (ps ph pw) (s h w) t -> b t (s ps) (h ph) (w pw)', s=s // self.ps, h=h // self.ph, w=w // self.pw,
                      ps=self.ps, ph=self.ph, pw=self.pw)  # (1, 64, 3, 28, 28)

        # Fusion
        x = self.conv3(x)  # (1, 48, 3, 28, 28)
        x = torch.cat((x, y), 1)  # (1, 96, 3, 28, 28)
        x = self.conv4(x)  # (1, 48, 3, 28, 28)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        self.gn = nn.GroupNorm(1, int(n_channels))
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 4),
                               padding=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn(x)))
        out = self.conv2(F.relu(self.gn(out)))
        out = self.conv3(F.relu(self.gn(out)))
        return out


class UpsamplingBlock(nn.Module):
    def __init__(self, n_channels, n_output_channels):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv1x1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.upsample(x)
        # print("upsample:", x.shape)
        x = self.conv1x1(x)
        # print("conv1x1:", x.shape)
        return x

class Mobilevit_trial(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(1, 2, 2),
                 dev_0=torch.device("cuda:4"), dev_1=torch.device("cuda:4"), dev_2=torch.device("cuda:4"),
                 dev_3=torch.device("cuda:4")):
        super().__init__()
        ih, iw = image_size
        ps, ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.dev_0 = dev_0
        self.dev_1 = dev_1
        self.dev_2 = dev_2
        self.dev_3 = dev_3

        self.conv1 = conv_nxn_bn(40, channels[0], stride=2).to(dev_0)

        self.mv2 = nn.ModuleList([]).to(dev_0)
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion).to(dev_0))  # 16, 16, 1, 2
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion).to(dev_0))  # 16, 24, 2, 1
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion).to(dev_0))  # 24, 24, 1, 2
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion).to(dev_0))  # 24, 24, 1, 2 (Repeat)
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion).to(dev_0))  # 24, 48, 2, 2
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion).to(dev_0))  # 48, 64, 2, 2
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion).to(dev_0))  # 64, 80, 2, 2

        self.mvit = nn.ModuleList([]).to(dev_0)
        self.mvit.append(MobilevitBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0] * 2)).to(dev_0))  # 64, 2, 48, 3, (1,2,2), 128
        self.mvit.append(MobilevitBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)).to(dev_0))  # 80, 4, 64, 3, (1,2,2), 320
        self.mvit.append(MobilevitBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)).to(dev_0))  # 96, 3, 80, 3, (1,2,2), 384

        self.resblock1 = ResidualBlock(28).to(dev_0)

        self.upsample1 = UpsamplingBlock(24, 12).to(dev_0)
        self.upsample2 = UpsamplingBlock(56, 28).to(dev_0)

        self.last_conv = nn.Conv3d(28, 5, kernel_size=(1, 1, 1)).to(dev_0)

    def forward(self, x):  # (1, 40, 20, 240, 240)
        x = self.conv1(x)  # (1, 16, 10, 120, 120)
        print(x.shape)
        mv2_0 = self.mv2[0](x)  # (1, 16, 10, 120, 120)
        print(mv2_0.shape)
        breakpoint()

        mv2_1 = self.mv2[1](mv2_0)  # (1, 24, 5, 60, 60)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 24, 5, 60, 60)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 24, 5, 60, 60), Repeat

        mvit_0 = self.mvit[0](mv2_3)  # (1, 24, 5, 60, 60)

        x = self.upsample1(mvit_0)  # (1, 12, 10, 120, 120)
        x = torch.cat((x, mv2_0), dim=1)  # (1, 28, 10, 120, 120)
        res1 = self.resblock1(x)  # (1, 28, 10, 120, 120)
        x = torch.cat((x, res1), dim=1)  # (1, 56, 10, 120, 120)

        x = self.upsample2(x)  # (1, 28, 20, 240, 240)

        reg_output = self.last_conv(x)  # (1, 5, 20, 240, 240)
        reg_output = torch.tanh(reg_output)

        return reg_output


class Mobilevit(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(1, 2, 2),
                 dev_0=torch.device("cuda:4"), dev_1=torch.device("cuda:4"), dev_2=torch.device("cuda:4"), dev_3=torch.device("cuda:4")):
        super().__init__()
        ih, iw = image_size
        ps, ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.dev_0 = dev_0
        self.dev_1 = dev_1
        self.dev_2 = dev_2
        self.dev_3 = dev_3

        self.conv1 = conv_nxn_bn(40, channels[0], stride=2).to(dev_0)

        self.mv2 = nn.ModuleList([]).to(dev_0)
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion).to(dev_0))  # 16, 16, 1, 2
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion).to(dev_0))  # 16, 24, 2, 1
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion).to(dev_0))  # 24, 24, 1, 2
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion).to(dev_0))  # 24, 24, 1, 2 (Repeat)
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion).to(dev_0))  # 24, 48, 2, 2
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion).to(dev_0))  # 48, 64, 2, 2
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion).to(dev_0))  # 64, 80, 2, 2

        self.mvit = nn.ModuleList([]).to(dev_0)
        self.mvit.append(MobilevitBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0] * 2)).to(dev_0))  # 64, 2, 48, 3, (1,2,2), 128
        self.mvit.append(MobilevitBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)).to(dev_0))  # 80, 4, 64, 3, (1,2,2), 320
        self.mvit.append(MobilevitBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)).to(dev_0))  # 96, 3, 80, 3, (1,2,2), 384

        self.resblock1 = ResidualBlock(28).to(dev_0)

        self.upsample1 = UpsamplingBlock(24, 12).to(dev_0)
        self.upsample2 = UpsamplingBlock(56, 28).to(dev_0)

        self.last_conv = nn.Conv3d(28, 5, kernel_size=(1, 1, 1)).to(dev_0)

    def forward(self, x):  # (1, 40, 20, 240, 240)
        x = self.conv1(x)  # (1, 16, 10, 120, 120)
        print(x.shape)
        mv2_0 = self.mv2[0](x)  # (1, 16, 10, 120, 120)
        print(mv2_0.shape)
        breakpoint()

        mv2_1 = self.mv2[1](mv2_0)  # (1, 24, 5, 60, 60)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 24, 5, 60, 60)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 24, 5, 60, 60), Repeat

        mvit_0 = self.mvit[0](mv2_3)  # (1, 24, 5, 60, 60)

        x = self.upsample1(mvit_0)  # (1, 12, 10, 120, 120)
        x = torch.cat((x, mv2_0), dim=1)  # (1, 28, 10, 120, 120)
        res1 = self.resblock1(x)  # (1, 28, 10, 120, 120)
        x = torch.cat((x, res1), dim=1)  # (1, 56, 10, 120, 120)

        x = self.upsample2(x)  # (1, 28, 20, 240, 240)

        reg_output = self.last_conv(x)  # (1, 5, 20, 240, 240)
        reg_output = torch.tanh(reg_output)

        return reg_output


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return Mobilevit((224, 224), dims, channels, num_classes=7, expansion=2)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return Mobilevit((224, 224), dims, channels, num_classes=7)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return Mobilevit((224, 224), dims, channels, num_classes=7)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.rand(1, 40, 20, 240, 240).to(torch.device("cuda:4"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> mobilevit_xxs")
    print(img.shape, "\n")
    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print("parameters:", count_parameters(vit))

    # print("\n>> mobilevit_xs")
    # vit = mobilevit_xs()
    # out = vit(img)
    # print(out.shape)
    # print("parameters:", count_parameters(vit))
    #
    # print("\n>> mobilevit_s")
    # vit = mobilevit_s()
    # out = vit(img)
    # print(out.shape)
    # print("parameters:", count_parameters(vit))