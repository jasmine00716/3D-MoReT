import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.ops import StochasticDepth
from torchsummary import summary

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
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(oup),
                nn.GroupNorm(1, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), groups=hidden_dim, bias=False),
                # nn.BatchNorm3d(hidden_dim),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                # nn.BatchNorm3d(oup),
                nn.GroupNorm(1, oup)
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
        y = x.clone()  # (1, 24, 20, 28, 28)                                        (1, 24, 20, 112, 112)
        # print(y.shape)

        # Local representations
        x = self.conv1(x)  # (1, 24, 20, 28, 28)                                         (1, 24, 20, 112, 112)
        # print(x.shape)
        x = self.conv2(x)  # (1, 64, 20, 28, 28)                                         (1, 64, 20, 112, 112)
        # print(x.shape)

        # Global representations
        _, t, s, h, w = x.shape  # (1, 64, 20, 112, 112)
        if h % 2 == 0 and w % 2 == 0:
            # unfold
            x = rearrange(x, 'b t (s ps) (h ph) (w pw) -> b (ps ph pw) (s h w) t', ps=self.ps, ph=self.ph, pw=self.pw)  # (1, 4, 3920, 64)                                         (1, 4, 62720, 64)
            # print(x.shape)

            x = self.transformer(x)  # (1, 4, 3920, 64)                                          (1, 4, 588, 64)
            # print(x.shape)

            #fold
            x = rearrange(x, 'b (ps ph pw) (s h w) t -> b t (s ps) (h ph) (w pw)', s=s // self.ps, h=h // self.ph, w=w // self.pw,
                          ps=self.ps, ph=self.ph, pw=self.pw)  # (1, 64, 20, 28, 28)                                         (1, 64, 3, 28, 28)
            # print(x.shape)
        else:
            # unfold
            x = rearrange(x, 'b t (s 1) (h 1) (w 1) -> b (1 1 1) (s h w) t')

            x = self.transformer(x)

            # fold
            x = rearrange(x, 'b (1 1 1) (s h w) t -> b t (s 1) (h 1) (w 1)', s=s, h=h, w=w)

        # Fusion
        x = self.conv3(x)  # (1, 24, 20, 28, 28)                                          (1, 48, 3, 28, 28)
        # print(x.shape)
        x = torch.cat((x, y), 1)  #  (1, 48, 20, 28, 28)                                         (1, 96, 3, 28, 28)
        # print(x.shape)
        x = self.conv4(x)  # (1, 24, 20, 28, 28)                                          (1, 48, 3, 28, 28)
        # print(x.shape)
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
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='nearest')
        self.conv1x1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(1, 1, 1))

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return x


class Mobilevit(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(1, 2, 2), device=torch.device("cuda:0")):
        super().__init__()
        ih, iw = image_size
        ps, ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.device = device

        self.conv1 = conv_nxn_bn(40, channels[0], stride=2).to(device)

        self.mv2 = nn.ModuleList([]).to(device)
        self.mv2.append(MV2Block(channels[0], channels[1], 2, expansion).to(device))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], 2, expansion).to(device))
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion).to(device))

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(MobilevitBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0] * 2)).to(device))
        self.mvit.append(MobilevitBlock(dims[1], L[1], channels[4], kernel_size, patch_size, int(dims[1] * 4)).to(device))

        # self.conv2 = conv_1x1_bn(channels[-2], channels[-1]).to(device)

        self.resblock1 = ResidualBlock(384).to(device)
        self.resblock2 = ResidualBlock(192).to(device)
        self.resblock3 = ResidualBlock(96).to(device)
        self.resblock4 = ResidualBlock(24).to(device)

        self.deconv0 = nn.ConvTranspose3d(512, 128, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(device)
        self.deconv1 = nn.ConvTranspose3d(768, 64, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(device)
        self.deconv2 = nn.ConvTranspose3d(384, 32, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(device)
        self.deconv3 = nn.ConvTranspose3d(192, 8, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(device)

        self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)
        '''기존 모델(3d mrod)과 같이 4번의 reduction하도록 수정'''

        conv1 = self.conv1(x)  # (1, 16, 20, 224, 224)

        mv2_0 = self.mv2[0](conv1)  # (1, 64, 20, 112, 112)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 128, 20, 56, 56)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 512, 20, 28, 28)

        mvit_0 = self.mvit[0](mv2_2)  # (1, 512, 20, 28, 28)

        mv2_3 = self.mv2[3](mvit_0)  # (1, 1024, 20, 14, 14)

        mvit_1 = self.mvit[1](mv2_3)  # (1, 1024, 20, 14, 14)

        deconv1 = self.deconv0(mvit_1)  # (1, 256, 20, 28, 28)
        x = torch.cat((deconv1, mvit_0), dim=1)  # (1, 778, 20, 28, 28)
        res1 = self.resblock1(x)
        x = torch.cat((x, res1), dim=1)  # (1, 1536, 20, 28, 28)

        deconv2 = self.deconv1(x)  # (1, 64, 20, 56, 56)
        x = torch.cat((deconv2, mv2_1), dim=1)  # (1, 192, 20, 56, 56)
        res2 = self.resblock2(x)  # 1, 576, 20, 56, 56
        x = torch.cat((x, res2), dim=1)  # 1, 1152, 20, 56, 56

        deconv3 = self.deconv2(x)  # (1, 32, 20, 112, 112)
        x = torch.cat((deconv3, mv2_0), dim=1)  # (1, 96, 20, 112, 112)
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)

        deconv4 = self.deconv3(x)  # (1, 8, 20, 224, 224)
        x = torch.cat((deconv4, conv1), dim=1)  # (1, 24, 20, 224, 224)
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)

        reg_output = self.last_conv(x)
        reg_output = torch.tanh(reg_output)

        return reg_output


class Mobilevit_nearest_neighbor_upsampling(nn.Module): # design2-채널수변경([32, 64, 128, 256, 512])
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(1, 2, 2), device=torch.device("cuda:4")):
        super().__init__()
        ih, iw = image_size
        ps, ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.device = device

        self.conv1 = conv_nxn_bn(40, channels[0], stride=2).to(device)

        self.mv2 = nn.ModuleList([]).to(device)
        self.mv2.append(MV2Block(channels[0], channels[1], 2, expansion).to(device))  # 16, 16, 1, 2
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion).to(device))  # 16, 24, 2, 1
        self.mv2.append(MV2Block(channels[2], channels[3], 2, expansion).to(device))  # 24, 24, 1, 2
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion).to(device))  # 24, 24, 1, 2 (Repeat)
        # self.mv2.append(MV2Block(channels[4], channels[4], 2, expansion).to(device))  # 24, 48, 2, 2

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(MobilevitBlock(dims[0], L[0], channels[3], kernel_size, patch_size, int(dims[0] * 2)).to(device))  # 64, 2, 48, 3, (1,2,2), 128
        self.mvit.append(MobilevitBlock(dims[1], L[1], channels[4], kernel_size, patch_size, int(dims[1] * 4)).to(device))  # 80, 4, 64, 3, (1,2,2), 320

        self.resblock1 = ResidualBlock(384).to(device)
        self.resblock2 = ResidualBlock(192).to(device)
        self.resblock3 = ResidualBlock(96).to(device)
        self.resblock4 = ResidualBlock(24).to(device)

        self.upsample1 = UpsamplingBlock(512, 128).to(device)
        self.upsample2 = UpsamplingBlock(768, 64).to(device)
        self.upsample3 = UpsamplingBlock(384, 32).to(device)
        self.upsample4 = UpsamplingBlock(192, 8).to(device)

        self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)
        '''기존 모델(3d mrod)과 같이 4번의 reduction하도록 수정'''

        conv1 = self.conv1(x)  # (1, 16, 20, 224, 224)

        mv2_0 = self.mv2[0](conv1)  # (1, 24, 20, 112, 112)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 40, 20, 56, 56)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 112, 20, 28, 28)

        mvit_0 = self.mvit[0](mv2_2)  # (1, 112, 20, 28, 28)

        mv2_3 = self.mv2[3](mvit_0)  # (1, 1280, 20, 14, 14)

        mvit_1 = self.mvit[1](mv2_3)  # (1, 1280, 20, 14, 14)

        upsample1 = self.upsample1(mvit_1)  # (1, 640, 20, 28, 28)
        x = torch.cat((upsample1, mvit_0), dim=1)  # (1, 752, 20, 28, 28)
        res1 = self.resblock1(x)
        x = torch.cat((x, res1), dim=1)  # (1, 1536, 20, 28, 28)

        upsample2 = self.upsample2(x)  # (1, 120, 20, 56, 56)
        x = torch.cat((upsample2, mv2_1), dim=1)  # (1, 160, 20, 56, 56)
        res2 = self.resblock2(x)  # 1, 576, 20, 56, 56
        x = torch.cat((x, res2), dim=1)  # 1, 1152, 20, 56, 56

        upsample3 = self.upsample3(x)  # (1, 40, 20, 112, 112)
        x = torch.cat((upsample3, mv2_0), dim=1)  # (1, 64, 20, 112, 112)
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)

        upsample4 = self.upsample4(x)  # (1, 8, 20, 224, 224)
        x = torch.cat((upsample4, conv1), dim=1)  # (1, 24, 20, 224, 224)
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)

        reg_output = self.last_conv(x)
        reg_output = torch.tanh(reg_output)

        return reg_output

def mobilevit_xxs():
    dims = [64, 80, 96]
    # channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # channels = [16, 16, 24, 24, 40, 40, 56, 56, 112, 112, 1280, 1280]
    # channels = [16, 24, 40, 112, 1280]
    channels = [16, 64, 128, 512, 1024]
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
    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:0"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> mobilevit_xxs")

    # device = torch.device("cuda:0")

    # vit = mobilevit_xxs()
    vit = Mobilevit_nearest_neighbor_upsampling(image_size = (240, 240),
                                                dims =[64, 80, 96],
                                                channels = [64, 128, 256, 512, 1024],  # [16, 24, 40, 112, 1280], [16, 64, 128, 512, 1024](design1-채널수변경)
                                                kernel_size = 3,
                                                patch_size = (1, 2, 2),
                                                num_classes = 7)

    out = vit(img)

    from thop import profile

    output = network(toy_input)
    print('params', output)
    flops, params, ret_layer_info = profile(network.to('cpu'), inputs=(img.to('cpu'),), ret_layer_info=True)
    print('params', params)
    print('FLOPs:', flops)

    #
    print(out.shape)
    # print("parameters:", count_parameters(vit)) # 1567697
    #
    vit.eval()
    with torch.no_grad():
        output = vit(img)
    memory_usage = output.element_size() * output.nelement() / (1024 * 1024)
    print(memory_usage, "MB")


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