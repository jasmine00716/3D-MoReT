import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torchsummary import summary
# from models.MobileViT_v3_3D.mobilevit_v3_block import MobileViTBlock
# from models.MobileViT_v3_3D.vit_block import SimpleViT
from mobilevit_v3_block import MobileViTBlock
from vit_block import SimpleViT

# mobilevit v3 로 수정
# python models/MobileViT_v3_3D/mobilevit_v3_trs

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

class MV2Block(nn.Module):
    def __init__(self, inp, oup, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), expansion=4):
    # def __init__(self, inp, oup, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1), expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (1, 3, 3), stride=(1, 1, 1), padding=(0,1,1), groups=hidden_dim, bias=False),
                # nn.Conv3d(hidden_dim, hidden_dim, (3, 3, 3), stride=(1, 1, 1), padding=(1,1,1), groups=hidden_dim, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, groups=hidden_dim, bias=False),
                nn.GroupNorm(1, oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=(1, stride, stride), padding=padding, groups=hidden_dim, bias=False),
                nn.GroupNorm(1, hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.GroupNorm(1, oup)
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


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
    def __init__(self, n_channels, n_output_channels, size=None, scale_factor=None):
        super(UpsamplingBlock, self).__init__()
        self.upsample = nn.Upsample(size=size, scale_factor=scale_factor, mode='nearest')
        self.conv1x1 = nn.Conv3d(n_channels, n_output_channels, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv1x1(x)
        return x


class Mobilevit(nn.Module):
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2, 2), device=torch.device("cuda:4")):
        super().__init__()
        i_s, ih, iw = image_size
        ps, ph, pw = patch_size
        assert i_s % ps == 0 and ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.device = device

        self.conv1 = conv_nxn_gn(40, channels[0], stride=2).to(device)

        self.mv2 = nn.ModuleList([]).to(device)
        self.mv2.append(MV2Block(channels[0], channels[1], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[1], channels[2], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[3], channels[4], stride=1, expansion=expansion).to(device))  # repeat
        self.mv2.append(MV2Block(channels[4], channels[5], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[5], channels[6], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[7], channels[8], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[9], channels[10], stride=2, padding=(0, 0, 0), expansion=expansion).to(device))

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[6], kernel_size, patch_size, int(dims[0] * 2)).to(device))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[8], kernel_size, patch_size, int(dims[1] * 4)).to(device))

        self.convMtoN = conv_1x1_gn(channels[10], 128).to(device)
        self.vit = SimpleViT(dims[2], L[2], 128, (1, 6, 6), heads=4, dim_head=32).to(device)
        self.convNtoM = conv_1x1_gn(128, channels[10]).to(device)
        ''''''
        self.resblock1 = ResidualBlock(channels[8] + 512).to(device)  # 256 + 320
        self.resblock2 = ResidualBlock(channels[6] + 256).to(device)  # 128 + 256
        self.resblock3 = ResidualBlock(64 + channels[4]).to(device)  # 64 + 64
        self.resblock4 = ResidualBlock(52 + channels[2]).to(device)  # 52 + 52
        ''''''
        self.deconv1 = nn.ConvTranspose3d(channels[11], channels[10], kernel_size=(3, 3, 3), stride=(1, 3, 3), padding=(1, 2, 2)).to(device)  # 640 + 320
        self.deconv2 = nn.ConvTranspose3d(2 * (channels[8] + channels[10]), channels[8], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2*(256 + 320), 256
        self.deconv3 = nn.ConvTranspose3d(2 * (channels[6] + channels[8]), channels[5], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (128 + 256), 64
        self.deconv4 = nn.ConvTranspose3d(2 * (2 * channels[4]), channels[2], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (64 + 64), 52
        self.deconv5 = nn.ConvTranspose3d(2 * (2 * channels[2]), 48, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)).to(device)  # 2 * (52 + 52), 48

        self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)

        conv1 = self.conv1(x)  # (1, 52, 20, 112, 112)

        mv2_0 = self.mv2[0](conv1)  # (1, 52, 20, 112, 112)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 52, 20, 112, 112)

        mv2_2 = self.mv2[2](mv2_1)  # (1, 64, 20, 56, 56)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 64, 20, 56, 56)
        mv2_4 = self.mv2[4](mv2_3)  # (1, 64, 20, 56, 56)
        mv2_5 = self.mv2[5](mv2_4)  # (1, 128, 20, 28, 28)
        mvit_0 = self.mvit[0](mv2_5)  # (1, 128, 20, 28, 28)
        mv2_6 = self.mv2[6](mvit_0)  # (1, 256, 20, 14, 14)
        mvit_1 = self.mvit[1](mv2_6)  # (1, 256, 20, 14, 14)
        mv2_7 = self.mv2[7](mvit_1)  # (1, 320, 20, 6, 6)
        convMtoN = self.convMtoN(mv2_7)
        vit = self.vit(convMtoN)
        convNtoM = self.convNtoM(vit)

        deconv1 = self.deconv1(convNtoM)  # (1, 320, 20, 14, 14)
        x = torch.cat((deconv1, mv2_6), dim=1)  # (1, 576, 20, 14, 14)
        res1 = self.resblock1(x)
        x = torch.cat((x, res1), dim=1)  # (1, 1152, 20, 14, 14)

        deconv2 = self.deconv2(x)  # (1, 256, 20, 28, 28)
        x = torch.cat((deconv2, mv2_5), dim=1)  # (1, 384, 20, 28, 28)
        res2 = self.resblock2(x)
        x = torch.cat((x, res2), dim=1)  # (1, 768, 20, 28, 28)

        deconv3 = self.deconv3(x)  # (1, 64, 20, 56, 56)
        x = torch.cat((deconv3, mv2_3), dim=1)  # (1, 128, 20, 56, 56)  # mv2_2
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)  # (1, 256, 10, 56, 56)
        deconv4 = self.deconv4(x)  # (1, 52, 20, 112, 112)
        x = torch.cat((deconv4, mv2_1), dim=1)  # (1, 104, 20, 112, 112)  #mv2_1
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)  # (1, 208, 20, 112, 112)
        deconv5 = self.deconv5(x) # (1, 48, 20, 224, 24)

        reg_output = self.last_conv(deconv5)

        reg_output = torch.tanh(reg_output)

        return reg_output


def mobilevit_xxs():
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    # channels = [16, 16, 24, 24, 40, 40, 56, 56, 112, 112, 1280, 1280]
    # channels = [16, 24, 40, 112, 1280]
    # channels = [16, 64, 128, 512, 1024]
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
    # img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:0"))  # (batch_size, time(==channel), slice, height, width)
    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:4"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> mobilevit(3d)")

    network_architecture = {
        "parameters": {
            "image_size": (20, 224, 224),
            "dims": [64, 80, 96],
            "channels": [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 512, 512],
            "kernel_size": 3,
            "patch_size": (2, 2, 2),
            "num_classes": 7,
            "expansion": 2,
            "device": torch.device("cuda:4"),
        }
    }
    params = network_architecture['parameters']
    network = Mobilevit(**params)

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

    print(total_time/repetitions/1000)
    # out = network(img)
    # print(out.shape)
    # summary(network, input_size=(batch_size, 40, 20, 224, 224), device="cpu")
    # # network = Mobilevit_nearest_neighbor_upsampling(**params)
    #
    # from thop import profile
    #
    # # output = network(img)
    # flops, params, ret_layer_info = profile(network.to(torch.device("cuda:4")), inputs=(img.to(torch.device("cuda:4")),), ret_layer_info=True)
    # print('params', params)
    # print('FLOPs:', flops)
    # #
    # # print("parameters:", count_parameters(network)) # 1567697
    # #
    #
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
    # import pytorch_model_summary
    # print(pytorch_model_summary.summary(network, torch.zeros(1, 40, 20, 224, 224), show_input=True))
    #
    # # from torchinfo import summary
    # # batch_size = 4

    # from torchsummary import summary
    # summary(network, (40, 20, 224, 224), batch_size=4, device="cpu")

    # out = network(img)
