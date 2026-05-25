import torch
import torch.nn as nn
import torch.nn.functional as F
from models.MobileViT_v1_3D.mobilevit_v1_block import MobileViTBlock

def conv_nxn_gn(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=kernel_size, stride=(stride, stride, stride), padding=(padding, padding, padding), bias=False),
        nn.GroupNorm(1, oup),
        nn.SiLU(),
    )



class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, padding=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (3, 3, 3), stride=stride, padding=padding, groups=hidden_dim, bias=False),
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
                nn.Conv3d(hidden_dim, hidden_dim, (3, 3, 3), stride=stride, padding=1, groups=hidden_dim, bias=False),
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
    def __init__(self, image_size, dims, channels, num_classes, expansion=4, kernel_size=3, patch_size=(2, 2, 2), device=torch.device("cuda:0")):
        super().__init__()
        i_s, ih, iw = image_size
        ps, ph, pw = patch_size
        assert i_s % ps == 0 and ih % ph == 0 and iw % pw == 0

        L = [2, 4, 3]  # depth

        self.device = device

        self.conv1 = conv_nxn_gn(40, channels[0], stride=2).to(device)

        self.mv2 = nn.ModuleList([]).to(device)
        self.mv2.append(MV2Block(channels[0], channels[1], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[1], channels[2], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=1, expansion=expansion).to(device))  # repeat
        self.mv2.append(MV2Block(channels[3], channels[4], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[5], channels[6], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[7], channels[8], stride=2, expansion=expansion).to(device))

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)).to(device))
        self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)).to(device))
        self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)).to(device))

        # self.resblock1 = ResidualBlock(384).to(device)
        # self.resblock2 = ResidualBlock(192).to(device)
        # self.resblock3 = ResidualBlock(96).to(device)
        # self.resblock4 = ResidualBlock(24).to(device)
        #
        # self.deconv1 = nn.ConvTranspose3d(512, 128, kernel_size=(3, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)).to(device)
        # self.deconv2 = nn.ConvTranspose3d(768, 64, kernel_size=(3, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)).to(device)
        # self.deconv3 = nn.ConvTranspose3d(384, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)).to(device)
        # self.deconv4 = nn.ConvTranspose3d(192, 8, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)).to(device)
        # self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(device)

        self.resblock1 = ResidualBlock(channels[4] + channels[8]).to(device)
        self.resblock2 = ResidualBlock(channels[4] + channels[5]).to(device)
        self.resblock3 = ResidualBlock(channels[4]).to(device)
        self.resblock4 = ResidualBlock(32).to(device)

        self.deconv1 = nn.ConvTranspose3d(channels[9], channels[7], kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(device)
        self.deconv2 = nn.ConvTranspose3d(2 * (channels[4] + channels[8]), channels[5], kernel_size=(3, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)).to(device)
        self.deconv3 = nn.ConvTranspose3d(2 * (channels[4] + channels[5]), channels[3], kernel_size=(3, 2, 2), stride=(2, 2, 2), padding=(1, 0, 0)).to(device)
        self.deconv4 = nn.ConvTranspose3d(2 * channels[4], channels[1], kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(device)
        self.deconv5 = nn.ConvTranspose3d(channels[6], 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(device)

        self.last_conv = nn.Conv3d(32, 5, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)

        conv1 = self.conv1(x)  # (1, 16, 10, 112, 112)
        # print("conv1", conv1.shape)

        mv2_0 = self.mv2[0](conv1)  # (1, 16, 10, 112, 112)
        # print("mv2_0", mv2_0.shape)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 24, 5, 56, 56)
        # print("mv2_1", mv2_1.shape)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 24, 5, 56, 56)
        # print("mv2_2", mv2_2.shape)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 24, 5, 56, 56)
        # print("mv2_3", mv2_3.shape)
        mv2_4 = self.mv2[4](mv2_3)  # (1, 48, 3, 28, 28)
        # print("mv2_4", mv2_4.shape)

        mvit_0 = self.mvit[0](mv2_4)  # (1, 48, 3, 28, 28)
        # print("mvit0", mvit_0.shape)

        mv2_5 = self.mv2[5](mvit_0)  # (1, 64, 2, 14, 14)
        # print("mv2_5", mv2_5.shape)

        mvit_1 = self.mvit[1](mv2_5)  # (1, 64, 2, 14, 14)
        # print("mvit1", mvit_1.shape)

        mv2_6 = self.mv2[6](mvit_1)  # (1, 80, 1, 7, 7)
        # print("mv2_6", mv2_6.shape)

        mvit_2 = self.mvit[2](mv2_6)  # (1, 80, 1, 7, 7)
        # print("mvit2", mvit_2.shape)

        deconv1 = self.deconv1(mvit_2)  # (1, 64, 2, 14, 14)
        # print("deconv1", deconv1.shape)
        x = torch.cat((deconv1, mvit_1), dim=1)  # (1, 128, 2, 14, 14)
        # print(x.shape)
        res1 = self.resblock1(x)
        # print(res1.shape)
        x = torch.cat((x, res1), dim=1)  # (1, 256, 2, 14, 14)
        # print(x.shape)

        deconv2 = self.deconv2(x)  # (1, 48, 3, 28, 28)
        # print(deconv2.shape)
        x = torch.cat((deconv2, mvit_0), dim=1)  # (1, 96, 3, 28, 28)
        # print(x.shape)
        res2 = self.resblock2(x)
        x = torch.cat((x, res2), dim=1)  # (1, 192, 3, 28, 28)
        # print("x", x.shape)

        deconv3 = self.deconv3(x)  # (1, 24, 5, 56, 56)
        # print("deconv3", deconv3.shape)
        x = torch.cat((deconv3, mv2_1), dim=1)  # (1, 48, 5, 56, 56)
        # print("x", x.shape)
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)  # (1, 96, 5, 56, 56)
        # print("x", x.shape)

        deconv4 = self.deconv4(x)  # (1, 16, 10, 112, 112)
        # print("deconv4", deconv4.shape)
        x = torch.cat((deconv4, conv1), dim=1)  # (1, 32, 10, 112, 112)
        # print("x", x.shape)
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)  # (1, 64, 10, 112, 112)
        # print(x.shape)

        deconv5 = self.deconv5(x)  # (1, 32, 20, 224, 224)
        # print("deconv5", deconv5.shape)

        reg_output = self.last_conv(deconv5)
        # print(reg_output.shape)
        reg_output = torch.tanh(reg_output)
        # print(reg_output.shape)
        # breakpoint()

        return reg_output


class Mobilevit_nearest_neighbor_upsampling(nn.Module):
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
        self.mv2.append(MV2Block(channels[1], channels[2], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=1, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[2], channels[3], stride=1, expansion=expansion).to(device))  # repeat
        self.mv2.append(MV2Block(channels[3], channels[4], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[5], channels[6], stride=2, expansion=expansion).to(device))
        self.mv2.append(MV2Block(channels[7], channels[8], stride=2, expansion=expansion).to(device))

        self.mvit = nn.ModuleList([]).to(device)
        self.mvit.append(
            MobileViTBlock(dims[0], L[0], channels[5], kernel_size, patch_size, int(dims[0] * 2)).to(device))
        self.mvit.append(
            MobileViTBlock(dims[1], L[1], channels[7], kernel_size, patch_size, int(dims[1] * 4)).to(device))
        self.mvit.append(
            MobileViTBlock(dims[2], L[2], channels[9], kernel_size, patch_size, int(dims[2] * 4)).to(device))

        self.resblock1 = ResidualBlock(channels[4] + channels[8]).to(device)
        self.resblock2 = ResidualBlock(channels[4] + channels[5]).to(device)
        self.resblock3 = ResidualBlock(channels[4]).to(device)
        self.resblock4 = ResidualBlock(32).to(device)

        self.upsample1 = UpsamplingBlock(channels[9], channels[7], scale_factor=(2,2,2)).to(device)
        self.upsample2 = UpsamplingBlock(2 * (channels[4] + channels[8]), channels[5], size=(3, 28, 28)).to(device)
        self.upsample3 = UpsamplingBlock(2 * (channels[4] + channels[5]), channels[3], size=(5, 56, 56)).to(device)
        self.upsample4 = UpsamplingBlock(2 * channels[4], channels[1], scale_factor=(2,2,2)).to(device)
        self.upsample5 = UpsamplingBlock(channels[6], 32, scale_factor=(2, 2, 2)).to(device)

        self.last_conv = nn.Conv3d(32, 5, kernel_size=(1, 1, 1)).to(device)

    def forward(self, x):  # (1, 40, 20, 224, 224)
        conv1 = self.conv1(x)  # (1, 16, 10, 112, 112)
        # print("conv1", conv1.shape)

        mv2_0 = self.mv2[0](conv1)  # (1, 16, 10, 112, 112)
        # print("mv2_0", mv2_0.shape)
        mv2_1 = self.mv2[1](mv2_0)  # (1, 24, 5, 56, 56)
        # print("mv2_1", mv2_1.shape)
        mv2_2 = self.mv2[2](mv2_1)  # (1, 24, 5, 56, 56)
        # print("mv2_2", mv2_2.shape)
        mv2_3 = self.mv2[3](mv2_2)  # (1, 24, 5, 56, 56)
        # print("mv2_3", mv2_3.shape)
        mv2_4 = self.mv2[4](mv2_3)  # (1, 48, 3, 28, 28)
        # print("mv2_4", mv2_4.shape)

        mvit_0 = self.mvit[0](mv2_4)  # (1, 48, 3, 28, 28)
        # print("mvit0", mvit_0.shape)

        mv2_5 = self.mv2[5](mvit_0)  # (1, 64, 2, 14, 14)
        # print("mv2_5", mv2_5.shape)

        mvit_1 = self.mvit[1](mv2_5)  # (1, 64, 2, 14, 14)
        # print("mvit1", mvit_1.shape)

        mv2_6 = self.mv2[6](mvit_1)  # (1, 80, 1, 7, 7)
        # print("mv2_6", mv2_6.shape)

        mvit_2 = self.mvit[2](mv2_6)  # (1, 80, 1, 7, 7)
        # print("mvit2", mvit_2.shape)

        upsample1 = self.upsample1(mvit_2)  # (1, 64, 2, 14, 14)
        x = torch.cat((upsample1, mvit_1), dim=1)
        res1 = self.resblock1(x)
        x = torch.cat((x, res1), dim=1)

        upsample2 = self.upsample2(x)
        # print(upsample2.shape)
        x = torch.cat((upsample2, mvit_0), dim=1)
        # print(x.shape)
        res2 = self.resblock2(x)
        # print(res2.shape)
        x = torch.cat((x, res2), dim=1)
        # print(x.shape)

        upsample3 = self.upsample3(x)  # (1, 24, 5, 56, 56)
        x = torch.cat((upsample3, mv2_1), dim=1)  # (1, 48, 5, 56, 56)
        # print("x", x.shape)
        res3 = self.resblock3(x)
        x = torch.cat((x, res3), dim=1)  # (1, 96, 5, 56, 56)
        # print("x", x.shape)

        upsample4 = self.upsample4(x)  # (1, 16, 10, 112, 112)
        # print("deconv4", deconv4.shape)
        x = torch.cat((upsample4, conv1), dim=1)  # (1, 32, 10, 112, 112)
        # print("x", x.shape)
        res4 = self.resblock4(x)
        x = torch.cat((x, res4), dim=1)  # (1, 64, 10, 112, 112)
        # print(x.shape)

        upsample5 = self.upsample5(x)

        reg_output = self.last_conv(upsample5)
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
    img = torch.rand(1, 40, 20, 224, 224).to(torch.device("cuda:4"))  # (batch_size, time(==channel), slice, height, width)

    print("\n>> mobilevit(3d)")

    network_architecture = {
        "parameters": {
            "image_size": (20, 240, 240),
            "dims": [64, 80, 96],
            "channels": [16, 16, 24, 24, 48, 48, 64, 64, 80, 80],  # [16, 64, 128, 256, 512],  # [16, 24, 48, 64, 80, 320]
            "kernel_size": 3,
            "patch_size": (2, 2, 2),
            "num_classes": 7,
            "expansion": 2,
            "device": torch.device("cuda:4"),
        }
    }
    params = network_architecture['parameters']
    # network = Mobilevit(**params)
    network = Mobilevit_nearest_neighbor_upsampling(**params)

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
    # print('model size: {:.3f}MB\n'.format(size_all_mb))

    out = network(img)
    # print(out.shape)

    # output = network(img)
    # print('params', output)
    # flops, params, ret_layer_info = profile(network.to('cpu'), inputs=(img.to('cpu'),), ret_layer_info=True)
    # print('params', params)
    # print('FLOPs:', flops)


    # print("parameters:", count_parameters(vit)) # 1567697

    # network.eval()
    # with torch.no_grad():
    #     output = network(img)
    # memory_usage = output.element_size() * output.nelement() / (1024 * 1024)
    # print(memory_usage, "MB")

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
