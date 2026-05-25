""" Full assembly of the parts to form the complete network """

# from models.unet_3d.unet_parts import *
from unet_parts import *
import torch.nn.functional as F
import torch

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, device, trilinear=True):
        super(UNet, self).__init__()
        # self.n_channels = n_channels
        self.n_classes = n_classes
        self.device = device
        self.trilinear = trilinear

        self.inc = DoubleConv(n_channels, 64).to(device)
        self.down1 = Down(64, 128).to(device)
        self.down2 = Down(128, 256).to(device)
        self.down3 = Down(256, 512).to(device)
        factor = 2 if trilinear else 1
        self.down4 = Down(512, 1024 // factor).to(device)
        self.up1 = Up(1024, 512 // factor, trilinear).to(device)
        self.up2 = Up(512, 256 // factor, trilinear).to(device)
        self.up3 = Up(256, 128 // factor, trilinear).to(device)
        self.up4 = Up(128, 64, trilinear).to(device)
        self.outc = OutConv(64, n_classes).to(device)

    def forward(self, x):
        if self.device:
            x = x.to(self.device, dtype=torch.float)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return torch.tanh(logits)

if __name__ == '__main__':
    # a = torch.rand((1, 30, 20, 176, 256))
    # b = torch.rand((1, 31, 20, 176, 256))
    # a = torch.rand((1, 33, 20, 176, 256))
    # a = torch.rand((1, 45, 20, 176, 256))
    # from utils import count_parameters
    img = torch.rand((1, 40, 20, 224, 224))
    network = UNet(40, 5, None)

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

    print(total_time / repetitions)

    totalparams = 0
    for name, param in network.named_parameters():
        if param.requires_grad:
            # print(f"Parameter name: {name}, {param.numel()}")
            totalparams += param.numel()
    print(f">> the total parameters: {totalparams}")

    buffer_size = 0
    for buffer in network.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (totalparams + buffer_size) / 1024 ** 2
    print('>> model size: {:.3f}MB\n'.format(size_all_mb))
    #
    # from thop import profile
    #
    # flops, params, ret_layer_info = profile(network.to(torch.device("cuda:6")),
    #                                         inputs=(img.to(torch.device("cuda:6")),), ret_layer_info=True)
    # print('params', params)
    # print('FLOPs:', flops)
    #

    # from torchsummary import summary
    #
    # summary(network, (40, 20, 224, 224), batch_size=1, device="cuda")

    # print(count_parameters(net))
    # _p = net(a)
    # print(_p.shape)



