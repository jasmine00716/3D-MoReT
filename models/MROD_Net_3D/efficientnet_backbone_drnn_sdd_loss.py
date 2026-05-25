import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientnet_pytorch import EfficientNet
# import models.efficientnet_pytorch
# from models.MROD_Net_3D import efficientnet_pytorch
import efficientnet_pytorch
import torch.autograd.profiler as profiler


class ResidualBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResidualBlock, self).__init__()
        # self.gn = nn.GroupNorm(int(n_channels), int(n_channels))
        self.gn = nn.GroupNorm(1, int(n_channels))
        # self.bn = nn.BatchNorm3d(n_channels)
        self.conv1 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv2 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 3, 3), groups=int(n_channels / 8),
                               padding=(1, 1, 1), bias=False)
        self.conv3 = nn.Conv3d(n_channels, n_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

    def forward(self, x):
        with profiler.record_function("Residual Block"):
            out = self.conv1(F.relu(self.gn(x)))
            out = self.conv2(F.relu(self.gn(out)))
            out = self.conv3(F.relu(self.gn(out)))
        return out

class SceneUnderstandingModule(nn.Module):
    def __init__(self, n_channels, scale_factor, device=torch.device("cuda:3"), dilations=[1, 3, 6, 9]):
        super(SceneUnderstandingModule, self).__init__()
        out_n_channels = 1280
        self.device = device
        self.n_channels = n_channels
        self.out_n_channels = out_n_channels
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)
        self.aspp1 = nn.Sequential(
            nn.Conv3d(n_channels, out_n_channels, kernel_size=1, stride=1, padding=0, dilation=dilations[0]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp2 = nn.Sequential(
            nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp3 = nn.Sequential(
            nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.aspp4 = nn.Sequential(
            nn.Conv3d(n_channels, out_n_channels, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3]),
            nn.GroupNorm(1, out_n_channels),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Conv3d(out_n_channels * 5, n_channels, 1),
            nn.GroupNorm(1, n_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.5)
        )

    def forward(self, x):
        d = x.shape[2]
        global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((d, 1, 1)),
                                             nn.Conv3d(self.n_channels, self.out_n_channels, 1, stride=1, bias=False),
                                             nn.GroupNorm(1, self.out_n_channels),
                                             nn.ReLU()).to(self.device)
        x0 = global_avg_pool(x)
        x0 = self.upsample(x0)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        # print(x.shape)
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x4.shape)

        x6 = torch.cat((x0, x1, x2, x3, x4), dim=1)
        # print('cat x6 size:', x6.size())
        out = self.concat_process(x6)
        return out


class OrdinalRegressionLayer(nn.Module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        with profiler.record_function("Ordinal Regression"):
            """
            :param x: N X H X W X C, N is batch_size, C is channels of features
            :return: ord_labels is ordinal outputs for each spatial locations , size is N x H X W X C (C = 2K, K is interval of SID)
                     decode_label is the ordinal labels for each position of Image I
            """
            # P is phase number. In this case, P = 5
            N, C, D, H, W = x.size()
            x = x.view(N, -1, 5, D, H, W)
            N, C, P, D, H, W = x.size()
            ord_num = C // 2

            """
            replace iter with matrix operation
            fast speed methods
            """
            # x = x.view(-1, 2, ord_num, P, D, H, W)
            # prob = nn.functional.softmax(x, dim=1)[:, 0, :, :, :, :, :]
            # prob = F.log_softmax(x, dim=1).view(-1, ord_num, P, D, H, W)
            # return prob
            A = x[:, ::2, :, :, :, :].clone()
            B = x[:, 1::2, :, :, :, :].clone()

            A = A.view(N, 1, ord_num * P * D * H * W)
            B = B.view(N, 1, ord_num * P * D * H * W)

            C = torch.cat((A, B), dim=1)
            C = torch.clamp(C, min=1e-8, max=1e8)  # prevent nans

            ord_c = nn.functional.softmax(C, dim=1)

            ord_c1 = ord_c[:, 1, :].clone()
            ord_c1 = ord_c1.view(-1, ord_num, P, D, H, W)
        return ord_c1

# Original DRNN is the neural network architecture which is the same to paper's NN architecture.
class EfficientnetBackboneDrnnSddLoss(nn.Module):
    # denblock_config = (6, 12, 24, 48)
    def __init__(self, n_channels, growth_rate, reduction, k_ordinal_class, dev_0, dev_1, dev_2, dev_3, model_name, scale_factor=(1, 14, 14)):
        super(EfficientnetBackboneDrnnSddLoss, self).__init__()
        self.dev_0 = dev_0
        self.dev_1 = dev_1
        self.dev_2 = dev_2
        self.dev_3 = dev_3
        # dense 1
        self.efficient_net = efficientnet_pytorch.EfficientNet.from_name(model_name, in_channels=n_channels, num_classes=1, drop_connect_rate=None).to(dev_0)

        self.scene_understanding_module = SceneUnderstandingModule(1280, scale_factor, device=dev_0).to(dev_0)

        n_channels = 1280
        self.transpose_conv1 = nn.ConvTranspose3d(n_channels, int(n_channels * reduction), kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)        # # resblock
        self.resblock1 = ResidualBlock(752).to(dev_0)
        n_channels = 1504
        # n_channels = int(n_channels * 6) # n_channels = 512
        self.transpose_conv2 = nn.ConvTranspose3d(n_channels, 120, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)
        self.resblock2 = ResidualBlock(160).to(dev_0)
        n_channels = 320
        self.transpose_conv3 = nn.ConvTranspose3d(n_channels, 40, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_0)
        # n_channels = int(n_channels // 12)  # n_channels = 32
        self.resblock3 = ResidualBlock(64).to(dev_1)
        self.transpose_conv4 = nn.ConvTranspose3d(128, 8, kernel_size=(3, 2, 2), stride=(1, 2, 2), padding=(1, 0, 0)).to(dev_1)
        # n_channels = int(n_channels // 12) # n_channels = 8
        self.resblock4 = ResidualBlock(24).to(dev_1)
        self.last_gn = nn.GroupNorm(32, 32)
        self.last_gn = nn.GroupNorm(1, 32).to(dev_1)
        self.ordinal_regression_conv1 = nn.Conv3d(48, k_ordinal_class * 10, kernel_size=(1, 1, 1)).to(dev_2)
        self.orl1 = OrdinalRegressionLayer().to(dev_2)
        self.last_conv = nn.Conv3d(48, 5, kernel_size=(1, 1, 1)).to(dev_0)

    def forward(self, x):
        # encoder
        x = x.to(self.dev_0, dtype=torch.float)

        with profiler.record_function("Feature Extraction"):
            endpoints = self.efficient_net.extract_endpoints(x)
            reduction_1, reduction_2, reduction_3, reduction_4, reduction_out = endpoints["reduction_1"], endpoints["reduction_2"], \
                                                                      endpoints["reduction_3"], endpoints["reduction_4"], \
                                                                      endpoints["reduction_5"]

            reduction_1, reduction_2, reduction_3, reduction_4, reduction_out = reduction_1.to(self.dev_1), reduction_2.to(self.dev_1), \
                                                                      reduction_3.to(self.dev_1), reduction_4.to(self.dev_1), \
                                                                      reduction_out.to(self.dev_1)
        # print(reduction_1.shape)
        # print(reduction_2.shape)
        # print(reduction_3.shape)
        # print(reduction_4.shape)
        # print(out.shape)
        aspp_out = self.scene_understanding_module(reduction_out)

        # decoder
        out = self.transpose_conv1(aspp_out)
        out = torch.cat((reduction_4, out), dim=1)
        del reduction_4
        res1 = self.resblock1(out)
        out = torch.cat((out, res1), dim=1)
        del res1

        out = self.transpose_conv2(out)
        out = torch.cat((reduction_3, out), dim=1)
        del reduction_3
        res2 = self.resblock2(out)
        out = torch.cat((out, res2), dim=1)
        del res2

        out = self.transpose_conv3(out)
        out = torch.cat((reduction_2, out), dim=1)
        # out = out.to(self.dev_1)
        del reduction_2
        res3 = self.resblock3(out)
        out = torch.cat((out, res3), dim=1)
        del res3

        out = self.transpose_conv4(out)
        out = torch.cat((reduction_1, out), dim=1)
        del reduction_1
        res4 = self.resblock4(out)
        out = torch.cat((out, res4), dim=1)
        del res4

        reg_output = self.last_conv(out)
        reg_output = torch.tanh(reg_output)

        ordinal_regression1 = self.ordinal_regression_conv1(out)
        ordinal_regression1 = self.orl1(ordinal_regression1)

        return (reg_output, ordinal_regression1)


if __name__ == '__main__':
    print(">> 3d mrod net")
    network = EfficientnetBackboneDrnnSddLoss(40, 12, 0.5, k_ordinal_class=15, dev_0=torch.device("cuda:4"), dev_1=torch.device("cuda:4"), dev_2=torch.device("cuda:4"), dev_3=torch.device("cuda:4"), model_name="efficientnet-b0")

    img = torch.randn(1, 40, 20, 224, 224)

    output = network(img)

    # from torchsummary import summary
    #
    # summary(net, (40, 20, 224, 224), batch_size=1, device='cpu')
    # breakpoint()

    # print(f"\ninput size: {dummy_input.shape}")
    # output1= net(dummy_input).to('cuda:4')  # warm-up
    # print(f"output1 size: {output1.shape}")
    # # print(f"output2 size: {output2.shape}")
    #
    # print("Model's state_dict:")
    # for param_tensor in net.state_dict():
    #     print(param_tensor, "\t", net.state_dict()[param_tensor].size())
    # breakpoint()

    # # count parameters
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
    # from thop import profile
    # flops, params, ret_layer_info = profile(network.to(torch.device("cuda:4")),
    #                                         inputs=(dummy_input.to(torch.device("cuda:4")),), ret_layer_info=True)
    # print('params', params)
    # print('FLOPs:', flops)

    # params = list(net.parameters())
    # reduction1_params = params[0]
    # reduction1_bias = params[1]
    # print(reduction1_params.shape)
    # print(reduction1_bias.shape)
    # reduction2_params = params[2]
    # reduction2_bias = params[3]
    # print(reduction2_params.shape)
    # print(reduction2_bias.shape)
    # reduction3_params = params[4]
    # reduction3_bias = params[5]
    # print(reduction3_params.shape)
    # print(reduction3_bias.shape)
    # reduction4_params = params[6]
    # reduction4_bias = params[7]
    # print(reduction4_params.shape)
    # print(reduction4_bias.shape)
    # out_params = params[8]
    # out_bias = params[9]
    # print(out_params.shape)
    # print(out_bias.shape)
    #
    # aspp_out_params = params[10]
    # aspp_out_bias = params[11]
    # print(aspp_out_params.shape)
    # print(aspp_out_bias.shape)
    #
    # print()
