import argparse
import random
import time

import numpy as np
import tqdm
import wandb

from fvcore.nn import FlopCountAnalysis, flop_count_table

# from models.MROD_Net_3D.efficientnet_backbone_drnn_sdd_loss import EfficientnetBackboneDrnnSddLoss

def get_3d_unetpp():
    from models.unetplusplus_3d.unet_plus_plus import UnetPlusPlus
    network_architecture = {
        "parameters": {
            "input_channels": 40,
            "output_channels": 5,
            "dev_0": torch.device("cuda:6"),
            "dev_1": torch.device("cuda:6")
        }
    }
    params = network_architecture['parameters']
    network = UnetPlusPlus(**params)
    return network


def get_3d_unet():
    from models.Unet_3D.unet3d import UNet3D
    network_architecture = {
        "file": "models/Unet_3D/unet3d",
        "parameters": {
            "in_channels": 40,
            "out_channels": 5,
            "image_size": (240, 240),
            "dims": [64, 80, 96],
            "channels": [16, 24, 40, 112, 1280],
            "kernel_size": 3,
            "patch_size": (1, 2, 2),
            "num_classes": 7,
            "dev_0": torch.device("cuda:6"),
            "dev_1": torch.device("cuda:6"),
            "dev_2": torch.device("cuda:6"),
            "dev_3": torch.device("cuda:6"),
        }
    }
    params = network_architecture['parameters']
    network = UNet3D(**params)
    return network

def get_3d_mrod():
    network_architecture = {
        "parameters": {
            'n_channels': 40,
            'growth_rate': 12,
            'reduction': 0.5,
            'k_ordinal_class': 10,
            'model_name': "efficientnet-b0",
            "dev_0": torch.device("cuda:6"),
            "dev_1": torch.device("cuda:6"),
            "dev_2": torch.device("cuda:6"),
            "dev_3": torch.device("cuda:6"),
        }
    }
    params = network_architecture['parameters']
    network = EfficientnetBackboneDrnnSddLoss(**params)
    return network


def get_mvit_3d():
    network_architecture = {
        "parameters": {
            "image_size": (20, 240, 240),
            "dims": [64, 80, 96],
            "channels": [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 320, 320, 640],
            # [16, 16, 24, 24, 48, 48, 64, 64, 80, 80],  # [16, 64, 128, 256, 512],  # [16, 24, 48, 64, 80, 320]
            "kernel_size": 3,
            "patch_size": (2, 2, 2),
            "num_classes": 7,
            "expansion": 2,
            "device": torch.device("cuda:6"),
        }
    }
    params = network_architecture['parameters']
    network = Mobilevit_unet(**params)

    return network

def get_mvit_3d_same_slice():
    network_architecture = {
        "parameters": {
            "image_size": (20, 240, 240),
            "dims": [64, 80, 96],
            "channels": [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 320, 320, 640],
            # [16, 16, 24, 24, 48, 48, 64, 64, 80, 80],  # [16, 64, 128, 256, 512],  # [16, 24, 48, 64, 80, 320]
            "kernel_size": 3,
            "patch_size": (2, 2, 2),
            "num_classes": 7,
            "expansion": 2,
            "device": torch.device("cuda:6"),
        }
    }
    params = network_architecture['parameters']
    network = mobilevit_3d_same_slice.Mobilevit_unet(**params)

    return network


def check_manual_seed(seed):
    """
    If manual seed is not specified, choose a random one and notify it to the user
    """
    seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print('Using manual seed: {seed}'.format(seed=seed))
    return

    return shape_augs, None


if __name__ == '__main__':
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_size', type=int, default=224)
    args = parser.parse_args()

    check_manual_seed(5)
    nr_procs_valid = 8
    batch_size = 1
    iters = 20
    warmup_iters = 10
    flag = 0

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    print('Count of using GPUs:', torch.cuda.device_count())
    print('Current cuda device:', torch.cuda.current_device())

    # model = get_3d_mrod()  # 162.254M
    # model = get_mvit_3d()  # 28.2885M
    # model = get_mvit_3d_same_slice()  #
    # model = get_3d_unet()
    model = get_3d_unetpp()
    model = model.to(device)
    dummy_input = torch.randn([1, 40, 20, args.input_size, args.input_size]).to(device)

    with torch.no_grad():
        with torch.autograd.profiler.profile(use_cuda=True) as prof:
            output = model(dummy_input)
        print(prof)

    overall_params = sum([p.numel() for p in model.parameters()])
    print("{:<20} = {}".format("Overall parameters", overall_params))
    flop_analyzer = FlopCountAnalysis(model, dummy_input)
    print(flop_count_table(flop_analyzer).split("\n")[2])

    for iter in tqdm.tqdm(range(iters)):
        x = dummy_input
        model.eval()
        if iter == warmup_iters:
            torch.cuda.cudart().cudaProfilerStart()
        if iter < warmup_iters:
            if iter % 1000 == 0:
                print("wait", iter)
            with torch.no_grad():
                x = model(x)
        else:
            if flag == 0:
                start = time.time()
                flag = 1
            if iter % 1000 == 0:
                print("run", iter)
            torch.cuda.nvtx.range_push(f"inference")
            with torch.no_grad():
                x = model(x)
            # with torch.no_grad():
            #     for name, child in model.named_children():
            #         torch.cuda.nvtx.range_push(name)
            #         if len(list(child.named_children())) == 0:
            #             x = child(x)
            #         else:
            #             for inner_name, inner_child in child.named_children():
            #                 if isinstance(inner_child, torch.nn.ModuleList):
            #                     for inner_name_2, inner_child_2 in inner_child.named_children():
            #                         torch.cuda.nvtx.range_push(name + '_' + inner_name + '_' + inner_name_2)
            #                         try:
            #                             x = inner_child_2(x)
            #                         except:
            #                             print(name + '_' + inner_name + '_' + inner_name_2)
            #                             raise AssertionError
            #                         torch.cuda.nvtx.range_pop()  # inner_block_name pop
            #                 else:
            #                     torch.cuda.nvtx.range_push(name + '_' + inner_name)
            #                     try:
            #                         x = inner_child(x)
            #                     except:
            #                         print(name + '_' + inner_name)
            #                         raise AssertionError
            #                     torch.cuda.nvtx.range_pop()  # inner_block_name pop
            #         torch.cuda.nvtx.range_pop()  # block_name pop
            torch.cuda.nvtx.range_pop()  # inference_i pop

    end = time.time()
    print(f"{end - start:.5f} sec")
    torch.cuda.cudart().cudaProfilerStop()
    wandb.finish()

# mvit
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas,vulkan  -s cpu  --cuda-memory-usage true --capture-range cudaProfilerApi --cudabacktrace true -x true -o MViT_3d python nsight_profile.py
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas,vulkan  -s cpu  --cuda-memory-usage true --capture-range cudaProfilerApi --cudabacktrace true -x true -o MViT_3d_Same_Slice python nsight_profile.py

# 3dmrod
# nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas,vulkan  -s cpu  --cuda-memory-usage true --capture-range cudaProfilerApi --cudabacktrace true -x true -o MROD_3d python nsight_profile.py
