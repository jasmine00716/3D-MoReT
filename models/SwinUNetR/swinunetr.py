import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

import torch
import torch.nn.functional as F
from thop import profile
import scipy.ndimage

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.data import (
    ThreadDataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    set_track_meta,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR

import torch
import torch.nn.functional as F
from thop import profile
import scipy.ndimage

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

img = torch.rand(1, 40, 20, 224, 224).to(device)
img = F.interpolate(img, size=(32, 224, 224), mode='nearest')

network = SwinUNETR(
        img_size=(32, 224, 224),
        in_channels=40,
        out_channels=5,
        feature_size=48,
        use_checkpoint=True,
    ).to(device)


print(">> swinunetr")
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


# totalparams = 0
# for name, param in network.named_parameters():
#     if param.requires_grad:
#         print(f"Parameter name: {name}, {param.numel()}")
        # totalparams += param.numel()
# print(f">> the total parameters: {totalparams}")

# buffer_size = 0
# for buffer in network.buffers():
#     buffer_size += buffer.nelement() * buffer.element_size()

# size_all_mb = (totalparams + buffer_size) / 1024 ** 2
# print('>> model size: {:.3f}MB\n'.format(size_all_mb))
#
# from thop import profile
#
# # output = network(img)
# flops, params, ret_layer_info = profile(network.to(torch.device("cuda:6")), inputs=(img.to(torch.device("cuda:6")),),
#                                         ret_layer_info=True)
# # print('params', params)
# print('FLOPs:', flops)