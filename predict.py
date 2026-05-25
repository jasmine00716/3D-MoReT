import argparse
import os
import torch
from utils_ import import_file, convert_str_from_underscore_to_camel
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from evaluation_metrics import Metrics_Evaluation
# import will_be_deleted.config_mobilevit as config
# import config
# import config_mrod
import config_mra as config

def load_nest_config(**kwargs):
    file_name = kwargs.get('file')
    parameters = kwargs.get('parameters')
    if "/" in file_name:
        cls_name = convert_str_from_underscore_to_camel(file_name.split("/")[-1])
    else:
        cls_name = convert_str_from_underscore_to_camel(file_name)
    if parameters is None:
        return getattr(import_file(file_name), cls_name)()
    return getattr(import_file(file_name), cls_name)(**parameters)


def predict(best_model_dir):
    print("\npredict...")
    # params['channels'] = [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 320, 320, 640]
    # params['channels'] = [52, 52, 52, 64, 64, 64, 128, 128, 256, 256, 512, 512, 1024]

    # network = load_nest_config(**config.network_architecture).to(torch.device("cuda:4"))

    # from models.MobileViT import mobilevit
    # network = mobilevit.Mobilevit(**params)
    # from models.MobileViT_v3_3D import mobilevit_3d_previous
    # network = mobilevit_3d_previous.Mobilevit(**params)
    # from models.MobileViT_v3_3D import mobilevit_3d
    # network = mobilevit_3d.Mobilevit_nearest_neighbor_upsampling(**params)

    # from models.MobileViT_v3_3D import mobilevit_v3_3d_same_slice
    # network = mobilevit_v3_3d_same_slice.Mobilevit(**params)

    """vit 1x6x6 patch"""
    from models.MobileViT_v3_3D.mobilevit_v3_trs_3d_1x6x6_same_slice import Mobilevit
    network = Mobilevit(**params)

    """vit 2x2x2 patch"""
    # from models.MobileViT_v3_3D.mobilevit_v3_trs_3d_2x2x2_same_slice import Mobilevit
    # network = Mobilevit(**params)

    # from models.MobileViT_v3_3D import mobilevit_v3_3d_same_slice
    # network = mobilevit_v3_3d_same_slice.Mobilevit(**params)

    # from models.MobileViT_v3_3D import mobilevit_3d
    # network = mobilevit_3d.Mobilevit(**params)

    # from models.unet_3d.u_net import UNet
    # network = UNet(40, 5, device)
    #
    # from models.unetplusplus_3d.unet_plus_plus import UnetPlusPlus
    # network = UnetPlusPlus(40, 5, device, device)

    """3d mrod net"""
    # from models.MROD_Net_3D.efficientnet_backbone_drnn_sdd_loss import EfficientnetBackboneDrnnSddLoss
    # params = config_mrod.network_architecture['parameters']
    # network = EfficientnetBackboneDrnnSddLoss(**params)

    """mvit 2x2x2patch/3TypesOfPatches, vit 1x6x6 patch"""
    # from models.MobileViT_v3_3D.mobilevit_v3_vit_3d_advanced_ver1 import Mobilevit
    # from models.MobileViT_v3_3D.mobilevit_v3_vit_3d_advanced_ver2 import Mobilevit
    # network = Mobilevit(**params)

    """unext"""
    # import config_unext
    # params = config_unext.network_architecture['parameters']
    # device = params['device']
    # from models.UNeXt.unext import UNext
    # network = UNext(**params)

    '''swin unetr'''
    # import os
    # import shutil
    # import tempfile
    # import matplotlib.pyplot as plt
    # from tqdm import tqdm
    # from monai.losses import DiceCELoss
    # from monai.inferers import sliding_window_inference
    # from monai.transforms import (
    #     AsDiscrete,
    #     Compose,
    #     CropForegroundd,
    #     LoadImaged,
    #     Orientationd,
    #     RandFlipd,
    #     RandCropByPosNegLabeld,
    #     RandShiftIntensityd,
    #     ScaleIntensityRanged,
    #     Spacingd,
    #     RandRotate90d,
    #     EnsureTyped,
    # )
    # from monai.data import (
    #     ThreadDataLoader,
    #     CacheDataset,
    #     load_decathlon_datalist,
    #     decollate_batch,
    #     set_track_meta,
    # )
    #
    # from monai.config import print_config
    # from monai.metrics import DiceMetric
    # from monai.networks.nets import SwinUNETR
    #
    # import torch
    # import torch.nn.functional as F
    # from thop import profile
    # import scipy.ndimage
    #
    # import argparse
    # from datetime import datetime
    # # import torchvision.models as models, ResNet50_Weights
    # # from models.Unet_3D.unet3d import UNet3D
    # import config
    # import torch.optim as optim
    # import torch.optim.lr_scheduler as lr_scheduler
    # import torch
    # from utils_ import load_nest_config
    # from torch.autograd import Variable
    # from torch.utils.tensorboard import SummaryWriter
    # from torch.utils import data
    # from pytz import timezone
    # import wandb
    # import os
    #
    # import os
    # import shutil
    # import tempfile
    # import matplotlib.pyplot as plt
    # from tqdm import tqdm
    # from monai.losses import DiceCELoss
    # from monai.inferers import sliding_window_inference
    # from monai.transforms import (
    #     AsDiscrete,
    #     Compose,
    #     CropForegroundd,
    #     LoadImaged,
    #     Orientationd,
    #     RandFlipd,
    #     RandCropByPosNegLabeld,
    #     RandShiftIntensityd,
    #     ScaleIntensityRanged,
    #     Spacingd,
    #     RandRotate90d,
    #     EnsureTyped,
    # )
    # from monai.data import (
    #     ThreadDataLoader,
    #     CacheDataset,
    #     load_decathlon_datalist,
    #     decollate_batch,
    #     set_track_meta,
    # )
    #
    # from monai.config import print_config
    # from monai.metrics import DiceMetric
    # from monai.networks.nets import SwinUNETR
    #
    # import torch
    # import torch.nn.functional as F
    # ''''''
    # device = params['device']
    # network = SwinUNETR(
    #     img_size=(32, 224, 224),
    #     in_channels=40,
    #     out_channels=5,
    #     feature_size=48,
    #     use_checkpoint=True,
    # ).to(device)
    ''''''
    device = params['device']
    checkpoint = torch.load(best_model_dir, map_location=device)
    network.load_state_dict(checkpoint['model_state_dict'])

    for index, (inputs, labels, mask, filename) in enumerate(test_loader):
        folder = '/'.join(filename[0].split('/')[-7:-3])
        print(f"{index+1}: {folder}")
        inputs, labels, mask = inputs.to(device), labels.to(device), mask.to(device)

        inputs = F.interpolate(inputs, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
        mask = F.interpolate(mask, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size
        labels = F.interpolate(labels, size=(28, 224, 224), mode='nearest')  # resize to SwinUNetR input size

        network.eval()
        predicts = network(inputs)
        del inputs

        directory = f'{predict_save_dir}/{folder}'
        if not os.path.isdir(directory):
            os.makedirs(directory)

        np.save(f'{predict_save_dir}/{folder}/gt.npy',torch.mul(labels, mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
        del labels
        np.save(f'{predict_save_dir}/{folder}/predict.npy', torch.mul(predicts[0], mask.repeat(1, 5, 1, 1, 1))[0].cpu().detach().numpy())
        del predicts
        np.save(f'{predict_save_dir}/{folder}/mask.npy', mask[0].cpu().detach().numpy())
        del mask

def evaluate(log_file_dir, predict_save_dir, evaluation_metrics_save_dir):
    print("\nevaluate...")
    evaluate_metrics = Metrics_Evaluation(log_file_dir, predict_save_dir, evaluation_metrics_save_dir, save_fig=True)
    evaluate_metrics.evaluate()


if __name__ == '__main__':

    cuda_visible_devices = [4, 5, 6, 7]

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', type=int, default=4)
    args = vars(parser.parse_args())

    params = config.network_architecture['parameters']
    # params = config_mrod.network_architecture['parameters']
    params['device'] = torch.device(f"cuda:{args['device']}")

    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevit3d_deconv_20231116_22h32m"
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevit3d_deconv_20231117_19h24m"

    # mobilevit v3 + vit
    # dir = "/data1/sumin/compu/mra_lightweight/mobilevit_mra_20240122_16h17m"  # mra

    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_deconv_20231126_12h41m"  # 2x2x2
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_deconv_20231126_17h08m"  # 20x6x6
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_2x2x2_deconv_20231128_00h32m"  # 2x2x2, ch: 52~512, batch 4
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_20x6x6_deconv_20231128_00h35m"  # 1x6x6, ch: 52~256, batch 4
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_1x6x6_deconv_20231129_01h06m"  # 1x6x6, ch: 52~512, batch 4
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevit_deconv_20231228_21h02m"  # unext
    # dir = "/data1/sumin/compu/mrp_lightweight/swinunetr_deconv_20231229_21h57m"  # swinunetr
    dir = "/data1/sumin/compu/mrp_lightweight/3dmrod_deconv_20231129_00h58m"
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevit_deconv_20231221_13h55m"
    # dir = "/data1/sumin/compu/mrp_lightweight/mobilevit_deconv_20231221_01h37m"

    # # unet best model
    # dir = "/data1/sumin/compu/mrp_lightweight/unet_deconv_20231119_16h43m"

    # # unet++ best model
    # dir = "/data1/sumin/compu/mrp_lightweight/unetplusplus_deconv_20231119_17h55m"

    log_file_dir = os.path.abspath(dir + "/log_file")
    model_save_dir = os.path.abspath(dir + "/pretrain")
    predict_save_dir = os.path.abspath(dir + "/result")
    evaluation_metrics_save_dir = os.path.abspath(dir + "/evaluation_metrics")
    summary_writer_folder_dir = os.path.abspath(dir + "/runs")

    board_writer = SummaryWriter(summary_writer_folder_dir)

    validation_leap = 4

    # test dataset
    dataset_config = config.dataset_test.get('dataset')
    loader_config = config.dataset_test.get('dataloader')
    # dataset_config = config_mrod.dataset_test.get('dataset')
    # loader_config = config_mrod.dataset_test.get('dataloader')

    test_dataset = load_nest_config(**dataset_config)
    test_loader = data.DataLoader(test_dataset, **loader_config)

    # best_model_dir = os.path.abspath(model_save_dir + "/model_208.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_160.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_220.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_176.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_236.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_168.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_244.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_232.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_164.pth")
    # best_model_dir = os.path.abspath(model_save_dir + "/model_264.pth")
    best_model_dir = os.path.abspath(model_save_dir + "/model_156.pth")
    predict(best_model_dir)
    evaluate(log_file_dir, predict_save_dir, evaluation_metrics_save_dir)