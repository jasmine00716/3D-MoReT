import torchvision.transforms as transforms
from data_augmentation.center_crop import CenterCrop
from data_augmentation.horizontal_flip import HorizontalFlip
import torch
from training_config.base_dsc_mrp_config import BaseDscMrpConfig
from constants import *


class _Config(BaseDscMrpConfig):
    def __init__(self, old_root_dir="",
                 new_root_dir="",
                 csv_file="./dataset/split_dsc_mrp_dataset.csv"):
        super(_Config, self).__init__(old_root_dir, new_root_dir, csv_file)

        self.log_file_dir = "/data1/sumin/compu/mrp/mrod_phasewise/log_file"
        self.model_save_dir = "/data1/sumin/compu/mrp/mrod_phasewise/pretrain"
        self.predict_save_dir = "/data1/sumin/compu/mrp/mrod_phasewise/result"
        self.evaluation_metrics_save_dir = "/data1/sumin/compu/mrp/mrod_phasewise/evaluation_metrics"
        self.summary_writer_folder_dir = "/data1/sumin/compu/mrp/mrod_phasewise/runs"

        self.model_parallel = False
        self.network_architecture = {
            "file": "network_architecture/efficientnet_backbone_drnn_sdd_loss",
            "parameters": {
                "n_channels": 40,
                "growth_rate": 12,
                "reduction": 0.5,
                "k_ordinal_class": 15,
                "dev_0": torch.device("cuda:4"),
                "dev_1": torch.device("cuda:4"),
                "dev_2": torch.device("cuda:5"),
                "dev_3": torch.device("cuda:5"),
                "model_name": "efficientnet-b0"
            }
        }
        # self.pretrain = '/data1/sumin/compu/mrp_v3/3d_unet_tgm_trm_entire_norm_input/pretrain/UnetBackboneTgmTrm_1AveragePhaseLoss_Adam_inputs_labels_labels_weight_mask_2022-08-07-15-36-29_seed42/model_24.pth'
        self.loss = {
            "loss1": {
                "file": "criteria/test_loss",
                "parameters": {
                    "loss_files": ["criteria/average_phase_loss", "criteria/ordinal_regression_loss"],
                    "device": torch.device("cuda:5")
                }
            }
        }
        self.loss_weights = [1]
        self.val_loss = {
            "loss1": {
                "file": "criteria/average_phase_loss",
                "parameters": {
                    "device": torch.device("cuda:5")
                }
            }
        }
        setattr(self, DatasetTypeString.TRAIN, {
            "dataset": {
                "file": "dsc_mrp_dataset",
                "parameters": {
                    "old_root_dir": f"{old_root_dir}",
                    "new_root_dir": f"{new_root_dir}",
                    "csv_file": f"{csv_file}",
                    "dataset_type": DatasetType.TRAIN,
                    "file_names": {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n_phasewise.npy',
                                   'ord_decreasing_labels_15_classes': 'phase_maps_or_sdd_label_medfilt_rs_15classes_n_phasewise.npy',
                                   'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase_phasewise.npy',
                                   'mask': 'mask_4d.npy'},
                    "transform": transforms.Compose([CenterCrop(224), HorizontalFlip(0.5)])
                }
            },
            "generator": {
                "batch_size": 1,
                "shuffle": True,
                "num_workers": 4
            }
        })


config = _Config().__dict__
