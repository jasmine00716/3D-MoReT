import os
import argparse
import torch
from constants import *
import torchvision.transforms as transforms
from data_augmentation import CenterCrop, HorizontalFlip
from datetime import datetime
from utils_ import import_file, convert_str_from_underscore_to_camel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils import data
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np

network_architecture = {
    "parameters": {
        "image_size": (20, 224, 224),
        "num_classes": 5,
        "device": torch.device("cuda:4"),
    }
}
loss_fn = {
    # "file": "criteria/average_phase_loss",
    "file": "criteria/berhu_loss",
    "parameters": {
        "device": torch.device("cuda:4")
    }
}
val_loss_fn = {
    # "file": "criteria/average_phase_loss",
    "file": "criteria/berhu_loss",
    "parameters": {
        "device": torch.device("cuda:4")
    }
}

optimizer_init={
    "name": "Adam",
    "parameters": {
        "init_setup": {
            "lr": 0.0001,
            "betas": (0.9, 0.999,),
            "eps": 10 ** -8,
            # "weight_decay": 0.01  # 0.1, 0.01
        }
    }
}

learning_rate_scheduler = {
    "name": "CosineAnnealingLR",  # ExponentialLR, CosineAnnealingLR
    "parameters": {
        "factor": 0.75,
        "patience": 3
    }
}

dataset_train = {
    "dataset": {
        "file": "dsc_mrp_dataset",
        "parameters": {
            "csv_file": "dataset/split_dsc_mrp_dataset.csv",
            # "csv_file": "dataset/split_dsc_mrp_dataset(sumin).csv",
            "dataset_type": DatasetType.TRAIN,
            "file_names": {
                'inputs': 'IMG_n01.npy',
                'labels': 'phase_maps_medfilt_rs_n.npy',
                'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                'mask': 'mask_4d.npy'
            },
            "transform": transforms.Compose([CenterCrop(224), HorizontalFlip(0.5)])
        }
    },
    "dataloader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4
    }
}

dataset_validation = {
    "dataset": {
        "file": "dsc_mrp_dataset",
        "parameters": {
            "csv_file": "dataset/split_dsc_mrp_dataset.csv",
            # "csv_file": "dataset/split_dsc_mrp_dataset(sumin).csv",
            "dataset_type": DatasetType.VAL,
            "file_names": {
                'inputs': 'IMG_n01.npy',
                'labels': 'phase_maps_medfilt_rs_n.npy',
                'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                'mask': 'mask_4d.npy'
            },
            "transform": transforms.Compose([CenterCrop(224)])
        }
    },
    "dataloader": {
        "batch_size": 8,
        "shuffle": True,
        "num_workers": 4
    }
}

dataset_test = {
        "dataset": {
            "file": "dsc_mrp_dataset",
            "parameters": {
                "csv_file": "dataset/split_dsc_mrp_dataset.csv",
                # "csv_file": "dataset/split_dsc_mrp_dataset(sumin).csv",
                "dataset_type": DatasetType.TEST,
                "file_names": {
                    'inputs': 'IMG_n01.npy',
                    'labels': 'phase_maps_medfilt_rs_n.npy',
                    'mask': 'mask_4d.npy'
                },
                "transform": transforms.Compose([CenterCrop(224)])
            }
        },
        "dataloader": {
            "batch_size": 8,
            "shuffle": False,
            "num_workers": 4
        }
    }
