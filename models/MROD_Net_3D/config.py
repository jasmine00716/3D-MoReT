import torch
from constants import *
import torchvision.transforms as transforms
from data_augmentation import CenterCrop, HorizontalFlip


network_architecture = {
    "file": "models/MROD_Net_3D/efficientnet_backbone_drnn_sdd_loss",
    "parameters": {
        'n_channels': 40,
        'growth_rate': 12,
        'reduction': 0.5,
        'k_ordinal_class': 10,
        'model_name': "efficientnet-b0",
        "dev_0": torch.device("cuda:3"),
        "dev_1": torch.device("cuda:3"),
        "dev_2": torch.device("cuda:3"),
        "dev_3": torch.device("cuda:3"),
    }
}

loss_fn = {
    "file": "criteria/average_phase_loss",
    "parameters": {
        "device": torch.device("cuda:3")
    }
}
val_loss_fn = {
    "file": "criteria/average_phase_loss",
    "parameters" : {
        "device": torch.device("cuda:3")
    }
}


optimizer_init={
    "name": "Adam",
    "parameters": {
        "init_setup": {
            "lr": 0.001,
            "betas": (0.9, 0.999,),
            "eps": 10 ** -8
        }
    }
}

learning_rate_scheduler = {
    "name": "ReduceLROnPlateau",
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
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 4
    }
}

dataset_validation = {
    "dataset": {
        "file": "dsc_mrp_dataset",
        "parameters": {
            "csv_file": "dataset/split_dsc_mrp_dataset.csv",
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
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 4
    }
}

dataset_test = {
        "dataset": {
            "file": "dsc_mrp_dataset",
            "parameters": {
                "csv_file": "dataset/split_dsc_mrp_dataset.csv",
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
            "batch_size": 1,
            "shuffle": True,
            "num_workers": 4
        }
    }