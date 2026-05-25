# contrastive learning

import torch
from constants import *
import torchvision.transforms as transforms
from utils_ import load_nest_config
from data_augmentation import HorizontalFlip, GaussianBlur
from torch.utils import data

dataset_all = {
    "dataset": {
        "file": "dsc_mrp_dataset",
        "parameters": {
            "csv_file": "dataset/split_dsc_mrp_dataset.csv",
            "dataset_type": DatasetType.ALL,
            "file_names": {
                'inputs': 'IMG_n01.npy',
                'labels': 'phase_maps_medfilt_rs_n.npy',
                'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                'mask': 'mask_4d.npy'
            },
            "transform": transforms.Compose([HorizontalFlip(0.5), GaussianBlur(3, 1)]),
            "n_views": 2
        }
    },
    "dataloader": {
        "batch_size": 1,
        "shuffle": True,
        "num_workers": 4,
    }
}

# dataset
dataset_config = dataset_all.get('dataset')
loader_config = dataset_all.get('dataloader')

dataset = load_nest_config(**dataset_config)
dataloader = data.DataLoader(dataset, **loader_config)

