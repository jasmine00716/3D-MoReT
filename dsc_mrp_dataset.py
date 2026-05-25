from torch.utils import data
import csv
import os
import numpy as np
from constants import DatasetType
import torch.nn.functional as F


class DscMrpDataset(data.Dataset):
    def __init__(self, csv_file, dataset_type, file_names, transform=None):
        if not isinstance(file_names, dict):
            raise TypeError("file_names must be dict")
        self.csv_file = csv_file
        self.dataset_type = dataset_type
        self.file_names = file_names
        self.transform = transform
        self._get_file_names(self.csv_file, self.dataset_type, self.file_names)

    def _get_file_names(self, dataset_file, dataset_type, file_names):
        for file_name in file_names.keys():
            setattr(self, file_name, [])
        with open(dataset_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            for row in csv_reader:
                if dataset_type != DatasetType.ALL:
                    if int(row[1]) == dataset_type:
                        dsc_file = row[0]
                        data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                        for k, v in file_names.items():
                            getattr(self, k).append(os.path.join(data_file_dir, v))
                else:
                    dsc_file = row[0]
                    data_file_dir = dsc_file.replace("/IMG_n01.npy", "")
                    for k, v in file_names.items():
                        getattr(self, k).append(os.path.join(data_file_dir, v))

    def __len__(self):
        return len(getattr(self, list(self.file_names.keys())[0]))

    def __getitem__(self, item):
        outputs = []
        file_name = ''
        for key in self.file_names.keys():
            b = getattr(self, key)[item]
            file_name = b
            sample = np.load(b)
            outputs.append(sample)
        if self.transform:
            outputs = self.transform(outputs)
        outputs.append(file_name)
        return tuple(outputs)

    def __repr__(self):
        label_names = [file_name for file_name in self.file_names]
        return "_".join(label_names)


if __name__ == '__main__':
    a = DscMrpDataset("./dataset/split_dsc_mrp_dataset.csv",
                        DatasetType.TRAIN, {'inputs': 'IMG_n01.npy',
                                   'labels': 'phase_maps_medfilt_rs_n.npy',
                                   'labels_weight_for_each_phase': 'phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy',
                                   'mask': 'mask_4d.npy'})

    for (input, label, label_weight, mask, file) in a:
        print(input.shape)
        print(label.shape)
        print(label_weight.shape)
        print(mask.shape)
        print(file)
