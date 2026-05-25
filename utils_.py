import importlib.util as im_util
import re
import os
import torch
import numpy as np
import random

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


def import_file(file_name):
    spec = im_util.spec_from_file_location(".", f"{file_name}.py")
    file = im_util.module_from_spec(spec)
    spec.loader.exec_module(file)
    return file

def convert_str_from_camel_to_underscore(input):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', input).lower()


def convert_str_from_underscore_to_camel(input):
    input_l = [*map(lambda x: x.capitalize(),input.split("_"))]
    return "".join(input_l)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# for normalization
def est_outlier_thr(x, thr):
    """"""
    x_s = np.sort(x.flat)
    lthr = x_s[int(np.floor(len(x_s) * thr) - 1)]  # lower threshold
    uthr = x_s[int(np.floor(len(x_s) * (1 - thr) - 1))]  # upper threshold
    return lthr, uthr


def norm_ab(x, a, b, predefined_minmax=None, exclude_outliers=False, mask=None, to_mask=False):
    """Normalize input to a-b range"""
    mask = np.ones_like(x) if mask is None else mask
    x_norm = x[:]

    if predefined_minmax is not None:
        xmin = predefined_minmax[0]
        xmax = predefined_minmax[1]
    else:
        if exclude_outliers:
            xmin, xmax = est_outlier_thr(x[mask > 0], 0.01)
            x[x > xmax] = xmax
            x[x < xmin] = xmin
        else:
            xmax = max(x[mask == 1])
            xmin = min(x[mask == 1])

    if to_mask:
        x_norm[mask > 0] = (b - a) * ((x[mask > 0] - xmin) / (xmax - xmin)) + a
        x_norm = x_norm * mask
    else:
        x_norm = (b - a) * ((x - xmin) / (xmax - xmin)) + a
        x_norm[x_norm < 0] = 0
    return x_norm

class EarlyStopping:
    # https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss