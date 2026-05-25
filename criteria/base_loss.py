import torch.nn as nn
import torch

class BaseLoss(nn.Module):

    def __init__(self, device=torch.device("cuda:4")):
        super(BaseLoss, self).__init__()
        self.loss = 0.0
        self.set_device(device)

    def set_device(self, device):
        self.device = device

    def __repr__(self):
        return self.__class__.__name__