from dataclasses import dataclass
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass(init=True)
class TargetSize:
    height: int
    width: int


class CenterCrop(object):
    def __init__(self, target_size):
        assert isinstance(target_size, (int, TargetSize))
        self.target_size = target_size

    def __call__(self, items):
        outputs = []
        for item in items:
            item = item.copy()
            channels, depth, height, width = item.shape
            if isinstance(self.target_size, int):
                start_x = width // 2 - (self.target_size // 2)
                start_y = height // 2 - (self.target_size // 2)
                outputs.append(
                    item[:, :, start_y:start_y + self.target_size, start_x:start_x + self.target_size])
            elif isinstance(self.target_size, TargetSize):
                start_x = width // 2 - (self.target_size.width // 2)
                start_y = height // 2 - (self.target_size.height // 2)
                outputs.append(item[:, :, start_y:start_y + self.target_size.height, start_x:start_x + self.target_size.width])
        return outputs


class HorizontalFlip(object):

    def __init__(self, prob=0.5):
        assert isinstance(prob, (float, int))
        self.prob = prob

    def __call__(self, items):
        if random.random() < self.prob:
            outputs = []
            for item in items:
                item = item[:, :, :, ::-1]
                item = item.copy()
                outputs.append(item)
            return outputs
        return items


class GaussianBlur(object):
    def __init__(self, k, r):
        self.k = k
        self.r = r
        self.blur_h = torch.nn.Conv3d(40, 40, kernel_size=(1, 1, k), bias=False)
        self.blur_v = torch.nn.Conv3d(40, 40, kernel_size=(1, k, 1), bias=False)

    def __call__(self, items):
        outputs = []
        for item in items:
            item = torch.from_numpy(item)
            sigma = np.random.uniform(0.1, 2.0)
            x = np.arange(-self.r, self.r + 1)
            x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
            x = x / x.sum()
            x = torch.from_numpy(x).view(1, 1, 1, -1).repeat(1, 3, 1, 1)  # Changed here

            with torch.no_grad():
                item = self.blur_h(item)
                item = self.blur_v(item)
                item = item.squeeze()
            outputs.append(item.numpy())
        return outputs


# class GaussianBlur(object):
#     def __init__(self, kernel_size):
#         radias = kernel_size // 2
#         kernel_size = radias * 2 + 1
#         self.blur_s = nn.Conv3d(40, 40, kernel_size=(kernel_size, 1, 1),
#                                 stride=1, padding=0, bias=False, groups=4)
#         self.blur_h = nn.Conv3d(40, 40, kernel_size=(1, kernel_size, 1),
#                                 stride=1, padding=0, bias=False, groups=4)
#         self.blur_w = nn.Conv3d(40, 40, kernel_size=(1, 1, kernel_size),
#                                 stride=1, padding=0, bias=False, groups=4)
#
#         self.k = kernel_size
#         self.r = radias
#
#         self.blur = nn.Sequential(
#             nn.ReflectionPad2d(radias),
#             self.blur_s,
#             self.blur_h,
#             self.blur_w,
#         )
#
#     def __call__(self, items):
#         outputs = []
#         for item in items:
#             torch.from_numpy(item)
#             sigma = np.random.uniform(0.1, 2.0)
#             x = np.arange(-self.r, self.r+1)
#             x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
#             x = x / x.sum()
#             x = torch.from_numpy(x).view(1, 1, -1).repeat(1, 3, 1)
#             self.blur_s.weight.data.copy_(x.view(40, 3, 1, self.k, 1, 1))
#             self.blur_h.weight.data.copy_(x.view(40, 1, self.k, 1))
#             self.blur_w.weight.data.copy_(x.view(40, 1, 1, self.k))
#
#             with torch.no_grad():
#                 item = self.blur(item)
#                 item = item.squeeze()
#
#             outputs.append(item.numpy())
#         return outputs

# unused
class Padding(object):
    def __init__(self, target_size):
        assert isinstance(target_size, (int, TargetSize))
        self.target_size = target_size

    def __call__(self, items):
        outputs = []
        for item in items:
            item = torch.from_numpy(item)
            _, _, h, w = item.shape
            if isinstance(self.target_size, int):
                w_diff = (self.target_size - w) // 2
                h_diff = (self.target_size - h) // 2
                if self.target_size >= w:
                    w_diff_plus = w_diff + 1 if (self.target_size - w) % 2 != 0 else w_diff
                else:
                    w_diff_plus = w_diff - 1 if (self.target_size - w) % 2 != 0 else w_diff
                if self.target_size >= h:
                    h_diff_plus = h_diff + 1 if (self.target_size - h) % 2 != 0 else h_diff
                else:
                    h_diff_plus = h_diff - 1 if (self.target_size - h) % 2 != 0 else h_diff
                output_item = F.pad(item, (w_diff, w_diff_plus, h_diff, h_diff_plus))
                outputs.append(output_item.numpy())
            elif isinstance(self.target_size, TargetSize):
                if self.target_size.width >= w:
                    w_diff = (self.target_size.width - w) // 2
                    w_diff_plus = w_diff + 1 if (self.target_size.width - w) % 2 != 0 else w_diff
                else:
                    w_diff = -((w - self.target_size.width) // 2)
                    w_diff_plus = w_diff - 1 if (self.target_size.width - w) % 2 != 0 else w_diff
                if self.target_size.height >= h:
                    h_diff = (self.target_size.height - h) // 2
                    h_diff_plus = h_diff + 1 if (self.target_size.height - h) % 2 != 0 else h_diff
                else:
                    h_diff = -((h - self.target_size.height) // 2)
                    h_diff_plus = h_diff - 1 if (self.target_size.height - h) % 2 != 0 else h_diff
                output_item = F.pad(item, (w_diff, w_diff_plus, h_diff, h_diff_plus))
                outputs.append(output_item.numpy())
        return outputs


# unused
class RandomDropImgs(object):

    def __init__(self, target_size=19, dim=1):
        assert isinstance(target_size, (float, int))
        assert isinstance(dim, int)
        self.target_size = target_size
        self.dim = dim

    def __call__(self, items):
        if items[0].shape[self.dim] != self.target_size:
            rm_idx = random.randint(0, self.target_size)
            outputs = []
            for item in items:
                item = np.delete(item, rm_idx, axis=self.dim)
                item = item.copy()
                outputs.append(item)
            return outputs
        return items


if __name__ == "__main__":
    from torchvision.transforms import transforms
    transforms = transforms.Compose([CenterCrop(224), GaussianBlur(3, 1)])

    npy_path_1 = "/path/to/mrp_npy/NpyFiles/IMG_n01.npy"
    img = np.load(npy_path_1)
    print(img.mean())
    print(img.shape)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2], img.shape[3]))
    print(img.shape)

    img = transforms(img)
    print(img[0].mean())