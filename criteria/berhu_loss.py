import torch
from criteria.base_loss import BaseLoss
import torch.nn.functional as F

class BerhuLoss(BaseLoss):
    def __init__(self, device=torch.device("cuda:4")):
        super(BerhuLoss, self).__init__()
        self.set_device(device)

    def forward(self, predict, target, weight_mask, mask, use_weight_mask=True):
        loss = 0
        predict = predict.to(self.device)
        target = target.to(self.device)
        weight_mask = weight_mask.to(self.device)
        mask = mask.to(self.device)

        # predict = F.interpolate(predict, size=(28, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        # target = F.interpolate(target, size=(28, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        # weight_mask = F.interpolate(weight_mask, size=(28, 224, 224), mode='nearest')  # swinunetr, 추가된 코드
        # mask = F.interpolate(mask, size=(28, 224, 224), mode='nearest')  # swinunetr, 추가된 코드

        predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
        target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))

        absolute_err = torch.abs(target_with_mask - predict_with_mask)
        c = 0.2 * torch.max(absolute_err).detach()
        bhloss = torch.mean(torch.where(absolute_err <= c, absolute_err, (absolute_err ** 2 + c ** 2) / (2 * c)))
        if use_weight_mask:
            matrix_loss = torch.mul(bhloss, weight_mask).sum(dim=(2, 3, 4))
        else:
            matrix_loss = bhloss.sum(dim=(2, 3, 4))
        elements_counts = mask.sum(dim=(2, 3, 4))
        loss_of_all_phase = matrix_loss / elements_counts
        loss += loss_of_all_phase.mean()

        return loss
