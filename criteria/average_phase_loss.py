import torch
from criteria.base_loss import BaseLoss

'''
loss function for regression task (not ordinal regression)
'''


class AveragePhaseLoss(BaseLoss):

    def __init__(self, device=torch.device("cuda:4")):
        super(AveragePhaseLoss, self).__init__()
        self.set_device(device)

    # input size: N, C, D, W, H
    def forward(self, predict, target, weight_mask, mask, use_weight_mask=True):
        loss = 0
        predict = predict.to(self.device)
        target = target.to(self.device)
        weight_mask = weight_mask.to(self.device)
        mask = mask.to(self.device)
        if len(predict.shape) == 5:
            predict_with_mask = torch.mul(predict, mask.repeat(1, 5, 1, 1, 1))
            target_with_mask = torch.mul(target, mask.repeat(1, 5, 1, 1, 1))
            # absolute_err = torch.abs(target_with_mask - predict_with_mask)
            sqrd_error = (target_with_mask - predict_with_mask) ** 2
            if use_weight_mask:
                matrix_loss = torch.mul(sqrd_error, weight_mask).sum(dim=(2, 3, 4))
            else:
                matrix_loss = sqrd_error.sum(dim=(2, 3, 4))
            elements_counts = mask.sum(dim=(2, 3, 4))
            loss_of_all_phase = matrix_loss / elements_counts
            loss += loss_of_all_phase.mean()
        elif len(predict.shape) == 4:
            predicts_with_mask = torch.mul(predict, mask)
            targets_with_mask = torch.mul(target, mask)
            b, n, _, h, w = mask.shape
            for predict, target, m, wm in zip(predicts_with_mask, targets_with_mask, mask, weight_mask):
                temp_loss = torch.mul(torch.abs(target - predict), wm).sum()
                element_count = m.sum()
                if element_count == 0:
                    element_count = w * h
                loss += temp_loss / element_count
            loss /= n
        return loss
