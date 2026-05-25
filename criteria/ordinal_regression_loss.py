import torch
from criteria.base_loss import BaseLoss


'''
loss function for 'ordinal' regression
'''

class OrdinalRegressionLoss(BaseLoss):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self, device=torch.device("cuda:5")):
        super(OrdinalRegressionLoss, self).__init__()
        self.set_device(device)
        self.loss = 0.0

    def forward(self, ord_labels, target, weight_mask ,mask):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """
        # assert pred.dim() == target.dim()
        # invalid_mask = target < 0
        # target[invalid_mask] = 0
        ord_labels = ord_labels.to(self.device)
        target = target.to(self.device)
        mask = mask.to(self.device)
        if len(ord_labels.shape) == 6:
            N, C, P, D, H, W = ord_labels.size()
            # print(N, C, P, D, H, W)
            # print(mask.shape)
            ord_num = C

            loss = 0.0

            # faster version
            if torch.cuda.is_available():
                K = torch.zeros((N, C, P, D, H, W), dtype=torch.int).to(self.device)
                #TODO: test whether remove ord_labels[:, i, :, :, :, :] = torch.mul(ord_labels[:, i, :, :, :, :], mask.repeat(1, 5, 1, 1, 1).to(self.device))
                for i in range(ord_num):
                    ord_labels[:, i, :, :, :, :] = torch.mul(ord_labels[:, i, :, :, :, :], mask.repeat(1, 5, 1, 1, 1).to(self.device))
                    K[:, i, :, :, :, :] = K[:, i, :, :, :, :] + i * torch.ones((N, P, D, H, W), dtype=torch.int).to(self.device)
            else:
                K = torch.zeros((N, C, P, D, H, W), dtype=torch.int)
                for i in range(ord_num):
                    ord_labels[:, i, :, :, :, :] = torch.mul(ord_labels[:, i, :, :, :, :], mask.repeat(1, 5, 1, 1, 1).to(self.device))
                    K[:, i, :, :, :, :] = K[:, i, :, :, :, :] + i * torch.ones((N, P, D, H, W), dtype=torch.int)

            mask_0 = (K <= target).detach()
            mask_1 = (K > target).detach()

            one = torch.ones(ord_labels[mask_1].size())
            if torch.cuda.is_available():
                one = one.to(self.device)
            loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                         + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

            N = N * P * int(mask.sum(dim=(2, 3, 4)))
            loss /= (-N)  # negative
            return loss

        elif len(ord_labels.shape == 4):
            N, C, H, W = ord_labels.size()
            ord_num = C

            self.loss = 0.0

            # faster version
            if torch.cuda.is_available():
                K = torch.zeros((N, C, H, W), dtype=torch.int).cuda()
                for i in range(ord_num):
                    K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).cuda()
            else:
                K = torch.zeros((N, C, H, W), dtype=torch.int)
                for i in range(ord_num):
                    K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int)

            mask_0 = (K <= target).detach()
            mask_1 = (K > target).detach()

            one = torch.ones(ord_labels[mask_1].size())
            if torch.cuda.is_available():
                one = one.cuda()

            self.loss += torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8))) \
                         + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8)))

            N = N * H * W
            self.loss /= (-N)  # negative
        return self.loss