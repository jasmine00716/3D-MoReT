import numpy as np
import torch
import os
import glob
from sklearn.metrics import mean_absolute_error as MAE
import pathlib
from writer import *
import matplotlib

matplotlib.use('Agg')
import pylab as plt
from skimage.util import montage
import wandb
from gen_dicoms_noHeader import mr_collateral_dsc_gen_dicoms

def Dice(pred, gt):
    smooth = 1.0
    assert pred.size() == gt.size()
    pred = pred[:, 0].contiguous().view(-1)
    gt = gt[:, 0].contiguous().view(-1)
    intersection = (pred * gt).sum()
    dsc = (2. * intersection + smooth) / (pred.sum() + gt.sum() + smooth)
    return 1. - dsc


def R_Squared(pred, gt):
    # sum of squared residuals
    SS_res = torch.sum((gt - pred)**2)

    # total sum of squares
    gt_mean = torch.mean(gt)
    SS_tot = torch.sum((gt - gt_mean)**2)

    r_squared = 1 - (SS_res / SS_tot)

    return r_squared


def TM(pred, gt):
    """input : vector"""
    euclidian_dist = torch.dist(pred, gt)
    pred = pred.view(1, -1)
    gt = gt.view(-1, 1)
    product = torch.matmul(pred, gt)
    tm = product / (product + euclidian_dist ** 2)
    return tm.item()


def SSIM(pred, gt):
    """input : vector"""
    dr = 1.8
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    cov = torch.sum((pred - pred.mean()) * (gt - gt.mean())) / pred.view(-1, 1).shape[0]

    ssim = ((2 * pred.mean() * gt.mean() + c1) * (2 * cov + c2)) / \
           ((pred.mean().pow(2) + gt.mean().pow(2) + c1) * (pred.var() + gt.var() + c2))
    return ssim.item()


def save_color_fig(pred, gt, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for ind, phase in enumerate(['Art', 'Cap', 'EVen', 'LVen', 'Del']):
        fig, ax = plt.subplots(1, 2, num='result', figsize=(20, 10))
        fig.suptitle(phase)
        ax[0].imshow(montage(pred[ind][:-2, :, :], grid_shape=(5, 5), multichannel=True))
        ax[1].imshow(montage(gt[ind][:-2, :, :], grid_shape=(5, 5), multichannel=True))
        plt.figure('result')
        plt.savefig(f'{save_path}/result_{phase}.png', dpi=150)
        plt.close('result')


class Metrics_Evaluation:
    def __init__(self, log_file_dir, predict_save_dir, evaluation_metrics_save_dir, save_fig=False):
        self.predict_save_dir = predict_save_dir
        self.result_list = glob.glob(f"{self.predict_save_dir}/*/*/*Y/*")
        self.evaluation_metrics_save_dir = evaluation_metrics_save_dir
        pathlib.Path(evaluation_metrics_save_dir).mkdir(parents=True, exist_ok=True)
        self.writers = [DscMrpCsvWriter(), DscMrpExcelWriter()]
        self.save_fig = save_fig
        self.log_file_dir = log_file_dir

    def evaluate(self):
        r_squared_list = []
        MAE_list = []
        TM_list = []
        SSIM_list = []
        for i, result_dir in enumerate(self.result_list):
            patient_info = '/'.join(result_dir.split('/')[-4:])
            print(i + 1, patient_info)

            try:
                r_squared_s = []
                MAEs = []
                TMs = []
                SSIMs = []
                pred_array_list = []
                label_array_list = []

                _mask = np.load((result_dir + "/" + "mask.npy"))
                mask = torch.from_numpy(_mask)

                for phase_idx in range(5):
                    _pred = np.load(result_dir + "/" + "predict.npy")[phase_idx]
                    _label = np.load(result_dir + "/" + "gt.npy")[phase_idx]
                    pred_array_list.append(_pred)
                    label_array_list.append(_label)
                    # calculate evaluation metrics
                    pre_predict = torch.from_numpy(_pred).type(torch.FloatTensor)[mask[0] > 0].numpy()
                    pre_label = torch.from_numpy(_label).type(torch.FloatTensor)[mask[0] > 0].numpy()
                    predict = (pre_predict - min(pre_predict)) / (max(pre_predict) - min(pre_predict))
                    label = (pre_label - min(pre_label)) / (max(pre_label) - min(pre_label))

                    r_squared_s.append(R_Squared(torch.from_numpy(predict), torch.from_numpy(label)))
                    MAEs.append(MAE(torch.from_numpy(label), torch.from_numpy(predict)))
                    TMs.append(TM(torch.from_numpy(predict), torch.from_numpy(label)))
                    SSIMs.append(SSIM(torch.from_numpy(predict), torch.from_numpy(label)))
                r_squared_list.append(r_squared_s)
                MAE_list.append(MAEs)
                TM_list.append(TMs)
                SSIM_list.append(SSIMs)

            except Exception as e:
                print(result_dir)
                print(e)
                continue

            data = {"R-Squared": r_squared_list, "MAE": MAE_list, "TM": TM_list, "SSIM": SSIM_list}
            r_squared_arr = np.array(r_squared_list)
            mae_arr = np.array(MAE_list)
            tm_arr = np.array(TM_list)
            ssim_arr = np.array(SSIM_list)
            np.save(f"{self.evaluation_metrics_save_dir}/r_squared.npy", r_squared_arr)
            np.save(f"{self.evaluation_metrics_save_dir}/mae.npy", mae_arr)
            np.save(f"{self.evaluation_metrics_save_dir}/tm.npy", tm_arr)
            np.save(f"{self.evaluation_metrics_save_dir}/ssim.npy", ssim_arr)

            for writer in self.writers:
                if type(writer) is DscMrpCsvWriter:
                    cf = {"data": data, "output_file": f"{self.evaluation_metrics_save_dir}/result.txt"}
                elif type(writer) is DscMrpExcelWriter:
                    cf = {"data": data, "output_file": f"{self.evaluation_metrics_save_dir}/result.xlsx"}
                writer.write(cf)
