import cv2
import numpy as np
import os
import glob
import utils_
from scipy.ndimage import gaussian_filter as gf


def postprocess(pred, mask):
    """Post-process prediction maps: clamp to [-0.9, 0.9] then normalize to [0, 1]"""
    predef_minmax = [-.9, .9]
    values_range = [0, 1]

    pred[pred < predef_minmax[0]] = predef_minmax[0]
    pred[pred > predef_minmax[1]] = predef_minmax[1]

    for j in range(pred.shape[0]):
        pred[j] = utils_.norm_ab(pred[j], values_range[0], values_range[1], predef_minmax, False, mask[0], True)
    return pred, mask


result_dir = '/data1/sumin/compu/mrp_lightweight/mobilevitvit3d_1x6x6_deconv_20231129_01h06m/result'
result_dir_list = [os.path.join(result_dir, folder) for folder in os.listdir(result_dir)]

for r in result_dir_list:
    for result_path in glob.glob(r + "/*/*/*"):
        patient = result_path.split('/')[-1]
        year = result_path.split('/')[-2]
        status = result_path.split('/')[-3]
        hospital = result_path.split('/')[-4]

        _pred = np.load(result_path + '/predict.npy')
        _gt = np.load(result_path + '/gt.npy')
        _mask = np.load(result_path + '/mask.npy')
        _mask = _mask.astype(np.uint8)

        save_path = os.path.join(result_dir, hospital, status, year, patient)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        phase = ['art', 'cap', 'even', 'lven', 'del']
        outliers = [2, 4, 8, 10, 10]

        _pred, _mask = postprocess(_pred, _mask)
        _gt, _mask = postprocess(_gt, _mask)

        for i in range(5):
            sigma = [0, 1, 1]
            sign_coef = [-1, 1]
            ColMask = _mask.mean(axis=0) > 0
            PreIMG = np.zeros_like(_pred) + _pred

            _pred[i] = gf(_pred[i].mean(axis=0) * ColMask[0], sigma[0]) * sign_coef[0] + _pred[i] * sign_coef[1]
            _gt[i] = gf(_gt[i].mean(axis=0) * ColMask[0], sigma[0]) * sign_coef[0] + _gt[i] * sign_coef[1]

            for j in range(20):
                cv2.imwrite(f'{save_path}/mask_{j}.png', _mask[0][j] * 255)
