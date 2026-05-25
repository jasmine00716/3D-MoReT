import os
import glob
import time
from utils_ import norm_ab
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset
import skimage.exposure as exposure
import pylab as plt
from skimage.util import montage
from copy import deepcopy
import torch.nn.functional as F
from skimage.color import label2rgb


def save_ds(ds, path):
    """Save DICOM file; prints a message if the path is too long."""
    try:
        ds.save_as(path)
    except Exception:
        print(f'Files cannot be saved. The file path {path} might be too long. Please try storing the input data in a parent folder!')


def get_color():
    from pylab import cm
    cm_jet = np.reshape(np.concatenate([cm.jet(i) for i in range(255)], axis=0), (255, 4))
    return np.vstack((np.array([0, 0, 0]), cm_jet[:, :3]))


JET = get_color()


class ColorBar:
    def __init__(self, height, width):
        bar_tmp = np.zeros((101, 10))
        for i in range(100):
            bar_tmp[i] = 99 - i + 1
        # bar position
        self.end_bar_x = int(width - 3)
        self.start_bar_x = int(self.end_bar_x - 10 + 1)
        self.start_bar_y = int(np.floor((height - 101) / 2))
        self.end_bar_y = int(self.start_bar_y + 101 - 1)
        # Index Image generation
        self.bar = bar_tmp * 2.551
        self.numimg = np.load('extra_data/mr_collateral_numing.npy')

    def insert_num_board(self, rgbIMG, maxIMG, maxWWL):
        # insert min number
        for ch in range(rgbIMG.shape[-1]):
            rgbIMG[self.end_bar_y:self.end_bar_y + 9, self.end_bar_x - 6:self.end_bar_x, ch] = self.numimg[..., 0, 0]

        # insert max number
        max_num_str = str((maxIMG * maxWWL * 255).astype('uint8'))
        str_length = len(max_num_str)
        num_board = np.zeros((9, str_length * 6, 3))
        for i in range(str_length):
            selected_num = int(max_num_str[i])
            num_board[:, i * 6:(i + 1) * 6, 0] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 1] = self.numimg[:, :, 0, selected_num]
            num_board[:, i * 6:(i + 1) * 6, 2] = self.numimg[:, :, 0, selected_num]
        rgbIMG[self.start_bar_y - 9:self.start_bar_y, self.end_bar_x - str_length * 6 + 1:self.end_bar_x + 1] = \
            num_board
        return rgbIMG

    def insert_color_bar(self, indIMG):
        indIMG[self.start_bar_y:self.end_bar_y + 1, self.start_bar_x:self.end_bar_x + 1] = self.bar
        return indIMG


def paste_center(img, new_size_x, new_size_y, center=None):
    """Paste img to the center of an image of size (new_size_y, new_size_x)"""
    size_x = img.shape[-1]
    size_y = img.shape[-2]

    new_img = np.zeros((img.shape[0], img.shape[1], new_size_y, new_size_x,))

    if not center:
        new_center = [np.floor(new_size_y / 2), np.floor(new_size_x / 2)]
    else:
        new_center = center

    start_x = int(new_center[1] - np.floor(size_x / 2))
    start_y = int(new_center[0] - np.floor(size_y / 2))

    if (start_x > 0) and (start_y > 0):
        new_img[..., start_y: start_y + size_y, start_x: start_x + size_x] = img
    else:
        new_img = img
    return new_img


def stretch_lim(img, tol_low, tol_high):
    """Mimic the stretchlim function in MATLAB"""
    nbins = 50
    N = np.histogram(img, nbins, [0, img.max()])[0]
    cdf = np.cumsum(N) / sum(N)
    ilow = np.where(cdf > tol_low)[0][0]
    ihigh = np.where(cdf >= tol_high)[0][0]
    if ilow == ihigh:
        ilowhigh = np.array([1, nbins])
    else:
        ilowhigh = np.array([ilow, ihigh])
    lowhigh = ilowhigh / (nbins - 1)
    return lowhigh


def mr_collateral_find_WWL(inIMG, mask, outlier):
    MIN = inIMG.min()
    MAX = inIMG.max()

    # Image Signal Normalization
    nIMG = inIMG / MAX
    outlier_low = (outlier / 100) / 2
    outlier_high = 1 - ((outlier / 100) / 2)
    WWL = stretch_lim(nIMG[mask > 0], outlier_low, outlier_high)
    minWWL = WWL.min()
    maxWWL = WWL.max()

    # Window width / level calculation
    WW = np.floor((MAX * maxWWL) - (MAX * minWWL))
    WL = np.floor(MAX * minWWL) + np.floor(WW / 2)
    return MIN, MAX, WW, WL


def save_grayscale_dcm(phase_map, ds, new_sn, k, slice_loop, outlier, col_dir_path, mask, col_sf=1, num_bits=16):
    """Save a phase map as a grayscale DICOM series"""
    SeriesDescription = [
        'DSC_Collateral_Arterial', 'DSC_Collateral_Capillary', 'DSC_Collateral_Early_Venous',
        'DSC_Collateral_Late_Venous', 'DSC_Collateral_Delay',
    ]
    phase_map = (phase_map * (2 ** num_bits - 1)).astype('uint%d' % num_bits)
    MIN, MAX, WW, WL = mr_collateral_find_WWL(phase_map * col_sf, mask, outlier[k])
    ds.SeriesDescription = SeriesDescription[k]
    ds.SeriesInstanceUID = str(new_sn + k)
    ds.SeriesNumber = new_sn + k
    ds.AcquisitionNumber = slice_loop
    ds.InstanceNumber = slice_loop
    ds.PixelSpacing = [1, 1]
    ds.PixelData = phase_map.tobytes()
    ds.Rows, ds.Columns = phase_map.shape
    ds.BitsAllocated = num_bits
    ds.SmallestImagePixelValue = int(MIN)
    ds.LargestImagePixelValue = int(MAX)
    ds.WindowCenter = int(WL)
    ds.WindowWidth = int(WW)
    ds.PixelRepresentation = 0
    path = "%s/%s_%03d.dcm" % (col_dir_path, ds.SeriesDescription, slice_loop)
    ds.save_as(path)


def save_color_dcm(phase_map, ds, new_sn, k, slice_loop, outlier, col_dir_path, mask, color_bar=None, col_sf=1,
                   num_bits=8, rescaling_first=False, prefix='DSC', SeriesDescription=None, WWL_list=None,
                   from_deep_learning=False, suffix=''):
    """Save a phase map as a color DICOM series"""
    WWL = WWL_list[k, slice_loop] if WWL_list is not None else None
    SeriesDescription = [
        'DSC_color_Collateral_Arterial', 'DSC_color_Collateral_Capillary', 'DSC_color_Collateral_Early_Venous',
        'DSC_color_Collateral_Late_Venous', 'DSC_color_Collateral_Delay',
    ]
    phase_map = phase_map * col_sf
    phase_map = mr_collateral_gen_color_image(phase_map, outlier, mask, rescaling_first, color_bar)
    phase_map = (phase_map * (2 ** num_bits - 1)).astype('uint%d' % num_bits)
    SeriesDescription = SeriesDescription[k]

    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.2'
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    ds = FileDataset('collateral_phase_maps', {}, file_meta=file_meta, preamble=b"\0" * 128)

    ds.SeriesInstanceUID = str(new_sn + k)
    ds.SeriesNumber = new_sn + k
    ds.AcquisitionNumber = k + 1
    ds.InstanceNumber = slice_loop + 1
    ds.PixelSpacing = [1, 1]
    ds.PixelData = phase_map.tobytes()
    ds.Rows, ds.Columns = phase_map.shape[:2]
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = 'RGB'
    ds.BitsAllocated = num_bits
    ds.BitsStored = num_bits
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.WindowCenter = 128
    ds.WindowWidth = 255
    path = f"{col_dir_path}_{prefix}_{slice_loop:03d}.dcm"
    save_ds(ds, path)
    return phase_map


def mr_collateral_gen_color_image(inIMG, outlier, mask=None, rescaling_first=True, color_bar=None, WWL=None,
                                  from_deep_learning=False):
    outlier = (outlier / 2, outlier / 2) if not isinstance(outlier, tuple) else outlier
    minIMG, maxIMG = inIMG.min(), max(inIMG.max(), 1)
    if WWL is None:
        if rescaling_first:
            # Image Signal Normalization
            nIMG = inIMG / maxIMG
            outlier_low = (outlier[0] / 100) / 2
            outlier_high = 1 - ((outlier[1] / 100) / 2)
            WWL = stretch_lim(nIMG, outlier_low, outlier_high)
            minWWL, maxWWL = WWL.min(), WWL.max()
            rsIMG = exposure.rescale_intensity(nIMG, tuple(WWL)) if not from_deep_learning else nIMG
        else:
            rsIMG = inIMG / maxIMG
            maxWWL = 1
    else:
        minWWL, maxWWL = WWL
        rsIMG = inIMG

    maxIMG *= 255 if maxIMG < 1.1 else 1
    indIMG = (rsIMG * 255).astype('uint8')
    indIMG = color_bar.insert_color_bar(indIMG) if color_bar else indIMG
    labels = np.unique(indIMG)
    rgb = np.zeros((indIMG.shape + (3,)))
    for label in labels:
        loc = np.where(indIMG == label)
        rgb[loc[0], loc[1], :] = JET[label]
    return rgb


def postprocess(pred, mask):
    """Post-process prediction maps: clamp to [-0.9, 0.9] then normalize to [0, 1]"""
    predef_minmax = [-.9, .9]
    values_range = [0, 1]

    # Enforce values to lie within the predef_minmax range
    pred[pred < predef_minmax[0]] = predef_minmax[0]
    pred[pred > predef_minmax[1]] = predef_minmax[1]

    # norm from predef_minmax to values_range
    for j in range(pred.shape[0]):
        pred[j] = norm_ab(pred[j], values_range[0], values_range[1], predef_minmax, False, mask[0], True)
    return pred, mask


def postprocess_dl(pred, mask, hdr, slicewise_rescaling=True):
    """Post-process deep learning predictions: clamp, normalize, optional slicewise rescaling"""
    predef_minmax = [-.9, .9]
    values_range = [0, 1]
    WWL = None

    if hdr is not None:
        pred = paste_center(pred, hdr[0].new_height, hdr[0].new_width)
        mask = paste_center(mask, hdr[0].new_height, hdr[0].new_width)

    # Enforce values to lie within the predef_minmax range
    pred[pred < predef_minmax[0]] = predef_minmax[0]
    pred[pred > predef_minmax[1]] = predef_minmax[1]

    # norm from predef_minmax to values_range
    for j in range(pred.shape[0]):
        pred[j] = norm_ab(pred[j], values_range[0], values_range[1], predef_minmax, False, mask[0], True)
        if slicewise_rescaling:
            for k in range(pred[j].shape[0]):
                maxIMG = pred[j][k].max()
                pred[j][k] /= maxIMG if maxIMG != 0 else 1
    return pred, mask, WWL


def postprocess_dsc(phase_maps, mask, hdr, slicewise_rescaling=True):
    outliers = [2, 4, 8, 10, 10]
    mask = np.ones(((1,) + phase_maps.shape[1:])) if mask is None else mask

    if hdr is not None:
        phase_maps = paste_center(phase_maps, hdr[0].new_height, hdr[0].new_width)
        mask = paste_center(mask, hdr[0].new_height, hdr[0].new_width)

    if slicewise_rescaling:
        WWL_list = np.zeros((phase_maps.shape[0], phase_maps.shape[1], 2))
        for i, (phase_map, outlier) in enumerate(zip(phase_maps, outliers)):
            outlier_low = (outlier / 100) / 2
            outlier_high = 1 - ((outlier / 100) / 2)
            for j, (pm, m) in enumerate(zip(phase_map, mask[0])):
                # Image Signal Normalization
                minIMG = pm.min()
                maxIMG = pm.max()
                nIMG = pm / maxIMG if maxIMG != 0 else pm
                WWL = stretch_lim(nIMG, outlier_low, outlier_high)
                # Rescaling so that values in every slice goes from 0 to 1
                phase_maps[i, j] = exposure.rescale_intensity(nIMG, tuple(WWL))
                phase_maps[i, j] *= m
                WWL_list[i, j] = WWL
    else:
        WWL_list = None
        for i, (phase_map, outlier) in enumerate(zip(phase_maps, outliers)):
            # Image Signal Normalization
            minIMG = phase_map[mask[0] > 0].min()
            maxIMG = phase_map[mask[0] > 0].max()
            nIMG = (phase_map - minIMG) / (maxIMG - minIMG)
            outlier_low = (outlier / 100) / 2
            outlier_high = 1 - ((outlier / 100) / 2)
            WWL = stretch_lim(nIMG[mask[0] > 0], outlier_low, outlier_high)
            # Rescaling so that values in every slice goes from 0 to 1
            phase_maps[i] = exposure.rescale_intensity(nIMG, tuple(WWL))
    return phase_maps, mask, WWL_list


def mr_collateral_dce_gen_dicoms(save_dir_path, hdr, phase_map, mask, suffix='', rescaling_first=True,
                                 insert_bar=False, gfs=3, predef_minmax=(-.9, .9), not_dl=False):
    print('Generating DICOM files...')
    tic = time.time()
    phase_map = deepcopy(phase_map)
    if not_dl:
        phase_map, mask = postprocess(phase_map, mask)
    else:
        phase_map, mask, WWL = postprocess_dl(phase_map, mask, hdr)

    phase_map = np.nan_to_num(phase_map)
    number_of_slice = phase_map.shape[1]
    col_sf = 10
    outliers_c_collateral = [(.01, 2), ] + [(4, 6), ] + [(3, 9.5), ] + [(2, 11.5), ] * 2

    new_sn_gray = 10000 + 1
    new_sn_color = 15000 + 1
    color_bar = ColorBar(*phase_map.shape[-2:]) if insert_bar else None
    color_phasemap = []

    for ind in range(phase_map.shape[0]):
        phasemap_result = []
        for slice_loop in range(number_of_slice):
            result = save_color_dcm(phase_map[ind, slice_loop], hdr, new_sn_color, ind, slice_loop,
                                    outliers_c_collateral[ind], None, mask[0, slice_loop], color_bar, col_sf,
                                    rescaling_first=rescaling_first)
            phasemap_result.append(result)
        color_phasemap.append(np.array(phasemap_result))

    return color_phasemap


def mr_collateral_dsc_gen_dicoms(save_dir_path, hdr, phase_map, mask, suffix='', rescaling_first=True,
                                 insert_bar=True, gfs=3, predef_minmax=(-.9, .9),
                                 from_deep_learning=False, window=None):
    print('Generating DICOM files...')
    tic = time.time()

    phase_map_org = phase_map.copy()
    phase_map, mask, WWL_list = postprocess_dsc(phase_map_org, mask, hdr,
                                                slicewise_rescaling=False) if not from_deep_learning \
        else postprocess_dl(phase_map, mask, hdr)

    outliers = [2, 4, 8, 10, 10]
    col_sf = 1
    number_of_slice = phase_map.shape[1]

    base_sn = 1e5
    new_sn_color = base_sn + 10000 + 1
    color_bar = ColorBar(*phase_map.shape[-2:]) if insert_bar else None

    # Save path
    presentations = ['Color']
    col_dir = 'DSC_GFS' + str(gfs) + suffix
    phases = ['Art', 'Cap', 'EVen', 'LVen', 'Del']
    col_dir_org = save_dir_path + '/' + col_dir
    os.makedirs(col_dir_org, exist_ok=True)
    col_dir_paths = {presentations[0]: {}}
    for presentation in presentations:
        for k in range(phase_map.shape[0]):
            col_dir_paths[presentation][k] = col_dir_org + '/' + presentation + '/' + '%d_%s' % (k, phases[k])
            if not os.path.exists(col_dir_paths[presentation][k]):
                os.makedirs(col_dir_paths[presentation][k])

    # Save DICOM files
    phase_map_color = phase_map.copy()
    if not from_deep_learning:
        phase_map_color, mask, WWL_list = postprocess_dsc(phase_map_color, mask, hdr, slicewise_rescaling=True)
    pm = []
    [save_color_dcm(phase_map[k, slice_loop], hdr, new_sn_color, k, slice_loop, outliers[k],
                    col_dir_paths['Color'][k], mask[0, slice_loop], color_bar, col_sf, rescaling_first=True)
     for k in range(phase_map.shape[0]) for slice_loop in range(number_of_slice)]


if __name__ == "__main__":
    import cv2
    from scipy.ndimage import gaussian_filter as gf

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
                    color_gt = (_gt[i][j] * (2 ** 8 - 1)).astype('uint%d' % 8)
                    color_gt = cv2.applyColorMap(color_gt, cv2.COLORMAP_JET)
                    color_gt[:, :, 0] *= _mask[0][j]
                    color_gt[:, :, 1] *= _mask[0][j]
                    color_gt[:, :, 2] *= _mask[0][j]
                    cv2.imwrite(f'{save_path}/gt_{phase[i]}_{j}.png', color_gt)

                    color_pred = (_pred[i][j] * (2 ** 8 - 1)).astype('uint%d' % 8)
                    color_pred = cv2.applyColorMap(color_pred, cv2.COLORMAP_JET)
                    color_pred[:, :, 0] *= _mask[0][j]
                    color_pred[:, :, 1] *= _mask[0][j]
                    color_pred[:, :, 2] *= _mask[0][j]
                    cv2.imwrite(f'{save_path}/pred_{phase[i]}_{j}.png', color_pred)
