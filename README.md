# 3D-MoReT — 3D Mobile Regression Vision Transformer
### 5-phase Collateral Map Generation from DSC-MR Perfusion in Acute Ischemic Stroke

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12.1-ee4c2c?logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-IJCARS%202024-blueviolet)](https://link.springer.com/article/10.1007/s11548-024-03229-5)
[![PMC](https://img.shields.io/badge/PMC-11442547-blue)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11442547/)

---

## Overview

**3D-MoReT** is a lightweight encoder-decoder network that generates five-phase collateral perfusion maps directly from 4D DSC-MR perfusion images. It replaces computationally expensive heuristic processing with a single forward pass, making real-time collateral assessment feasible in clinical settings.

**Published in:** *International Journal of Computer Assisted Radiology and Surgery*, Vol. 19, pp. 2043–2054, 2024  
**Authors:** Sumin Jung, Hyun Yang, Hyun Jeong Kim, Hong Gee Roh, Jin Tae Kwak

---

## Architecture

3D-MoReT follows an encoder-decoder design with three key building blocks:
  ![3D-MoReT Architecture](assets/architecture.png)

| Block | Role |
|---|---|
| MobileNet V2 blocks | Efficient spatial feature extraction |
| MobileViT blocks | Volumetric (3D) relationship modeling |
| Vision Transformer layers | Inter-slice temporal interaction |

**Input:** 4D DSC-MRP — `(40 time points × 20 slices × 224 × 224)`  
**Output:** 5-phase collateral maps — arterial / capillary / early venous / late venous / delayed

---

## Results

Evaluated on 337 test subjects from two medical centers (CMC, KU).

| Metric | Score |
|---|---|
| R² | ≥ 0.926 |
| MAE | ≤ 0.048 |
| SSIM | ≥ 0.913 |
| Tanimoto Measure | ≥ 0.947 |

**Efficiency vs. baselines:**

| Model | Params (K) | Size (MB) | FLOPs (×10¹¹) | Inference (s) |
|---|---|---|---|---|
| **3D-MoReT (ours)** | **26,736** | **25.5** | **7.14** | **3.56** |
| 3D-SwinUNetR | 62,194 | 237.4 | 35.68 | 18.73 |
| 3D-MROD-Net | 163,841 | 625.2 | 11.46 | 5.55 |
| 3D-UNet | 19,077 | 72.8 | 5.78 | 3.18 |

3D-MoReT matches SwinUNetR-level accuracy at **~4× fewer parameters** and **~5× lower FLOPs**.

---

## Installation

```bash
git clone https://github.com/jasmine00716/3D-MoReT.git
cd 3D-MoReT
pip install -r requirements.txt
```

**Core dependencies:** PyTorch 1.12.1 · torchvision 0.13.1 · scikit-image · pydicom · wandb

---

## Dataset Preparation

Preprocessed data is stored as NumPy arrays (`.npy`). Each subject directory should contain:

```
<subject_dir>/NpyFiles/
    IMG_n01.npy                                        # 4D DSC-MRP input
    phase_maps_medfilt_rs_n.npy                        # 5-phase ground truth
    phase_maps_medfilt_rs_n_wm_50_bins_for_each_phase.npy  # phase weights
    mask_4d.npy                                        # brain mask
```

Populate `dataset/split_dsc_mrp_dataset.csv` with columns:

```
img_path,split
/path/to/subject/NpyFiles/IMG_n01.npy,0   # 0=train, 1=val, 2=test
```

> **Note:** CSV files containing subject paths are excluded from this repository (`.gitignore`). Prepare your own split file following the format above.

---

## Training

Edit `config.py` to set your data paths and GPU device, then:

```bash
# Train 3D-MoReT on DSC-MRP
python train_script.py -m mobilevit3d -u deconv -d 6

# Train on DCE-MRA
python train_script_mra.py -m mobilevit3d -u deconv -d 6
```

**Key training settings** (see `config.py`):

| Setting | Value |
|---|---|
| Loss | BerHu |
| Optimizer | Adam (lr = 1e-3) |
| Scheduler | ReduceLROnPlateau (factor=0.75, patience=3) |
| Batch size | 4 |
| Epochs | 300 |
| Augmentation | CenterCrop(224), HorizontalFlip(p=0.5) |

Set your W&B API key before training:
```bash
export WANDB_API_KEY="your_key_here"
```

---

## Inference & Evaluation

```bash
python predict.py -m mobilevit3d -d 0
```

Evaluation metrics (R², MAE, SSIM, Tanimoto) are computed in `evaluation_metrics.py` and saved to `evaluation_metrics/`.

---

## Project Structure

```
3D-MoReT/
├── models/
│   ├── MobileViT_v3_3D/          # 3D-MoReT (proposed model)
│   │   ├── 3d_moret.py           # model entry point & test script
│   │   ├── mobilevit_v3_trs_3d_1x6x6_same_slice.py  # main architecture
│   │   ├── mobilevit_v3_block.py # MobileViT block
│   │   └── vit_block.py          # Vision Transformer block
│   ├── MROD_Net_3D/              # 3D-MROD-Net baseline
│   ├── SwinUNetR/                # 3D-SwinUNetR baseline
│   ├── SwinTransformer_3D/       # Swin Transformer variants
│   ├── Unet_3D/                  # 3D-UNet baseline
│   └── unetplusplus_3d/          # 3D-UNet++ baseline
├── criteria/
│   ├── berhu_loss.py             # BerHu loss (primary)
│   ├── uncertainty_loss.py
│   └── ordinal_regression_loss.py
├── dataset/                      # CSV split files (gitignored)
├── data/                         # Raw data CSVs (gitignored)
├── config.py                     # MRP training config
├── config_mra.py                 # MRA training config
├── config_mrod.py                # MROD-Net config
├── config_unext.py               # UNeXt config
├── dsc_mrp_dataset.py            # PyTorch Dataset class
├── data_augmentation.py          # CenterCrop, HorizontalFlip
├── train_script.py               # MRP training entry point
├── train_script_mra.py           # MRA training entry point
├── predict.py                    # Inference script
├── evaluation_metrics.py         # R², MAE, SSIM, Tanimoto
├── gen_dicoms_noHeader.py        # Export predictions as DICOM
├── visualize.py                  # Prediction visualization
└── utils_.py                     # Seeding, normalization, early stopping
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{jung20243dmoret,
  title   = {3D mobile regression vision transformer for collateral imaging in acute ischemic stroke},
  author  = {Jung, Sumin and Yang, Hyun and Kim, Hyun Jeong and Roh, Hong Gee and Kwak, Jin Tae},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  volume  = {19},
  pages   = {2043--2054},
  year    = {2024},
  doi     = {10.1007/s11548-024-03229-5}
}
```
