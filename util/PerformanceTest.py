import lpips
import torch
from skimage.metrics import structural_similarity as ssim

import cv2
import numpy as np


def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)  # multichannel=True for color images


def calculate_lpips(img1, img2):
    # img1과 img2를 Tensor로 변환하고 (1, 3, H, W)로 reshape 합니다.
    img1 = torch.tensor(img1).permute(2, 0, 1).unsqueeze(0) * 2 - 1  # normalize to [-1, 1]
    img2 = torch.tensor(img2).permute(2, 0, 1).unsqueeze(0) * 2 - 1

    loss_fn = lpips.LPIPS(net='alex')
    lpips_score = loss_fn(img1, img2)
    return lpips_score.item()
