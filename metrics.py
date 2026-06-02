import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def compute_ssim(img1, img2):
    """SSIM between two RGB numpy arrays [0, 255]."""
    channel_axis = 2 if img1.ndim == 3 else None
    return ssim(img1, img2, data_range=255, channel_axis=channel_axis)


def compute_psnr(img1, img2):
    """PSNR between two numpy arrays [0, 255]."""
    return psnr(img1, img2, data_range=255)
