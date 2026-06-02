import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as T

from filters import apply_filter, N_FILTERS, FILTER_NAMES, STOP_ACTION
from metrics import compute_ssim
from noise import (
    add_gaussian_noise, add_salt_pepper, add_periodic_noise,
    add_scratches, add_historical_photo_degradation,
)

IMG_SIZE = 128
SSIM_IMPROVEMENT_THRESHOLD = 0.003  # melhora mínima para considerar filtro útil

# Normalização padrão ImageNet
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def image_to_tensor(image_rgb):
    """Converte numpy RGB [0,255] → tensor normalizado."""
    t = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    return _NORMALIZE(t)


def find_best_action(current_bgr, clean_bgr):
    """
    Testa cada filtro e retorna o índice do que maximiza o ganho de SSIM.
    Retorna STOP se nenhum filtro melhora além do limiar.
    """
    current_rgb = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2RGB)
    clean_rgb   = cv2.cvtColor(clean_bgr,   cv2.COLOR_BGR2RGB)

    base_ssim = compute_ssim(current_rgb, clean_rgb)
    best_action = STOP_ACTION
    best_ssim   = base_ssim

    for action in range(N_FILTERS):
        filtered_bgr = apply_filter(current_bgr, action)
        filtered_rgb = cv2.cvtColor(filtered_bgr, cv2.COLOR_BGR2RGB)
        s = compute_ssim(filtered_rgb, clean_rgb)
        if s > best_ssim + SSIM_IMPROVEMENT_THRESHOLD:
            best_ssim   = s
            best_action = action

    return best_action


# ── Geradores de amostras de treino ─────────────────────────────────────────

def _make_sample(clean_bgr):
    """
    Cria uma amostra (imagem degradada, label) a partir de uma imagem limpa.
    Aleatoriamente aplica 0 ou mais filtros antes de gerar o label,
    para que a rede aprenda a agir em diferentes estágios da restauração.
    """
    current = clean_bgr.copy()

    # Degradação inicial aleatória
    # 70% histórica (sépia, fading, manchas), 30% sintética genérica
    if np.random.random() < 0.7:
        current = add_historical_photo_degradation(current)
    else:
        degradation = np.random.randint(0, 3)
        if degradation == 0:
            current = add_gaussian_noise(current, sigma=np.random.uniform(10, 35))
        elif degradation == 1:
            current = add_salt_pepper(current, amount=np.random.uniform(0.02, 0.07))
        else:
            current = add_periodic_noise(current,
                                         freq=np.random.randint(5, 25),
                                         amplitude=np.random.randint(20, 45))

    # Aplica 0-2 filtros aleatórios antes de rotular (ensina a rede
    # a continuar a partir de estados parcialmente restaurados)
    n_pre = np.random.randint(0, 3)
    for _ in range(n_pre):
        action = np.random.randint(0, N_FILTERS)
        current = apply_filter(current, action)

    label = find_best_action(current, clean_bgr)
    return current, label


class RestorationDataset(Dataset):
    def __init__(self, image_paths, samples_per_image=25):
        self.paths = list(image_paths)
        self.spi   = samples_per_image

    def __len__(self):
        return len(self.paths) * self.spi

    def __getitem__(self, idx):
        path = self.paths[idx % len(self.paths)]

        bgr   = cv2.imread(str(path))
        bgr   = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        degraded_bgr, label = _make_sample(bgr)
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)

        return image_to_tensor(degraded_rgb), label
