import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms.functional as TF
from skimage.metrics import structural_similarity as _skimage_ssim

from filters import apply_filter, N_FILTERS, FILTER_NAMES, STOP_ACTION
from noise import (
    add_gaussian_noise, add_salt_pepper, add_periodic_noise,
    add_scratches, add_historical_photo_degradation,
    add_sepia_tone, add_fading, add_vignette, add_film_grain, add_age_spots,
)

IMG_SIZE  = 224   # ResNet-18 espera 224x224
_SSIM_SIZE = 64   # tamanho para cálculo de SSIM nos soft labels (12x mais rápido)

SSIM_IMPROVEMENT_THRESHOLD = 0.003   # ganho mínimo — reduzido para detectar melhorias sutis
SOFT_LABEL_TEMPERATURE     = 0.04    # temperatura do softmax sobre ganhos SSIM

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def image_to_tensor(image_rgb):
    """Converte numpy RGB [0,255] → tensor normalizado ImageNet."""
    t = torch.from_numpy(image_rgb.transpose(2, 0, 1)).float() / 255.0
    for c in range(3):
        t[c] = (t[c] - _MEAN[c]) / _STD[c]
    return t


# ── Augmentação espacial ──────────────────────────────────────────────────────

def augment_clean(image_bgr):
    """Flip horizontal e random crop para aumentar variação do dataset."""
    if np.random.random() < 0.5:
        image_bgr = cv2.flip(image_bgr, 1)

    h, w   = image_bgr.shape[:2]
    scale  = np.random.uniform(0.8, 1.0)
    new_h  = int(h * scale)
    new_w  = int(w * scale)
    top    = np.random.randint(0, h - new_h + 1)
    left   = np.random.randint(0, w - new_w + 1)
    image_bgr = image_bgr[top:top + new_h, left:left + new_w]
    return cv2.resize(image_bgr, (w, h))


# ── Métrica híbrida: SSIM (estrutura) + color cast (cor global) ─────────────

_SSIM_WEIGHT  = 0.5
_COLOR_WEIGHT = 0.5

# Normalização do color cast: reduzido de 30 → 20 para aumentar sensibilidade
# a dominantes sutis (tom dourado leve, sépia fraca, papel envelhecido).
# Desvio de 20 unidades LAB a/b = score_cor 0; desvio de 5 = score_cor 0.75.
_CAST_NORM = 20.0


def _perceptual_score(img_bgr, ref_bgr):
    """
    Métrica híbrida perceptual:
      score = 0.5 * SSIM  +  0.5 * acuracia_de_cor_global

    SSIM captura estrutura, bordas e contraste — bom para ruído, blur e contraste.

    Color cast compara a MEDIA dos canais a/b (LAB) da imagem vs referência.
    Isso é sensível a dominante sépia/amarela (que desloca a média dos canais),
    mas insensível a ruído (que tem media zero e não altera a média dos canais).
    Resultado: color_correction vence claramente em imagens com dominante de cor.

    Calculado em resolução _SSIM_SIZE para equilibrar qualidade e velocidade.
    """
    s   = _SSIM_SIZE
    img = cv2.resize(img_bgr, (s, s))
    ref = cv2.resize(ref_bgr, (s, s))

    # Componente 1: SSIM no espaço RGB
    ssim_val = _skimage_ssim(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(ref, cv2.COLOR_BGR2RGB),
        channel_axis=2, data_range=255,
    )

    # Componente 2: distância entre o tom de cor médio (color cast)
    # Mede desvio da média dos canais a/b — sensível a sépia, insensível a ruído
    img_lab  = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab  = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_ab   = img_lab[:, :, 1:].mean(axis=(0, 1))
    ref_ab   = ref_lab[:, :, 1:].mean(axis=(0, 1))
    cast_dist = float(np.sqrt(((img_ab - ref_ab) ** 2).sum()))
    color_acc = max(0.0, 1.0 - cast_dist / _CAST_NORM)

    return _SSIM_WEIGHT * ssim_val + _COLOR_WEIGHT * color_acc


def compute_soft_labels(current_bgr, clean_bgr):
    """
    Retorna distribuição de probabilidade sobre as ações.

    Usa métrica híbrida SSIM + cor LAB: cada filtro recebe peso proporcional
    ao ganho de score que proporciona sobre a imagem atual.
    Se nenhum filtro melhora além do limiar, STOP recebe probabilidade 1.0.
    """
    base_score = _perceptual_score(current_bgr, clean_bgr)
    gains      = np.zeros(len(FILTER_NAMES), dtype=np.float32)

    for action in range(N_FILTERS):
        filtered_score = _perceptual_score(apply_filter(current_bgr, action), clean_bgr)
        gains[action]  = max(0.0, filtered_score - base_score)

    gains[STOP_ACTION] = 0.0

    if gains[:N_FILTERS].max() < SSIM_IMPROVEMENT_THRESHOLD:
        soft = np.zeros(len(FILTER_NAMES), dtype=np.float32)
        soft[STOP_ACTION] = 1.0
        return soft

    scaled = gains / SOFT_LABEL_TEMPERATURE
    scaled -= scaled.max()
    exp    = np.exp(scaled)
    return (exp / exp.sum()).astype(np.float32)


# ── Geração de amostras ───────────────────────────────────────────────────────

def _add_warm_cast(image_bgr, strength=0.3):
    """
    Tom quente/dourado — simula papel fotográfico envelhecido, pinturas antigas,
    digitalização com balanço de branco quente. Diferente da sépia uniforme:
    levanta R e G (amarelo-âmbar), reduz B, preservando variação espacial.
    Cobre tonalidades como as da Monalisa, fotos de família dos anos 70-80.
    """
    img = image_bgr.astype(np.float32)
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 + 0.30 * strength), 0, 255)  # R sobe
    img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 + 0.12 * strength), 0, 255)  # G sobe leve
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 - 0.20 * strength), 0, 255)  # B desce
    return img.astype(np.uint8)


def _add_blue_cast(image_bgr, strength=0.5):
    """
    Dominante azulada (WB frio, flash errado, digitalização com tonalidade fria).
    """
    img = image_bgr.astype(np.float32)
    img[:, :, 0] = np.clip(img[:, :, 0] * (1.0 + 0.40 * strength), 0, 255)  # B sobe
    img[:, :, 2] = np.clip(img[:, :, 2] * (1.0 - 0.30 * strength), 0, 255)  # R desce
    img[:, :, 1] = np.clip(img[:, :, 1] * (1.0 - 0.10 * strength), 0, 255)  # G leve
    return img.astype(np.uint8)


def _add_dark_exposure(image_bgr, factor=0.4):
    """Subexposição: escurece uniformemente todos os canais."""
    return np.clip(image_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _add_blur_degradation(image_bgr, sigma=2.0):
    """
    Desfoque por movimento/foco — ensina o modelo a usar high_pass para recuperar
    nitidez. Mais realista que ruído para simular fotos tiradas sem tripé.
    """
    ksize = int(sigma * 4) | 1  # kernel ímpar
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), sigma)


def _add_dust_spots(image_bgr, n_spots=15):
    """
    Pequenas manchas claras (poeira no scanner, foxing claro) — ensina morph_open.
    Círculos brancos pequenos espalhados aleatoriamente.
    """
    result = image_bgr.copy()
    h, w = image_bgr.shape[:2]
    for _ in range(n_spots):
        x, y = np.random.randint(0, w), np.random.randint(0, h)
        r    = np.random.randint(2, 6)
        cv2.circle(result, (x, y), r, (220, 220, 220), -1)
    return result


def _make_sample(clean_bgr):
    """
    17 cenários cobrindo todos os 20 filtros com variantes de intensidade.
    Cada faixa é projetada para que um filtro específico (ou família) vença
    claramente na métrica híbrida SSIM + color_cast.
    """
    current = augment_clean(clean_bgr)
    roll    = np.random.random()

    if roll < 0.08:
        # Foto ja limpa / bem preservada — modelo deve aprender a PARAR
        # Nenhuma degradacao aplicada; STOP ganha porque nenhum filtro melhora
        pass

    elif roll < 0.15:
        # Tom quente muito sutil — papel envelhecido, pintura ambar, monalisa-like
        current = _add_warm_cast(current, strength=np.random.uniform(0.15, 0.45))

    elif roll < 0.15:
        # Tom quente moderado — foto de familia anos 70-80, papel amarelado
        current = _add_warm_cast(current, strength=np.random.uniform(0.50, 0.90))
        if np.random.random() < 0.4:
            current = add_fading(current, factor=np.random.uniform(0.75, 0.92))

    elif roll < 0.23:
        # Sepia moderada — amarelamento quimico medio
        current = add_sepia_tone(current, intensity=np.random.uniform(0.30, 0.60))

    elif roll < 0.31:
        # Sepia forte — oxidacao severa, fotos muito antigas
        current = add_sepia_tone(current, intensity=np.random.uniform(0.65, 1.00))
        if np.random.random() < 0.4:
            current = add_fading(current, factor=np.random.uniform(0.65, 0.88))

    elif roll < 0.39:
        # Tom azulado — WB frio, scanner com temperatura errada
        current = _add_blue_cast(current, strength=np.random.uniform(0.30, 0.90))

    elif roll < 0.47:
        # Subexposicao forte — foto muito escura
        current = _add_dark_exposure(current, factor=np.random.uniform(0.12, 0.32))

    elif roll < 0.54:
        # Subexposicao leve — foto levemente escura
        current = _add_dark_exposure(current, factor=np.random.uniform(0.38, 0.65))

    elif roll < 0.62:
        # Desbotamento/fading — compressao de contraste, pretos acinzentados
        current = add_fading(current, factor=np.random.uniform(0.30, 0.75))
        if np.random.random() < 0.55:
            current = add_vignette(current, strength=np.random.uniform(0.3, 0.65))

    elif roll < 0.68:
        # Foto desfocada — movimento ou foco errado; high_pass deve restaurar
        current = _add_blur_degradation(current, sigma=np.random.uniform(1.5, 4.0))
        if np.random.random() < 0.3:
            current = add_sepia_tone(current, intensity=np.random.uniform(0.1, 0.35))

    elif roll < 0.75:
        # Granulado leve — pelicula de baixa ISO
        current = add_film_grain(current, sigma=np.random.uniform(6, 18))
        if np.random.random() < 0.25:
            current = add_salt_pepper(current, amount=np.random.uniform(0.003, 0.015))

    elif roll < 0.82:
        # Granulado forte — pelicula de alta ISO, muito ruido
        current = add_film_grain(current, sigma=np.random.uniform(20, 45))
        if np.random.random() < 0.5:
            current = add_salt_pepper(current, amount=np.random.uniform(0.02, 0.08))

    elif roll < 0.88:
        # Sal-e-pimenta predominante (defeito de sensor ou emulsao)
        current = add_salt_pepper(current, amount=np.random.uniform(0.03, 0.12))

    elif roll < 0.93:
        # Textura periodica (scanner com interferencia, papel texturizado)
        current = add_periodic_noise(current,
                                     freq=np.random.randint(5, 22),
                                     amplitude=np.random.randint(20, 60))

    elif roll < 0.96:
        # Manchas claras: poeira no scanner, foxing — morph_open
        current = _add_dust_spots(current, n_spots=np.random.randint(10, 30))
        if np.random.random() < 0.5:
            current = add_age_spots(current, n_spots=np.random.randint(3, 10))

    elif roll < 0.98:
        # Arranhoes e fissuras escuros — morph_close
        current = add_scratches(current, n_scratches=np.random.randint(3, 10))

    elif roll < 0.99:
        # Degradacao historica composta realista
        current = add_historical_photo_degradation(current)

    else:
        # Ruido gaussiano puro
        current = add_gaussian_noise(current, sigma=np.random.uniform(8, 50))

    # Aplica 0-2 filtros aleatorios para treinar estados parcialmente restaurados
    for _ in range(np.random.randint(0, 3)):
        current = apply_filter(current, np.random.randint(0, N_FILTERS))

    soft_label = compute_soft_labels(current, clean_bgr)
    return current, soft_label


class RestorationDataset(Dataset):
    def __init__(self, image_paths, samples_per_image=25):
        self.paths = list(image_paths)
        self.spi   = samples_per_image

    def __len__(self):
        return len(self.paths) * self.spi

    def __getitem__(self, idx):
        bgr = cv2.imread(str(self.paths[idx % len(self.paths)]))
        bgr = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))

        degraded_bgr, soft_label = _make_sample(bgr)
        degraded_rgb = cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB)

        return image_to_tensor(degraded_rgb), torch.from_numpy(soft_label)
