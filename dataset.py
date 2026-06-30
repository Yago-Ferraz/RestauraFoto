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


# ── Hard label balanceado ─────────────────────────────────────────────────────

def _make_hard_label(target_action, n_actions=21, weight=0.75):
    """Label com suavização: target=weight, restante=uniforme."""
    label = np.full(n_actions, (1.0 - weight) / (n_actions - 1), dtype=np.float32)
    label[target_action] = weight
    return label


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
    Distribuição uniforme: 1 cenário por filtro (21 total).
    Label fixo = 0.75 no filtro correto — elimina o viés da métrica.

    Mapeamento degradação → filtro esperado:
      0  mean              ← ruído gaussiano pesado (uniforme)
      1  gaussian_soft     ← grão leve (σ 5-13)
      2  gaussian_medium   ← grão médio (σ 13-22)
      3  gaussian_strong   ← grão pesado (σ 22-45)
      4  bilateral         ← grão médio (bilateral preserva bordas)
      5  median_fine       ← sal-e-pimenta leve
      6  median_strong     ← sal-e-pimenta intenso
      7  high_pass_gentle  ← blur leve (σ 0.8-1.8)
      8  high_pass_strong  ← blur forte (σ 2-4.5)
      9  color_corr gentle ← cast leve (quente ou frio)
     10  color_corr strong ← sépia intensa ou cast forte
     11  clahe_subtle      ← fading leve (pretos acinzentados)
     12  clahe_strong      ← fading forte + vinheta
     13  gamma_bright      ← subexposição leve (fator 0.40-0.60)
     14  gamma_very_bright ← subexposição forte (fator 0.12-0.32)
     15  log_transform     ← subexposição moderada (fator 0.25-0.45)
     16  contrast_stretch  ← histograma comprimido (lavado)
     17  morph_open        ← manchas de poeira / foxing
     18  morph_close       ← riscos escuros / fissuras
     19  fourier           ← textura periódica forte
     20  STOP              ← imagem limpa
    """
    current = augment_clean(clean_bgr)
    target  = np.random.randint(0, len(FILTER_NAMES))  # uniforme 0-20

    if target == 0:    # mean — ruído gaussiano pesado
        current = add_gaussian_noise(current, sigma=np.random.uniform(20, 40))

    elif target == 1:  # gaussian_soft — grão leve
        current = add_film_grain(current, sigma=np.random.uniform(5, 13))

    elif target == 2:  # gaussian_medium — grão médio
        current = add_film_grain(current, sigma=np.random.uniform(13, 22))

    elif target == 3:  # gaussian_strong — grão pesado
        current = add_film_grain(current, sigma=np.random.uniform(22, 45))

    elif target == 4:  # bilateral — grão com estrutura de borda
        current = add_film_grain(current, sigma=np.random.uniform(12, 28))

    elif target == 5:  # median_fine — sal-e-pimenta leve
        current = add_salt_pepper(current, amount=np.random.uniform(0.01, 0.04))

    elif target == 6:  # median_strong — sal-e-pimenta intenso
        current = add_salt_pepper(current, amount=np.random.uniform(0.05, 0.15))

    elif target == 7:  # high_pass_gentle — blur leve
        current = _add_blur_degradation(current, sigma=np.random.uniform(0.8, 1.8))

    elif target == 8:  # high_pass_strong — blur forte
        current = _add_blur_degradation(current, sigma=np.random.uniform(2.0, 4.5))

    elif target == 9:  # color_correction_gentle — cast leve
        if np.random.random() < 0.5:
            current = _add_warm_cast(current, strength=np.random.uniform(0.15, 0.40))
        else:
            current = _add_blue_cast(current, strength=np.random.uniform(0.15, 0.35))

    elif target == 10:  # color_correction_strong — sépia/cast forte
        if np.random.random() < 0.6:
            current = add_sepia_tone(current, intensity=np.random.uniform(0.50, 0.95))
        else:
            current = _add_warm_cast(current, strength=np.random.uniform(0.55, 0.95))

    elif target == 11:  # clahe_subtle — fading leve
        current = add_fading(current, factor=np.random.uniform(0.60, 0.80))

    elif target == 12:  # clahe_strong — fading forte
        current = add_fading(current, factor=np.random.uniform(0.30, 0.55))
        if np.random.random() < 0.5:
            current = add_vignette(current, strength=np.random.uniform(0.30, 0.55))

    elif target == 13:  # gamma_bright — subexposição leve
        current = _add_dark_exposure(current, factor=np.random.uniform(0.40, 0.60))

    elif target == 14:  # gamma_very_bright — subexposição forte
        current = _add_dark_exposure(current, factor=np.random.uniform(0.12, 0.32))

    elif target == 15:  # log_transform — subexposição moderada
        current = _add_dark_exposure(current, factor=np.random.uniform(0.25, 0.50))

    elif target == 16:  # contrast_stretch — histograma comprimido (lavado)
        lo = np.random.uniform(40, 80)
        hi = np.random.uniform(175, 215)
        current = ((current.astype(np.float32) / 255.0) * (hi - lo) + lo).clip(0, 255).astype(np.uint8)

    elif target == 17:  # morph_open — manchas de poeira
        current = _add_dust_spots(current, n_spots=np.random.randint(20, 60))
        if np.random.random() < 0.4:
            current = add_age_spots(current, n_spots=np.random.randint(3, 12))

    elif target == 18:  # morph_close — riscos escuros
        current = add_scratches(current, n_scratches=np.random.randint(4, 12))

    elif target == 19:  # fourier — textura periódica forte
        current = add_periodic_noise(current,
                                     freq=np.random.randint(5, 20),
                                     amplitude=np.random.randint(30, 70))

    # target == 20 → STOP: imagem limpa, sem degradação

    return current, _make_hard_label(target)


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
