import cv2
import numpy as np

FILTER_NAMES = [
    "mean",              # 0 — suavização geral
    "gaussian",          # 1 — granulado de filme
    "median",            # 2 — manchas e poeira
    "color_correction",  # 3 — remove dominante amarela/sépia
    "morph_close",       # 4 — arranhões e buracos
    "fourier",           # 5 — textura periódica do papel
    "clahe",             # 6 — restaura contraste de fotos desbotadas
    "STOP",              # 7
]

N_FILTERS   = len(FILTER_NAMES) - 1
STOP_ACTION = len(FILTER_NAMES) - 1


def apply_filter(image, action_idx):
    dispatch = {
        0: apply_mean,
        1: apply_gaussian,
        2: apply_median,
        3: apply_color_correction,
        4: apply_morph_close,
        5: apply_fourier_bandreject,
        6: apply_clahe,
    }
    fn = dispatch.get(action_idx)
    return fn(image) if fn is not None else image


# ── Filtros Espaciais ────────────────────────────────────────────────────────

def apply_mean(image, ksize=5):
    return cv2.blur(image, (ksize, ksize))


def apply_gaussian(image, ksize=5, sigma=1.5):
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_median(image, ksize=5):
    return cv2.medianBlur(image, ksize)


# ── Correção de Cor ──────────────────────────────────────────────────────────

def apply_color_correction(image, strength=0.6):
    """
    Remove dominante de cor amarela/sépia balanceando os canais.
    strength controla o quanto corrige (0=nada, 1=correção total).
    Valor parcial evita extrapolar para dominante oposta.
    """
    img_float = image.astype(np.float32)
    means     = img_float.mean(axis=(0, 1))
    overall   = means.mean()

    for c in range(3):
        if means[c] > 1:
            scale = overall / means[c]
            # Interpola entre sem correção (1.0) e correção total (scale)
            scale = 1.0 + (scale - 1.0) * strength
            img_float[:, :, c] *= scale

    return np.clip(img_float, 0, 255).astype(np.uint8)


# ── Morfologia Matemática ────────────────────────────────────────────────────

def apply_morph_close(image, ksize=3):
    """Fechamento: preenche arranhões e buracos finos."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# ── Domínio da Frequência ────────────────────────────────────────────────────

def apply_fourier_bandreject(image, radius_low=8, radius_high=40):
    """Remove textura periódica (padrão do papel fotográfico antigo)."""
    def _filter_channel(ch):
        f       = np.fft.fft2(ch.astype(np.float32))
        fshift  = np.fft.fftshift(f)
        rows, cols = ch.shape
        crow, ccol = rows // 2, cols // 2
        Y, X    = np.ogrid[:rows, :cols]
        dist    = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
        mask    = np.ones((rows, cols), np.float32)
        mask[(dist > radius_low) & (dist < radius_high)] = 0
        result  = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift * mask)))
        return np.clip(result, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        return cv2.merge([_filter_channel(c) for c in cv2.split(image)])
    return _filter_channel(image)


# ── Restauração de Contraste ─────────────────────────────────────────────────

def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Muito melhor que equalização global para fotos desbotadas:
    restaura contraste local sem estourar áreas já claras.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe.apply(image)
