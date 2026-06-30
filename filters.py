import cv2
import numpy as np

# Cobertura completa das técnicas da disciplina de Processamento de Imagens:
#   Filtragem espacial passa-baixa  → mean, gaussian_soft, gaussian_strong, bilateral
#   Filtragem não-linear            → median
#   Filtragem passa-alta            → high_pass
#   Transformações de intensidade   → gamma_bright, log_transform, contrast_stretch
#   Histograma                      → clahe
#   Correção de cor                 → color_correction
#   Morfologia matemática           → morph_open, morph_close
#   Domínio da frequência           → fourier

FILTER_NAMES = [
    "mean",                    #  0 — Filtro da Média ksize=5
    "gaussian_soft",           #  1 — Gaussiano σ=0.8  (leve)
    "gaussian_medium",         #  2 — Gaussiano σ=1.5  (médio)
    "gaussian_strong",         #  3 — Gaussiano σ=2.5  (forte)
    "bilateral",               #  4 — Bilateral d=9 (preserva bordas)
    "median_fine",             #  5 — Mediana ksize=3  (sal e pimenta leve)
    "median_strong",           #  6 — Mediana ksize=7  (sal e pimenta forte)
    "high_pass_gentle",        #  7 — Passa-Alta suave  (α=0.4, nitidez leve)
    "high_pass_strong",        #  8 — Passa-Alta forte  (α=1.0, nitidez agressiva)
    "color_correction_gentle", #  9 — Correção de cor suave  (cast leve, strength=0.25)
    "color_correction_strong", # 10 — Correção de cor forte  (sépia intensa, strength=0.65)
    "clahe_subtle",            # 11 — CLAHE suave  (clip=1.5, contraste moderado)
    "clahe_strong",            # 12 — CLAHE forte  (clip=3.0, contraste agressivo)
    "gamma_bright",            # 13 — Potência γ=0.6  (aclara moderado)
    "gamma_very_bright",       # 14 — Potência γ=0.35 (aclara forte, fotos muito escuras)
    "log_transform",           # 15 — Logarítmica (realça detalhes nas sombras)
    "contrast_stretch",        # 16 — Alargamento de Contraste linear
    "morph_open",              # 17 — Abertura Morfológica (remove manchas/poeira)
    "morph_close",             # 18 — Fechamento Morfológico (preenche fissuras)
    "fourier",                 # 19 — Rejeita-Banda de Fourier (textura periódica)
    "STOP",                    # 20
]

N_FILTERS   = len(FILTER_NAMES) - 1   # 20
STOP_ACTION = len(FILTER_NAMES) - 1   # 20


def apply_filter(image, action_idx):
    dispatch = {
        0:  apply_mean,
        1:  lambda img: apply_gaussian(img, ksize=3, sigma=0.8),
        2:  lambda img: apply_gaussian(img, ksize=5, sigma=1.5),
        3:  lambda img: apply_gaussian(img, ksize=9, sigma=2.5),
        4:  apply_bilateral,
        5:  lambda img: apply_median(img, ksize=3),
        6:  lambda img: apply_median(img, ksize=7),
        7:  lambda img: apply_high_pass(img, strength=0.4),
        8:  lambda img: apply_high_pass(img, strength=1.0),
        9:  lambda img: apply_color_correction(img, strength=0.25),
        10: lambda img: apply_color_correction(img, strength=0.65),
        11: lambda img: apply_clahe(img, clip_limit=1.5),
        12: lambda img: apply_clahe(img, clip_limit=3.0),
        13: lambda img: apply_gamma(img, gamma=0.60),
        14: lambda img: apply_gamma(img, gamma=0.35),
        15: apply_log_transform,
        16: apply_contrast_stretch,
        17: apply_morph_open,
        18: apply_morph_close,
        19: apply_fourier_bandreject,
    }
    fn = dispatch.get(action_idx)
    return fn(image) if fn is not None else image


# ── 1. Filtragem Espacial Passa-Baixa ────────────────────────────────────────

def apply_mean(image, ksize=5):
    """Filtro da Média: suavização por corte seco (box filter)."""
    return cv2.blur(image, (ksize, ksize))


def apply_gaussian(image, ksize=5, sigma=1.5):
    """Filtro Gaussiano: suavização com peso decrescente pelo desvio padrão σ."""
    return cv2.GaussianBlur(image, (ksize, ksize), sigma)


def apply_bilateral(image, d=9, sigma_color=75, sigma_space=75):
    """Filtro Bilateral: suavização que preserva bordas — melhor para granulado."""
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


# ── 2. Filtragem Não-Linear ───────────────────────────────────────────────────

def apply_median(image, ksize=5):
    """Filtro da Mediana: excelente para ruído impulsivo (sal e pimenta)."""
    return cv2.medianBlur(image, ksize)


# ── 3. Filtragem Passa-Alta ───────────────────────────────────────────────────

def apply_high_pass(image, strength=0.7):
    """
    Filtro Passa-Alta via Unsharp Mask.
    Subtrai a versão suavizada e adiciona de volta à original com ganho,
    realçando as altas frequências (bordas e detalhes finos).
    resultado = (1 + α) × original − α × gaussiana
    """
    blurred = cv2.GaussianBlur(image, (5, 5), 1.0)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


# ── 4. Transformações de Intensidade (Domínio Espacial) ──────────────────────

def apply_gamma(image, gamma=0.5):
    """
    Transformação de Potência (Power-Law): s = c × r^γ
    γ < 1 → aclara (fotos escuras); γ > 1 → escurece.
    Usa tabela LUT para eficiência.
    """
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in range(256)], dtype=np.uint8)
    return cv2.LUT(image, table)


def apply_log_transform(image):
    """
    Transformação Logarítmica: s = c × log(1 + r)
    Comprime as altas intensidades e expande as baixas,
    revelando detalhes perdidos nas sombras de fotos escuras.
    """
    img_float = image.astype(np.float32) + 1.0
    log_img   = np.log(img_float)
    c         = 255.0 / np.log(256.0)
    return np.clip(c * log_img, 0, 255).astype(np.uint8)


def apply_contrast_stretch(image, low_pct=2, high_pct=98):
    """
    Alargamento de Contraste (Contrast Stretching).
    Normalização linear do histograma: expande o intervalo de intensidades
    de [p_low, p_high] para [0, 255], corrigindo fotos desbotadas/lavadas.
    """
    result = np.zeros_like(image)
    for c in range(image.shape[2] if image.ndim == 3 else 1):
        ch     = image[:, :, c] if image.ndim == 3 else image
        p_low  = np.percentile(ch, low_pct)
        p_high = np.percentile(ch, high_pct)
        if p_high > p_low:
            stretched = (ch.astype(np.float32) - p_low) / (p_high - p_low) * 255.0
            out = np.clip(stretched, 0, 255).astype(np.uint8)
        else:
            out = ch
        if image.ndim == 3:
            result[:, :, c] = out
        else:
            result = out
    return result


# ── 5. Histograma ─────────────────────────────────────────────────────────────

def apply_clahe(image, clip_limit=2.0, tile_size=8):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Equalização de histograma por regiões com limite de amplificação —
    restaura contraste local sem estourar áreas já claras.
    Trabalha no canal L do espaço LAB para não distorcer a cor.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(tile_size, tile_size))
    if image.ndim == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return clahe.apply(image)


# ── 6. Processamento de Cor ───────────────────────────────────────────────────

def apply_color_correction(image, strength=0.35):
    """
    Correção de dominante de cor via balanceamento de canais.
    Detecta o canal com média mais alta (dominante sépia = canal R elevado)
    e escala todos os canais para a média global. strength=0.35 é conservador
    para evitar inversão de dominante.
    """
    img_float = image.astype(np.float32)
    means     = img_float.mean(axis=(0, 1))
    overall   = means.mean()
    for c in range(3):
        if means[c] > 1:
            scale = overall / means[c]
            scale = 1.0 + (scale - 1.0) * strength
            img_float[:, :, c] *= scale
    return np.clip(img_float, 0, 255).astype(np.uint8)


# ── 7. Morfologia Matemática ──────────────────────────────────────────────────

def apply_morph_open(image, ksize=3):
    """
    Abertura Morfológica (Erosão → Dilatação).
    Remove pequenos objetos claros (manchas de poeira, foxing, grãos de prata)
    sem alterar as regiões maiores da imagem.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def apply_morph_close(image, ksize=3):
    """
    Fechamento Morfológico (Dilatação → Erosão).
    Preenche pequenos buracos escuros e fissuras finas (arranhões no papel)
    sem alterar as regiões intactas ao redor.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


# ── 8. Domínio da Frequência ─────────────────────────────────────────────────

def apply_fourier_bandreject(image, radius_low=8, radius_high=40):
    """
    Filtro Rejeita-Banda no Domínio da Frequência (Fourier).
    Implementação conforme o pipeline da disciplina:
      1. Padding para 2M × 2N (evita aliasing/wrap-around)
      2. Multiplicação por (-1)^(x+y) para centralizar o espectro
      3. DFT → H(u,v) (máscara rejeita-banda) → produto → IDFT
    Remove texturas periódicas do papel fotográfico antigo.
    """
    def _filter_channel(ch):
        h, w   = ch.shape
        ph, pw = 2 * h, 2 * w

        # 1. Padding com zeros (P=2M, Q=2N)
        padded = np.zeros((ph, pw), dtype=np.float32)
        padded[:h, :w] = ch.astype(np.float32)

        # 2. Centralizar: multiplicar por (-1)^(x+y)
        x, y   = np.meshgrid(np.arange(pw), np.arange(ph))
        padded *= ((-1) ** (x + y)).astype(np.float32)

        # 3. DFT
        F = np.fft.fft2(padded)

        # 4. Construir H(u,v) — máscara rejeita-banda simétrica
        crow, ccol = ph // 2, pw // 2
        Y, X  = np.ogrid[:ph, :pw]
        dist  = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
        H     = np.ones((ph, pw), np.float32)
        H[(dist > radius_low) & (dist < radius_high)] = 0

        # 5-6. Produto no domínio da frequência e IDFT
        result = np.abs(np.fft.ifft2(F * H))

        # Desfaz centralização e remove padding
        result *= ((-1) ** (x + y)).astype(np.float32)
        result = result[:h, :w]
        return np.clip(result, 0, 255).astype(np.uint8)

    if image.ndim == 3:
        return cv2.merge([_filter_channel(c) for c in cv2.split(image)])
    return _filter_channel(image)
