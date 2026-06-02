import numpy as np
import cv2


# ── Degradações genéricas ────────────────────────────────────────────────────

def add_gaussian_noise(image, sigma=25):
    noise = np.random.normal(0, sigma, image.shape).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(image, amount=0.05):
    noisy = image.copy()
    h, w  = image.shape[:2]
    n     = int(amount * h * w)

    ys, xs = np.random.randint(0, h, n), np.random.randint(0, w, n)
    noisy[ys[:n//2], xs[:n//2]] = 255
    noisy[ys[n//2:], xs[n//2:]] = 0
    return noisy


def add_periodic_noise(image, freq=10, amplitude=35):
    rows, cols = image.shape[:2]
    pattern = amplitude * np.sin(2 * np.pi * freq * np.arange(cols) / cols)
    pattern = pattern[np.newaxis, :].astype(np.int16)
    if image.ndim == 3:
        pattern = pattern[:, :, np.newaxis]
    return np.clip(image.astype(np.int16) + pattern, 0, 255).astype(np.uint8)


def add_scratches(image, n_scratches=5):
    noisy = image.copy()
    h, w  = image.shape[:2]
    for _ in range(n_scratches):
        x      = np.random.randint(0, w)
        length = np.random.randint(h // 4, h // 2)
        y0     = np.random.randint(0, h - length)
        color  = (255, 255, 255) if image.ndim == 3 else 255
        cv2.line(noisy, (x, y0), (x, y0 + length), color, np.random.randint(1, 3))
    return noisy


# ── Degradações de foto histórica ────────────────────────────────────────────

def add_sepia_tone(image, intensity=0.7):
    """Amarelamento/sépia por oxidação química do papel."""
    gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sepia = np.zeros_like(image, dtype=np.float32)
    sepia[:, :, 2] = np.clip(gray * 1.08, 0, 255)   # canal R (mais alto)
    sepia[:, :, 1] = np.clip(gray * 0.88, 0, 255)   # canal G (médio)
    sepia[:, :, 0] = np.clip(gray * 0.68, 0, 255)   # canal B (reduzido)
    result = cv2.addWeighted(image.astype(np.float32), 1 - intensity,
                             sepia, intensity, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def add_fading(image, factor=0.65):
    """Desbotamento: levanta os pretos e comprime o contraste."""
    faded = image.astype(np.float32) * factor + 255 * (1 - factor) * 0.45
    return np.clip(faded, 0, 255).astype(np.uint8)


def add_vignette(image, strength=0.55):
    """Escurecimento das bordas (envelhecimento das extremidades do papel)."""
    h, w  = image.shape[:2]
    Y, X  = np.ogrid[:h, :w]
    dist  = np.sqrt(((X - w / 2) / (w / 2)) ** 2 + ((Y - h / 2) / (h / 2)) ** 2)
    mask  = np.clip(1 - dist * strength, 0.2, 1.0)
    if image.ndim == 3:
        mask = mask[:, :, np.newaxis]
    return np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def add_age_spots(image, n_spots=8):
    """Manchas marrons de envelhecimento (foxing)."""
    result = image.copy().astype(np.float32)
    h, w   = image.shape[:2]
    Y, X   = np.ogrid[:h, :w]

    for _ in range(n_spots):
        cx  = np.random.randint(0, w)
        cy  = np.random.randint(0, h)
        r   = np.random.randint(4, 25)
        amt = np.random.uniform(0.3, 0.8)
        blob = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * r ** 2))
        if image.ndim == 3:
            result[:, :, 0] -= blob * 70 * amt   # menos azul
            result[:, :, 1] -= blob * 25 * amt   # menos verde

    return np.clip(result, 0, 255).astype(np.uint8)


def add_film_grain(image, sigma=12):
    """Granulado de filme: ruído com padrão mais orgânico que gaussiano puro."""
    grain = np.random.normal(0, sigma, image.shape[:2]).astype(np.float32)
    # Aplica o mesmo grão nos 3 canais (comportamento de filme preto-e-branco digitalizado)
    if image.ndim == 3:
        grain = grain[:, :, np.newaxis]
    return np.clip(image.astype(np.float32) + grain, 0, 255).astype(np.uint8)


# ── Composição realista de foto histórica ─────────────────────────────────────

def add_historical_photo_degradation(image):
    """
    Simula o conjunto de degradações típicas de uma fotografia histórica.
    Combinações aleatórias para variedade no dataset de treino.
    """
    result = image.copy()

    # Desbotamento e sépia são quase universais em fotos antigas
    result = add_fading(result,     factor=np.random.uniform(0.5, 0.8))
    result = add_sepia_tone(result, intensity=np.random.uniform(0.3, 0.85))

    # Vinheta (comum em fotografias do séc. XIX/XX)
    if np.random.random() < 0.65:
        result = add_vignette(result, strength=np.random.uniform(0.3, 0.7))

    # Manchas de envelhecimento
    if np.random.random() < 0.55:
        result = add_age_spots(result, n_spots=np.random.randint(3, 15))

    # Granulado de filme
    if np.random.random() < 0.75:
        result = add_film_grain(result, sigma=np.random.uniform(5, 18))

    # Arranhões físicos
    if np.random.random() < 0.4:
        result = add_scratches(result, n_scratches=np.random.randint(1, 6))

    # Textura periódica do papel
    if np.random.random() < 0.3:
        result = add_periodic_noise(result,
                                    freq=np.random.randint(8, 20),
                                    amplitude=np.random.randint(10, 25))

    return result


def add_compound_noise(image):
    """Degradação composta aleatória — mantida para compatibilidade."""
    return add_historical_photo_degradation(image)
