"""
Baixa o BSDS300 (Berkeley Segmentation Dataset) — 300 fotos naturais nítidas
em resolução ~320x480, padrão acadêmico para tarefas de restauração de imagens.
As imagens são salvas em data/clean/ como arquivos PNG.

Uso:
    python download_data.py
"""

import urllib.request
import tarfile
import shutil
from pathlib import Path
from PIL import Image
import numpy as np

BSDS_URL  = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
TMP_FILE  = Path("data/BSDS300-images.tgz")
BSDS_DIR  = Path("data/BSDS300")
OUT_DIR   = Path("data/clean")


def _progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    print(f"\r  Baixando... {pct}%", end="", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ── Download ─────────────────────────────────────────────────────────────
    if not TMP_FILE.exists():
        print(f"Baixando BSDS300 (~50 MB)...")
        urllib.request.urlretrieve(BSDS_URL, TMP_FILE, reporthook=_progress)
        print()
    else:
        print("Arquivo já baixado, pulando download.")

    # ── Extração ──────────────────────────────────────────────────────────────
    if not BSDS_DIR.exists():
        print("Extraindo...")
        with tarfile.open(TMP_FILE, "r:gz") as tar:
            tar.extractall("data/")
        # O tgz extrai para data/BSDS300/
        print("Extração concluída.")

    # ── Copia imagens para data/clean/ ────────────────────────────────────────
    # Limpa imagens antigas do CIFAR-10 (img_00000.png ... img_01999.png)
    removed = 0
    for f in OUT_DIR.glob("img_?????.png"):
        f.unlink()
        removed += 1
    if removed:
        print(f"Removidas {removed} imagens antigas do CIFAR-10.")

    # Copia as imagens do BSDS300 (pasta train/ e test/)
    src_dirs = list(BSDS_DIR.rglob("images")) or [BSDS_DIR]
    copied = 0
    for src_dir in BSDS_DIR.rglob("*.jpg"):
        dest = OUT_DIR / f"bsds_{copied:04d}.png"
        img = Image.open(src_dir).convert("RGB")
        img.save(dest)
        copied += 1

    print(f"Copiadas {copied} imagens nítidas para '{OUT_DIR}/'")
    print("Pronto! Agora rode: python train.py")


if __name__ == "__main__":
    main()
