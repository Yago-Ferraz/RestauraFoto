"""
Baixa o DIV2K — 800 fotos em resolução 2K, padrão acadêmico para restauração.
As imagens são salvas em data/clean/ como arquivos PNG.

Uso:
    python download_data.py
"""

import urllib.request
import zipfile
from pathlib import Path
from PIL import Image

DIV2K_URL = "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip"
TMP_ZIP   = Path("data/DIV2K_train_HR.zip")
DIV2K_DIR = Path("data/DIV2K_train_HR")
OUT_DIR   = Path("data/clean")


def _progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(count * block_size * 100 // total_size, 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%", end="", flush=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_ZIP.parent.mkdir(parents=True, exist_ok=True)

    # ── Download ──────────────────────────────────────────────────────────────
    if not TMP_ZIP.exists():
        print("Baixando DIV2K (~3.3 GB) — pode demorar alguns minutos...\n")
        urllib.request.urlretrieve(DIV2K_URL, TMP_ZIP, reporthook=_progress)
        print("\nDownload concluído.")
    else:
        print("Arquivo já baixado, pulando download.")

    # ── Extração ──────────────────────────────────────────────────────────────
    if not DIV2K_DIR.exists():
        print("Extraindo...")
        with zipfile.ZipFile(TMP_ZIP, "r") as zf:
            zf.extractall("data/")
        print("Extração concluída.")

    # ── Remove imagens antigas e copia DIV2K ─────────────────────────────────
    removed = sum(1 for f in OUT_DIR.glob("*.png") if f.unlink() is None)
    if removed:
        print(f"Removidas {removed} imagens antigas.")

    print("Copiando para data/clean/...")
    copied = 0
    for src in sorted(DIV2K_DIR.glob("*.png")):
        Image.open(src).convert("RGB").save(OUT_DIR / f"div2k_{copied:04d}.png")
        copied += 1
        if copied % 100 == 0:
            print(f"  {copied}/800")

    print(f"\nPronto! {copied} imagens em '{OUT_DIR}/'")
    print("Agora rode: python train.py")


if __name__ == "__main__":
    main()
