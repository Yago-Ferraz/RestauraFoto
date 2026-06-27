"""
Demo do agente de restauração iterativa de fotografias históricas.

Modos:
  synthetic  — aplica degradação conhecida em imagem limpa, restaura e compara com ground truth
  real       — restaura uma foto antiga real (sem referência de ground truth)

Uso:
    python demo.py imagem.jpg --mode synthetic
    python demo.py foto_antiga.jpg --mode real
"""

import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

from agent import RestorationAgent
from noise import add_salt_pepper, add_gaussian_noise, add_historical_photo_degradation
from metrics import compute_ssim, compute_psnr
from filters import FILTER_NAMES

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)


def _stamp(prefix):
    """Gera nome de arquivo com timestamp para não sobrescrever resultados anteriores."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"{prefix}_{ts}"


def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -- Visualizações -------------------------------------------------------------

def plot_restoration_steps(history, reference_bgr=None, base_path=None):
    n        = len(history)
    has_ssim = any(s["ssim"] is not None for s in history)
    n_cols   = n + (1 if reference_bgr is not None else 0)
    n_rows   = 2 if has_ssim else 1

    fig = plt.figure(figsize=(min(4 * n_cols, 28), 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.05)

    for i, step in enumerate(history):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(_bgr_to_rgb(step["image"]))
        title = step["action"]
        if step["ssim"] is not None:
            title += f"\nSSIM {step['ssim']:.3f}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")

    if reference_bgr is not None:
        ax_ref = fig.add_subplot(gs[0, n])
        ax_ref.imshow(_bgr_to_rgb(reference_bgr))
        ax_ref.set_title("Ground Truth", fontsize=8)
        ax_ref.axis("off")

    if has_ssim:
        ax_m   = fig.add_subplot(gs[1, :])
        valid  = [(i, s) for i, s in enumerate(history) if s["ssim"] is not None]
        xs     = [i for i, _ in valid]
        ssims  = [s["ssim"] for _, s in valid]
        labels = [s["action"] for _, s in valid]
        ax_m.plot(xs, ssims, "b-o", markersize=7, linewidth=1.5)
        ax_m.set_xticks(xs)
        ax_m.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_m.set_ylabel("SSIM")
        ax_m.set_title("Evolução do SSIM durante a restauração")
        ax_m.grid(True, alpha=0.3)

    path = f"{base_path}_passos.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Passos salvos em '{path}'")
    plt.show()


def plot_action_probs(history, base_path=None):
    steps = [s for s in history[1:] if s["probs"] is not None]
    if not steps:
        return

    data   = np.array([s["probs"] for s in steps])
    iters  = [s["action"] for s in steps]
    n_act  = data.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_act))

    fig, ax = plt.subplots(figsize=(10, 4))
    bottom  = np.zeros(len(steps))
    for a in range(n_act):
        ax.bar(np.arange(len(steps)), data[:, a], bottom=bottom,
               label=FILTER_NAMES[a], color=colors[a])
        bottom += data[:, a]

    ax.set_xticks(np.arange(len(steps)))
    ax.set_xticklabels([f"passo {i+1}\n{name}" for i, name in enumerate(iters)], fontsize=8)
    ax.set_ylabel("Probabilidade")
    ax.set_title("Distribuição de probabilidades por iteração")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1)

    path = f"{base_path}_probs.png"
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Probabilidades salvas em '{path}'")
    plt.show()


# -- Modos de demo -------------------------------------------------------------

def demo_synthetic(image_path, model_path):
    clean_bgr = cv2.imread(str(image_path))
    if clean_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")
    clean_bgr = cv2.resize(clean_bgr, (512, 512))

    degraded = add_historical_photo_degradation(clean_bgr)

    print("=== Demo Sintético ===")
    print("Degradação: histórica composta (sépia, fading, manchas)\n")

    agent = RestorationAgent(model_path)
    restored, history = agent.restore(degraded, max_iterations=4, min_steps=0,
                                       reference_bgr=clean_bgr, verbose=True)

    init_ssim  = compute_ssim(_bgr_to_rgb(degraded),  _bgr_to_rgb(clean_bgr))
    final_ssim = compute_ssim(_bgr_to_rgb(restored),  _bgr_to_rgb(clean_bgr))
    init_psnr  = compute_psnr(_bgr_to_rgb(degraded),  _bgr_to_rgb(clean_bgr))
    final_psnr = compute_psnr(_bgr_to_rgb(restored),  _bgr_to_rgb(clean_bgr))

    print(f"\n{'':-<40}")
    print(f"  SSIM inicial  : {init_ssim:.4f}")
    print(f"  SSIM final    : {final_ssim:.4f}  (+{final_ssim - init_ssim:.4f})")
    print(f"  PSNR inicial  : {init_psnr:.2f} dB")
    print(f"  PSNR final    : {final_psnr:.2f} dB  (+{final_psnr - init_psnr:.2f} dB)")
    print(f"{'':-<40}")
    print(f"  Filtros aplicados: {[s['action'] for s in history[1:]]}")

    base = _stamp("synthetic")
    cv2.imwrite(f"{base}_original.png", clean_bgr)
    cv2.imwrite(f"{base}_degradada.png", degraded)
    cv2.imwrite(f"{base}_restaurada.png", restored)
    plot_restoration_steps(history, reference_bgr=clean_bgr, base_path=base)
    plot_action_probs(history, base_path=base)


def demo_real(image_path, model_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    print("=== Demo Foto Real ===\n")

    agent = RestorationAgent(model_path)
    restored, history = agent.restore(image_bgr, max_iterations=4, min_steps=0, verbose=True)

    print(f"\nFiltros aplicados: {[s['action'] for s in history[1:]]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(_bgr_to_rgb(image_bgr))
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")
    n_steps = len(history) - 1
    axes[1].imshow(_bgr_to_rgb(restored))
    axes[1].set_title(f"Restaurada ({n_steps} filtro{'s' if n_steps != 1 else ''} aplicado{'s' if n_steps != 1 else ''})",
                      fontsize=12)
    axes[1].axis("off")

    base = _stamp("real")
    cv2.imwrite(f"{base}_original.png", image_bgr)
    cv2.imwrite(f"{base}_restaurada.png", restored)

    plt.tight_layout()
    comparacao = f"{base}_comparacao.png"
    plt.savefig(comparacao, dpi=150, bbox_inches="tight")
    print(f"Comparação salva em '{comparacao}'")
    plt.show()

    plot_action_probs(history, base_path=base)


# -- Entrada -------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: restauração de fotos históricas")
    parser.add_argument("image", help="Caminho para a imagem de entrada")
    parser.add_argument("--model", default="model.pth")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()

    if args.mode == "synthetic":
        demo_synthetic(args.image, args.model)
    else:
        demo_real(args.image, args.model)
