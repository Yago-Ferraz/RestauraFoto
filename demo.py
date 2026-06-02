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

from agent import RestorationAgent
from noise import add_salt_pepper, add_gaussian_noise, add_periodic_noise
from metrics import compute_ssim, compute_psnr
from filters import FILTER_NAMES, STOP_ACTION


# ── Visualizações ────────────────────────────────────────────────────────────

def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def plot_restoration_steps(history, reference_bgr=None, save_path="resultado_passos.png"):
    """Mostra cada passo da restauração + evolução do SSIM."""
    n = len(history)
    has_metrics = any(s["ssim"] is not None for s in history)

    n_rows = 2 if has_metrics else 1
    fig = plt.figure(figsize=(min(4 * (n + 1), 24), 4 * n_rows))
    gs  = gridspec.GridSpec(n_rows, n + (1 if reference_bgr is not None else 0),
                             hspace=0.4, wspace=0.05)

    # Linha 1: imagens de cada passo
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

    # Linha 2: evolução do SSIM
    if has_metrics:
        ax_m = fig.add_subplot(gs[1, :])
        steps_with_ssim = [(i, s) for i, s in enumerate(history) if s["ssim"] is not None]
        xs     = [i for i, _ in steps_with_ssim]
        ssims  = [s["ssim"] for _, s in steps_with_ssim]
        labels = [s["action"] for _, s in steps_with_ssim]

        ax_m.plot(xs, ssims, "b-o", markersize=7, linewidth=1.5)
        ax_m.set_xticks(xs)
        ax_m.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax_m.set_ylabel("SSIM")
        ax_m.set_title("Evolução do SSIM durante a restauração")
        ax_m.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Resultado salvo em '{save_path}'")
    plt.show()


def plot_action_probs(history, save_path="probabilidades.png"):
    """Plota as probabilidades de cada ação ao longo das iterações."""
    steps = [s for s in history[1:] if s["probs"] is not None]
    if not steps:
        return

    data   = np.array([s["probs"] for s in steps])
    iters  = [s["action"] for s in steps]
    n_act  = data.shape[1]
    colors = plt.cm.tab10(np.linspace(0, 1, n_act))

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(steps))
    bottom = np.zeros(len(steps))

    for a in range(n_act):
        ax.bar(x, data[:, a], bottom=bottom, label=FILTER_NAMES[a], color=colors[a])
        bottom += data[:, a]

    ax.set_xticks(x)
    ax.set_xticklabels([f"passo {i+1}\n{name}" for i, name in enumerate(iters)], fontsize=8)
    ax.set_ylabel("Probabilidade")
    ax.set_title("Distribuição de probabilidades por iteração")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Probabilidades salvas em '{save_path}'")
    plt.show()


# ── Modos de demo ─────────────────────────────────────────────────────────────

def demo_synthetic(image_path, model_path):
    """
    Degrada artificialmente uma imagem limpa e testa o agente.
    Permite comparação direta com ground truth via SSIM/PSNR.
    """
    clean_bgr = cv2.imread(str(image_path))
    if clean_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    clean_bgr = cv2.resize(clean_bgr, (512, 512))

    # Degradação composta
    degraded = add_gaussian_noise(clean_bgr, sigma=20)
    degraded = add_salt_pepper(degraded, amount=0.04)

    print("=== Demo Sintético ===")
    print("Degradação: ruído gaussiano + sal-e-pimenta\n")

    agent = RestorationAgent(model_path)
    restored, history = agent.restore(degraded, max_iterations=8,
                                       reference_bgr=clean_bgr, verbose=True)

    # Métricas finais
    init_ssim = compute_ssim(cv2.cvtColor(degraded,  cv2.COLOR_BGR2RGB),
                              cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
    final_ssim = compute_ssim(cv2.cvtColor(restored,  cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
    init_psnr  = compute_psnr(cv2.cvtColor(degraded,  cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))
    final_psnr = compute_psnr(cv2.cvtColor(restored,  cv2.COLOR_BGR2RGB),
                               cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2RGB))

    print(f"\n{'':─<40}")
    print(f"  SSIM inicial  : {init_ssim:.4f}")
    print(f"  SSIM final    : {final_ssim:.4f}  (+{final_ssim - init_ssim:.4f})")
    print(f"  PSNR inicial  : {init_psnr:.2f} dB")
    print(f"  PSNR final    : {final_psnr:.2f} dB  (+{final_psnr - init_psnr:.2f} dB)")
    print(f"{'':─<40}")
    print(f"  Filtros aplicados: {[s['action'] for s in history[1:]]}")

    plot_restoration_steps(history, reference_bgr=clean_bgr)
    plot_action_probs(history)


def demo_real(image_path, model_path):
    """
    Restaura uma fotografia histórica real (sem ground truth).
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    print("=== Demo Foto Real ===\n")

    agent = RestorationAgent(model_path)
    restored, history = agent.restore(image_bgr, max_iterations=8, verbose=True)

    print(f"\nFiltros aplicados: {[s['action'] for s in history[1:]]}")

    # Comparação lado a lado
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(_bgr_to_rgb(image_bgr))
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(restored))
    n_steps = len(history) - 1
    axes[1].set_title(f"Restaurada ({n_steps} filtro{'s' if n_steps != 1 else ''} aplicado{'s' if n_steps != 1 else ''})", fontsize=12)
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("resultado_real.png", dpi=150, bbox_inches="tight")
    print("Resultado salvo em 'resultado_real.png'")
    plt.show()

    plot_action_probs(history)


# ── Entrada ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo: restauração de fotos históricas")
    parser.add_argument("image", help="Caminho para a imagem de entrada")
    parser.add_argument("--model", default="model.pth", help="Caminho para o modelo treinado")
    parser.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                        help="'synthetic': degrada e restaura com ground truth | 'real': restaura foto antiga diretamente")
    args = parser.parse_args()

    if args.mode == "synthetic":
        demo_synthetic(args.image, args.model)
    else:
        demo_real(args.image, args.model)
