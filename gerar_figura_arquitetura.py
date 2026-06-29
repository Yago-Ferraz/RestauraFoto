"""
Gera a Figura 1 — Fluxo de processamento do RestauraFoto.
Salva como figura1_arquitetura.png na pasta do projeto.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(12, 7.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7.5)
ax.axis("off")
fig.patch.set_facecolor("white")

# ── Paleta ────────────────────────────────────────────────────────────────────
C_INPUT   = "#dbeafe"   # azul claro  — entrada/saída
C_MODEL   = "#ede9fe"   # roxo claro  — modelo
C_AGENT   = "#dcfce7"   # verde claro — agente
C_FILTER  = "#fef9c3"   # amarelo     — filtros
C_STOP    = "#fee2e2"   # vermelho    — STOP
BORDER    = "#334155"
ARROW     = "#475569"

def box(ax, x, y, w, h, label, sublabel=None, color="#f8fafc", fontsize=10):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.07",
                          linewidth=1.5, edgecolor=BORDER,
                          facecolor=color, zorder=3)
    ax.add_patch(rect)
    dy = 0.12 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight="bold",
            color="#1e293b", zorder=4, wrap=True)
    if sublabel:
        ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                fontsize=8, color="#64748b", zorder=4)

def arrow(ax, x1, y1, x2, y2, label=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=ARROW,
                                lw=1.5, mutation_scale=16),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.12, my, label, fontsize=7.5, color="#475569",
                ha="left", va="center", style="italic")

# ── Layout: coluna central vertical ──────────────────────────────────────────
#  y positions (top → bottom)
Y = [6.9, 5.9, 4.8, 3.7, 2.65, 1.55, 0.55]
X = 6.0   # centro

# 0. Foto degradada
box(ax, X, Y[0], 3.2, 0.7, "Foto Degradada (entrada)", color=C_INPUT)

# seta
arrow(ax, X, Y[0]-0.35, X, Y[1]+0.40, "resize 224×224\nnormalização ImageNet")

# 1. Backbone
box(ax, X, Y[1], 4.2, 0.72,
    "ResNet-18  Backbone",
    "4 blocos residuais → Global Avg Pooling → vetor 512 features",
    color=C_MODEL, fontsize=10)

arrow(ax, X, Y[1]-0.36, X, Y[2]+0.36, "512 features")

# 2. Cabeça
box(ax, X, Y[2], 4.8, 0.72,
    "Cabeça de Classificação",
    "Linear(512→256) → ReLU → Dropout(0.4) → Linear(256→21)",
    color=C_MODEL, fontsize=10)

arrow(ax, X, Y[2]-0.36, X, Y[3]+0.36, "21 logits → Softmax")

# 3. Agente
box(ax, X, Y[3], 4.2, 0.72,
    "Agente Iterativo",
    "seleciona ação de maior prob. (com guardas de segurança)",
    color=C_AGENT, fontsize=10)

# seta para STOP (direita) e para filtros (baixo)
arrow(ax, X + 2.1, Y[3], 9.8, Y[3])   # para STOP
arrow(ax, X, Y[3]-0.36, X, Y[4]+0.38)

# STOP box (lado direito)
box(ax, 10.5, Y[3], 1.6, 0.60, "STOP", color=C_STOP, fontsize=11)
ax.text(8.05, Y[3]+0.14, "prob. máx.\n= STOP", fontsize=7.5,
        color="#475569", ha="center", va="center", style="italic")

# 4. Filtros
box(ax, X, Y[4], 5.2, 0.72,
    "Camada de Filtros  (20 opções)",
    "gaussiano · bilateral · mediana · CLAHE · gamma · log · fourier · ...",
    color=C_FILTER, fontsize=10)

arrow(ax, X, Y[4]-0.36, X, Y[5]+0.36, "imagem filtrada")

# 5. Verificação SSIM
box(ax, X, Y[5], 4.2, 0.72,
    "Verificação de Qualidade",
    "SSIM consecutivo · SSIM vs. entrada · limite de blur",
    color=C_AGENT, fontsize=10)

# seta de retorno (esquerda)
ax.annotate("", xy=(X - 2.75, Y[1]),
            xytext=(X - 2.75, Y[5]),
            arrowprops=dict(arrowstyle="-|>", color="#7c3aed",
                            lw=1.8, mutation_scale=16,
                            connectionstyle="arc3,rad=0"),
            zorder=2)
ax.plot([X - 2.1, X - 2.75], [Y[5], Y[5]], color="#7c3aed", lw=1.8, zorder=2)
ax.plot([X - 2.75, X - 2.1], [Y[1], Y[1]], color="#7c3aed", lw=1.8, zorder=2)
ax.text(X - 3.55, (Y[1]+Y[5])/2, "próxima\niteração", fontsize=8,
        color="#7c3aed", ha="center", va="center", fontweight="bold")

arrow(ax, X + 2.1, Y[5], 9.8, Y[5])   # seta para STOP (baixo)
box(ax, 10.5, Y[5], 1.6, 0.60, "STOP\n(SSIM)", color=C_STOP, fontsize=9)
ax.text(8.05, Y[5]+0.14, "qualidade\ndegradou", fontsize=7.5,
        color="#475569", ha="center", va="center", style="italic")

arrow(ax, X, Y[5]-0.36, X, Y[6]+0.36)

# 6. Saída
box(ax, X, Y[6], 3.2, 0.70, "Foto Restaurada (saída)", color=C_INPUT)

# ── Legenda ───────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color=C_INPUT,  label="Entrada / Saída"),
    mpatches.Patch(color=C_MODEL,  label="Modelo ResNet-18"),
    mpatches.Patch(color=C_AGENT,  label="Agente / Verificação"),
    mpatches.Patch(color=C_FILTER, label="Filtros de imagem"),
    mpatches.Patch(color=C_STOP,   label="Condição de parada"),
]
ax.legend(handles=legend_items, loc="lower left",
          bbox_to_anchor=(0.01, 0.01), fontsize=8.5,
          framealpha=0.9, edgecolor=BORDER)

ax.set_title("Figura 1 – Fluxo de processamento do RestauraFoto",
             fontsize=11, pad=8, color="#1e293b")

plt.tight_layout()
plt.savefig("figura1_arquitetura.png", dpi=150, bbox_inches="tight",
            facecolor="white")
print("Figura salva: figura1_arquitetura.png")
