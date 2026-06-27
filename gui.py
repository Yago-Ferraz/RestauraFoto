"""
RestauraFoto — Interface Gráfica
Restauração iterativa de fotografias históricas com agente ResNet-18.

Uso:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRect
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ── Importações do projeto ────────────────────────────────────────────────────
try:
    from agent import RestorationAgent
    from noise import add_historical_photo_degradation
    from filters import FILTER_NAMES
    from metrics import compute_ssim, compute_psnr
except ImportError as e:
    import sys
    print(f"Erro ao importar módulos do projeto: {e}")
    sys.exit(1)

MODEL_PATH   = "model.pth"
DISPLAY_SIZE = (480, 360)   # tamanho de exibição de cada imagem na GUI
BG_COLOR     = "#1e1e2e"
PANEL_COLOR  = "#2a2a3e"
ACCENT       = "#7c3aed"
TEXT_COLOR   = "#e2e8f0"
TEXT_DIM     = "#94a3b8"
SUCCESS      = "#22c55e"
WARNING      = "#f59e0b"


def _bgr_to_photoimage(img_bgr, size=DISPLAY_SIZE):
    """Converte imagem BGR do OpenCV para PhotoImage do Tkinter."""
    rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil   = Image.fromarray(rgb)
    pil   = pil.resize(size, Image.LANCZOS)
    return ImageTk.PhotoImage(pil)


def _no_ref_quality(img_bgr):
    """
    Score de qualidade estimado (0–100) sem imagem de referência.

    Combina três componentes perceptuais independentes:
      - Nitidez     (Laplacian variance — sobe com high_pass/deblur)
      - Brilho      (proximidade ao ponto médio 128 — sobe com gamma/log)
      - Neutralidade de cor (baixo desvio entre canais — sobe com color_correction)

    Sobe quando a restauração melhora a foto; cai se um filtro piorar.
    """
    f    = img_bgr.astype(np.float32)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Nitidez: variância do Laplaciano (limiar ~600 para foto nítida típica)
    lap_var     = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharp_score = min(1.0, lap_var / 600.0)

    # Brilho: ideal ~128 em 8-bit; margem de tolerância de ±90
    brightness   = float(f.mean())
    bright_score = max(0.0, 1.0 - abs(brightness - 128.0) / 90.0)

    # Neutralidade de cor: desvio-padrão entre médias B/G/R; 35 = cast forte
    means       = f.mean(axis=(0, 1))
    color_score = max(0.0, 1.0 - float(np.std(means)) / 35.0)

    return (0.35 * sharp_score + 0.25 * bright_score + 0.40 * color_score) * 100.0


class RestauraFotoApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RestauraFoto — Restauração de Fotografias Históricas")
        self.configure(bg=BG_COLOR)
        self.resizable(True, True)

        self._agent        = None
        self._image_bgr    = None   # imagem de entrada carregada
        self._result_bgr   = None   # imagem restaurada
        self._photo_orig   = None   # referência PhotoImage (evita garbage collection)
        self._photo_rest   = None
        self._btn_dl_orig  = None   # botão salvar imagem original
        self._btn_dl_rest  = None   # botão salvar imagem restaurada

        self._build_ui()
        self._try_load_model()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_viewer()
        self._build_statusbar()

    def _build_sidebar(self):
        side = tk.Frame(self, bg=PANEL_COLOR, width=220)
        side.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=10)
        side.grid_propagate(False)

        # Título
        tk.Label(side, text="RestauraFoto", font=("Segoe UI", 14, "bold"),
                 bg=PANEL_COLOR, fg=ACCENT).pack(pady=(18, 2))
        tk.Label(side, text="Restauração com IA", font=("Segoe UI", 9),
                 bg=PANEL_COLOR, fg=TEXT_DIM).pack(pady=(0, 18))

        ttk.Separator(side).pack(fill="x", padx=14, pady=4)

        # Selecionar imagem
        tk.Label(side, text="Imagem", font=("Segoe UI", 10, "bold"),
                 bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=16, pady=(12, 4))
        self._lbl_file = tk.Label(side, text="Nenhuma imagem selecionada",
                                  font=("Segoe UI", 8), bg=PANEL_COLOR, fg=TEXT_DIM,
                                  wraplength=190, justify="left")
        self._lbl_file.pack(anchor="w", padx=16, pady=(0, 6))
        tk.Button(side, text="Selecionar Foto...", command=self._select_image,
                  bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2", padx=10, pady=6).pack(padx=16, fill="x")

        ttk.Separator(side).pack(fill="x", padx=14, pady=14)

        # Modo
        tk.Label(side, text="Modo", font=("Segoe UI", 10, "bold"),
                 bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=16, pady=(0, 6))
        self._mode = tk.StringVar(value="real")
        for val, lbl, desc in [
            ("real",      "Real",      "Foto antiga original"),
            ("synthetic", "Sintético", "Aplica degradação para teste"),
        ]:
            f = tk.Frame(side, bg=PANEL_COLOR)
            f.pack(anchor="w", padx=16, pady=1)
            tk.Radiobutton(f, text=lbl, variable=self._mode, value=val,
                           bg=PANEL_COLOR, fg=TEXT_COLOR, selectcolor=PANEL_COLOR,
                           activebackground=PANEL_COLOR, font=("Segoe UI", 9)).pack(side="left")
            tk.Label(f, text=desc, font=("Segoe UI", 8), bg=PANEL_COLOR,
                     fg=TEXT_DIM).pack(side="left", padx=4)

        ttk.Separator(side).pack(fill="x", padx=14, pady=14)

        # Máximo de iterações
        tk.Label(side, text="Max. iterações", font=("Segoe UI", 10, "bold"),
                 bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=16, pady=(0, 4))
        self._max_iter = tk.IntVar(value=6)
        frm_iter = tk.Frame(side, bg=PANEL_COLOR)
        frm_iter.pack(padx=16, fill="x")
        tk.Scale(frm_iter, from_=1, to=12, orient="horizontal",
                 variable=self._max_iter, bg=PANEL_COLOR, fg=TEXT_COLOR,
                 highlightthickness=0, troughcolor="#3a3a5e",
                 activebackground=ACCENT).pack(fill="x")
        self._lbl_iter = tk.Label(frm_iter, textvariable=self._max_iter,
                                  font=("Segoe UI", 9), bg=PANEL_COLOR, fg=TEXT_DIM)
        self._lbl_iter.pack()

        ttk.Separator(side).pack(fill="x", padx=14, pady=14)

        # Botão restaurar
        self._btn_restore = tk.Button(
            side, text="Restaurar", command=self._run_restoration,
            bg="#16a34a", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", cursor="hand2", padx=10, pady=10, state="disabled")
        self._btn_restore.pack(padx=16, fill="x")

        # Salvar resultado (atalho no sidebar)
        self._btn_save = tk.Button(
            side, text="Salvar Resultado", command=self._save_result,
            bg="#1d4ed8", fg="white", font=("Segoe UI", 9),
            relief="flat", cursor="hand2", padx=10, pady=6, state="disabled")
        self._btn_save.pack(padx=16, fill="x", pady=(8, 0))

        # Log de filtros
        ttk.Separator(side).pack(fill="x", padx=14, pady=14)
        tk.Label(side, text="Filtros aplicados", font=("Segoe UI", 10, "bold"),
                 bg=PANEL_COLOR, fg=TEXT_COLOR).pack(anchor="w", padx=16, pady=(0, 4))
        self._txt_log = tk.Text(side, height=8, bg="#111120", fg=SUCCESS,
                                font=("Consolas", 8), relief="flat",
                                state="disabled", wrap="word")
        self._txt_log.pack(padx=16, fill="x")

    def _build_viewer(self):
        viewer = tk.Frame(self, bg=BG_COLOR)
        viewer.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=10)
        viewer.columnconfigure(0, weight=1)
        viewer.columnconfigure(1, weight=1)

        W, H = DISPLAY_SIZE
        GH   = 220   # altura dos gráficos

        # ── Linha 0: cabeçalhos das fotos com botões de download ─────────────
        for col, txt in [(0, "Original"), (1, "Restaurada")]:
            hdr = tk.Frame(viewer, bg=BG_COLOR)
            hdr.grid(row=0, column=col, pady=(0, 6))
            tk.Label(hdr, text=txt, font=("Segoe UI", 11, "bold"),
                     bg=BG_COLOR, fg=TEXT_COLOR).pack(side="left")
            cmd = self._save_original if col == 0 else self._save_result
            btn = tk.Button(hdr, text="⬇ Salvar", command=cmd,
                            bg=PANEL_COLOR, fg=TEXT_DIM, font=("Segoe UI", 8),
                            relief="flat", cursor="hand2", padx=6, pady=2, state="disabled")
            btn.pack(side="left", padx=(10, 0))
            if col == 0:
                self._btn_dl_orig = btn
            else:
                self._btn_dl_rest = btn

        # ── Linha 1: canvas das fotos (tamanho fixo) ─────────────────────────
        self._canvas_orig = tk.Canvas(viewer, width=W, height=H,
                                      bg=PANEL_COLOR, highlightthickness=0)
        self._canvas_orig.grid(row=1, column=0, padx=(0, 4), sticky="n")
        self._canvas_orig.create_text(W//2, H//2, text="Selecione uma imagem\npara comecar",
                                      fill=TEXT_DIM, font=("Segoe UI", 10), justify="center")

        self._canvas_rest = tk.Canvas(viewer, width=W, height=H,
                                      bg=PANEL_COLOR, highlightthickness=0)
        self._canvas_rest.grid(row=1, column=1, padx=(4, 0), sticky="n")
        self._canvas_rest.create_text(W//2, H//2, text="Resultado aparece\naqui apos restaurar",
                                      fill=TEXT_DIM, font=("Segoe UI", 10), justify="center")

        # ── Linha 2: cabeçalhos dos gráficos com botões ℹ ────────────────────
        for col, txt, info_cmd in [
            (0, "Distribuicao de filtros por passo", self._info_filters),
            (1, "Evolucao da qualidade",              self._info_metrics),
        ]:
            hdr = tk.Frame(viewer, bg=BG_COLOR)
            hdr.grid(row=2, column=col, pady=(10, 2))
            tk.Label(hdr, text=txt, font=("Segoe UI", 9, "bold"),
                     bg=BG_COLOR, fg=TEXT_DIM).pack(side="left")
            tk.Button(hdr, text=" ℹ ", command=info_cmd,
                      bg=PANEL_COLOR, fg="#60a5fa", font=("Segoe UI", 9, "bold"),
                      relief="flat", cursor="hand2", padx=3, pady=0
                      ).pack(side="left", padx=(6, 0))

        # ── Linha 3: gráficos com toolbars interativas ────────────────────────

        # -- Distribuição de filtros --
        frm_f = tk.Frame(viewer, bg=BG_COLOR)
        frm_f.grid(row=3, column=0, padx=(0, 4), sticky="n")

        fig_f = Figure(figsize=(W/100, GH/100), dpi=100, facecolor=PANEL_COLOR)
        self._ax_filters = fig_f.add_subplot(111)
        self._ax_filters.set_facecolor(PANEL_COLOR)
        self._ax_filters.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self._ax_filters.spines.values():
            spine.set_edgecolor("#3a3a5e")
        self._ax_filters.text(0.5, 0.5, "Sem dados", transform=self._ax_filters.transAxes,
                              ha="center", va="center", color=TEXT_DIM, fontsize=9)
        fig_f.tight_layout(pad=0.5)

        self._mpl_filters = FigureCanvasTkAgg(fig_f, master=frm_f)
        self._mpl_filters.get_tk_widget().pack(side="top")
        toolbar_f = NavigationToolbar2Tk(self._mpl_filters, frm_f, pack_toolbar=False)
        toolbar_f.update()
        toolbar_f.pack(side="top", fill="x")
        tk.Button(frm_f, text="⬇ Salvar Gráfico", command=self._save_chart_filters,
                  bg=PANEL_COLOR, fg=TEXT_DIM, font=("Segoe UI", 7),
                  relief="flat", cursor="hand2", padx=6, pady=2).pack(side="top", pady=(2, 0))

        # -- Evolução de métricas --
        frm_m = tk.Frame(viewer, bg=BG_COLOR)
        frm_m.grid(row=3, column=1, padx=(4, 0), sticky="n")

        fig_m = Figure(figsize=(W/100, GH/100), dpi=100, facecolor=PANEL_COLOR)
        self._ax_metrics = fig_m.add_subplot(111)
        self._ax_metrics.set_facecolor(PANEL_COLOR)
        self._ax_metrics.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in self._ax_metrics.spines.values():
            spine.set_edgecolor("#3a3a5e")
        self._ax_metrics.text(0.5, 0.5, "Sem dados", transform=self._ax_metrics.transAxes,
                              ha="center", va="center", color=TEXT_DIM, fontsize=9)
        fig_m.tight_layout(pad=0.5)

        self._mpl_metrics = FigureCanvasTkAgg(fig_m, master=frm_m)
        self._mpl_metrics.get_tk_widget().pack(side="top")
        toolbar_m = NavigationToolbar2Tk(self._mpl_metrics, frm_m, pack_toolbar=False)
        toolbar_m.update()
        toolbar_m.pack(side="top", fill="x")
        tk.Button(frm_m, text="⬇ Salvar Gráfico", command=self._save_chart_metrics,
                  bg=PANEL_COLOR, fg=TEXT_DIM, font=("Segoe UI", 7),
                  relief="flat", cursor="hand2", padx=6, pady=2).pack(side="top", pady=(2, 0))

        # ── Linha 4: rótulo de métricas numéricas ─────────────────────────────
        self._lbl_ssim = tk.Label(viewer, text="", font=("Segoe UI", 9),
                                  bg=BG_COLOR, fg=TEXT_DIM)
        self._lbl_ssim.grid(row=4, column=0, columnspan=2, pady=(4, 0))

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=PANEL_COLOR, height=28)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew")
        self._lbl_status = tk.Label(bar, text="Pronto.", font=("Segoe UI", 9),
                                    bg=PANEL_COLOR, fg=TEXT_DIM, anchor="w")
        self._lbl_status.pack(side="left", padx=12)
        self._progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self._progress.pack(side="right", padx=12, pady=4)

    # ── Lógica ────────────────────────────────────────────────────────────────

    def _try_load_model(self):
        if not Path(MODEL_PATH).exists():
            self._set_status("model.pth nao encontrado — treine primeiro: python train.py", WARNING)
            return
        self._set_status("Carregando modelo...")
        self._progress.start(10)
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        try:
            agent = RestorationAgent(MODEL_PATH)
            self.after(0, lambda: self._on_model_loaded(agent))
        except Exception as e:
            self.after(0, lambda: self._on_model_error(str(e)))

    def _on_model_loaded(self, agent):
        self._agent = agent
        self._progress.stop()
        self._set_status(f"Modelo carregado ({agent.device}). Selecione uma foto.")

    def _on_model_error(self, msg):
        self._progress.stop()
        self._set_status("Erro ao carregar modelo.", WARNING)
        messagebox.showerror("Erro ao carregar modelo", msg)

    def _select_image(self):
        path = filedialog.askopenfilename(
            title="Selecionar fotografia",
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff"), ("Todos", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Erro", f"Não foi possível abrir: {path}")
            return
        self._image_bgr  = img
        self._result_bgr = None
        self._lbl_file.config(text=Path(path).name)
        self._show_image(self._canvas_orig, img)
        W, H = DISPLAY_SIZE
        self._canvas_rest.delete("all")
        self._canvas_rest.create_text(W // 2, H // 2, text="Pronto para restaurar",
                                      fill=TEXT_DIM, font=("Segoe UI", 10), justify="center")
        self._lbl_ssim.config(text="")
        self._log_clear()
        self._clear_charts()
        self._btn_restore.config(state="normal" if self._agent else "disabled")
        self._btn_save.config(state="disabled")
        self._btn_dl_orig.config(state="normal")
        self._btn_dl_rest.config(state="disabled")
        self._set_status("Imagem carregada. Clique em Restaurar.")

    def _run_restoration(self):
        if self._image_bgr is None or self._agent is None:
            return
        self._btn_restore.config(state="disabled")
        self._btn_save.config(state="disabled")
        self._btn_dl_rest.config(state="disabled")
        self._progress.start(10)
        self._set_status("Restaurando...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            img = self._image_bgr.copy()
            mode = self._mode.get()

            if mode == "synthetic":
                reference = img.copy()
                img = add_historical_photo_degradation(img)
                self.after(0, lambda: self._show_image(self._canvas_orig, img))
            else:
                reference = None

            restored, history = self._agent.restore(
                img,
                max_iterations=self._max_iter.get(),
                min_steps=0,
                reference_bgr=reference,
                verbose=False,
            )
            self.after(0, lambda: self._on_done(img, restored, history, reference))
        except Exception as e:
            self.after(0, lambda: self._on_error(str(e)))

    def _on_done(self, degraded, restored, history, reference):
        self._result_bgr = restored
        self._progress.stop()
        self._btn_restore.config(state="normal")
        self._btn_save.config(state="normal")
        self._btn_dl_rest.config(state="normal")

        self._show_image(self._canvas_orig, degraded)
        self._show_image(self._canvas_rest, restored)

        # Log de filtros
        applied = [s["action"] for s in history[1:]]
        self._log_clear()
        if applied:
            for i, name in enumerate(applied, 1):
                self._log_append(f"[{i}] {name}\n")
        else:
            self._log_append("Nenhum filtro necessario (STOP)\n")

        # Métricas (se modo sintético com referência)
        if reference is not None:
            ref_rgb  = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
            deg_rgb  = cv2.cvtColor(degraded,  cv2.COLOR_BGR2RGB)
            res_rgb  = cv2.cvtColor(restored,  cv2.COLOR_BGR2RGB)
            ssim_ini = compute_ssim(deg_rgb, ref_rgb)
            ssim_fin = compute_ssim(res_rgb, ref_rgb)
            psnr_ini = compute_psnr(deg_rgb, ref_rgb)
            psnr_fin = compute_psnr(res_rgb, ref_rgb)
            delta_s  = ssim_fin - ssim_ini
            delta_p  = psnr_fin - psnr_ini
            sign_s   = "+" if delta_s >= 0 else ""
            sign_p   = "+" if delta_p >= 0 else ""
            self._lbl_ssim.config(
                text=(f"SSIM: {ssim_ini:.3f} -> {ssim_fin:.3f} ({sign_s}{delta_s:.3f})   "
                      f"PSNR: {psnr_ini:.1f} dB -> {psnr_fin:.1f} dB ({sign_p}{delta_p:.1f} dB)"),
                fg=SUCCESS if delta_s >= 0 else WARNING,
            )

        self._update_charts(history, reference)

        n = len(applied)
        self._set_status(f"Concluido. {n} filtro{'s' if n != 1 else ''} aplicado{'s' if n != 1 else ''}.")

    def _update_charts(self, history, reference):
        steps = history[1:]   # exclui o passo "input"
        if not steps:
            return

        input_bgr = history[0]["image"]

        # ── Gráfico 1: barras empilhadas de probabilidade por passo ─────────
        ax = self._ax_filters
        ax.clear()
        ax.set_facecolor(PANEL_COLOR)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#3a3a5e")

        if any(s["probs"] is not None for s in steps):
            n_steps   = len(steps)
            x         = np.arange(n_steps)
            all_probs = np.array([s["probs"] for s in steps])   # (n_steps, 21)

            # Índice do filtro REALMENTE aplicado (após guards do agente)
            chosen_ids = [
                FILTER_NAMES.index(s["action"]) if s["action"] in FILTER_NAMES else -1
                for s in steps
            ]

            # Filtros a mostrar: os escolhidos em cada passo (garantidos) +
            # complemento até 7 filtros, priorizando maior prob. máxima
            chosen_set = {c for c in chosen_ids if c >= 0}
            top_order  = [int(i) for i in np.argsort(all_probs.max(axis=0))[::-1]
                          if int(i) not in chosen_set]
            extra_slots = max(0, 7 - len(chosen_set))
            shown_fids  = sorted(chosen_set | set(top_order[:extra_slots]))

            COLORS = ["#7c3aed", "#22c55e", "#f59e0b", "#3b82f6",
                      "#ec4899", "#ef4444", "#06b6d4", "#84cc16"]

            bottom      = np.zeros(n_steps)
            # guarda (bottom_inicio, altura) de cada segmento escolhido para destacar
            chosen_segs = {}

            for rank, fid in enumerate(shown_fids):
                color = COLORS[rank % len(COLORS)]
                vals  = all_probs[:, fid]
                ax.bar(x, vals, bottom=bottom, color=color, alpha=0.85,
                       label=FILTER_NAMES[fid][:15])

                # Percentual dentro do segmento (só se altura ≥ 4%)
                for si, (val, bot) in enumerate(zip(vals, bottom)):
                    if val >= 0.04:
                        ax.text(si, bot + val / 2, f"{val:.0%}",
                                ha="center", va="center",
                                fontsize=5.5, color="white", fontweight="bold")

                # Salva posição para o destaque posterior
                for si, cid in enumerate(chosen_ids):
                    if cid == fid:
                        chosen_segs[si] = (bottom[si], vals[si])

                bottom += vals

            # Borda branca ao redor do segmento do filtro realmente aplicado
            BAR_W = 0.8
            for si, (bot, val) in chosen_segs.items():
                ax.add_patch(MplRect(
                    (si - BAR_W / 2, bot), BAR_W, val,
                    linewidth=2, edgecolor="white", facecolor="none", zorder=5
                ))

            ax.set_xticks(x)
            ax.set_xticklabels(
                [f"P{i+1}\n{s['action'][:11]}" for i, s in enumerate(steps)],
                fontsize=6, color=TEXT_DIM
            )
            ax.set_ylabel("Probabilidade", fontsize=7, color=TEXT_DIM)
            ax.set_ylim(0, 1)
            ax.legend(loc="upper right", fontsize=5, ncol=1,
                      facecolor=PANEL_COLOR, labelcolor=TEXT_DIM, edgecolor="#3a3a5e")
            ax.set_title("Prob. por passo  (□ branco = filtro aplicado)",
                         fontsize=7, color=TEXT_DIM, pad=3)

        self._mpl_filters.figure.tight_layout(pad=0.5)
        self._mpl_filters.draw()

        # ── Gráfico 2: evolução da qualidade por passo ───────────────────────
        ax2 = self._ax_metrics
        ax2.clear()
        ax2.set_facecolor(PANEL_COLOR)
        ax2.tick_params(colors=TEXT_DIM, labelsize=7)
        for spine in ax2.spines.values():
            spine.set_edgecolor("#3a3a5e")

        step_labels = ["Entrada"] + [s["action"][:9] for s in steps]

        if reference is not None:
            # Modo sintético: SSIM vs ground truth em cada passo
            ref_rgb   = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
            ssim_vals = []
            for s in history:
                rgb = cv2.cvtColor(s["image"], cv2.COLOR_BGR2RGB)
                ssim_vals.append(compute_ssim(rgb, ref_rgb))
            ax2.plot(ssim_vals, "o-", color="#22c55e", linewidth=1.5, markersize=4)
            ax2.set_ylabel("SSIM vs. referencia", fontsize=7, color=TEXT_DIM)
            ax2.set_title("SSIM por passo (modo sintetico)", fontsize=7, color=TEXT_DIM, pad=3)
            ax2.set_ylim(max(0, min(ssim_vals) - 0.05), 1.0)
        else:
            # Modo real: score de qualidade absoluto + cast de cor
            #   1. Qualidade estimada 0-100 (sobe quando a foto melhora)
            #   2. Cast de cor — desvio entre canais BGR (cai quando cor é corrigida)
            quality_scores = []
            color_cast     = []
            for s in history:
                bgr = s["image"]
                quality_scores.append(_no_ref_quality(bgr))
                f     = bgr.astype(np.float32)
                means = f.mean(axis=(0, 1))
                color_cast.append(float(np.std(means)))

            ax2_r = ax2.twinx()
            ax2_r.set_facecolor(PANEL_COLOR)
            ax2_r.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax2_r.spines.values():
                spine.set_edgecolor("#3a3a5e")

            ax2.plot(quality_scores, "o-", color="#22c55e", linewidth=1.5,
                     markersize=4, label="Qualidade (0-100) ↑")
            ax2_r.plot(color_cast, "s--", color="#f59e0b", linewidth=1.5,
                       markersize=4, label="Cast de cor ↓")
            ax2.set_ylabel("Qualidade estimada (0-100)", fontsize=7, color="#22c55e")
            ax2_r.set_ylabel("Cast de cor (↓ melhor)", fontsize=7, color="#f59e0b")
            ax2.set_ylim(0, 100)
            ax2.set_title("Qualidade por passo (modo real)", fontsize=7, color=TEXT_DIM, pad=3)

            lines1, lab1 = ax2.get_legend_handles_labels()
            lines2, lab2 = ax2_r.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, lab1 + lab2, fontsize=5,
                       facecolor=PANEL_COLOR, labelcolor=TEXT_DIM,
                       edgecolor="#3a3a5e", loc="lower left")

        xs = range(len(step_labels))
        ax2.set_xticks(list(xs))
        ax2.set_xticklabels(step_labels, fontsize=6, color=TEXT_DIM, rotation=15, ha="right")
        ax2.grid(True, alpha=0.15, color=TEXT_DIM)

        self._mpl_metrics.figure.tight_layout(pad=0.5)
        self._mpl_metrics.draw()

    def _on_error(self, msg):
        self._progress.stop()
        self._btn_restore.config(state="normal")
        messagebox.showerror("Erro na restauração", msg)
        self._set_status("Erro.", WARNING)

    # ── Salvar ────────────────────────────────────────────────────────────────

    def _save_original(self):
        if self._image_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")],
            title="Salvar imagem original",
        )
        if path:
            cv2.imwrite(path, self._image_bgr)
            self._set_status(f"Original salvo em {Path(path).name}")

    def _save_result(self):
        if self._result_bgr is None:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")],
            title="Salvar imagem restaurada",
        )
        if path:
            cv2.imwrite(path, self._result_bgr)
            self._set_status(f"Salvo em {Path(path).name}")

    def _save_chart_filters(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("Todos", "*.*")],
            title="Salvar grafico de distribuicao de filtros",
        )
        if path:
            self._mpl_filters.figure.savefig(
                path, dpi=150, facecolor=PANEL_COLOR, bbox_inches="tight")
            self._set_status(f"Grafico salvo em {Path(path).name}")

    def _save_chart_metrics(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"), ("Todos", "*.*")],
            title="Salvar grafico de evolucao de qualidade",
        )
        if path:
            self._mpl_metrics.figure.savefig(
                path, dpi=150, facecolor=PANEL_COLOR, bbox_inches="tight")
            self._set_status(f"Grafico salvo em {Path(path).name}")

    # ── Diálogos informativos ─────────────────────────────────────────────────

    def _info_filters(self):
        self._show_info_dialog(
            "Distribuição de Probabilidades por Passo",
            "COMO LER O GRÁFICO:\n\n"
            "  • Cada barra vertical = um passo de restauração (P1, P2...)\n"
            "  • Cada cor/segmento = um filtro e sua probabilidade\n"
            "  • O % dentro do segmento = prob. exata atribuída pela rede\n"
            "  • Borda branca = filtro que foi REALMENTE aplicado\n\n"
            "QUAIS FILTROS APARECEM:\n\n"
            "  Os filtros aplicados em cada passo aparecem sempre.\n"
            "  Os demais slots são preenchidos pelos filtros com maior\n"
            "  probabilidade máxima ao longo da sessão.\n\n"
            "POR QUE O APLICADO PODE NÃO SER O MAIS ALTO?\n\n"
            "  O agente tem guardas que podem redirecionar:\n"
            "    - high_pass_strong exige ≥ 30% de confiança\n"
            "    - fourier exige ≥ 45% de confiança\n"
            "    - color_correction bloqueado em fotos P&B\n"
            "    - mesma ação não repete consecutivamente\n"
            "  Nesses casos a borda branca fica num segmento menor,\n"
            "  revelando exatamente o redirecionamento do agente.\n\n"
            "  Use a barra de ferramentas abaixo para zoom e pan."
        )

    def _info_metrics(self):
        self._show_info_dialog(
            "Evolução da Qualidade por Passo",
            "MODO REAL (sem imagem de referência):\n\n"
            "  Linha verde — Qualidade estimada (0–100)  ↑ sobe = melhora\n"
            "    Score composto que SOBE quando a restauração melhora a foto.\n"
            "    Combina três fatores independentes:\n"
            "      • Nitidez (35%): variância do Laplaciano — sobe quando\n"
            "        high_pass ou deblur recuperam detalhes.\n"
            "      • Brilho (25%): proximidade ao ponto médio ideal (128) —\n"
            "        sobe quando gamma/log corrigem subexposição.\n"
            "      • Neutralidade de cor (40%): baixo desvio entre canais —\n"
            "        sobe quando color_correction remove sépia ou tom azul.\n\n"
            "  Linha laranja — Cast de cor  ↓ cai = melhora\n"
            "    Desvio-padrão entre as médias dos canais R, G e B.\n"
            "    Zero = imagem completamente neutra (sem dominante de cor).\n"
            "    Queda confirma que a correção de cor está funcionando.\n\n"
            "MODO SINTÉTICO (com imagem de referência):\n\n"
            "  Linha verde — SSIM vs. Referência  ↑ sobe = melhora\n"
            "    Similaridade com a imagem limpa original (ground truth).\n"
            "    Valores crescentes = restauração indo na direção certa.\n"
            "    Objetivo: chegar o mais próximo de 1.0.\n\n"
            "O QUE É O SSIM?\n"
            "  Structural Similarity Index — mede semelhança perceptual em\n"
            "  três dimensões: luminância, contraste e estrutura. Imita como\n"
            "  o olho humano compara imagens. Varia de 0 a 1."
        )

    def _show_info_dialog(self, title, body):
        win = tk.Toplevel(self)
        win.title(title)
        win.configure(bg=PANEL_COLOR)
        win.resizable(False, False)
        win.grab_set()
        tk.Label(win, text=title, font=("Segoe UI", 11, "bold"),
                 bg=PANEL_COLOR, fg=ACCENT).pack(padx=24, pady=(16, 8))
        tk.Message(win, text=body, font=("Segoe UI", 9),
                   bg=PANEL_COLOR, fg=TEXT_COLOR, width=460, justify="left").pack(padx=24, pady=(0, 8))
        tk.Button(win, text="Fechar", command=win.destroy,
                  bg=ACCENT, fg="white", font=("Segoe UI", 9, "bold"),
                  relief="flat", cursor="hand2", padx=16, pady=6).pack(pady=(0, 16))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _clear_charts(self):
        """Reseta ambos os gráficos para o estado vazio.

        Usa figure.clear() em vez de ax.clear() para garantir que eixos
        secundários (twinx) criados na restauração anterior sejam removidos
        — caso contrário acumulam e distorcem o próximo gráfico.
        """
        for canvas, attr in [
            (self._mpl_filters, "_ax_filters"),
            (self._mpl_metrics, "_ax_metrics"),
        ]:
            fig = canvas.figure
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor(PANEL_COLOR)
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a3a5e")
            ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                    ha="center", va="center", color=TEXT_DIM, fontsize=9)
            fig.tight_layout(pad=0.5)
            setattr(self, attr, ax)
            canvas.draw()

    def _show_image(self, canvas, img_bgr):
        W, H  = DISPLAY_SIZE
        photo = _bgr_to_photoimage(img_bgr, size=(W, H))
        canvas.delete("all")
        canvas.create_image(W // 2, H // 2, anchor="center", image=photo)
        canvas.image = photo   # mantém referência para o garbage collector

    def _set_status(self, msg, color=None):
        self._lbl_status.config(text=msg, fg=color or TEXT_DIM)

    def _log_clear(self):
        self._txt_log.config(state="normal")
        self._txt_log.delete("1.0", "end")
        self._txt_log.config(state="disabled")

    def _log_append(self, text):
        self._txt_log.config(state="normal")
        self._txt_log.insert("end", text)
        self._txt_log.config(state="disabled")


if __name__ == "__main__":
    app = RestauraFotoApp()
    app.mainloop()
