import cv2
import numpy as np
import torch
import torchvision.transforms as T

from model import ImageAnalyzer
from filters import apply_filter, FILTER_NAMES, STOP_ACTION
from metrics import compute_ssim, compute_psnr
from dataset import IMG_SIZE, image_to_tensor

# Para se a melhora de SSIM entre iterações cair abaixo deste limiar
SSIM_STOP_THRESHOLD = 0.003


class RestorationAgent:
    """
    Agente iterativo de restauração de imagens.
    A cada passo, a rede CNN observa a imagem atual e escolhe o próximo filtro.
    O loop continua até a rede emitir STOP ou a melhora de SSIM cair abaixo do limiar.
    """

    def __init__(self, model_path="model.pth", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ImageAnalyzer(n_actions=len(FILTER_NAMES))
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)

    def _predict(self, image_bgr):
        """Retorna (action_idx, probabilidades) para a imagem BGR atual."""
        rgb     = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        tensor  = image_to_tensor(resized).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        return int(probs.argmax()), probs

    def restore(self, image_bgr, max_iterations=8, min_steps=2, reference_bgr=None, verbose=True):
        """
        Restaura iterativamente uma imagem.

        Parâmetros
        ----------
        image_bgr      : imagem degradada (numpy BGR)
        max_iterations : número máximo de passos permitidos
        reference_bgr  : imagem limpa de referência para calcular SSIM/PSNR (opcional)
        verbose        : imprime log a cada passo

        Retorna
        -------
        (restored_bgr, history)
            history é uma lista de dicts com chaves:
            'image', 'action', 'ssim', 'psnr', 'probs'
        """
        current = image_bgr.copy()

        # Detecta se a imagem é essencialmente P&B (desvio entre canais RGB < 8).
        # Para fotos P&B, color_correction não ajuda e pode criar dominante azul.
        def _is_grayscale(img_bgr, threshold=8):
            f   = img_bgr.astype(np.float32)
            std = np.std([f[:,:,0].mean(), f[:,:,1].mean(), f[:,:,2].mean()])
            return std < threshold

        is_grayscale_photo = _is_grayscale(image_bgr)
        # Ambas as variantes de color_correction bloqueadas para fotos P&B
        COLOR_CORRECTION_ACTIONS = {9, 10}

        def _metrics(img):
            if reference_bgr is None:
                return None, None
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
            return compute_ssim(rgb, ref), compute_psnr(rgb, ref)

        # SSIM acumulado em relação à imagem de ENTRADA (modo real sem referência).
        # Impede que o agente aplique filtros que distorcem demais a foto original.
        def _ssim_vs_input(img):
            a = cv2.cvtColor(img,           cv2.COLOR_BGR2RGB)
            b = cv2.cvtColor(image_bgr,     cv2.COLOR_BGR2RGB)
            return compute_ssim(a, b)

        init_ssim, init_psnr = _metrics(current)
        history = [{
            "image":  current.copy(),
            "action": "input",
            "ssim":   init_ssim,
            "psnr":   init_psnr,
            "probs":  None,
        }]

        prev_ssim   = init_ssim
        last_action = None
        blur_count  = 0  # quantos blurs (gaussian/bilateral) já foram aplicados

        # Índices conforme FILTER_NAMES (20 filtros + STOP=20)
        FOURIER_ACTION       = 19
        HIGH_PASS_GENTLE     = 7
        HIGH_PASS_STRONG     = 8
        # mean(0) gaussian_soft(1) gaussian_medium(2) gaussian_strong(3) bilateral(4)
        BLUR_ACTIONS         = {0, 1, 2, 3, 4}
        # Filtros destrutivos excluídos ao forçar saída do STOP
        DESTRUCTIVE_ACTIONS  = {FOURIER_ACTION, HIGH_PASS_STRONG}

        for step in range(1, max_iterations + 1):
            action, probs = self._predict(current)

            # Para fotos P&B, color_correction não tem efeito útil e pode criar cast
            if is_grayscale_photo and action in COLOR_CORRECTION_ACTIONS:
                masked = probs.copy()
                for c in COLOR_CORRECTION_ACTIONS:
                    masked[c] = 0
                masked[STOP_ACTION] = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] color_correction bloqueado (foto P&B) -> '{FILTER_NAMES[action]}'")

            # Filtros que requerem confiança mínima por risco de artefatos
            FOURIER_MIN_PROB   = 0.45
            HIGH_PASS_MIN_PROB = 0.30  # high_pass amplifica ruído se a imagem não está limpa
            if action == HIGH_PASS_STRONG and probs[HIGH_PASS_STRONG] < HIGH_PASS_MIN_PROB:
                masked = probs.copy()
                masked[HIGH_PASS_STRONG] = 0
                masked[STOP_ACTION]      = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] high_pass_strong bloqueado (confianca {probs[HIGH_PASS_STRONG]:.2f}) "
                          f"-> '{FILTER_NAMES[action]}'")
            if action == FOURIER_ACTION and probs[FOURIER_ACTION] < FOURIER_MIN_PROB:
                masked = probs.copy()
                masked[FOURIER_ACTION] = 0
                masked[STOP_ACTION]    = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] Fourier bloqueado (confianca {probs[FOURIER_ACTION]:.2f}) "
                          f"-> '{FILTER_NAMES[action]}'")

            if action == STOP_ACTION:
                if step <= min_steps:
                    masked = probs.copy()
                    masked[STOP_ACTION] = 0
                    for d in DESTRUCTIVE_ACTIONS:
                        masked[d] = 0
                    action = int(masked.argmax())
                    if verbose:
                        print(f"[{step}] STOP ignorado (minimo {min_steps} passos) -> '{FILTER_NAMES[action]}'")
                else:
                    if verbose:
                        print(f"[{step}] Rede decidiu PARAR ({probs[STOP_ACTION]*100:.1f}%).")
                    break

            # Impede apenas repetição imediata (A→A), mas permite A→B→A
            # O modelo pode oscilar entre variantes de intensidade até achar o tom certo
            if action == last_action:
                masked = probs.copy()
                masked[last_action] = 0
                masked[STOP_ACTION] = 0
                for d in DESTRUCTIVE_ACTIONS:
                    masked[d] = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] Mesma acao consecutiva bloqueada -> '{FILTER_NAMES[action]}'")

            # Limite de blurs — aplicado por último, depois de todos os redirecionamentos.
            # Garante que no máximo 2 passes de suavização sejam aplicados por sessão,
            # independente de qual caminho de redirect foi tomado acima.
            if blur_count >= 2 and action in BLUR_ACTIONS:
                masked = probs.copy()
                for b in BLUR_ACTIONS:
                    masked[b] = 0
                masked[STOP_ACTION] = 0
                for d in DESTRUCTIVE_ACTIONS:
                    masked[d] = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] Blur limitado ({blur_count} passes) -> '{FILTER_NAMES[action]}'")

            last_action = action
            if action in BLUR_ACTIONS:
                blur_count += 1

            filtered = apply_filter(current, action)

            # Se o filtro distorceu demais, tenta a proxima melhor opcao em vez de desistir
            threshold = 0.55 if action in (FOURIER_ACTION, HIGH_PASS_STRONG) else 0.25
            ssim_consecutivo = compute_ssim(
                cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB),
                cv2.cvtColor(current,  cv2.COLOR_BGR2RGB)
            )
            if ssim_consecutivo < threshold:
                if verbose:
                    print(f"[{step}] '{FILTER_NAMES[action]}' distorceu (SSIM={ssim_consecutivo:.3f}), "
                          f"tentando proxima opcao...")
                masked = probs.copy()
                masked[action]      = 0
                masked[STOP_ACTION] = 0
                for d in DESTRUCTIVE_ACTIONS:
                    masked[d] = 0
                action   = int(masked.argmax())
                filtered = apply_filter(current, action)
                ssim2 = compute_ssim(
                    cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(current,  cv2.COLOR_BGR2RGB)
                )
                if ssim2 < threshold:
                    if verbose:
                        print(f"[{step}] '{FILTER_NAMES[action]}' tambem distorceu, parando.")
                    break
                if verbose:
                    print(f"[{step}] Redirecionado para '{FILTER_NAMES[action]}'")

            cur_ssim, cur_psnr = _metrics(filtered)

            # Parada por SSIM estagnado (só quando há referência).
            # Não aplica nas primeiras min_steps iterações.
            if prev_ssim is not None and cur_ssim is not None and step > min_steps:
                improvement = cur_ssim - prev_ssim
                if improvement < SSIM_STOP_THRESHOLD:
                    if verbose:
                        print(f"[{step}] Melhora de SSIM insuficiente ({improvement:.4f}), parando.")
                    break
                prev_ssim = cur_ssim
            elif cur_ssim is not None:
                prev_ssim = cur_ssim

            # Parada sem referência: compara resultado acumulado com a imagem de entrada.
            # Se o conjunto de filtros aplicados até agora distorceu demais a foto
            # (SSIM vs entrada < 0.80), descarta o filtro atual e para.
            # Isso evita que o agente borre ou altere excessivamente fotos reais.
            if reference_bgr is None and step > min_steps:
                ssim_vs_orig = _ssim_vs_input(filtered)
                if ssim_vs_orig < 0.80:
                    if verbose:
                        print(f"[{step}] Imagem muito distante do original (SSIM={ssim_vs_orig:.3f}), parando.")
                    break

            current = filtered
            filter_name = FILTER_NAMES[action]

            if verbose:
                ssim_str = f" | SSIM: {cur_ssim:.4f}" if cur_ssim else ""
                psnr_str = f" | PSNR: {cur_psnr:.2f} dB" if cur_psnr else ""
                print(f"[{step}] Filtro: '{filter_name}'{ssim_str}{psnr_str}")

            history.append({
                "image":  current.copy(),
                "action": filter_name,
                "ssim":   cur_ssim,
                "psnr":   cur_psnr,
                "probs":  probs,
            })

        return current, history
