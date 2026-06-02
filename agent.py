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
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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

        def _metrics(img):
            if reference_bgr is None:
                return None, None
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ref = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2RGB)
            return compute_ssim(rgb, ref), compute_psnr(rgb, ref)

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

        for step in range(1, max_iterations + 1):
            action, probs = self._predict(current)

            if action == STOP_ACTION:
                if step <= min_steps:
                    action = int(probs[:STOP_ACTION].argmax())
                    if verbose:
                        print(f"[{step}] STOP ignorado (mínimo {min_steps} passos) → forçando '{FILTER_NAMES[action]}'")
                else:
                    if verbose:
                        print(f"[{step}] Rede decidiu PARAR.")
                    break

            # Impede repetir o mesmo filtro em sequência
            if action == last_action:
                masked = probs.copy()
                masked[last_action]  = 0
                masked[STOP_ACTION]  = 0
                action = int(masked.argmax())
                if verbose:
                    print(f"[{step}] Filtro repetido bloqueado → alternando para '{FILTER_NAMES[action]}'")

            last_action = action

            filtered = apply_filter(current, action)
            cur_ssim, cur_psnr = _metrics(filtered)

            # Parada por SSIM estagnado (só quando há referência)
            if prev_ssim is not None and cur_ssim is not None:
                improvement = cur_ssim - prev_ssim
                if improvement < SSIM_STOP_THRESHOLD:
                    if verbose:
                        print(f"[{step}] Melhora de SSIM insuficiente ({improvement:.4f}), parando.")
                    break
                prev_ssim = cur_ssim

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
