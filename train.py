"""
Treina a rede ImageAnalyzer para decidir qual filtro aplicar dado o estado
atual da imagem. Labels são gerados automaticamente pela métrica SSIM.

Uso:
    python train.py --data data/clean --epochs 40
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ImageAnalyzer
from dataset import RestorationDataset
from filters import FILTER_NAMES, STOP_ACTION


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detectada: {name} ({vram:.1f} GB VRAM) — usando CUDA")
    else:
        device = torch.device("cpu")
        print("GPU não detectada — usando CPU")
        print("  (para usar GPU instale PyTorch com CUDA: ver instruções no README)")
    return device


def train(data_dir, epochs=40, batch_size=32, lr=1e-3, save_path="model.pth"):
    device = get_device()

    # ── Dataset ──────────────────────────────────────────────────────────────
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    paths = [p for ext in exts for p in Path(data_dir).rglob(ext)]
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em '{data_dir}'")
    print(f"{len(paths)} imagens encontradas")

    dataset = RestorationDataset(paths, samples_per_image=12)
    n_train = int(0.85 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=0, pin_memory=device.type == "cuda")
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=0, pin_memory=device.type == "cuda")

    # ── Modelo ────────────────────────────────────────────────────────────────
    model     = ImageAnalyzer(n_actions=len(FILTER_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # STOP recebe peso menor para o modelo não aprender a parar por padrão
    weights = torch.ones(len(FILTER_NAMES), device=device)
    weights[STOP_ACTION] = 0.4
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_losses, val_losses, val_accs = [], [], []
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Treino
        model.train()
        total_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                out = model(imgs)
                val_loss += criterion(out, labels).item()
                correct  += (out.argmax(1) == labels).sum().item()
                total    += labels.size(0)

        scheduler.step()

        avg_train = total_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        acc       = correct    / total

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        val_accs.append(acc)

        print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f} | Acc: {acc:.3f}")

        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), save_path)

    print(f"\nMelhor acurácia de validação: {best_val_acc:.3f}")
    print(f"Modelo salvo em '{save_path}'")

    # ── Curvas de treino ──────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(val_accs)
    ax2.set_title("Acurácia de Validação")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Curvas salvas em 'training_curves.png'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    default="data/clean", help="Pasta com imagens limpas para treino")
    parser.add_argument("--epochs",  type=int, default=40)
    parser.add_argument("--batch",   type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--model",   default="model.pth")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.lr, args.model)
