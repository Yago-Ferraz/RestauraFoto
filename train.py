"""
Treina a rede ImageAnalyzer (ResNet-18) para decidir qual filtro aplicar.
Soft labels gerados por MSE. Treino em duas fases para transfer learning.

Uso:
    python train.py --epochs 40 --batch 384
    python train.py --epochs 40 --batch 384 --resume   ← continua do checkpoint
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import ImageAnalyzer
from dataset import RestorationDataset
from filters import FILTER_NAMES

PHASE1_EPOCHS   = 10
CHECKPOINT_PATH = Path("checkpoint.pth")


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU detectada: {name} ({vram:.1f} GB VRAM) — usando CUDA")
    else:
        device = torch.device("cpu")
        print("GPU não detectada — usando CPU")
    return device


def soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


def make_optimizer(model, epoch, epochs):
    """Cria o optimizer correto para a fase atual."""
    if epoch <= PHASE1_EPOCHS:
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        ), torch.optim.lr_scheduler.CosineAnnealingLR(
            torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3),
            T_max=PHASE1_EPOCHS
        )
    else:
        opt = torch.optim.Adam([
            {"params": model.backbone.parameters(),   "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4},
        ])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - PHASE1_EPOCHS)
        return opt, sch


def run_epoch(model, loader, optimizer, device, training):
    model.train(training)
    total_loss = correct = total = 0

    desc = "treino" if training else "val"
    with torch.set_grad_enabled(training):
        for imgs, soft_labels in tqdm(loader, desc=desc, leave=False):
            imgs        = imgs.to(device)
            soft_labels = soft_labels.to(device)

            logits = model(imgs)
            loss   = soft_cross_entropy(logits, soft_labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            pred    = logits.argmax(1)
            target  = soft_labels.argmax(1)
            correct += (pred == target).sum().item()
            total   += imgs.size(0)

    return total_loss / len(loader), correct / total


def train(data_dir, epochs=40, batch_size=128, save_path="model.pth", resume=False):
    device = get_device()

    # ── Dataset ───────────────────────────────────────────────────────────────
    exts  = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff")
    paths = [p for ext in exts for p in Path(data_dir).rglob(ext)]
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em '{data_dir}'")
    print(f"{len(paths)} imagens encontradas")

    dataset = RestorationDataset(paths, samples_per_image=25)
    n_train = int(0.85 * len(dataset))
    n_val   = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=6, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=6, pin_memory=False)

    # ── Modelo e estado inicial ───────────────────────────────────────────────
    model        = ImageAnalyzer(n_actions=len(FILTER_NAMES), freeze_backbone=True).to(device)
    start_epoch  = 1
    best_val_acc = 0.0
    train_losses, val_losses, val_accs = [], [], []

    # ── Resume ────────────────────────────────────────────────────────────────
    if resume and CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        train_losses = ckpt.get("train_losses", [])
        val_losses   = ckpt.get("val_losses",   [])
        val_accs     = ckpt.get("val_accs",     [])
        print(f"Checkpoint carregado — continuando da época {start_epoch} "
              f"(melhor val acc: {best_val_acc:.3f})")

        if start_epoch > PHASE1_EPOCHS:
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam([
                {"params": model.backbone.parameters(),   "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - PHASE1_EPOCHS
            )
            print(f"── Fase 2 (épocas {PHASE1_EPOCHS+1}-{epochs}): fine-tuning completo ──")
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=PHASE1_EPOCHS
            )
            print(f"── Fase 1 (épocas 1-{PHASE1_EPOCHS}): backbone congelado ──")

        # Só restaura o estado do optimizer se estiver na mesma fase do checkpoint
        ckpt_phase = 1 if ckpt["epoch"] <= PHASE1_EPOCHS else 2
        curr_phase = 1 if start_epoch <= PHASE1_EPOCHS else 2
        if ckpt_phase == curr_phase:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        else:
            print("Transição de fase detectada — optimizer reiniciado para fase 2.")
    else:
        print(f"\n── Fase 1 (épocas 1-{PHASE1_EPOCHS}): backbone congelado, treinando cabeça ──")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    # ── Loop de treino ────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs + 1):

        if epoch == PHASE1_EPOCHS + 1:
            print(f"\n── Fase 2 (épocas {PHASE1_EPOCHS+1}-{epochs}): fine-tuning completo ──")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam([
                {"params": model.backbone.parameters(),   "lr": 1e-5},
                {"params": model.classifier.parameters(), "lr": 1e-4},
            ])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs - PHASE1_EPOCHS
            )

        train_loss, train_acc = run_epoch(model, train_loader, optimizer, device, training=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   optimizer, device, training=False)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} ({train_acc:.3f}) "
              f"| Val: {val_loss:.4f} ({val_acc:.3f})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)

        # Salva checkpoint completo a cada época
        torch.save({
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_acc":         best_val_acc,
            "train_losses":         train_losses,
            "val_losses":           val_losses,
            "val_accs":             val_accs,
        }, CHECKPOINT_PATH)

    print(f"\nMelhor acurácia de validação: {best_val_acc:.3f}")
    print(f"Modelo salvo em '{save_path}'")

    # ── Curvas ────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses,   label="Val")
    ax1.axvline(PHASE1_EPOCHS - 1, color="gray", linestyle="--", label="início fase 2")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(val_accs)
    ax2.axvline(PHASE1_EPOCHS - 1, color="gray", linestyle="--", label="início fase 2")
    ax2.set_title("Acurácia de Validação")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("Curvas salvas em 'training_curves.png'")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="data/clean")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch",  type=int, default=128)
    parser.add_argument("--model",  default="model.pth")
    parser.add_argument("--resume", action="store_true",
                        help="Continua o treino do último checkpoint salvo")
    args = parser.parse_args()

    train(args.data, args.epochs, args.batch, args.model, args.resume)
