"""
train_classifier.py
-------------------
Fine-tunes a pretrained ResNet-50 on ACNE04 patches for binary
acne vs. clear skin classification.

Usage:
    python part2_classification/train_classifier.py

Outputs:
    - Best weights saved to outputs/checkpoints/classifier/best.pt
    - Last weights saved to outputs/checkpoints/classifier/last.pt
    - Training log saved to outputs/checkpoints/classifier/train_log.csv
    - Loss/accuracy curves saved to outputs/checkpoints/classifier/training_curves.png
"""

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.models import ResNet50_Weights

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append(str(Path(__file__).resolve().parent))
from dataset import (
    AcnePatchDataset,
    get_train_transforms,
    get_val_transforms,
)

# PATHS
ROOT       = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "checkpoints" / "classifier"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# HYPERPARAMETERS
NUM_CLASSES  = 2        # acne vs clear skin
EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-4     # low LR for fine-tuning pretrained weights
WEIGHT_DECAY = 1e-4
LR_STEP      = 8        # drop LR every N epochs
LR_GAMMA     = 0.5
PATIENCE     = 8        # early stopping patience


# MODEL
def build_model(num_classes: int = NUM_CLASSES):
    """
    Load pretrained ResNet-50 and replace the final FC layer
    for binary classification.

    We use a two-stage fine-tuning strategy:
        Stage 1: freeze backbone, train only the head (fast convergence)
        Stage 2: unfreeze all layers, train end-to-end at low LR
    """
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )

    return model


def freeze_backbone(model):
    """Freeze all layers except the final FC head."""
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
    print("Backbone frozen — training head only")


def unfreeze_all(model):
    """Unfreeze all layers for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen — end-to-end fine-tuning")


# TRAINING UTILITIES
def run_epoch(model, loader, criterion, optimizer, device, is_train: bool):
    """
    Run one epoch of training or validation.
    Returns (avg_loss, accuracy).
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    correct    = 0
    total      = 0

    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total    if total > 0 else 0.0
    return avg_loss, accuracy


def plot_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Save loss and accuracy curves to a PNG file."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, train_losses, label="Train Loss", color="steelblue")
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="coral")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, train_accs, label="Train Acc", color="steelblue")
    ax2.plot(epochs, val_accs,   label="Val Acc",   color="coral")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to: {save_path}")


# MAIN TRAINING LOOP
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # ── Datasets ──
    train_dataset = AcnePatchDataset(split="train", transform=get_train_transforms())
    val_dataset   = AcnePatchDataset(split="val",   transform=get_val_transforms())

    # Use weighted sampler to handle class imbalance
    sampler = train_dataset.get_weighted_sampler()

    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,        # replaces shuffle=True
        num_workers = 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Model ──
    model = build_model()
    model.to(device)

    # ── Loss ──
    # Label smoothing helps prevent overconfidence and improves generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── Logging ──
    log_path = OUTPUT_DIR / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "stage", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])

    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    best_val_acc  = 0.0
    epochs_no_imp = 0   # for early stopping

    # STAGE 1: Train head only (epochs 1 to EPOCHS//3)
    print("=" * 55)
    print("STAGE 1: Training classification head only")
    print("=" * 55)

    freeze_backbone(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR * 5,           # higher LR for head-only training
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    stage1_epochs = max(5, EPOCHS // 3)

    for epoch in range(1, stage1_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        scheduler.step()
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        print(f"[S1] Epoch {epoch:02d}/{stage1_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        writer.writerow([epoch, "head_only", train_loss, train_acc, val_loss, val_acc, current_lr])
        log_file.flush()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            print(f"  ✓ New best model saved (val acc: {best_val_acc:.4f})")

    # STAGE 2: Fine-tune all layers (remaining epochs)
    print("\n" + "=" * 55)
    print("STAGE 2: End-to-end fine-tuning (all layers)")
    print("=" * 55)

    unfreeze_all(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    stage2_epochs = EPOCHS - stage1_epochs
    epochs_no_imp = 0

    for epoch in range(1, stage2_epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, None,      device, is_train=False)

        scheduler.step()
        elapsed    = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]

        global_epoch = stage1_epochs + epoch
        print(f"[S2] Epoch {epoch:02d}/{stage2_epochs} (global {global_epoch}) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | {elapsed:.1f}s")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        writer.writerow([global_epoch, "finetune", train_loss, train_acc, val_loss, val_acc, current_lr])
        log_file.flush()

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            print(f"  ✓ New best model saved (val acc: {best_val_acc:.4f})")
        else:
            epochs_no_imp += 1
            if epochs_no_imp >= PATIENCE:
                print(f"\nEarly stopping triggered after {epoch} stage-2 epochs.")
                break

        torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")

    log_file.close()

    # ── Save training curves ──
    plot_curves(
        train_losses, val_losses,
        train_accs,   val_accs,
        save_path = OUTPUT_DIR / "training_curves.png",
    )

    print(f"\n── Training complete ──")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Best weights:      {OUTPUT_DIR / 'best.pt'}")
    print(f"Training log:      {OUTPUT_DIR / 'train_log.csv'}")


# INFERENCE HELPER  (used by evaluate_dermnet.py and gradcam.py)
def load_model(weights_path: str = None):
    """Load trained classifier from saved weights."""
    if weights_path is None:
        weights_path = OUTPUT_DIR / "best.pt"

    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


if __name__ == "__main__":
    train() 