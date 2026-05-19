"""
evaluate_dermnet.py
-------------------
Evaluates the trained classifier on the DermNet test set.
Reports accuracy, F1-score, and AUROC for acne vs. non-acne classification.

Also runs a quick sanity check on 20 DermNet training samples to
assess the domain gap before full evaluation.

Usage:
    python part2_classification/evaluate_dermnet.py

Outputs:
    - Printed metrics table (accuracy, F1, AUROC)
    - outputs/checkpoints/classifier/dermnet_results.txt
    - outputs/checkpoints/classifier/confusion_matrix.png
    - outputs/checkpoints/classifier/roc_curve.png
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving figures

from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    classification_report,
)

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import DermNetDataset, get_val_transforms, get_tta_transforms
from train_classifier import load_model
from domain_adapt import get_adapted_val_transforms

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = ROOT / "outputs" / "checkpoints" / "classifier"
DERMNET_DIR  = ROOT / "data" / "dermnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────
BATCH_SIZE      = 32
CONF_THRESHOLD  = 0.5     # probability threshold for predicting acne
USE_TTA         = True    # test-time augmentation for more robust predictions
USE_COLOR_NORM  = True    # apply Reinhard normalization before inference

# INFERENCE
def run_inference(model, dataset, device, use_tta=USE_TTA):
    """
    Run inference on a dataset.

    Returns:
        all_probs  : numpy array of acne probabilities, shape (N,)
        all_labels : numpy array of ground truth labels, shape (N,)
    """
    model.eval()

    if use_tta:
        return run_inference_tta(model, dataset, device)

    loader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)[:, 1]   # probability of class 1 (acne)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_labels)


def run_inference_tta(model, dataset, device, n_augments=5):
    """
    Test-Time Augmentation: run inference with multiple augmented versions
    of each image and average the probabilities.
    """
    split          = getattr(dataset, "split", "test")
    max_samples    = getattr(dataset, "max_samples", None)
    seed           = getattr(dataset, "seed", 42)
    use_color_norm = USE_COLOR_NORM and split == "test"
    all_labels     = np.array(dataset.labels)
    all_probs      = np.zeros(len(all_labels), dtype=np.float32)

    tta_augs = [
        transforms.Lambda(lambda img: img),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Compose([
            transforms.Resize((246, 246)),
            transforms.CenterCrop(224),
        ]),
        transforms.RandomRotation(degrees=10),
    ][:n_augments]

    for t_idx, tta_aug in enumerate(tta_augs):
        probs_this_aug = []

        transform_steps = []
        if use_color_norm:
            # Keep color normalization in the TTA path instead of replacing it.
            transform_steps.append(get_adapted_val_transforms().transforms[0])
        transform_steps.extend([
            tta_aug,
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        temp_dataset = DermNetDataset(
            dermnet_dir = DERMNET_DIR,
            split       = split,
            transform   = transforms.Compose(transform_steps),
            max_samples = max_samples,
            seed        = seed,
        )
        temp_loader = DataLoader(
            temp_dataset,
            batch_size  = BATCH_SIZE,
            shuffle     = False,
            num_workers = 0,
        )

        with torch.no_grad():
            for images, _ in temp_loader:
                images = images.to(device)
                logits = model(images)
                probs  = F.softmax(logits, dim=1)[:, 1]
                probs_this_aug.extend(probs.cpu().numpy())

        all_probs += np.array(probs_this_aug[:len(all_probs)])
        print(f"  TTA augment {t_idx + 1}/{len(tta_augs)} done")

    all_probs /= len(tta_augs)   # average across augmentations
    return all_probs, all_labels

# METRICS
def compute_metrics(probs, labels, threshold=CONF_THRESHOLD):
    """
    Compute accuracy, F1, and AUROC from predicted probabilities.

    Returns dict of metrics.
    """
    preds = (probs >= threshold).astype(int)

    accuracy = accuracy_score(labels, preds)
    f1       = f1_score(labels, preds, zero_division=0)
    try:
        auroc = roc_auc_score(labels, probs)
    except ValueError:
        auroc = float("nan")   # only one class present

    return {
        "accuracy": accuracy,
        "f1":       f1,
        "auroc":    auroc,
        "preds":    preds,
        "probs":    probs,
        "labels":   labels,
    }

# VISUALIZATIONS
def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im)

    classes    = ["Non-Acne", "Acne"]
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix — DermNet Test Set", fontsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def plot_roc_curve(labels, probs, auroc, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"ROC (AUROC = {auroc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve — DermNet Test Set", fontsize=13)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to: {save_path}")


def print_results_table(results_dict):
    """Print a formatted results table."""
    divider = "─" * 45

    print("\n")
    print("=" * 45)
    print("  DERMNET EVALUATION RESULTS")
    print("=" * 45)

    for split_name, metrics in results_dict.items():
        print(f"\n  [{split_name}]")
        print(divider)
        print(f"  {'Accuracy':<20} {metrics['accuracy']:.4f}")
        print(f"  {'F1-Score':<20} {metrics['f1']:.4f}")
        print(f"  {'AUROC':<20} {metrics['auroc']:.4f}")
        print(divider)

        if "labels" in metrics and "preds" in metrics:
            print(f"\n  Classification Report:")
            report = classification_report(
                metrics["labels"],
                metrics["preds"],
                target_names=["Non-Acne", "Acne"],
                zero_division=0,
            )
            for line in report.split("\n"):
                print(f"  {line}")

    print("=" * 45)


def save_results_txt(results_dict, save_path):
    """Save results to a text file."""
    with open(save_path, "w") as f:
        f.write("DERMNET EVALUATION RESULTS\n")
        f.write("=" * 45 + "\n\n")

        for split_name, metrics in results_dict.items():
            f.write(f"[{split_name}]\n")
            f.write(f"Accuracy : {metrics['accuracy']:.4f}\n")
            f.write(f"F1-Score : {metrics['f1']:.4f}\n")
            f.write(f"AUROC    : {metrics['auroc']:.4f}\n\n")

            if "labels" in metrics and "preds" in metrics:
                report = classification_report(
                    metrics["labels"],
                    metrics["preds"],
                    target_names=["Non-Acne", "Acne"],
                    zero_division=0,
                )
                f.write("Classification Report:\n")
                f.write(report + "\n")

    print(f"Results saved to: {save_path}")

# MAIN
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("\nLoading classifier...")
    model = load_model()
    model.to(device)
    model.eval()

    # Choose transforms
    if USE_COLOR_NORM:
        print("Using Reinhard color normalization for inference")
        val_transform = get_adapted_val_transforms()
    else:
        val_transform = get_val_transforms()

    results_dict = {}

    # ── 1. Quick check on 20 DermNet train samples ──
    print("\n── Step 1: 20-sample domain gap check ──")
    try:
        dev_dataset = DermNetDataset(
            split       = "train",
            transform   = val_transform,
            max_samples = 20,
        )
        dev_probs, dev_labels = run_inference(model, dev_dataset, device, use_tta=False)
        dev_metrics = compute_metrics(dev_probs, dev_labels)
        results_dict["20-sample dev set"] = dev_metrics
        print(f"Dev set — Acc: {dev_metrics['accuracy']:.4f} | "
              f"F1: {dev_metrics['f1']:.4f} | "
              f"AUROC: {dev_metrics['auroc']:.4f}")
    except FileNotFoundError as e:
        print(f"Skipping dev set: {e}")

    # ── 2. Full DermNet test set evaluation ──
    print("\n── Step 2: Full DermNet test set ──")
    try:
        test_dataset = DermNetDataset(
            split     = "test",
            transform = val_transform,
        )

        if USE_TTA:
            print(f"Running inference with TTA...")
        else:
            print(f"Running inference...")

        test_probs, test_labels = run_inference(model, test_dataset, device, use_tta=USE_TTA)
        test_metrics = compute_metrics(test_probs, test_labels)
        results_dict["DermNet test set"] = test_metrics

        # Save visualizations
        plot_confusion_matrix(
            test_labels,
            test_metrics["preds"],
            save_path = OUTPUT_DIR / "confusion_matrix.png",
        )
        plot_roc_curve(
            test_labels,
            test_probs,
            auroc     = test_metrics["auroc"],
            save_path = OUTPUT_DIR / "roc_curve.png",
        )

    except FileNotFoundError as e:
        print(f"Skipping test set: {e}")

    # ── Print and save results ──
    print_results_table(results_dict)
    save_results_txt(results_dict, OUTPUT_DIR / "dermnet_results.txt")


if __name__ == "__main__":
    evaluate()