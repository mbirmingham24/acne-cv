"""
train_yolo.py
-------------
Fine-tunes YOLOv5 on the ACNE04 dataset using the Ultralytics library.

Usage:
    python part1_detection/train_yolo.py

Outputs:
    - Trained weights saved to outputs/checkpoints/yolo/weights/best.pt
    - Training logs and metrics saved alongside weights
"""

import os
from pathlib import Path
from ultralytics import YOLO

# PATHS  —  adjust if your folder layout differs
ROOT        = Path(__file__).resolve().parent.parent   # acne-cv/
DATA_YAML   = ROOT / "data" / "acne04_yolo" / "data.yaml"
OUTPUT_DIR  = ROOT / "outputs" / "checkpoints" / "yolo"

# HYPERPARAMETERS
MODEL_WEIGHTS = "yolov5su.pt"   # pretrained YOLOv5s (downloaded automatically)
EPOCHS        = 50              # increase to 100 for better results if time allows
IMG_SIZE      = 640             # standard YOLOv5 input resolution
BATCH_SIZE    = 16              # reduce to 8 if you get out-of-memory errors
PATIENCE      = 10              # early stopping: stop if no improvement for 10 epochs

def train():
    # Sanity check — make sure the data.yaml exists before starting
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"Could not find data.yaml at {DATA_YAML}\n"
            "Make sure you downloaded the ACNE04 YOLO format dataset from Roboflow "
            "and placed it in data/acne04_yolo/"
        )

    print(f"Loading pretrained model: {MODEL_WEIGHTS}")
    print(f"Dataset config:           {DATA_YAML}")
    print(f"Output directory:         {OUTPUT_DIR}")
    print(f"Training for {EPOCHS} epochs with image size {IMG_SIZE}px\n")

    # Load a pretrained YOLOv5s model
    # 'yolov5su.pt' is the Ultralytics-format YOLOv5s — downloaded automatically on first run
    model = YOLO(MODEL_WEIGHTS)

    # Fine-tune on ACNE04
    results = model.train(
        data       = str(DATA_YAML),
        epochs     = EPOCHS,
        imgsz      = IMG_SIZE,
        batch      = BATCH_SIZE,
        patience   = PATIENCE,
        project    = str(OUTPUT_DIR),   # where to save runs
        name       = "acne04",          # subfolder name inside project
        exist_ok   = True,              # overwrite previous run if re-running
        pretrained = True,
        verbose    = True,
    )

    # Print final validation metrics
    print("\n── Training complete ──")
    print(f"Best weights saved to: {OUTPUT_DIR / 'acne04' / 'weights' / 'best.pt'}")
    print(f"mAP@0.5:     {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"mAP@0.5:0.95:{results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"Precision:   {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"Recall:      {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")

    return results


def validate():
    """
    Run validation on the test set using the best saved weights.
    Call this separately after training to get clean test-set numbers.
    """
    best_weights = OUTPUT_DIR / "acne04" / "weights" / "best.pt"

    if not best_weights.exists():
        raise FileNotFoundError(
            f"No trained weights found at {best_weights}. Run train() first."
        )

    print(f"\nRunning validation on test set with: {best_weights}")
    model = YOLO(str(best_weights))

    metrics = model.val(
        data   = str(DATA_YAML),
        split  = "test",            # evaluate on test split specifically
        imgsz  = IMG_SIZE,
        batch  = BATCH_SIZE,
        verbose= True,
    )

    print("\n── Test Set Results ──")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    return metrics


if __name__ == "__main__":
    train()
    validate()