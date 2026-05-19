"""
train_faster_rcnn.py
--------------------
Fine-tunes a Faster R-CNN (ResNet-50 FPN backbone) on ACNE04 using torchvision.
 
Usage:
    python part1_detection/train_faster_rcnn.py
 
Outputs:
    - Best weights saved to outputs/checkpoints/faster_rcnn/best.pt
    - Last weights saved to outputs/checkpoints/faster_rcnn/last.pt
    - Training log saved to outputs/checkpoints/faster_rcnn/train_log.csv
"""
 
import os
import csv
import json
import time
from pathlib import Path
 
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
 
import numpy as np
from PIL import Image
 
# PATHS
ROOT       = Path(__file__).resolve().parent.parent
COCO_DIR   = ROOT / "data" / "acne04_coco"
OUTPUT_DIR = ROOT / "outputs" / "checkpoints" / "faster_rcnn"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
 
# HYPERPARAMETERS
NUM_CLASSES = 5          # 4 acne classes + 1 background (background is always class 0)
EPOCHS      = 20         # Faster R-CNN converges slower, but 20 is reasonable
BATCH_SIZE  = 4          # Faster R-CNN is memory-heavy — keep this low
LR          = 0.005
MOMENTUM    = 0.9
WEIGHT_DECAY= 0.0005
LR_STEP     = 5          # drop LR every 5 epochs
LR_GAMMA    = 0.5        # multiply LR by this each step
IMG_SIZE    = 640
 
# DATASET
class ACNE04CocoDataset(Dataset):
    """
    Loads ACNE04 in COCO format for torchvision Faster R-CNN.
 
    Faster R-CNN expects each sample to be:
        image  : FloatTensor [3, H, W] in range [0, 1]
        target : dict with keys:
                    boxes  — FloatTensor [N, 4] in xyxy format
                    labels — Int64Tensor [N]
    """
 
    def __init__(self, split: str):
        """
        Args:
            split: one of 'train', 'valid', 'test'
        """
        self.img_dir = COCO_DIR / split 
        ann_path     = COCO_DIR / split / "_annotations.coco.json"
 
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                "Make sure you downloaded ACNE04 in COCO format from Roboflow "
                "and placed it in data/acne04_coco/"
            )
 
        with open(ann_path) as f:
            coco = json.load(f)
 
        # Build lookup: image_id → file_name
        self.id_to_filename = {img["id"]: img["file_name"] for img in coco["images"]}
        self.image_ids      = list(self.id_to_filename.keys())
 
        # Build lookup: image_id → list of annotations
        self.annotations = {img_id: [] for img_id in self.image_ids}
        for ann in coco["annotations"]:
            self.annotations[ann["image_id"]].append(ann)
 
    def __len__(self):
        return len(self.image_ids)
 
    def __getitem__(self, idx):
        image_id  = self.image_ids[idx]
        file_name = self.id_to_filename[image_id]
        img_path  = self.img_dir / file_name
 
        # Load image and convert to tensor
        img = Image.open(img_path).convert("RGB")
        img = T.functional.to_tensor(img)   # [3, H, W], range [0, 1]
 
        # Load annotations
        anns = self.annotations[image_id]
 
        if len(anns) == 0:
            # Image with no annotations — return empty target
            target = {
                "boxes":    torch.zeros((0, 4), dtype=torch.float32),
                "labels":   torch.zeros((0,),   dtype=torch.int64),
                "image_id": torch.tensor([image_id]),
            }
            return img, target
 
        boxes  = []
        labels = []
 
        for ann in anns:
            # COCO format: [x_min, y_min, width, height]
            # Faster R-CNN expects: [x_min, y_min, x_max, y_max]
            x, y, w, h = ann["bbox"]
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
 
            # Skip degenerate boxes
            if x_max <= x_min or y_max <= y_min:
                continue
 
            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(ann["category_id"])   # 1-indexed, background = 0
 
        target = {
            "boxes":    torch.tensor(boxes,  dtype=torch.float32),
            "labels":   torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([image_id]),
        }
 
        return img, target
 
 
def collate_fn(batch):
    """
    Custom collate — Faster R-CNN takes a list of (image, target) pairs,
    not a stacked tensor, because images can have different numbers of boxes.
    """
    return tuple(zip(*batch))
 
# MODEL
def build_model(num_classes: int):
    """
    Load pretrained Faster R-CNN and replace the classification head
    to output num_classes instead of the default 91 COCO classes.
    """
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
 
    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
 
    return model
 
# TRAINING LOOP
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("Warning: Training on CPU will be very slow. A GPU is strongly recommended.\n")
 
    # Datasets and loaders
    train_dataset = ACNE04CocoDataset("train")
    valid_dataset = ACNE04CocoDataset("valid")
 
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = True,
        collate_fn  = collate_fn,
        num_workers = 0,    # set to 2-4 on Linux/Mac for speed; keep 0 on Windows
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        collate_fn  = collate_fn,
        num_workers = 0,
    )
 
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}\n")
 
    # Model
    model = build_model(NUM_CLASSES)
    model.to(device)
 
    # Optimizer and scheduler
    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)
 
    # Logging
    log_path = OUTPUT_DIR / "train_log.csv"
    log_file = open(log_path, "w", newline="")
    writer   = csv.writer(log_file)
    writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
 
    best_val_loss = float("inf")
 
    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []
        t0 = time.time()
 
        for batch_idx, (images, targets) in enumerate(train_loader):
            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
            loss_dict = model(images, targets)
            losses    = sum(loss for loss in loss_dict.values())
 
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
 
            train_losses.append(losses.item())
 
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {losses.item():.4f}")
 
        avg_train_loss = np.mean(train_losses)
 
        # ── Validate ──
        # Note: Faster R-CNN returns losses only in train mode.
        # We switch to train mode but disable gradient updates to get val loss.
        val_losses = []
        with torch.no_grad():
            for images, targets in valid_loader:
                images  = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
                # model must be in train mode to return loss dict
                model.train()
                loss_dict = model(images, targets)
                losses    = sum(loss for loss in loss_dict.values())
                val_losses.append(losses.item())
 
        avg_val_loss = np.mean(val_losses)
        elapsed      = time.time() - t0
        current_lr   = scheduler.get_last_lr()[0]
 
        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")
 
        writer.writerow([epoch, avg_train_loss, avg_val_loss, current_lr])
        log_file.flush()
 
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")
            print(f"  ✓ New best model saved (val loss: {best_val_loss:.4f})")
 
        # Save latest model
        torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")
 
        scheduler.step()
 
    log_file.close()
    print(f"\n── Training complete ──")
    print(f"Best weights: {OUTPUT_DIR / 'best.pt'}")
    print(f"Training log: {log_path}")
 
 
# INFERENCE HELPER  (used by evaluate.py later)
def load_model(weights_path: str = None):
    """
    Load the trained Faster R-CNN model from saved weights.
    Defaults to the best checkpoint.
    """
    if weights_path is None:
        weights_path = OUTPUT_DIR / "best.pt"
 
    model = build_model(NUM_CLASSES)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model
 
 
if __name__ == "__main__":
    train()