"""
visualize_detections.py
-----------------------
Runs both trained models on sample test images and saves side-by-side
visualizations with predicted and ground truth bounding boxes overlaid.

Usage:
    python part1_detection/visualize_detections.py

Outputs:
    - Individual comparison images saved to outputs/detections/
    - One image per test sample showing: original | YOLOv5 | Faster R-CNN
"""

import sys
import random
import torch
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

sys.path.append(str(Path(__file__).resolve().parent))
from train_faster_rcnn import load_model as load_frcnn, NUM_CLASSES

# PATHS
ROOT         = Path(__file__).resolve().parent.parent
YOLO_WEIGHTS = ROOT / "outputs" / "checkpoints" / "yolo" / "acne04" / "weights" / "best.pt"
FRCNN_WEIGHTS= ROOT / "outputs" / "checkpoints" / "faster_rcnn" / "best.pt"
TEST_IMG_DIR = ROOT / "data" / "acne04_yolo" / "test" / "images"
TEST_LBL_DIR = ROOT / "data" / "acne04_yolo" / "test" / "labels"
OUTPUT_DIR   = ROOT / "outputs" / "detections"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SETTINGS
NUM_SAMPLES    = 10       # how many test images to visualize
CONF_THRESHOLD = 0.05     # minimum confidence to show a box
RANDOM_SEED    = 42       # for reproducible sample selection

# Class names matching ACNE04 (index 0 = background for Faster R-CNN)
CLASS_NAMES = [
    "background",             # index 0 — Faster R-CNN only
    "nodules and cysts",      # index 1
    "papules",                # index 2
    "pustules",               # index 3
    "whitehead and blackhead" # index 4
]

# YOLO class names (0-indexed, no background)
YOLO_CLASS_NAMES = CLASS_NAMES[1:]

# COLORS  (BGR for OpenCV)
GT_COLOR   = (0, 255, 0)      # green  — ground truth boxes
YOLO_COLOR = (0, 0, 255)      # red    — YOLOv5 predictions
FRCNN_COLOR= (255, 128, 0)    # orange — Faster R-CNN predictions

# Per-class colors for ground truth (makes multi-class easier to read)
CLASS_COLORS = [
    (0, 255, 0),      # nodules and cysts   — green
    (255, 255, 0),    # papules             — yellow
    (0, 255, 255),    # pustules            — cyan
    (255, 0, 255),    # whitehead/blackhead — magenta
]

# DRAWING UTILITIES
def draw_box(img, box, label, color, thickness=2):
    """
    Draw a single bounding box with a label on an image (in place).

    Args:
        img       : numpy array (H, W, 3) BGR
        box       : [x1, y1, x2, y2] in pixels
        label     : string to display above the box
        color     : BGR tuple
        thickness : line thickness
    """
    x1, y1, x2, y2 = [int(v) for v in box]

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thick = 1
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, font_thick)

    label_y = max(y1 - 4, th + 4)
    cv2.rectangle(img, (x1, label_y - th - 4), (x1 + tw + 4, label_y), color, -1)
    cv2.putText(img, label, (x1 + 2, label_y - 2), font, font_scale, (0, 0, 0), font_thick)

    return img


def draw_boxes(img, boxes, labels, scores, color, class_names=None, per_class_colors=False):
    """
    Draw multiple boxes on a copy of the image.

    Args:
        img              : numpy array BGR
        boxes            : list of [x1, y1, x2, y2]
        labels           : list of class indices
        scores           : list of confidence scores (None for GT)
        color            : default BGR color
        class_names      : list of class name strings
        per_class_colors : if True, use CLASS_COLORS per class index
    """
    img = img.copy()
    for box, label_idx, score in zip(boxes, labels, scores):
        # Pick color
        c = CLASS_COLORS[(label_idx - 1) % len(CLASS_COLORS)] if per_class_colors else color

        # Build label string
        name = class_names[label_idx] if class_names and label_idx < len(class_names) else str(label_idx)
        if score is not None:
            text = f"{name} {score:.2f}"
        else:
            text = name

        draw_box(img, box, text, c)

    return img


def add_title_bar(img, title, bar_height=30):
    """Add a colored title bar above an image."""
    bar = np.zeros((bar_height, img.shape[1], 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)   # dark gray
    cv2.putText(bar, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.vstack([bar, img])

# GROUND TRUTH LOADER
def load_gt_boxes(label_path, img_w, img_h):
    """
    Load YOLO-format ground truth boxes and convert to pixel xyxy.
    Returns (boxes, labels) where labels are 0-indexed class ids.
    """
    boxes, labels = [], []

    if not label_path.exists():
        return boxes, labels

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, w, h = map(float, parts)
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls))   # 0-indexed for YOLO

    return boxes, labels

# MAIN VISUALIZATION
def visualize(num_samples=NUM_SAMPLES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load models ──
    print("Loading YOLOv5...")
    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLOv5 weights not found at {YOLO_WEIGHTS}. Run train_yolo.py first.")
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    print("Loading Faster R-CNN...")
    if not FRCNN_WEIGHTS.exists():
        raise FileNotFoundError(f"Faster R-CNN weights not found at {FRCNN_WEIGHTS}. Run train_faster_rcnn.py first.")
    frcnn_model = load_frcnn(str(FRCNN_WEIGHTS))
    frcnn_model.to(device)
    frcnn_model.eval()

    # ── Sample test images ──
    all_images = sorted(TEST_IMG_DIR.glob("*.jpg")) + sorted(TEST_IMG_DIR.glob("*.png"))
    if len(all_images) == 0:
        raise FileNotFoundError(f"No images found in {TEST_IMG_DIR}")

    random.seed(RANDOM_SEED)
    samples = random.sample(all_images, min(num_samples, len(all_images)))
    print(f"\nVisualizing {len(samples)} test images → {OUTPUT_DIR}\n")

    for i, img_path in enumerate(samples):
        print(f"  Processing {i+1}/{len(samples)}: {img_path.name}")

        # Load image as BGR numpy array (OpenCV format)
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  Warning: could not read {img_path}, skipping.")
            continue

        img_h, img_w = img_bgr.shape[:2]

        # ── Ground truth ──
        label_path = TEST_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes, gt_labels = load_gt_boxes(label_path, img_w, img_h)

        gt_img = draw_boxes(
            img_bgr,
            gt_boxes,
            gt_labels,
            scores        = [None] * len(gt_labels),
            color         = GT_COLOR,
            class_names   = YOLO_CLASS_NAMES,   # 0-indexed
            per_class_colors = True,
        )
        gt_img = add_title_bar(gt_img, f"Ground Truth ({len(gt_boxes)} boxes)")

        # ── YOLOv5 predictions ──
        yolo_results = yolo_model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        yolo_result  = yolo_results[0]

        yolo_boxes, yolo_labels, yolo_scores = [], [], []
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            yolo_boxes  = yolo_result.boxes.xyxy.cpu().numpy().tolist()
            yolo_labels = yolo_result.boxes.cls.cpu().numpy().astype(int).tolist()
            yolo_scores = yolo_result.boxes.conf.cpu().numpy().tolist()

        yolo_img = draw_boxes(
            img_bgr,
            yolo_boxes,
            yolo_labels,
            scores      = yolo_scores,
            color       = YOLO_COLOR,
            class_names = YOLO_CLASS_NAMES,
        )
        yolo_img = add_title_bar(yolo_img, f"YOLOv5s ({len(yolo_boxes)} predictions)")

        # ── Faster R-CNN predictions ──
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage

        pil_img   = PILImage.open(img_path).convert("RGB")
        img_tensor= TF.to_tensor(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            frcnn_preds = frcnn_model(img_tensor)[0]

        keep         = frcnn_preds["scores"] >= CONF_THRESHOLD
        frcnn_boxes  = frcnn_preds["boxes"][keep].cpu().numpy().tolist()
        frcnn_labels = frcnn_preds["labels"][keep].cpu().numpy().astype(int).tolist()
        frcnn_scores = frcnn_preds["scores"][keep].cpu().numpy().tolist()

        frcnn_img = draw_boxes(
            img_bgr,
            frcnn_boxes,
            frcnn_labels,
            scores      = frcnn_scores,
            color       = FRCNN_COLOR,
            class_names = CLASS_NAMES,   # 1-indexed (0 = background)
        )
        frcnn_img = add_title_bar(frcnn_img, f"Faster R-CNN ({len(frcnn_boxes)} predictions)")

        # ── Combine side by side ──
        # All three panels: GT | YOLO | Faster R-CNN
        combined = np.hstack([gt_img, yolo_img, frcnn_img])

        # Add overall title bar
        title_bar = np.zeros((35, combined.shape[1], 3), dtype=np.uint8)
        title_bar[:] = (20, 20, 20)
        title = f"Image: {img_path.name}  |  Green=GT  Red=YOLOv5  Orange=FasterRCNN"
        cv2.putText(title_bar, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        combined = np.vstack([title_bar, combined])

        # Save
        out_path = OUTPUT_DIR / f"detection_{i+1:02d}_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"    Saved → {out_path.name}")

    print(f"\nDone. {len(samples)} visualizations saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    visualize()