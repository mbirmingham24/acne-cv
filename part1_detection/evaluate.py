"""
evaluate.py
-----------
Evaluates both trained models (YOLOv5 and Faster R-CNN) on the ACNE04 test set
and prints a side-by-side comparison table of mAP, Precision, Recall, and IoU.

Usage:
    python part1_detection/evaluate.py

Requirements:
    - YOLOv5 best weights at:      outputs/checkpoints/yolo/acne04/weights/best.pt
    - Faster R-CNN best weights at: outputs/checkpoints/faster_rcnn/best.pt
    - ACNE04 YOLO format at:        data/acne04_yolo/
    - ACNE04 COCO format at:        data/acne04_coco/
"""

import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms as T
from ultralytics import YOLO

# Import our Faster R-CNN loader from the training script
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from train_faster_rcnn import load_model as load_frcnn, ACNE04CocoDataset, collate_fn, NUM_CLASSES
from torch.utils.data import DataLoader

# PATHS
ROOT         = Path(__file__).resolve().parent.parent
DATA_YAML    = ROOT / "data" / "acne04_yolo" / "data.yaml"
YOLO_WEIGHTS = ROOT / "outputs" / "checkpoints" / "yolo" / "acne04" / "weights" / "best.pt"
FRCNN_WEIGHTS= ROOT / "outputs" / "checkpoints" / "faster_rcnn" / "best.pt"
COCO_TEST    = ROOT / "data" / "acne04_coco" / "test" / "_annotations.coco.json"

# Confidence threshold for filtering predictions
CONF_THRESHOLD = 0.25
IOU_THRESHOLD  = 0.5      # IoU threshold for a prediction to count as a true positive

# IoU UTILITIES
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xyxy format.
    box1, box2: [x_min, y_min, x_max, y_max]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0
    return intersection / union


def compute_average_iou(pred_boxes, gt_boxes):
    """
    For each predicted box, find the best-matching ground truth box
    and return the mean IoU across all predictions.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0.0

    ious = []
    for pred in pred_boxes:
        best_iou = max(compute_iou(pred, gt) for gt in gt_boxes)
        ious.append(best_iou)
    return float(np.mean(ious))

# PRECISION / RECALL HELPERS

def match_predictions(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD):
    """
    Match predicted boxes to ground truth boxes.
    Returns (tp, fp, fn) counts for this image.

    A prediction is a True Positive if it overlaps a GT box with IoU >= threshold
    and that GT box hasn't already been matched.
    """
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    tp, fp = 0, 0

    for pred in pred_boxes:
        best_iou  = 0
        best_gt_i = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou  = iou
                best_gt_i = i

        if best_iou >= iou_threshold and best_gt_i >= 0:
            tp += 1
            matched_gt.add(best_gt_i)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

# YOLO EVALUATION
def evaluate_yolo():
    """
    Use Ultralytics built-in validator for mAP,
    then run manual inference for per-image IoU/P/R.
    """
    print("─" * 50)
    print("Evaluating YOLOv5...")
    print("─" * 50)

    if not YOLO_WEIGHTS.exists():
        print(f"ERROR: YOLOv5 weights not found at {YOLO_WEIGHTS}")
        print("Run train_yolo.py first.\n")
        return None

    model = YOLO(str(YOLO_WEIGHTS))

    # Built-in mAP evaluation on test split
    metrics = model.val(
        data    = str(DATA_YAML),
        split   = "test",
        imgsz   = 640,
        verbose = False,
    )

    map50     = metrics.box.map50
    map50_95  = metrics.box.map
    precision = metrics.box.mp
    recall    = metrics.box.mr

    # Manual IoU computation on test images
    test_img_dir = ROOT / "data" / "acne04_yolo" / "test" / "images"
    test_lbl_dir = ROOT / "data" / "acne04_yolo" / "test" / "labels"
    image_paths  = sorted(test_img_dir.glob("*.jpg")) + sorted(test_img_dir.glob("*.png"))

    all_ious = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for img_path in image_paths:
        # Load ground truth from YOLO label file
        label_path = test_lbl_dir / (img_path.stem + ".txt")
        gt_boxes_norm = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        gt_boxes_norm.append([cx, cy, w, h])

        # Run inference
        results = model(str(img_path), conf=CONF_THRESHOLD, verbose=False)
        result  = results[0]

        img_w = result.orig_shape[1]
        img_h = result.orig_shape[0]

        # Convert GT from normalized xywh → pixel xyxy
        gt_boxes = []
        for cx, cy, w, h in gt_boxes_norm:
            x1 = (cx - w / 2) * img_w
            y1 = (cy - h / 2) * img_h
            x2 = (cx + w / 2) * img_w
            y2 = (cy + h / 2) * img_h
            gt_boxes.append([x1, y1, x2, y2])

        # Get predicted boxes in xyxy pixel format
        pred_boxes = []
        if result.boxes is not None and len(result.boxes) > 0:
            pred_boxes = result.boxes.xyxy.cpu().numpy().tolist()

        iou = compute_average_iou(pred_boxes, gt_boxes)
        all_ious.append(iou)

        tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    avg_iou = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "model":      "YOLOv5s",
        "mAP@0.5":    map50,
        "mAP@0.5:95": map50_95,
        "Precision":  precision,
        "Recall":     recall,
        "Avg IoU":    avg_iou,
        "TP":         total_tp,
        "FP":         total_fp,
        "FN":         total_fn,
    }

# FASTER R-CNN EVALUATION
def evaluate_faster_rcnn():
    """
    Run Faster R-CNN on the COCO test set and compute mAP, P, R, IoU.
    Uses pycocotools for mAP to match the standard evaluation protocol.
    """
    print("─" * 50)
    print("Evaluating Faster R-CNN...")
    print("─" * 50)

    if not FRCNN_WEIGHTS.exists():
        print(f"ERROR: Faster R-CNN weights not found at {FRCNN_WEIGHTS}")
        print("Run train_faster_rcnn.py first.\n")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_frcnn(str(FRCNN_WEIGHTS))
    model.to(device)
    model.eval()

    test_dataset = ACNE04CocoDataset("test")
    test_loader  = DataLoader(
        test_dataset,
        batch_size = 4,
        shuffle    = False,
        collate_fn = collate_fn,
        num_workers= 0,
    )

    all_ious = []
    total_tp, total_fp, total_fn = 0, 0, 0

    # Collect predictions in COCO format for pycocotools mAP
    coco_results = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = [img.to(device) for img in images]
            preds  = model(images)   # list of dicts: boxes, labels, scores

            for pred, target in zip(preds, targets):
                image_id  = target["image_id"].item()
                gt_boxes  = target["boxes"].numpy().tolist()

                # Filter by confidence
                keep       = pred["scores"] >= CONF_THRESHOLD
                pred_boxes = pred["boxes"][keep].cpu().numpy().tolist()
                pred_scores= pred["scores"][keep].cpu().numpy().tolist()
                pred_labels= pred["labels"][keep].cpu().numpy().tolist()

                # IoU and P/R
                iou = compute_average_iou(pred_boxes, gt_boxes)
                all_ious.append(iou)

                tp, fp, fn = match_predictions(pred_boxes, gt_boxes)
                total_tp += tp
                total_fp += fp
                total_fn += fn

                # Format for pycocotools
                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    x1, y1, x2, y2 = box
                    coco_results.append({
                        "image_id":   image_id,
                        "category_id": label,
                        "bbox":        [x1, y1, x2 - x1, y2 - y1],  # xyxy → xywh
                        "score":       score,
                    })

    # Compute mAP with pycocotools
    map50, map50_95 = compute_coco_map(coco_results)

    # Compute precision and recall from TP/FP/FN
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    avg_iou   = float(np.mean(all_ious)) if all_ious else 0.0

    return {
        "model":      "Faster R-CNN",
        "mAP@0.5":    map50,
        "mAP@0.5:95": map50_95,
        "Precision":  precision,
        "Recall":     recall,
        "Avg IoU":    avg_iou,
        "TP":         total_tp,
        "FP":         total_fp,
        "FN":         total_fn,
    }


def compute_coco_map(coco_results):
    """
    Use pycocotools to compute mAP@0.5 and mAP@0.5:0.95.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        coco_gt   = COCO(str(COCO_TEST))
        coco_dt   = coco_gt.loadRes(coco_results) if coco_results else coco_gt.loadRes([])
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        map50_95 = coco_eval.stats[0]   # mAP@0.5:0.95
        map50    = coco_eval.stats[1]   # mAP@0.5

        return map50, map50_95

    except Exception as e:
        print(f"pycocotools mAP computation failed: {e}")
        print("Returning 0.0 for mAP values.")
        return 0.0, 0.0

# PRINT COMPARISON TABLE
def print_comparison_table(yolo_results, frcnn_results):
    metrics = ["mAP@0.5", "mAP@0.5:95", "Precision", "Recall", "Avg IoU"]

    col_w = 18
    header = f"{'Metric':<20}" + f"{'YOLOv5s':>{col_w}}" + f"{'Faster R-CNN':>{col_w}}"
    divider = "─" * len(header)

    print("\n")
    print("=" * len(header))
    print("  MODEL COMPARISON — ACNE04 TEST SET")
    print("=" * len(header))
    print(header)
    print(divider)

    for m in metrics:
        y_val = yolo_results.get(m, 0.0)  if yolo_results  else 0.0
        f_val = frcnn_results.get(m, 0.0) if frcnn_results else 0.0
        print(f"{m:<20}{y_val:>{col_w}.4f}{f_val:>{col_w}.4f}")

    print(divider)

    # TP / FP / FN
    for k in ["TP", "FP", "FN"]:
        y_val = yolo_results.get(k, 0)  if yolo_results  else 0
        f_val = frcnn_results.get(k, 0) if frcnn_results else 0
        print(f"{k:<20}{y_val:>{col_w}}{f_val:>{col_w}}")

    print("=" * len(header))
    print()

# MAIN
if __name__ == "__main__":
    yolo_results  = evaluate_yolo()
    frcnn_results = evaluate_faster_rcnn()
    print_comparison_table(yolo_results, frcnn_results)    