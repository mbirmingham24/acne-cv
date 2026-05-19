"""
extract_patches.py
------------------
Creates a binary classification dataset from ACNE04 by cropping:
    - Positive patches: regions inside bounding boxes (acne lesions)
    - Negative patches: random regions that don't overlap any bounding box (clear skin)

Usage:
    python part2_classification/extract_patches.py

Outputs:
    outputs/patches/
    ├── positive/
    │   ├── nodules_and_cysts/
    │   ├── papules/
    │   ├── pustules/
    │   └── whitehead_and_blackhead/
    └── negative/
        └── clear_skin/
"""

import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# PATHS
ROOT         = Path(__file__).resolve().parent.parent
YOLO_DIR     = ROOT / "data" / "acne04_yolo"
OUTPUT_DIR   = ROOT / "outputs" / "patches"

# SETTINGS
PATCH_SIZE      = 128      # all patches resized to this square size
PADDING         = 10       # pixels of padding added around each positive box crop
NEG_PER_IMAGE   = 10        # how many negative patches to sample per image
NEG_IOU_THRESH  = 0.1      # a negative patch must have IoU < this with all GT boxes
MAX_NEG_ATTEMPTS= 50       # max random attempts to find a valid negative region
RANDOM_SEED     = 42
SPLITS          = ["train", "valid", "test"]

# ACNE04 class names (0-indexed, matches YOLO label files)
CLASS_NAMES = [
    "nodules_and_cysts",
    "papules",
    "pustules",
    "whitehead_and_blackhead",
]

# SETUP OUTPUT FOLDERS
def setup_output_dirs():
    # Clear and recreate output directory for fresh run
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for class_name in CLASS_NAMES:
        (OUTPUT_DIR / "positive" / class_name).mkdir(parents=True, exist_ok=True)

    (OUTPUT_DIR / "negative" / "clear_skin").mkdir(parents=True, exist_ok=True)
    print(f"Output directory ready: {OUTPUT_DIR}\n")

# IoU UTILITY
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in xyxy pixel format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0

# LABEL LOADER
def load_yolo_boxes(label_path, img_w, img_h):
    """
    Load YOLO format annotations and convert to pixel xyxy.

    Returns:
        list of (class_idx, [x1, y1, x2, y2]) tuples
    """
    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls, cx, cy, w, h = map(float, parts)
            cls = int(cls)

            # Convert normalized xywh → pixel xyxy
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            # Clamp to image boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append((cls, [x1, y1, x2, y2]))

    return boxes

# PATCH EXTRACTORS
def extract_positive_patch(img, box, padding=PADDING):
    """
    Crop the region inside a bounding box with optional padding.
    Returns resized patch or None if crop is invalid.
    """
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = box

    # Add padding and clamp
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(img_w, x2 + padding)
    y2 = min(img_h, y2 + padding)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))


def extract_negative_patch(img, gt_boxes, patch_size=PATCH_SIZE,
                            iou_thresh=NEG_IOU_THRESH,
                            max_attempts=MAX_NEG_ATTEMPTS):
    """
    Sample a random crop from the image that doesn't overlap any GT box.
    Returns resized patch or None if no valid region found.
    """
    img_h, img_w = img.shape[:2]

    # Can't crop a patch larger than the image
    if img_w < patch_size or img_h < patch_size:
        return None

    for _ in range(max_attempts):
        # Random top-left corner
        x1 = random.randint(0, img_w - patch_size)
        y1 = random.randint(0, img_h - patch_size)
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        candidate = [x1, y1, x2, y2]

        # Check IoU against all GT boxes
        valid = all(
            compute_iou(candidate, gt_box) < iou_thresh
            for _, gt_box in gt_boxes
        )

        if valid:
            crop = img[y1:y2, x1:x2]
            return cv2.resize(crop, (PATCH_SIZE, PATCH_SIZE))

    return None   # couldn't find a clean negative region

# MAIN EXTRACTION LOOP
def extract_patches():
    random.seed(RANDOM_SEED)
    setup_output_dirs()

    pos_counts = {name: 0 for name in CLASS_NAMES}
    neg_count  = 0
    skip_count = 0

    for split in SPLITS:
        img_dir = YOLO_DIR / split / "images"
        lbl_dir = YOLO_DIR / split / "labels"

        if not img_dir.exists():
            print(f"Warning: {img_dir} not found, skipping split '{split}'")
            continue

        image_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"Processing split '{split}': {len(image_paths)} images")

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                skip_count += 1
                continue

            img_h, img_w = img.shape[:2]
            label_path   = lbl_dir / (img_path.stem + ".txt")
            gt_boxes     = load_yolo_boxes(label_path, img_w, img_h)

            # ── Positive patches ──
            for cls_idx, box in gt_boxes:
                patch = extract_positive_patch(img, box)
                if patch is None:
                    continue

                class_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else "unknown"
                save_dir   = OUTPUT_DIR / "positive" / class_name
                save_path  = save_dir / f"{img_path.stem}_box{pos_counts[class_name]:04d}.jpg"

                cv2.imwrite(str(save_path), patch)
                pos_counts[class_name] += 1

            # ── Negative patches ──
            # Only sample negatives from images that have at least one GT box
            # so we know the rest of the face is actually clear skin
            if len(gt_boxes) > 0:
                for _ in range(NEG_PER_IMAGE):
                    patch = extract_negative_patch(img, gt_boxes)
                    if patch is None:
                        continue

                    save_path = OUTPUT_DIR / "negative" / "clear_skin" / \
                                f"{img_path.stem}_neg{neg_count:05d}.jpg"
                    cv2.imwrite(str(save_path), patch)
                    neg_count += 1

    # ── Summary ──
    print("\n── Patch Extraction Complete ──")
    print(f"Positive patches:")
    total_pos = 0
    for name, count in pos_counts.items():
        print(f"  {name:<30} {count}")
        total_pos += count
    print(f"  {'TOTAL':<30} {total_pos}")
    print(f"Negative patches (clear skin):     {neg_count}")
    print(f"Skipped images:                    {skip_count}")
    print(f"\nClass balance ratio (pos:neg): {total_pos}:{neg_count}")

    if neg_count > total_pos * 2:
        print("Note: negatives outnumber positives significantly.")
        print("Consider reducing NEG_PER_IMAGE or using weighted sampling in training.")
    elif total_pos > neg_count * 2:
        print("Note: positives outnumber negatives significantly.")
        print("Consider increasing NEG_PER_IMAGE.")

    print(f"\nPatches saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    extract_patches()