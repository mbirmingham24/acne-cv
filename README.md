# Acne-CV

Computer vision pipeline for acne analysis with two parts:

- `part1_detection`: lesion detection on ACNE04 using YOLOv5 and Faster R-CNN
- `part2_classification`: binary acne vs. non-acne classification using ResNet-50, plus DermNet evaluation and Grad-CAM visualizations

---

## Project Structure

```text
acne-cv/
├── part1_detection/
│   ├── train_yolo.py
│   ├── train_faster_rcnn.py
│   ├── evaluate.py
│   └── visualize_detections.py
├── part2_classification/
│   ├── dataset.py
│   ├── train_classifier.py
│   ├── evaluate_dermnet.py
│   ├── extract_patches.py
│   ├── gradcam.py
│   └── domain_adapt.py
├── data/
├── outputs/
├── requirements.txt
└── requirements_cuda.txt
```

---

## Features

- YOLOv5 lesion detection on ACNE04
- Faster R-CNN lesion detection on ACNE04
- Side-by-side detector evaluation on the ACNE04 test set
- ResNet-50 patch classifier for acne vs. clear skin
- Cross-domain evaluation on DermNet
- Test-time augmentation (TTA) for DermNet evaluation
- Optional Reinhard color normalization for domain adaptation
- Grad-CAM visualizations for classifier predictions

---

## Requirements

- Python 3.10+ recommended
- Windows works; the scripts use `num_workers=0`, which is suitable there
- A CUDA-capable GPU is strongly recommended for training

### Step 1 — Install PyTorch

Install PyTorch manually before running either requirements file, as the correct
command depends on your system.

**CPU only:**

```powershell
pip install torch torchvision
```

**CUDA (NVIDIA GPU — strongly recommended):**

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

If the above fails due to a network or firewall issue, add trusted host flags:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

To verify CUDA is working after install:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name())"
```

This should print `True` and your GPU name. If it prints `False`, you got the CPU-only
build — uninstall and reinstall using the CUDA command above.

### Step 2 — Install remaining dependencies

**CPU:**

```powershell
pip install -r requirements.txt
```

**CUDA:**

```powershell
pip install -r requirements_cuda.txt
```

---

## Expected Data Layout

The scripts expect local datasets under `data/` and generated artifacts under `outputs/`.

### Detection datasets

```text
data/
├── acne04_yolo/
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
└── acne04_coco/
    ├── train/
    │   ├── images/
    │   └── _annotations.coco.json
    ├── valid/
    │   ├── images/
    │   └── _annotations.coco.json
    └── test/
        ├── images/
        └── _annotations.coco.json
```

### Classification datasets

```text
data/
└── dermnet/
    ├── train/
    └── test/

outputs/
└── patches/
    ├── positive/
    └── negative/
        └── clear_skin/
```

Notes:

- `part1_detection/train_yolo.py` expects `data/acne04_yolo/data.yaml`
- `part1_detection/train_faster_rcnn.py` expects COCO annotations under `data/acne04_coco/`
- `part2_classification/train_classifier.py` expects extracted image patches under `outputs/patches/`
- `part2_classification/evaluate_dermnet.py` and `part2_classification/gradcam.py` expect DermNet under `data/dermnet/`

---

## How To Run

Run all commands from the repository root:

```powershell
cd c:\Users\mbirm\acne-cv
```

### 1. Train YOLOv5 detector

```powershell
python part1_detection/train_yolo.py
```

Outputs:

- `outputs/checkpoints/yolo/acne04/weights/best.pt`

### 2. Train Faster R-CNN detector

```powershell
python part1_detection/train_faster_rcnn.py
```

Outputs:

- `outputs/checkpoints/faster_rcnn/best.pt`
- `outputs/checkpoints/faster_rcnn/last.pt`

### 3. Compare both detectors on ACNE04

```powershell
python part1_detection/evaluate.py
```

Prints a comparison table with [mAP@0.5](mailto:mAP@0.5), [mAP@0.5](mailto:mAP@0.5):95, precision, recall, average IoU, and TP/FP/FN counts.

### 4. Visualize detections

```powershell
python part1_detection/visualize_detections.py
```

Outputs:

- `outputs/detections/` — side-by-side images showing ground truth, YOLOv5, and Faster R-CNN predictions

### 5. Extract acne/clear-skin patches from ACNE04

```powershell
python part2_classification/extract_patches.py
```

Outputs:

- `outputs/patches/positive/` (class-specific acne patches)
- `outputs/patches/negative/clear_skin/` (clear-skin negatives)

### 6. Train the patch classifier

```powershell
python part2_classification/train_classifier.py
```

Outputs:

- `outputs/checkpoints/classifier/best.pt`
- `outputs/checkpoints/classifier/last.pt`
- `outputs/checkpoints/classifier/train_log.csv`
- `outputs/checkpoints/classifier/training_curves.png`

### 7. Evaluate the classifier on DermNet

```powershell
python part2_classification/evaluate_dermnet.py
```

Outputs:

- `outputs/checkpoints/classifier/dermnet_results.txt`
- `outputs/checkpoints/classifier/confusion_matrix.png`
- `outputs/checkpoints/classifier/roc_curve.png`

### 8. Generate Grad-CAM visualizations

```powershell
python part2_classification/gradcam.py
```

Outputs:

- `outputs/gradcam/`
- `outputs/gradcam/gradcam_grid.png`

---

## Recommended Workflow

If you are starting from scratch:

1. Install PyTorch manually (see Requirements above).
2. Install remaining dependencies from `requirements.txt` or `requirements_cuda.txt`.
3. Place the ACNE04 YOLO and COCO datasets in `data/`.
4. Train the detection models (steps 1 and 2 above).
5. Run `part1_detection/evaluate.py` to compare them.
6. Run `part2_classification/extract_patches.py` to build `outputs/patches/`.
7. Train the classifier with `part2_classification/train_classifier.py`.
8. Evaluate on DermNet with `part2_classification/evaluate_dermnet.py`.
9. Generate explanations with `part2_classification/gradcam.py`.

---

## Notes On Evaluation

- DermNet evaluation supports TTA — the model predicts on several augmented versions of each image and averages the probabilities.
- DermNet evaluation also supports optional Reinhard color normalization to reduce the source-target domain gap.
- Both options are enabled by default in `evaluate_dermnet.py`.

---

## Common Issues

### Package not found errors

If any `pip install` command fails with "Could not find a version that satisfies the requirement",
your network may be blocking PyPI. Add trusted host flags to any install command:

```powershell
pip install <package-name> --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

For example:

```powershell
pip install ultralytics --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

Updating pip first can also help:

```powershell
python -m pip install --upgrade pip
```

### CUDA not detected

If `torch.cuda.is_available()` returns `False`, you have the CPU-only version of PyTorch installed.
Uninstall and reinstall with the CUDA index URL:

```powershell
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### data.yaml paths on Windows

If YOLOv5 throws a `FileNotFoundError` about missing image paths, open
`data/acne04_yolo/data.yaml` and make sure the paths are relative with no leading slash:

```yaml
train: train/images
val: valid/images
test: test/images
```

### Missing dataset files

If a script raises `FileNotFoundError`, double-check the expected folder layout above.
Most scripts validate their input paths early and print a helpful message.

### CPU training is too slow

Install the CUDA version of PyTorch (see Requirements) and verify with:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### `pycocotools` install problems on Windows

Try the Windows-specific build:

```powershell
pip install pycocotools-windows --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

---

## Outputs

Generated outputs are written under:

- `outputs/checkpoints/` — model weights and training logs
- `outputs/detections/` — bounding box visualizations
- `outputs/gradcam/` — Grad-CAM heatmap overlays
- `outputs/patches/` — extracted classification patches

These are local artifacts and are not tracked by git.