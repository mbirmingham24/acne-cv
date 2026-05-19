"""
gradcam.py
----------
Generates Grad-CAM heatmap visualizations for DermNet predictions.
Shows which regions of the image the classifier attends to when
making its acne vs. non-acne decision.

Usage:
    python part2_classification/gradcam.py

Outputs:
    - outputs/gradcam/  — one overlay image per sample
    - outputs/gradcam/gradcam_grid.png  — summary grid of all samples
"""

import sys
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from PIL import Image as PILImage
from torchvision import transforms

sys.path.append(str(Path(__file__).resolve().parent))
from dataset import DermNetDataset, get_val_transforms
from train_classifier import load_model, build_model, NUM_CLASSES
from domain_adapt import get_adapted_val_transforms

# PATHS
ROOT         = Path(__file__).resolve().parent.parent
OUTPUT_DIR   = ROOT / "outputs" / "gradcam"
DERMNET_DIR  = ROOT / "data" / "dermnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# SETTINGS
NUM_SAMPLES     = 12      # number of DermNet images to visualize (>= 10 required)
RANDOM_SEED     = 42
USE_COLOR_NORM  = True
PATCH_SIZE      = 224
CLASS_NAMES     = ["Non-Acne", "Acne"]

# GRAD-CAM IMPLEMENTATION
"""
Grad-CAM (Gradient-weighted Class Activation Mapping):
    1. Register a hook on the last convolutional layer to capture
       its output feature maps (activations)
    2. Run a forward pass and compute the class score
    3. Run a backward pass to get gradients of the class score
       w.r.t. those feature maps
    4. Global-average-pool the gradients → importance weights
    5. Weighted sum of feature maps → raw heatmap
    6. ReLU + normalize → final heatmap in [0, 1]
    7. Resize and overlay on original image

The heatmap shows which spatial regions contributed most
to the model's prediction.
"""

class GradCAM:
    def __init__(self, model, target_layer):
        """
        Args:
            model        : trained PyTorch model
            target_layer : the layer to hook (usually last conv layer)
        """
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activations)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            input_tensor : preprocessed image tensor, shape (1, 3, H, W)
            class_idx    : class to explain (None = predicted class)

        Returns:
            heatmap : numpy array shape (H, W), values in [0, 1]
            pred_class : predicted class index
            confidence : predicted probability
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(input_tensor)
        probs  = F.softmax(output, dim=1)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        confidence = probs[0, class_idx].item()

        # Backward pass for target class
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        # Global average pool gradients → weights
        gradients   = self.gradients[0]          # (C, H, W)
        activations = self.activations[0]        # (C, H, W)
        weights     = gradients.mean(dim=(1, 2)) # (C,)

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam, class_idx, confidence

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def get_target_layer(model):
    """
    Get the last convolutional layer of ResNet-50.
    This is layer4[-1].conv3 — the deepest spatial feature map.
    """
    return model.layer4[-1].conv3

# VISUALIZATION HELPERS
def overlay_heatmap(original_img_rgb, heatmap, alpha=0.45):
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img_rgb : numpy array (H, W, 3) RGB uint8
        heatmap          : numpy array (h, w) float in [0, 1]
        alpha            : heatmap opacity

    Returns:
        overlay : numpy array (H, W, 3) RGB uint8
    """
    H, W = original_img_rgb.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Apply colormap (jet: blue=low, red=high attention)
    heatmap_colored = cv2.applyColorMap(
        (heatmap_resized * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original
    overlay = (
        (1 - alpha) * original_img_rgb.astype(np.float32) +
        alpha       * heatmap_rgb.astype(np.float32)
    ).astype(np.uint8)

    return overlay


def make_panel(original_rgb, overlay_rgb, pred_class, confidence, true_label, img_name):
    """
    Create a two-panel image: original | grad-cam overlay.
    Adds prediction info as title.
    """
    H, W = original_rgb.shape[:2]
    panel = np.hstack([original_rgb, overlay_rgb])

    # Convert to PIL for text rendering
    pil  = PILImage.fromarray(panel)
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont
    draw = ImageDraw.Draw(pil)

    correct   = pred_class == true_label
    color     = (0, 200, 0) if correct else (220, 50, 50)
    pred_name = CLASS_NAMES[pred_class]
    true_name = CLASS_NAMES[true_label]
    text      = f"Pred: {pred_name} ({confidence:.2f}) | GT: {true_name} | {'✓' if correct else '✗'}"

    draw.rectangle([(0, 0), (panel.shape[1], 22)], fill=(30, 30, 30))
    draw.text((6, 4), text, fill=color)

    return np.array(pil)

# MAIN
def run_gradcam(num_samples=NUM_SAMPLES):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Load model ──
    print("\nLoading classifier...")
    model = load_model()
    model.to(device)
    model.eval()

    # ── Set up Grad-CAM ──
    target_layer = get_target_layer(model)
    gradcam      = GradCAM(model, target_layer)

    # ── Load DermNet test images ──
    if USE_COLOR_NORM:
        transform = get_adapted_val_transforms()
    else:
        transform = get_val_transforms()

    try:
        dataset = DermNetDataset(split="test", transform=None)  # no transform yet
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Sample a mix of acne and non-acne images for better visualization
    acne_indices    = [i for i, l in enumerate(dataset.labels) if l == 1]
    nonacne_indices = [i for i, l in enumerate(dataset.labels) if l == 0]

    random.seed(RANDOM_SEED)
    n_acne    = min(num_samples // 2, len(acne_indices))
    n_nonacne = min(num_samples - n_acne, len(nonacne_indices))

    sampled_indices = (
        random.sample(acne_indices, n_acne) +
        random.sample(nonacne_indices, n_nonacne)
    )
    random.shuffle(sampled_indices)

    print(f"Visualizing {len(sampled_indices)} images "
          f"({n_acne} acne, {n_nonacne} non-acne) → {OUTPUT_DIR}\n")

    # ── Preprocessing transform (separate from loading) ──
    preprocess = transform if USE_COLOR_NORM else get_val_transforms()

    panels = []

    for i, idx in enumerate(sampled_indices):
        img_path, true_label = dataset.samples[idx]
        img_name = Path(img_path).stem

        print(f"  [{i+1:02d}/{len(sampled_indices)}] {img_name} (GT: {CLASS_NAMES[true_label]})")

        # Load original image for display
        original_pil = PILImage.open(img_path).convert("RGB")
        original_rgb = np.array(original_pil.resize((PATCH_SIZE, PATCH_SIZE)))

        # Preprocess for model
        input_tensor = preprocess(original_pil).unsqueeze(0).to(device)

        # Generate Grad-CAM
        heatmap, pred_class, confidence = gradcam.generate(input_tensor)

        # Overlay heatmap
        overlay_rgb = overlay_heatmap(original_rgb, heatmap, alpha=0.45)

        # Build panel
        panel = make_panel(
            original_rgb = original_rgb,
            overlay_rgb  = overlay_rgb,
            pred_class   = pred_class,
            confidence   = confidence,
            true_label   = true_label,
            img_name     = img_name,
        )
        panels.append(panel)

        # Save individual image
        out_path = OUTPUT_DIR / f"gradcam_{i+1:02d}_{img_name}.jpg"
        PILImage.fromarray(panel).save(str(out_path))
        print(f"     Pred: {CLASS_NAMES[pred_class]} ({confidence:.2f}) | "
              f"{'CORRECT' if pred_class == true_label else 'WRONG'}")

    gradcam.remove_hooks()

    # ── Save summary grid ──
    save_grid(panels, save_path=OUTPUT_DIR / "gradcam_grid.png")
    print(f"\nDone. Individual images and grid saved to {OUTPUT_DIR}")


def save_grid(panels, save_path, ncols=3):
    """
    Arrange panels into a grid and save as a single PNG.
    Good for including in your report.
    """
    n      = len(panels)
    nrows  = (n + ncols - 1) // ncols

    # Pad panels list to fill grid
    while len(panels) < nrows * ncols:
        blank = np.ones_like(panels[0]) * 240
        panels.append(blank)

    rows = []
    for r in range(nrows):
        row_panels = panels[r * ncols: (r + 1) * ncols]
        rows.append(np.hstack(row_panels))

    grid = np.vstack(rows)

    fig, ax = plt.subplots(figsize=(18, nrows * 4))
    ax.imshow(grid)
    ax.axis("off")
    ax.set_title(
        "Grad-CAM Visualizations on DermNet\n"
        "Left = Original | Right = Grad-CAM Overlay | "
        "Green title = Correct | Red title = Wrong",
        fontsize=13, pad=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Summary grid saved to: {save_path}")


if __name__ == "__main__":
    run_gradcam()