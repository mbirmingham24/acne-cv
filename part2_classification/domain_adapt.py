"""
domain_adapt.py
---------------
Domain adaptation utilities to close the gap between ACNE04 (source domain)
and DermNet (target domain).

Techniques implemented:
    1. Reinhard color normalization  — matches color statistics of source to target
    2. Histogram matching            — matches full pixel distribution
    3. Enhanced augmentation pipeline — used in dataset.py at training time

These can be applied as preprocessing steps before inference on DermNet,
or as part of the training pipeline to make the model more robust.

Usage (standalone — normalizes a folder of images):
    python part2_classification/domain_adapt.py

Usage (as a module):
    from domain_adapt import reinhard_normalize, histogram_match, get_adapted_transforms
"""

import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
PATCHES_DIR = ROOT / "outputs" / "patches" / "positive"
DERMNET_DIR = ROOT / "data" / "dermnet" / "train" / "Acne and Rosacea Photos"

PATCH_SIZE  = 224

# TECHNIQUE 1: REINHARD COLOR NORMALIZATION
"""
Reinhard et al. (2001) — transfers color appearance from a target image
to a source image by matching mean and std in LAB color space.

Why LAB? LAB separates luminance (L) from color (A, B), so we can
normalize color and brightness somewhat independently. Works better
than matching in RGB directly.

In our case:
    source = ACNE04 patch (consumer camera, close-up face photo)
    target = DermNet image (clinical camera, different color profile)

We compute mean/std statistics over a reference set of DermNet images
and apply those statistics to ACNE04 patches at training time,
making ACNE04 look more like DermNet.
"""

def compute_lab_stats(image_paths: list, max_images: int = 100):
    """
    Compute mean and std of LAB channels over a set of images.
    Used to get target domain statistics.

    Args:
        image_paths : list of Path objects
        max_images  : cap to avoid slow computation

    Returns:
        (mean, std) each shape (3,) for L, A, B channels
    """
    l_vals, a_vals, b_vals = [], [], []

    sample = image_paths[:max_images]
    for path in sample:
        img = cv2.imread(str(path))
        if img is None:
            continue

        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_vals.append(lab[:, :, 0].mean())
        a_vals.append(lab[:, :, 1].mean())
        b_vals.append(lab[:, :, 2].mean())

    if not l_vals:
        # Fallback to neutral stats
        return np.array([128.0, 128.0, 128.0]), np.array([20.0, 5.0, 5.0])

    mean = np.array([np.mean(l_vals), np.mean(a_vals), np.mean(b_vals)])
    std  = np.array([np.std(l_vals),  np.std(a_vals),  np.std(b_vals)])
    std  = np.clip(std, 1.0, None)   # avoid division by zero

    return mean, std


def reinhard_normalize(
    img_bgr: np.ndarray,
    target_mean: np.ndarray,
    target_std:  np.ndarray,
) -> np.ndarray:
    """
    Apply Reinhard color normalization to a single BGR image.

    Transfers the color statistics (mean, std per LAB channel)
    from the target domain to the source image.

    Args:
        img_bgr     : source image as BGR numpy array (H, W, 3) uint8
        target_mean : LAB channel means of target domain, shape (3,)
        target_std  : LAB channel stds  of target domain, shape (3,)

    Returns:
        Normalized BGR image as uint8
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    for i in range(3):
        channel      = lab[:, :, i]
        src_mean     = channel.mean()
        src_std      = channel.std()
        src_std      = max(src_std, 1.0)

        # Shift and scale to match target statistics
        lab[:, :, i] = (channel - src_mean) * (target_std[i] / src_std) + target_mean[i]

    # Clip to valid LAB range and convert back
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# TECHNIQUE 2: HISTOGRAM MATCHING
"""
Histogram matching transforms the pixel distribution of a source image
to match the pixel distribution of a reference (target) image.

More aggressive than Reinhard — it matches the full distribution,
not just mean and std. Can sometimes be too aggressive and look
unnatural, so we offer a blend parameter.
"""

def match_histograms_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match histogram of a single channel from source to reference.

    Args:
        source    : 2D numpy array (single channel)
        reference : 2D numpy array (single channel)

    Returns:
        Histogram-matched source channel
    """
    src_hist, bins = np.histogram(source.flatten(), 256, [0, 256])
    ref_hist, _    = np.histogram(reference.flatten(), 256, [0, 256])

    # Compute CDFs
    src_cdf = src_hist.cumsum().astype(np.float64)
    ref_cdf = ref_hist.cumsum().astype(np.float64)

    # Normalize CDFs
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Build lookup table: for each source pixel value, find closest ref value
    lookup = np.zeros(256, dtype=np.uint8)
    ref_idx = 0
    for src_val in range(256):
        while ref_idx < 255 and ref_cdf[ref_idx] < src_cdf[src_val]:
            ref_idx += 1
        lookup[src_val] = ref_idx

    return lookup[source]


def histogram_match(
    source_bgr:    np.ndarray,
    reference_bgr: np.ndarray,
    blend:         float = 0.7,
) -> np.ndarray:
    """
    Match the histogram of source to reference image, per channel.

    Args:
        source_bgr    : source image BGR uint8
        reference_bgr : reference image BGR uint8
        blend         : 0.0 = keep source, 1.0 = full match to reference

    Returns:
        Histogram-matched source image BGR uint8
    """
    matched = np.zeros_like(source_bgr)

    for c in range(3):
        matched[:, :, c] = match_histograms_channel(
            source_bgr[:, :, c],
            reference_bgr[:, :, c],
        )

    # Blend between original and matched
    result = cv2.addWeighted(source_bgr, 1.0 - blend, matched, blend, 0)
    return result.astype(np.uint8)

# TECHNIQUE 3: PYTORCH TRANSFORM WRAPPERS
class ReinhardNormalizeTransform:
    """
    PyTorch-compatible transform that applies Reinhard normalization.
    Precomputes target domain statistics once from a set of reference images.

    Usage:
        transform = ReinhardNormalizeTransform(reference_dir)
        normalized_pil = transform(pil_image)
    """

    def __init__(self, reference_dir: Path, max_ref_images: int = 100):
        """
        Args:
            reference_dir  : folder of reference (target domain) images
            max_ref_images : how many reference images to use for stats
        """
        reference_dir = Path(reference_dir)

        if reference_dir.exists():
            ref_paths = list(reference_dir.glob("*.jpg")) + \
                        list(reference_dir.glob("*.png")) + \
                        list(reference_dir.glob("*.jpeg"))
            self.target_mean, self.target_std = compute_lab_stats(ref_paths, max_ref_images)
            print(f"ReinhardNormalize: computed stats from {min(len(ref_paths), max_ref_images)} "
                  f"reference images in {reference_dir.name}")
        else:
            # Fallback neutral stats — no normalization effect
            print(f"Warning: reference dir {reference_dir} not found. "
                  f"Using neutral Reinhard stats (no effect).")
            self.target_mean = np.array([128.0, 128.0, 128.0])
            self.target_std  = np.array([20.0,  5.0,   5.0])

    def __call__(self, img: Image.Image) -> Image.Image:
        """Apply Reinhard normalization to a PIL image."""
        img_bgr      = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        normalized   = reinhard_normalize(img_bgr, self.target_mean, self.target_std)
        img_rgb      = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)


class HistogramMatchTransform:
    """
    PyTorch-compatible transform that applies histogram matching
    to a random reference image from the target domain.
    """

    def __init__(self, reference_dir: Path, blend: float = 0.5, max_ref_images: int = 50):
        reference_dir = Path(reference_dir)

        self.blend       = blend
        self.ref_images  = []

        if reference_dir.exists():
            ref_paths = list(reference_dir.glob("*.jpg")) + \
                        list(reference_dir.glob("*.png")) + \
                        list(reference_dir.glob("*.jpeg"))
            ref_paths = ref_paths[:max_ref_images]

            for p in ref_paths:
                img = cv2.imread(str(p))
                if img is not None:
                    self.ref_images.append(img)

            print(f"HistogramMatch: loaded {len(self.ref_images)} reference images")
        else:
            print(f"Warning: reference dir {reference_dir} not found. "
                  f"Histogram matching will be skipped.")

    def __call__(self, img: Image.Image) -> Image.Image:
        if not self.ref_images:
            return img

        ref_bgr  = random.choice(self.ref_images)
        img_bgr  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        matched  = histogram_match(img_bgr, ref_bgr, blend=self.blend)
        img_rgb  = cv2.cvtColor(matched, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

# ADAPTED TRANSFORM PIPELINES

def get_adapted_train_transforms(reference_dir: Optional[Path] = None):
    """
    Training transforms with domain adaptation.
    Adds Reinhard normalization on top of standard augmentation.

    Use this instead of get_train_transforms() from dataset.py
    when you want to apply color normalization during training.
    """
    if reference_dir is None:
        reference_dir = DERMNET_DIR

    transform_list = []

    # Color normalization first (before augmentation)
    if Path(reference_dir).exists():
        transform_list.append(ReinhardNormalizeTransform(reference_dir))

    transform_list += [
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(
            brightness=0.5,
            contrast=0.5,
            saturation=0.4,
            hue=0.15,
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(p=0.15),
    ]

    return transforms.Compose(transform_list)


def get_adapted_val_transforms(reference_dir: Optional[Path] = None):
    """
    Validation/test transforms with color normalization only — no augmentation.
    Use this for DermNet inference.
    """
    if reference_dir is None:
        reference_dir = DERMNET_DIR

    transform_list = []

    if Path(reference_dir).exists():
        transform_list.append(ReinhardNormalizeTransform(reference_dir))

    transform_list += [
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ]

    return transforms.Compose(transform_list)

# STANDALONE: NORMALIZE A FOLDER OF IMAGES
def normalize_folder(
    source_dir:    Path,
    reference_dir: Path,
    output_dir:    Path,
    method:        str = "reinhard",
    max_images:    int = 200,
):
    """
    Apply color normalization to all images in source_dir and save to output_dir.
    Useful for visualizing what normalization looks like before/after.

    Args:
        source_dir    : folder of images to normalize (e.g. ACNE04 patches)
        reference_dir : folder of reference images (e.g. DermNet)
        output_dir    : where to save normalized images
        method        : 'reinhard' or 'histogram'
        max_images    : max source images to process
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = list(Path(source_dir).glob("*.jpg")) + \
                   list(Path(source_dir).glob("*.png"))
    source_paths = source_paths[:max_images]

    ref_paths = list(Path(reference_dir).glob("*.jpg")) + \
                list(Path(reference_dir).glob("*.png"))

    if method == "reinhard":
        target_mean, target_std = compute_lab_stats(ref_paths)
        print(f"Reinhard stats — mean: {target_mean}, std: {target_std}")

    elif method == "histogram":
        ref_images = [cv2.imread(str(p)) for p in ref_paths[:50] if cv2.imread(str(p)) is not None]
    else:
        raise ValueError(f"method must be 'reinhard' or 'histogram', got '{method}'")

    print(f"Normalizing {len(source_paths)} images using {method}...")

    for i, src_path in enumerate(source_paths):
        img = cv2.imread(str(src_path))
        if img is None:
            continue

        if method == "reinhard":
            result = reinhard_normalize(img, target_mean, target_std)
        else:
            ref = random.choice(ref_images)
            result = histogram_match(img, ref, blend=0.6)

        out_path = output_dir / src_path.name
        cv2.imwrite(str(out_path), result)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(source_paths)} done")

    print(f"Saved normalized images to {output_dir}")


if __name__ == "__main__":
    # Demo: normalize a sample of ACNE04 patches to look like DermNet
    sample_source = ROOT / "outputs" / "patches" / "positive" / "papules"
    sample_output = ROOT / "outputs" / "patches" / "normalized_demo"

    if sample_source.exists() and DERMNET_DIR.exists():
        normalize_folder(
            source_dir    = sample_source,
            reference_dir = DERMNET_DIR,
            output_dir    = sample_output,
            method        = "reinhard",
            max_images    = 20,
        )
        print(f"\nCheck {sample_output} to see normalization results.")
    else:
        print("Run extract_patches.py first and make sure DermNet is downloaded.")
        print("Then re-run this script to see normalization demo.")