"""
dataset.py
----------
PyTorch Dataset classes for loading:
    1. ACNE04 patches (for training the classifier)
    2. DermNet images (for cross-domain evaluation)

These are used by train_classifier.py and evaluate_dermnet.py.
"""

import os
import random
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

# PATHS
ROOT        = Path(__file__).resolve().parent.parent
PATCHES_DIR = ROOT / "outputs" / "patches"
DERMNET_DIR = ROOT / "data" / "dermnet"

# CONSTANTS
PATCH_SIZE  = 224     # ResNet-50 native input size — upsample from 128
RANDOM_SEED = 42

# DermNet acne-related folder names
# These are the folders in DermNet that correspond to acne
DERMNET_ACNE_FOLDERS = [
    "Acne and Rosacea Photos",
]


# TRANSFORMS
def get_train_transforms():
    """
    Augmented transforms for training.
    Aggressive augmentation helps close the domain gap between
    ACNE04 and DermNet by making the model robust to lighting,
    color, and geometric variation.
    """
    return transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.4,   # random brightness shift
            contrast=0.4,     # random contrast shift
            saturation=0.3,   # random saturation shift
            hue=0.1,          # slight hue shift
        ),
        transforms.RandomGrayscale(p=0.05),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
        transforms.RandomErasing(p=0.1),  # randomly mask small regions
    ])


def get_val_transforms():
    """
    Clean transforms for validation and test — no augmentation,
    just resize and normalize.
    """
    return transforms.Compose([
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_tta_transforms(n_augments=5):
    """
    Test-Time Augmentation (TTA) transforms.
    Returns a list of transform pipelines — we run inference
    with each and average the predictions for more robust results.
    """
    base = [
        transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]

    augment_options = [
        transforms.Compose(base),  # original
        transforms.Compose([transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                            transforms.RandomHorizontalFlip(p=1.0),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                            transforms.ColorJitter(brightness=0.2, contrast=0.2),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((int(PATCH_SIZE*1.1), int(PATCH_SIZE*1.1))),
                            transforms.CenterCrop(PATCH_SIZE),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
        transforms.Compose([transforms.Resize((PATCH_SIZE, PATCH_SIZE)),
                            transforms.RandomRotation(degrees=10),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]),
    ]

    return augment_options[:n_augments]


# DATASET 1: ACNE04 PATCHES
class AcnePatchDataset(Dataset):
    """
    Loads positive (acne) and negative (clear skin) patches
    extracted by extract_patches.py.

    Binary labels:
        1 = acne (positive)
        0 = clear skin (negative)
    """

    def __init__(
        self,
        patches_dir: Path = PATCHES_DIR,
        split: str = "train",
        val_fraction: float = 0.15,
        test_fraction: float = 0.10,
        transform=None,
        seed: int = RANDOM_SEED,
    ):
        """
        Args:
            patches_dir   : root of outputs/patches/
            split         : 'train', 'val', or 'test'
            val_fraction  : fraction of data for validation
            test_fraction : fraction of data for test
            transform     : torchvision transform pipeline
            seed          : random seed for reproducible splits
        """
        self.transform = transform
        self.samples   = []   # list of (path, label) tuples
        self.labels    = []   # parallel list of labels for WeightedRandomSampler

        # Collect all positive patches (label = 1)
        pos_dir = patches_dir / "positive"
        if not pos_dir.exists():
            raise FileNotFoundError(
                f"Positive patches not found at {pos_dir}\n"
                "Run extract_patches.py first."
            )

        for class_folder in sorted(pos_dir.iterdir()):
            if class_folder.is_dir():
                for img_path in sorted(class_folder.glob("*.jpg")):
                    self.samples.append((img_path, 1))
                    self.labels.append(1)

        # Collect all negative patches (label = 0)
        neg_dir = patches_dir / "negative" / "clear_skin"
        if not neg_dir.exists():
            raise FileNotFoundError(
                f"Negative patches not found at {neg_dir}\n"
                "Run extract_patches.py first."
            )

        for img_path in sorted(neg_dir.glob("*.jpg")):
            self.samples.append((img_path, 0))
            self.labels.append(0)

        # Shuffle and split
        random.seed(seed)
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        self.samples, self.labels = zip(*combined)
        self.samples = list(self.samples)
        self.labels  = list(self.labels)

        n = len(self.samples)
        n_test = int(n * test_fraction)
        n_val  = int(n * val_fraction)
        n_train= n - n_test - n_val

        if split == "train":
            self.samples = self.samples[:n_train]
            self.labels  = self.labels[:n_train]
        elif split == "val":
            self.samples = self.samples[n_train:n_train + n_val]
            self.labels  = self.labels[n_train:n_train + n_val]
        elif split == "test":
            self.samples = self.samples[n_train + n_val:]
            self.labels  = self.labels[n_train + n_val:]
        else:
            raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

        print(f"AcnePatchDataset [{split}]: {len(self.samples)} samples "
              f"({sum(self.labels)} positive, {len(self.labels)-sum(self.labels)} negative)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def get_weighted_sampler(self):
        """
        Returns a WeightedRandomSampler that balances positive and negative
        samples during training — prevents the model from learning to always
        predict the majority class.
        """
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos

        if n_pos == 0 or n_neg == 0:
            return None

        weight_pos = 1.0 / n_pos
        weight_neg = 1.0 / n_neg

        weights = [weight_pos if l == 1 else weight_neg for l in self.labels]
        weights = torch.tensor(weights, dtype=torch.float)

        return WeightedRandomSampler(
            weights     = weights,
            num_samples = len(weights),
            replacement = True,
        )

# DATASET 2: DERMNET
class DermNetDataset(Dataset):
    """
    Loads DermNet images for cross-domain evaluation.

    DermNet folder structure:
        dermnet/
        ├── train/
        │   ├── acne-and-rosacea-photos/
        │   ├── eczema-photos/
        │   └── ...
        └── test/
            ├── acne-and-rosacea-photos/
            ├── eczema-photos/
            └── ...

    Binary labels:
        1 = acne (folder is in DERMNET_ACNE_FOLDERS)
        0 = non-acne (everything else)
    """

    def __init__(
        self,
        dermnet_dir: Path = DERMNET_DIR,
        split: str = "test",
        transform=None,
        max_samples: Optional[int] = None,
        seed: int = RANDOM_SEED,
    ):
        """
        Args:
            dermnet_dir : root of data/dermnet/
            split       : 'train' or 'test'
            transform   : torchvision transform pipeline
            max_samples : if set, randomly sample this many images
                          (used for the 20-sample development set)
            seed        : random seed
        """
        self.split     = split
        self.transform = transform
        self.samples   = []   # list of (path, label) tuples
        self.labels    = []

        split_dir = dermnet_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"DermNet {split} directory not found at {split_dir}\n"
                "Make sure you downloaded DermNet from Kaggle and placed it in data/dermnet/"
            )

        # Walk all condition folders
        for condition_folder in sorted(split_dir.iterdir()):
            if not condition_folder.is_dir():
                continue

            # Binary label: is this an acne folder?
            label = 1 if condition_folder.name in DERMNET_ACNE_FOLDERS else 0

            for img_path in sorted(condition_folder.glob("*.jpg")) + \
                            sorted(condition_folder.glob("*.png")) + \
                            sorted(condition_folder.glob("*.jpeg")):
                self.samples.append((img_path, label))
                self.labels.append(label)

        # Optionally subsample (for the 20-sample development set)
        if max_samples is not None and max_samples < len(self.samples):
            random.seed(seed)
            combined = list(zip(self.samples, self.labels))
            random.shuffle(combined)
            combined = combined[:max_samples]
            self.samples, self.labels = zip(*combined)
            self.samples = list(self.samples)
            self.labels  = list(self.labels)

        n_acne    = sum(self.labels)
        n_nonacne = len(self.labels) - n_acne
        print(f"DermNetDataset [{split}]: {len(self.samples)} samples "
              f"({n_acne} acne, {n_nonacne} non-acne)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def get_image_paths(self):
        """Return just the file paths — used by gradcam.py."""
        return [s[0] for s in self.samples]

# QUICK TEST
if __name__ == "__main__":
    print("Testing AcnePatchDataset...")
    try:
        train_ds = AcnePatchDataset(split="train", transform=get_train_transforms())
        val_ds   = AcnePatchDataset(split="val",   transform=get_val_transforms())
        test_ds  = AcnePatchDataset(split="test",  transform=get_val_transforms())

        img, label = train_ds[0]
        print(f"Sample image shape: {img.shape}")
        print(f"Sample label: {label}")
        print(f"Sampler: {train_ds.get_weighted_sampler()}\n")
    except FileNotFoundError as e:
        print(f"Skipping patch dataset test: {e}\n")

    print("Testing DermNetDataset (20-sample dev set)...")
    try:
        dev_ds = DermNetDataset(split="train", max_samples=20, transform=get_val_transforms())
        img, label = dev_ds[0]
        print(f"Sample image shape: {img.shape}")
        print(f"Sample label: {label}")
    except FileNotFoundError as e:
        print(f"Skipping DermNet dataset test: {e}\n")