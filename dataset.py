"""Dataset and data loading for real vs AI-generated image classification."""
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# Class indices: 0 = real, 1 = AI-generated
CLASS_NAMES = ("real", "ai")


def get_transforms(image_size=224, train=True):
    """Standard transforms; optional augmentation for training."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


class RealVsAIDataset(Dataset):
    """Dataset from folder structure: data/real/ and data/ai/."""

    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []  # (path, label): 0=real, 1=ai

        for label, folder in enumerate(CLASS_NAMES):
            folder_path = self.root / folder
            if not folder_path.is_dir():
                continue
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
                for path in folder_path.glob(ext):
                    self.samples.append((str(path), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label
