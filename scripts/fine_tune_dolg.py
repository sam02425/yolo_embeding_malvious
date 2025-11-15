#!/usr/bin/env python3
"""
Fine-tune the DOLG embedding backbone on YOLO-style grocery datasets.

This trains the EfficientNet-based embedding head on cropped SKU images so the
Milvus embeddings become SKU-discriminative instead of generic ImageNet
features. The resulting checkpoint is compatible with populate_milvus_embeddings.py.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml


def resolve_path(path: Path | str, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _resolve_split_entry(data: dict, split: str) -> str:
    """Handle different YAML naming conventions (val vs valid)."""
    aliases = {
        "train": ["train"],
        "valid": ["valid", "val", "validation"],
        "val": ["val", "valid", "validation"],
        "test": ["test"],
    }
    candidates = aliases.get(split, [split])
    for key in candidates:
        if key in data:
            return data[key]
    raise KeyError(f"Split '{split}' not found in dataset YAML (checked {candidates})")


class YoloCropDataset(Dataset):
    """Create per-object crops from a YOLO dataset for classification training."""

    def __init__(
        self,
        dataset_yaml: Path,
        split: str = "train",
        min_box_area: int = 1000,
        augment: bool = True,
    ):
        self.dataset_yaml = Path(dataset_yaml).resolve()
        with open(self.dataset_yaml, "r") as f:
            data = yaml.safe_load(f)

        self.class_names = data["names"]
        base_dir = self.dataset_yaml.parent

        split_entry = _resolve_split_entry(data, split)
        split_path = resolve_path(split_entry, base_dir)
        if split_path.name == "images":
            images_dir = split_path
            labels_dir = split_path.parent / "labels"
        else:
            images_dir = split_path / "images"
            labels_dir = split_path / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory missing: {images_dir}")

        self.samples: List[Tuple[Path, Tuple[int, int, int, int], int]] = []
        for label_path in sorted(labels_dir.glob("*.txt")):
            with open(label_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            if not lines:
                continue
            image_path = images_dir / label_path.with_suffix(".jpg").name
            if not image_path.exists():
                image_path = images_dir / label_path.with_suffix(".png").name
            if not image_path.exists():
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                continue
            h, w = img.shape[:2]

            for line in lines:
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                if (x2 - x1) * (y2 - y1) < min_box_area:
                    continue
                self.samples.append((image_path, (x1, y1, x2, y2), class_id))

        if not self.samples:
            raise RuntimeError("No labeled crops found for DOLG fine-tuning.")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_tfms = [
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.ToTensor(),
            normalize,
        ]
        eval_tfms = [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
        self.transforms = transforms.Compose(train_tfms if augment else eval_tfms)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, (x1, y1, x2, y2), class_id = self.samples[idx]
        image = cv2.imread(str(image_path))
        crop = image[y1:y2, x1:x2]
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transforms(crop)
        return tensor, class_id


class ProductionDOLGModel(nn.Module):
    """EfficientNet-B0 backbone with an embedding layer."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        from torchvision import models

        self.backbone = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class DOLGClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.encoder = ProductionDOLGModel(embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)


@dataclass
class TrainConfig:
    dataset_yaml: Path
    output_path: Path
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda:0"
    embedding_dim: int = 128
    min_box_area: int = 1000


def train_dolg(config: TrainConfig) -> None:
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    train_ds = YoloCropDataset(config.dataset_yaml, "train",
                               min_box_area=config.min_box_area, augment=True)
    val_ds = YoloCropDataset(config.dataset_yaml, "valid",
                             min_box_area=config.min_box_area, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size,
                              shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    model = DOLGClassifier(config.embedding_dim, len(train_ds.class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val = 0.0
    config.output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / max(1, val_total)
        print(f"[Epoch {epoch}/{config.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.encoder.state_dict(), config.output_path)
            print(f"✅ Saved improved encoder weights to {config.output_path}")

    print(f"Training complete. Best val accuracy: {best_val:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune DOLG embeddings on grocery crops.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the YOLO dataset YAML (e.g., data/grocery_augmented/grocery_augmented.yaml)")
    parser.add_argument("--output", type=str, default="dolg_models/dolg_grocery_best.pth",
                        help="Where to save the fine-tuned encoder weights.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of fine-tuning epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--min-box-area", type=int, default=1000,
                        help="Skip boxes smaller than this many pixels (removes noise).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = TrainConfig(
        dataset_yaml=Path(args.dataset),
        output_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        embedding_dim=args.embedding_dim,
        min_box_area=args.min_box_area,
    )
    train_dolg(config)


if __name__ == "__main__":
    main()
