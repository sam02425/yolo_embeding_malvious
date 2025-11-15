#!/usr/bin/env python3
"""
Train DOLG embedding model on retail dataset for fine-grained product recognition

This script fine-tunes the DOLG model specifically on grocery/retail items
to learn discriminative embeddings that can distinguish between similar products.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
import cv2
import numpy as np
from pathlib import Path
import yaml
from tqdm import tqdm
import json
from typing import List, Tuple, Dict
import random


class RetailProductDataset(Dataset):
    """Dataset for training DOLG embeddings on retail products"""
    
    def __init__(self, images_dir: str, labels_dir: str, transform=None, 
                 min_crop_size: int = 64, augment: bool = True):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.min_crop_size = min_crop_size
        self.augment = augment
        
        # Collect all valid crops
        self.crops = []
        self._load_crops()
        
        if len(self.crops) == 0:
            raise ValueError(f"No crops found! Check paths:\n  Images: {self.images_dir}\n  Labels: {self.labels_dir}")
        
        print(f"✅ Loaded {len(self.crops)} product crops for training")
        
        # Count samples per class
        class_counts = {}
        for _, class_id in self.crops:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        print(f"📊 Classes: {len(class_counts)}, Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}, avg={len(self.crops)/len(class_counts):.1f}")
    
    def _load_crops(self):
        """Extract all product crops from images and labels"""
        label_files = sorted(self.labels_dir.glob('*.txt'))
        
        for label_file in tqdm(label_files, desc="Loading crops"):
            # Get corresponding image
            img_file = self.images_dir / label_file.with_suffix('.jpg').name
            if not img_file.exists():
                img_file = self.images_dir / label_file.with_suffix('.png').name
            if not img_file.exists():
                continue
            
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Read labels
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    
                    # Convert to pixel coordinates
                    x1 = int((x_center - width / 2) * w)
                    y1 = int((y_center - height / 2) * h)
                    x2 = int((x_center + width / 2) * w)
                    y2 = int((y_center + height / 2) * h)
                    
                    # Validate crop
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 - x1 < self.min_crop_size or y2 - y1 < self.min_crop_size:
                        continue
                    
                    # Extract crop
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    
                    self.crops.append((crop.copy(), class_id))
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop, class_id = self.crops[idx]
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation if enabled
        if self.augment:
            crop_rgb = self._augment_crop(crop_rgb)
        
        # Apply transforms
        if self.transform:
            crop_rgb = self.transform(crop_rgb)
        
        return crop_rgb, class_id
    
    def _augment_crop(self, crop: np.ndarray) -> np.ndarray:
        """Apply data augmentation"""
        # Random horizontal flip
        if random.random() > 0.5:
            crop = cv2.flip(crop, 1)
        
        # Random brightness/contrast
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2)  # Contrast
            beta = random.randint(-20, 20)    # Brightness
            crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
        
        # Random rotation (small angles)
        if random.random() > 0.7:
            angle = random.uniform(-15, 15)
            h, w = crop.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        
        return crop


class DOLGModel(nn.Module):
    """DOLG (Deep Orthogonal Local and Global) model for retail embeddings"""
    
    def __init__(self, num_classes: int, embedding_dim: int = 128, 
                 backbone: str = 'efficientnet_b0', pretrained: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, 
                                         features_only=True, out_indices=[3])
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)[0]
            feat_dim = features.shape[1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # L2 normalization for embeddings
        self.l2_norm = lambda x: nn.functional.normalize(x, p=2, dim=1)
    
    def forward(self, x, return_embeddings=False):
        # Extract features
        features = self.backbone(x)[0]  # [B, C, H, W]
        
        # Global pooling
        global_feat = self.global_pool(features).flatten(1)  # [B, C]
        
        # Get embeddings
        embeddings = self.embedding_head(global_feat)
        embeddings = self.l2_norm(embeddings)
        
        if return_embeddings:
            return embeddings
        
        # Get logits
        logits = self.classifier(embeddings)
        
        return logits, embeddings


class ArcFaceLoss(nn.Module):
    """ArcFace loss for better embedding learning"""
    
    def __init__(self, embedding_dim: int, num_classes: int, 
                 scale: float = 30.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.threshold = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, embeddings, labels):
        # Normalize weights
        weight_norm = nn.functional.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity
        cosine = nn.functional.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Compute phi = cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        
        # One-hot encoding
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Combine
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale
        
        return output


def train_dolg_model(
    dataset_yaml: str,
    output_dir: str = 'dolg_retail_model',
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    embedding_dim: int = 128,
    device: str = 'cuda:0',
    use_arcface: bool = True
):
    """Train DOLG model on retail dataset"""
    
    print(f"\n{'='*80}")
    print("🚀 TRAINING DOLG MODEL ON RETAIL DATASET")
    print(f"{'='*80}\n")
    
    # Load dataset config
    with open(dataset_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    num_classes = len(dataset_info['names'])
    print(f"📊 Dataset: {num_classes} classes")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    print("\n📦 Loading training data...")
    # Handle both path formats: ending with /images or not
    train_path = dataset_info['train']
    if train_path.endswith('/images'):
        train_images_dir = train_path
        train_labels_dir = train_path.replace('/images', '/labels')
    else:
        train_images_dir = train_path + '/images'
        train_labels_dir = train_path + '/labels'
    
    train_dataset = RetailProductDataset(
        images_dir=train_images_dir,
        labels_dir=train_labels_dir,
        transform=train_transform,
        augment=True
    )
    
    print("\n📦 Loading validation data...")
    val_path = dataset_info['val']
    if val_path.endswith('/images'):
        val_images_dir = val_path
        val_labels_dir = val_path.replace('/images', '/labels')
    else:
        val_images_dir = val_path + '/images'
        val_labels_dir = val_path + '/labels'
    
    val_dataset = RetailProductDataset(
        images_dir=val_images_dir,
        labels_dir=val_labels_dir,
        transform=val_transform,
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"\n🏗️  Creating DOLG model (embedding_dim={embedding_dim})...")
    model = DOLGModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        backbone='efficientnet_b0',
        pretrained=True
    ).to(device)
    
    # Loss and optimizer
    if use_arcface:
        print("📐 Using ArcFace loss for better embedding learning")
        criterion = ArcFaceLoss(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            scale=30.0,
            margin=0.5
        ).to(device)
        ce_loss = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    print(f"\n🏋️  Starting training for {epochs} epochs...\n")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if use_arcface:
                logits, embeddings = model(images)
                arcface_logits = criterion(embeddings, labels)
                loss = ce_loss(arcface_logits, labels)
            else:
                logits, embeddings = model(images)
                loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            if use_arcface:
                _, predicted = torch.max(arcface_logits, 1)
            else:
                _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]  "):
                images, labels = images.to(device), labels.to(device)
                
                if use_arcface:
                    logits, embeddings = model(images)
                    arcface_logits = criterion(embeddings, labels)
                    loss = ce_loss(arcface_logits, labels)
                    _, predicted = torch.max(arcface_logits, 1)
                else:
                    logits, embeddings = model(images)
                    loss = criterion(logits, labels)
                    _, predicted = torch.max(logits, 1)
                
                val_loss += loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"\n📊 Epoch {epoch+1}/{epochs}:")
        print(f"   Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.6f}\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
                'num_classes': num_classes
            }, output_path / 'dolg_retail_best.pth')
            print(f"✅ Saved best model (val_acc: {val_acc:.2f}%)\n")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'embedding_dim': embedding_dim,
                'num_classes': num_classes
            }, output_path / f'dolg_retail_epoch_{epoch+1}.pth')
    
    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
        'embedding_dim': embedding_dim,
        'num_classes': num_classes
    }, output_path / 'dolg_retail_final.pth')
    
    # Save training history
    with open(output_path / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Model saved to: {output_path}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DOLG model on retail dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--output-dir", type=str, default="dolg_retail_model", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--no-arcface", action="store_true", help="Don't use ArcFace loss")
    
    args = parser.parse_args()
    
    train_dolg_model(
        dataset_yaml=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        embedding_dim=args.embedding_dim,
        device=args.device,
        use_arcface=not args.no_arcface
    )
