#!/usr/bin/env python3
"""
SAM-Based Dataset Augmentation for Retail Products

This script creates a new augmented dataset by:
1. Using SAM to segment products from grocery/liquor datasets
2. Compositing products onto empty shelf backgrounds
3. Ensuring products don't overlap with existing annotations
4. Creating new YOLO format labels

Usage:
    python3 sam_augmentation.py \
        --output-dir data/sam_augmented \
        --num-samples 5000 \
        --device cuda:0
"""

import argparse
import os
import sys
from pathlib import Path
import random
import shutil
from typing import List, Tuple, Dict, Optional
import yaml
import json

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import urllib.request

# Import SAM
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    print("⚠️  Segment Anything Model not available. Please install: pip install git+https://github.com/facebookresearch/segment-anything.git")
    SAM_AVAILABLE = False


class SAMAugmentor:
    """SAM-based dataset augmentation for retail products"""

    def __init__(self, device: str = "cuda:0"):
        if not SAM_AVAILABLE:
            raise ImportError("Segment Anything Model is required. Please install: pip install git+https://github.com/facebookresearch/segment-anything.git")

        self.device = device
        self.sam_model = None
        self.mask_generator = None
        self.predictor = None

        # Download SAM model if not exists
        self._setup_sam_model()

    def _setup_sam_model(self):
        """Download and setup SAM model"""
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)

        model_path = model_dir / "sam_vit_h_4b8939.pth"

        if not model_path.exists():
            print("📥 Downloading SAM model (this may take a few minutes)...")
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            urllib.request.urlretrieve(url, model_path)
            print("✅ SAM model downloaded")

        print("🔧 Loading SAM model...")
        self.sam_model = sam_model_registry["vit_h"](checkpoint=str(model_path))
        self.sam_model.to(self.device)

        # Create mask generator for automatic segmentation
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,
        )

        # Create predictor for interactive segmentation
        self.predictor = SamPredictor(self.sam_model)
        print("✅ SAM model loaded successfully")

    def segment_product(self, image: np.ndarray, bbox: List[float]) -> Optional[np.ndarray]:
        """
        Segment a product using SAM with bounding box prompt

        Args:
            image: RGB image array
            bbox: [x1, y1, x2, y2] in pixel coordinates

        Returns:
            Binary mask of the product
        """
        try:
            self.predictor.set_image(image)

            # Convert bbox to points for SAM
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            # Use multiple points for better segmentation
            input_points = np.array([
                [center_x, center_y],  # center
                [x1 + (x2-x1)*0.25, y1 + (y2-y1)*0.25],  # top-left quarter
                [x2 - (x2-x1)*0.25, y2 - (y2-y1)*0.25],  # bottom-right quarter
            ])
            input_labels = np.array([1, 1, 1])  # all foreground

            masks, scores, logits = self.predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )

            # Use the mask with highest score
            if len(masks) > 0:
                best_mask_idx = np.argmax(scores)
                return masks[best_mask_idx]

        except Exception as e:
            print(f"⚠️  SAM segmentation failed: {e}")

        return None

    def load_product_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load product images and annotations from YOLO dataset

        Args:
            dataset_path: Path to YOLO dataset directory

        Returns:
            List of product entries with image path and annotations
        """
        dataset_path = Path(dataset_path)
        products = []

        # Load data.yaml to get class names
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            print(f"⚠️  No data.yaml found in {dataset_path}")
            return products

        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)

        class_names = data_config.get('names', [])
        print(f"📊 Loading {len(class_names)} classes from {dataset_path.name}")

        # Process train, valid, test splits
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'

            if not images_dir.exists() or not labels_dir.exists():
                continue

            print(f"   Processing {split} split...")

            for label_file in tqdm(list(labels_dir.glob('*.txt')), desc=f"Loading {split}"):
                image_file = images_dir / label_file.with_suffix('.jpg').name
                if not image_file.exists():
                    image_file = images_dir / label_file.with_suffix('.png').name
                if not image_file.exists():
                    continue

                # Read annotations
                with open(label_file, 'r') as f:
                    annotations = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center, y_center, width, height = map(float, parts[1:5])

                            # Convert to pixel coordinates (we'll do this when we load the image)
                            annotations.append({
                                'class_id': class_id,
                                'class_name': class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                                'bbox_norm': [x_center, y_center, width, height]
                            })

                if annotations:  # Only add if there are annotations
                    products.append({
                        'image_path': str(image_file),
                        'annotations': annotations,
                        'split': split,
                        'dataset': dataset_path.name
                    })

        print(f"✅ Loaded {len(products)} product images from {dataset_path.name}")
        return products

    def load_background_dataset(self, dataset_path: str) -> List[Dict]:
        """
        Load background images from empty shelf dataset

        Args:
            dataset_path: Path to empty shelf dataset

        Returns:
            List of background entries
        """
        dataset_path = Path(dataset_path)
        backgrounds = []

        # Process all splits
        for split in ['train', 'valid', 'test']:
            images_dir = dataset_path / split / 'images'
            labels_dir = dataset_path / split / 'labels'

            if not images_dir.exists():
                continue

            print(f"   Processing {split} backgrounds...")

            for image_file in tqdm(list(images_dir.glob('*.jpg')), desc=f"Loading {split} backgrounds"):
                label_file = labels_dir / image_file.with_suffix('.txt').name

                annotations = []
                if label_file.exists():
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                x_center, y_center, width, height = map(float, parts[1:5])
                                annotations.append({
                                    'class_id': class_id,
                                    'bbox_norm': [x_center, y_center, width, height]
                                })

                backgrounds.append({
                    'image_path': str(image_file),
                    'annotations': annotations,  # Existing products/shelves to avoid
                    'split': split
                })

        print(f"✅ Loaded {len(backgrounds)} background images")
        return backgrounds

    def composite_product_on_background(self, product_mask: np.ndarray, product_image: np.ndarray,
                                      background_image: np.ndarray, background_annotations: List[Dict],
                                      target_position: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Composite a segmented product onto a background image

        Args:
            product_mask: Binary mask of the product
            product_image: Original product image
            background_image: Background image
            background_annotations: Existing annotations to avoid
            target_position: (x, y) position to place product, or None for random

        Returns:
            Composited image and new annotation
        """
        h_bg, w_bg = background_image.shape[:2]

        # Find product bounding box from mask
        if product_mask is None:
            return background_image, []

        # Get product region
        y_indices, x_indices = np.where(product_mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return background_image, []

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()

        product_crop = product_image[y_min:y_max+1, x_min:x_max+1]
        product_mask_crop = product_mask[y_min:y_max+1, x_min:x_max+1]

        product_h, product_w = product_crop.shape[:2]

        # Try to place product, avoiding existing annotations
        max_attempts = 50
        for attempt in range(max_attempts):
            if target_position is not None:
                x_pos, y_pos = target_position
            else:
                # Random position with margin
                margin = 50
                x_pos = random.randint(margin, w_bg - product_w - margin)
                y_pos = random.randint(margin, h_bg - product_h - margin)

            # Check if this position overlaps with existing annotations
            overlaps = False
            for ann in background_annotations:
                x_center, y_center, width, height = ann['bbox_norm']

                # Convert to pixel coordinates
                ann_x = int(x_center * w_bg)
                ann_y = int(y_center * h_bg)
                ann_w = int(width * w_bg)
                ann_h = int(height * h_bg)

                ann_x1 = ann_x - ann_w // 2
                ann_y1 = ann_y - ann_h // 2
                ann_x2 = ann_x + ann_w // 2
                ann_y2 = ann_y + ann_h // 2

                # Check overlap with product placement
                product_x2 = x_pos + product_w
                product_y2 = y_pos + product_h

                if not (x_pos >= ann_x2 or product_x2 <= ann_x1 or
                       y_pos >= ann_y2 or product_y2 <= ann_y1):
                    overlaps = True
                    break

            if not overlaps:
                break
        else:
            # Couldn't find non-overlapping position
            return background_image, []

        # Composite the product
        result = background_image.copy()

        # Create alpha mask from product mask
        if product_mask_crop.ndim == 2:
            alpha = product_mask_crop.astype(np.uint8) * 255
        else:
            alpha = product_mask_crop[:, :, 0].astype(np.uint8) * 255

        # Ensure alpha is single channel
        if alpha.ndim == 3:
            alpha = alpha[:, :, 0]

        # Resize alpha to match product_crop if needed
        if alpha.shape != product_crop.shape[:2]:
            alpha = cv2.resize(alpha, (product_crop.shape[1], product_crop.shape[0]))

        # Apply the mask - direct replacement for better product visibility
        for c in range(3):  # RGB channels
            result[y_pos:y_pos+product_h, x_pos:x_pos+product_w, c] = \
                np.where(alpha > 128,
                        product_crop[:, :, c],
                        result[y_pos:y_pos+product_h, x_pos:x_pos+product_w, c])

        # Apply brightness/contrast enhancement to make product more visible
        # Slightly increase contrast and brightness of the product region
        product_region = result[y_pos:y_pos+product_h, x_pos:x_pos+product_w]
        alpha_float = alpha.astype(np.float32) / 255.0

        # Apply mild enhancement where mask is active
        enhanced_region = product_region.copy().astype(np.float32)
        enhanced_region = cv2.convertScaleAbs(enhanced_region, alpha=1.1, beta=10)  # slight contrast and brightness boost

        # Blend enhanced region with original based on mask strength
        for c in range(3):
            result[y_pos:y_pos+product_h, x_pos:x_pos+product_w, c] = \
                result[y_pos:y_pos+product_h, x_pos:x_pos+product_w, c] * (1 - alpha_float * 0.3) + \
                enhanced_region[:, :, c] * (alpha_float * 0.3)

    def create_augmented_dataset(self, output_dir: str, num_samples: int = 5000,
                               grocery_path: str = "data/grocery.v3i.yolov11",
                               liquor_path: str = "data/Liquor-data.v4i.yolov11",
                               background_path: str = "data/empty-shelf-detection.v1i.yolov11"):
        """
        Create augmented dataset using SAM segmentation

        Args:
            output_dir: Output directory for augmented dataset
            num_samples: Number of augmented samples to create
            grocery_path: Path to grocery dataset
            liquor_path: Path to liquor dataset
            background_path: Path to background dataset
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Creating SAM-augmented dataset with {num_samples} samples")
        print(f"📁 Output directory: {output_dir}")

        # Load datasets
        print("\n📦 Loading product datasets...")
        grocery_products = self.load_product_dataset(grocery_path)
        liquor_products = self.load_product_dataset(liquor_path)
        all_products = grocery_products + liquor_products

        print(f"\n📦 Loading background dataset...")
        backgrounds = self.load_background_dataset(background_path)

        if not all_products or not backgrounds:
            print("❌ No products or backgrounds loaded!")
            return

        # Create output structure
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)

        # Create class mapping (combine grocery and liquor classes)
        class_mapping = {}
        next_class_id = 0

        # Grocery classes (0-58)
        grocery_yaml = Path(grocery_path) / "data.yaml"
        if grocery_yaml.exists():
            with open(grocery_yaml, 'r') as f:
                grocery_config = yaml.safe_load(f)
                for i, name in enumerate(grocery_config.get('names', [])):
                    class_mapping[f"grocery_{name}"] = next_class_id
                    next_class_id += 1

        # Liquor classes (59-472)
        liquor_yaml = Path(liquor_path) / "data.yaml"
        if liquor_yaml.exists():
            with open(liquor_yaml, 'r') as f:
                liquor_config = yaml.safe_load(f)
                for i, name in enumerate(liquor_config.get('names', [])):
                    class_mapping[f"liquor_{name}"] = next_class_id
                    next_class_id += 1

        print(f"📊 Total classes in augmented dataset: {len(class_mapping)}")

        # Generate augmented samples
        successful_samples = 0
        failed_samples = 0

        with tqdm(total=num_samples, desc="Creating augmented samples") as pbar:
            while successful_samples < num_samples:
                try:
                    # Randomly select product and background
                    product_entry = random.choice(all_products)
                    background_entry = random.choice(backgrounds)

                    # Load images
                    product_image = cv2.imread(product_entry['image_path'])
                    background_image = cv2.imread(background_entry['image_path'])

                    if product_image is None or background_image is None:
                        failed_samples += 1
                        pbar.update(1)
                        continue

                    product_image = cv2.cvtColor(product_image, cv2.COLOR_BGR2RGB)
                    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

                    # Get product annotation (use first one for simplicity)
                    if not product_entry['annotations']:
                        failed_samples += 1
                        pbar.update(1)
                        continue

                    ann = random.choice(product_entry['annotations'])

                    # Convert normalized bbox to pixel coordinates
                    h_prod, w_prod = product_image.shape[:2]
                    x_center, y_center, width, height = ann['bbox_norm']
                    x1 = int((x_center - width/2) * w_prod)
                    y1 = int((y_center - height/2) * h_prod)
                    x2 = int((x_center + width/2) * w_prod)
                    y2 = int((y_center + height/2) * h_prod)

                    # Segment product using SAM
                    product_mask = self.segment_product(product_image, [x1, y1, x2, y2])

                    if product_mask is None:
                        failed_samples += 1
                        pbar.update(1)
                        continue

                    # Composite product on background
                    composited_image, new_annotations = self.composite_product_on_background(
                        product_mask, product_image, background_image,
                        background_entry['annotations']
                    )

                    if not new_annotations:
                        failed_samples += 1
                        pbar.update(1)
                        continue

                    # Save augmented sample
                    sample_id = f"{successful_samples:06d}"
                    image_filename = f"{sample_id}.jpg"
                    label_filename = f"{sample_id}.txt"

                    # Save image
                    cv2.imwrite(str(images_dir / image_filename),
                              cv2.cvtColor(composited_image, cv2.COLOR_RGB2BGR))

                    # Save label
                    with open(labels_dir / label_filename, 'w') as f:
                        for ann in new_annotations:
                            class_id = ann['class_id']
                            x_center, y_center, width, height = ann['bbox_norm']
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                    successful_samples += 1
                    pbar.update(1)

                except Exception as e:
                    print(f"⚠️  Error creating sample: {e}")
                    failed_samples += 1
                    pbar.update(1)

        # Create data.yaml for the augmented dataset
        data_config = {
            'train': 'images',
            'val': 'images',  # Use same images for both (small dataset)
            'test': 'images',
            'nc': len(class_mapping),
            'names': list(class_mapping.keys())
        }

        with open(output_dir / "data.yaml", 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)

        # Save metadata
        metadata = {
            'total_samples': successful_samples,
            'failed_samples': failed_samples,
            'grocery_products': len(grocery_products),
            'liquor_products': len(liquor_products),
            'background_images': len(backgrounds),
            'class_mapping': class_mapping,
            'creation_date': str(torch.tensor(0).to(self.device).device)  # Just to get current time
        }

        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print("\n✅ SAM augmentation complete!")
        print(f"📊 Created {successful_samples} augmented samples")
        print(f"📁 Dataset saved to: {output_dir}")
        print(f"📋 Metadata saved to: {output_dir}/metadata.json")
        print(f"🏷️  Labels saved to: {output_dir}/data.yaml")


def main():
    parser = argparse.ArgumentParser(description='SAM-based dataset augmentation')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for augmented dataset')
    parser.add_argument('--num-samples', type=int, default=5000,
                       help='Number of augmented samples to create')
    parser.add_argument('--grocery-path', type=str, default='data/grocery.v3i.yolov11',
                       help='Path to grocery dataset')
    parser.add_argument('--liquor-path', type=str, default='data/Liquor-data.v4i.yolov11',
                       help='Path to liquor dataset')
    parser.add_argument('--background-path', type=str, default='data/empty-shelf-detection.v1i.yolov11',
                       help='Path to background dataset')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use for SAM')

    args = parser.parse_args()

    # Check if CUDA is available
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print("⚠️  CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Create augmentor and generate dataset
    augmentor = SAMAugmentor(device=args.device)
    augmentor.create_augmented_dataset(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        grocery_path=args.grocery_path,
        liquor_path=args.liquor_path,
        background_path=args.background_path
    )


if __name__ == '__main__':
    main()