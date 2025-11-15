#!/usr/bin/env python3
"""
Simplified test - just check first image IoUs
"""

import sys
sys.path.append('yolo_vs_embeding_malvious')

from ultralytics import YOLO
from experimental_framework import (
    DOLGEmbeddingExtractor,
    MilvusRetailDB,
    HybridYOLODetector,
    RetailEvaluator
)
from pathlib import Path
import yaml
import cv2

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / (union_area + 1e-8)

# Load model
model = YOLO('grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt')

# Create hybrid detector
embedding_extractor = DOLGEmbeddingExtractor(device='cuda:0')
milvus_db = MilvusRetailDB(
    db_path='experiment_results/milvus_retail.db',
    collection_name='retail_items',
    auto_setup=False,
    connect_only=True
)

hybrid_detector = HybridYOLODetector(
    yolo_model=model,
    embedding_extractor=embedding_extractor,
    milvus_db=milvus_db,
    similarity_threshold=0.7,
    device='cuda:0'
)

# Get first validation image
dataset_yaml = 'data/grocery_augmented/grocery_augmented.yaml'
with open(dataset_yaml, 'r') as f:
    dataset_info = yaml.safe_load(f)

val_dir = Path(dataset_info['val'])
val_images = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))

img_path = str(val_images[0])
label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.png', '.txt')

print(f"Testing: {Path(img_path).name}\n")

# Load image and get predictions
image = cv2.imread(img_path)
h, w = image.shape[:2]
print(f"Image size: {w}x{h}")

predictions, _ = hybrid_detector.predict(image)
print(f"\nPredictions: {len(predictions)}")
for i, p in enumerate(predictions):
    print(f"  {i}: class={p['class_id']}, conf={p['confidence']:.3f}, bbox={[int(x) for x in p['bbox']]}")

# Load ground truth
with open(label_path, 'r') as f:
    gt_boxes = []
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 5:
            cls_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = (x_center - width / 2) * w
            y1 = (y_center - height / 2) * h
            x2 = (x_center + width / 2) * w
            y2 = (y_center + height / 2) * h
            gt_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'class_id': cls_id
            })

print(f"\nGround Truth: {len(gt_boxes)}")
for i, g in enumerate(gt_boxes):
    print(f"  {i}: class={g['class_id']}, bbox={[int(x) for x in g['bbox']]}")

# Calculate IoU matrix
print(f"\nIoU Matrix:")
for i, pred in enumerate(predictions):
    for j, gt in enumerate(gt_boxes):
        if pred['class_id'] == gt['class_id']:
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            print(f"  Pred{i}(cls{pred['class_id']}) vs GT{j}(cls{gt['class_id']}): IoU={iou:.3f}")

milvus_db.close()
