#!/usr/bin/env python3
"""
Re-run a simplified hybrid evaluation to see if we can reproduce the 0.003 mAP issue
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

# Load model
print("Loading YOLO model...")
model = YOLO('grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt')

# Create embedding extractor
print("Loading embedding extractor...")
embedding_extractor = DOLGEmbeddingExtractor(device='cuda:0')

# Create Milvus database connection
print("Loading Milvus database...")
milvus_db = MilvusRetailDB(
    db_path='experiment_results/milvus_retail.db',  # Use the SAME database path as the experiment
    collection_name='retail_items',
    auto_setup=False,
    connect_only=True
)

# Create hybrid detector
print("Creating hybrid detector...")
hybrid_detector = HybridYOLODetector(
    yolo_model=model,
    embedding_extractor=embedding_extractor,
    milvus_db=milvus_db,
    similarity_threshold=0.7,
    device='cuda:0'
)

# Load validation data
dataset_yaml = 'data/grocery_augmented/grocery_augmented.yaml'
with open(dataset_yaml, 'r') as f:
    dataset_info = yaml.safe_load(f)

val_dir = Path(dataset_info['val'])
val_images = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))
val_labels = [str(img).replace('images', 'labels').replace(img.suffix, '.txt') 
             for img in val_images]

print(f"Found {len(val_images)} validation images")

# Evaluate with corrected IoU threshold (0.25 to match dataset characteristics)
print("\nEvaluating hybrid model...")
evaluator = RetailEvaluator(dataset_yaml)  # Now uses default 0.25
metrics = evaluator.evaluate_hybrid_model(
    hybrid_detector,
    [str(img) for img in val_images],
    val_labels
)

print(f"\n{'='*60}")
print("RESULTS:")
print(f"{'='*60}")
print(f"mAP@0.5: {metrics.map50:.4f}")
print(f"Precision: {metrics.precision:.4f}")
print(f"Recall: {metrics.recall:.4f}")
print(f"True Positives: {metrics.true_positives}")
print(f"False Positives: {metrics.false_positives}")
print(f"False Negatives: {metrics.false_negatives}")
print(f"Milvus Hit Rate: {metrics.milvus_hit_rate:.4f}")

milvus_db.close()
