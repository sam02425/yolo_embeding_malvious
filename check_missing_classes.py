#!/usr/bin/env python3
"""Detailed class mapping comparison"""

import yaml
from pathlib import Path
from ultralytics import YOLO
from pymilvus import MilvusClient

# Load YOLO model
yolo_path = "grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt"
yolo = YOLO(yolo_path)
yolo_classes = dict(yolo.names)

# Load dataset YAML
with open("data/grocery_augmented/grocery_augmented.yaml", 'r') as f:
    dataset = yaml.safe_load(f)
dataset_classes = dict(enumerate(dataset['names']))

# Load Milvus classes
client = MilvusClient(uri="experiment_results/milvus_retail.db")
results = client.query(
    collection_name='retail_items',
    filter='',
    output_fields=['class_id', 'class_name'],
    limit=10000
)

milvus_classes = {}
for entity in results:
    class_id = entity['class_id']
    class_name = entity['class_name']
    if class_id not in milvus_classes:
        milvus_classes[class_id] = class_name

client.close()

# Find missing classes
dataset_ids = set(dataset_classes.keys())
milvus_ids = set(milvus_classes.keys())

missing_in_milvus = dataset_ids - milvus_ids

print(f"Dataset has {len(dataset_classes)} classes")
print(f"Milvus has {len(milvus_classes)} classes")
print(f"\nMissing in Milvus ({len(missing_in_milvus)} classes):")

for cid in sorted(missing_in_milvus):
    print(f"   {cid}: {dataset_classes[cid]}")

# Check if these classes exist in training data
train_dir = Path("data/grocery_augmented/train/labels")
if train_dir.exists():
    print(f"\nChecking which classes have training examples...")
    
    all_class_ids = set()
    for label_file in train_dir.glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    all_class_ids.add(int(parts[0]))
    
    print(f"\nClasses found in training labels: {len(all_class_ids)}")
    
    for cid in sorted(missing_in_milvus):
        has_training = "✅ HAS TRAINING DATA" if cid in all_class_ids else "❌ NO TRAINING DATA"
        print(f"   {cid}: {dataset_classes[cid]:40s} {has_training}")
