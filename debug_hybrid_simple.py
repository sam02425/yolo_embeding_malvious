#!/usr/bin/env python3
"""Simple debugging script to see what hybrid detector actually outputs"""

import cv2
from ultralytics import YOLO
import sys
sys.path.append('yolo_vs_embeding_malvious')

from experimental_framework import DOLGEmbeddingExtractor, MilvusRetailDB, HybridYOLODetector

# Load components
print("Loading model...")
model = YOLO('grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt')

print("Loading embedding extractor...")
embedding_extractor = DOLGEmbeddingExtractor(device='cuda:0')

print("Loading Milvus database...")
milvus_db = MilvusRetailDB(
    db_path='experiments/milvus_release/databases/milvus_retail.db',
    collection_name='retail_items',
    embedding_dim=128
)

print("Creating hybrid detector...")
hybrid = HybridYOLODetector(
    yolo_model=model,
    embedding_extractor=embedding_extractor,
    milvus_db=milvus_db,
    similarity_threshold=0.7,
    device='cuda:0'
)

# Test on one image
test_img_path = 'data/grocery_augmented/test/images/23_jpg.rf.959c6c1a3d1de40bc5561c7ea15216e1.jpg'
test_label_path = 'data/grocery_augmented/test/labels/23_jpg.rf.959c6c1a3d1de40bc5561c7ea15216e1.txt'

print(f"\n{'='*60}")
print(f"Testing image: {test_img_path}")
print(f"{'='*60}\n")

# Load image
image = cv2.imread(test_img_path)

# Get YOLO predictions directly
print("1️⃣ DIRECT YOLO PREDICTIONS:")
yolo_results = model(image, verbose=False)[0]
print(f"  Found {len(yolo_results.boxes)} detections")
for i, box in enumerate(yolo_results.boxes):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = yolo_results.names[cls_id]
    print(f"  Detection {i}: class_id={cls_id:2d}, conf={conf:.3f}, name={cls_name}")

# Get hybrid predictions
print(f"\n2️⃣ HYBRID DETECTOR PREDICTIONS:")
hybrid_preds, timings = hybrid.predict(image)
print(f"  Found {len(hybrid_preds)} detections")
for i, pred in enumerate(hybrid_preds):
    print(f"  Detection {i}:")
    print(f"    class_id: {pred['class_id']:2d}")
    print(f"    class_name: {pred['class_name']}")
    print(f"    confidence: {pred['confidence']:.3f}")
    print(f"    yolo_class: {pred['yolo_class']:2d}")
    print(f"    used_milvus: {pred['used_milvus']}")
    print(f"    milvus_similarity: {pred['milvus_similarity']:.4f}")

# Load ground truth
print(f"\n3️⃣ GROUND TRUTH:")
with open(test_label_path, 'r') as f:
    for i, line in enumerate(f):
        parts = line.strip().split()
        if parts:
            cls_id = int(parts[0])
            print(f"  Object {i}: class_id={cls_id}")

print(f"\n4️⃣ MILVUS HIT RATE:")
print(f"  {timings.get('milvus_hit_rate', 0)*100:.1f}%")
