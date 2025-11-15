#!/usr/bin/env python3
"""
Diagnostic script to understand why hybrid mAP is 0.003 instead of 0.94
"""
import sys
sys.path.insert(0, 'yolo_vs_embeding_malvious')

from experimental_framework import HybridYOLODetector, MilvusRetailDB
from ultralytics import YOLO
import cv2
import glob
import numpy as np

print("="*70)
print("DIAGNOSTIC: Hybrid Detection Issue")
print("="*70)

# Load test image
test_imgs = glob.glob('data/grocery_augmented/valid/images/*.jpg')[:1]
img_path = test_imgs[0]
label_path = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')

print(f"\nTest image: {img_path}")
print(f"Label file: {label_path}")

image = cv2.imread(img_path)
h, w = image.shape[:2]

# Load ground truth
with open(label_path, 'r') as f:
    print("\nGround Truth:")
    for line in f:
        parts = line.strip().split()
        cls_id = int(parts[0])
        print(f"  Class {cls_id}")

# Test pure YOLO
print("\n" + "="*70)
print("TEST 1: Pure YOLO Detection")
print("="*70)
yolo_model = YOLO('grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt')
yolo_results = yolo_model(image, conf=0.25, verbose=False)[0]

print(f"YOLO detected {len(yolo_results.boxes)} objects:")
for i, box in enumerate(yolo_results.boxes[:5]):
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    cls_name = yolo_results.names[cls_id]
    print(f"  {i+1}. Class {cls_id} ({cls_name}), conf={conf:.3f}")

# Test Hybrid
print("\n" + "="*70)
print("TEST 2: Hybrid Detection")
print("="*70)

milvus_db = MilvusRetailDB(
    db_path='experiment_results/milvus_retail.db',
    collection_name='retail_items'
)

hybrid_detector = HybridYOLODetector(
    yolo_model_path='grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt',
    milvus_db=milvus_db,
    dolg_model_path='dolg_model.pth',
    similarity_threshold=0.7,
    device='cuda:0'
)

detections, timings = hybrid_detector.predict(image, conf_threshold=0.25)

print(f"Hybrid detected {len(detections)} objects:")
print(f"Milvus hit rate: {timings.get('milvus_hit_rate', 0):.1%}")

for i, det in enumerate(detections[:5]):
    print(f"\n  Detection {i+1}:")
    print(f"    class_id: {det['class_id']}")
    print(f"    class_name: {det['class_name']}")
    print(f"    confidence: {det['confidence']:.3f}")
    print(f"    yolo_class: {det['yolo_class']}")
    print(f"    milvus_similarity: {det['milvus_similarity']:.3f}")
    print(f"    used_milvus: {det['used_milvus']}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)
print(f"YOLO detections: {len(yolo_results.boxes)}")
print(f"Hybrid detections: {len(detections)}")

if len(yolo_results.boxes) > 0 and len(detections) > 0:
    yolo_cls = int(yolo_results.boxes[0].cls[0])
    hybrid_cls = detections[0]['class_id']
    print(f"\nFirst detection class:")
    print(f"  YOLO: {yolo_cls}")
    print(f"  Hybrid: {hybrid_cls}")
    print(f"  Match: {yolo_cls == hybrid_cls}")
