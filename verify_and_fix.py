#!/usr/bin/env python3
"""
Quick fix script for Milvus hybrid approach issues
"""

import yaml
from pathlib import Path
from ultralytics import YOLO
from pymilvus import MilvusClient

def check_class_consistency():
    """Verify class name consistency across all components"""
    
    print(f"\n{'='*80}")
    print("CLASS CONSISTENCY CHECK")
    print(f"{'='*80}\n")
    
    # 1. Load YOLO model classes
    yolo_v8_path = "grocery/runs/yolov8_grocery_baseline_20251114_194239/weights/best.pt"
    yolo_v11_path = "grocery/runs/yolo11_grocery_baseline_20251114_1953262/weights/best.pt"
    
    print("📦 Loading YOLO models...")
    yolo_v8 = YOLO(yolo_v8_path)
    yolo_v11 = YOLO(yolo_v11_path)
    
    yolo_v8_classes = dict(yolo_v8.names)
    yolo_v11_classes = dict(yolo_v11.names)
    
    print(f"   YOLOv8 classes: {len(yolo_v8_classes)}")
    print(f"   YOLOv11 classes: {len(yolo_v11_classes)}")
    
    # 2. Load dataset YAML
    dataset_yaml = "data/grocery_augmented/grocery_augmented.yaml"
    print(f"\n📄 Loading dataset YAML: {dataset_yaml}")
    
    with open(dataset_yaml, 'r') as f:
        dataset = yaml.safe_load(f)
    
    dataset_classes = dict(enumerate(dataset['names']))
    print(f"   Dataset classes: {len(dataset_classes)}")
    
    # 3. Load Milvus classes
    milvus_db_path = "experiment_results/milvus_retail.db"
    print(f"\n🗄️  Loading Milvus database: {milvus_db_path}")
    
    client = MilvusClient(uri=milvus_db_path)
    
    # Get all unique class names from Milvus
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
    
    print(f"   Milvus classes: {len(milvus_classes)}")
    
    client.close()
    
    # 4. Compare class sets
    print(f"\n🔍 Comparing class mappings...")
    
    yolo_v8_set = set(yolo_v8_classes.items())
    yolo_v11_set = set(yolo_v11_classes.items())
    dataset_set = set(dataset_classes.items())
    milvus_set = set(milvus_classes.items())
    
    # Check if all match
    all_match = (yolo_v8_set == yolo_v11_set == dataset_set == milvus_set)
    
    if all_match:
        print("   ✅ ALL CLASS MAPPINGS ARE CONSISTENT!")
    else:
        print("   ❌ CLASS MAPPINGS ARE INCONSISTENT!")
        
        # Find differences
        if yolo_v8_set != dataset_set:
            only_yolo = dict(yolo_v8_set - dataset_set)
            only_dataset = dict(dataset_set - yolo_v8_set)
            
            if only_yolo:
                print(f"\n   Classes only in YOLOv8 model ({len(only_yolo)}):")
                for cid, cname in sorted(only_yolo.items())[:5]:
                    print(f"      {cid}: {cname}")
                if len(only_yolo) > 5:
                    print(f"      ... and {len(only_yolo) - 5} more")
            
            if only_dataset:
                print(f"\n   Classes only in dataset ({len(only_dataset)}):")
                for cid, cname in sorted(only_dataset.items())[:5]:
                    print(f"      {cid}: {cname}")
                if len(only_dataset) > 5:
                    print(f"      ... and {len(only_dataset) - 5} more")
        
        if milvus_set != dataset_set:
            only_milvus = dict(milvus_set - dataset_set)
            only_dataset = dict(dataset_set - milvus_set)
            
            if only_milvus:
                print(f"\n   Classes only in Milvus ({len(only_milvus)}):")
                for cid, cname in sorted(only_milvus.items())[:5]:
                    print(f"      {cid}: {cname}")
                if len(only_milvus) > 5:
                    print(f"      ... and {len(only_milvus) - 5} more")
            
            if only_dataset:
                print(f"\n   Classes only in dataset (not in Milvus) ({len(only_dataset)}):")
                for cid, cname in sorted(only_dataset.items())[:5]:
                    print(f"      {cid}: {cname}")
                if len(only_dataset) > 5:
                    print(f"      ... and {len(only_dataset) - 5} more")
    
    return all_match, {
        'yolov8': yolo_v8_classes,
        'yolov11': yolo_v11_classes,
        'dataset': dataset_classes,
        'milvus': milvus_classes
    }


def check_collection_names():
    """Check Milvus collection names"""
    
    print(f"\n{'='*80}")
    print("MILVUS COLLECTION NAME CHECK")
    print(f"{'='*80}\n")
    
    milvus_db_path = "experiment_results/milvus_retail.db"
    client = MilvusClient(uri=milvus_db_path)
    
    collections = client.list_collections()
    print(f"Available collections: {collections}")
    
    # Check what the code expects
    config_file = "yolo_vs_embeding_malvious/experiment_config.yaml"
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    expected_collection = config['milvus']['collection_name']
    print(f"\nExpected collection (from config): '{expected_collection}'")
    
    if expected_collection in collections:
        print(f"✅ Collection '{expected_collection}' exists!")
    else:
        print(f"❌ Collection '{expected_collection}' NOT FOUND!")
        print(f"   Available: {collections}")
        print(f"\n   💡 FIX: Update experiment_config.yaml:")
        print(f"      milvus:")
        print(f"        collection_name: '{collections[0]}'  # Change to actual collection name")
    
    client.close()
    
    return expected_collection in collections


def main():
    print(f"\n{'='*80}")
    print("MILVUS HYBRID APPROACH - DIAGNOSTIC & FIX TOOL")
    print(f"{'='*80}")
    
    # Check 1: Collection names
    collection_ok = check_collection_names()
    
    # Check 2: Class consistency
    classes_ok, class_mappings = check_class_consistency()
    
    # Summary
    print(f"\n{'='*80}")
    print("DIAGNOSTIC SUMMARY")
    print(f"{'='*80}\n")
    
    issues_found = []
    
    if not collection_ok:
        issues_found.append("❌ Collection name mismatch")
    else:
        print("✅ Collection name is correct")
    
    if not classes_ok:
        issues_found.append("❌ Class mapping inconsistency")
    else:
        print("✅ Class mappings are consistent")
    
    if issues_found:
        print(f"\n🔧 ISSUES FOUND ({len(issues_found)}):")
        for issue in issues_found:
            print(f"   {issue}")
        
        print(f"\n📋 RECOMMENDED FIXES:")
        print(f"   1. Update experiment_config.yaml with correct collection_name")
        print(f"   2. Re-populate Milvus with correct class mappings if needed")
        print(f"   3. Re-run experiments after fixes")
        print(f"\n   See MILVUS_FAILURE_ANALYSIS.md for detailed instructions")
    else:
        print(f"\n✅ ALL CHECKS PASSED!")
        print(f"   The Milvus hybrid approach should work correctly.")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
