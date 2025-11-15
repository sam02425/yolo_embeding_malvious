#!/usr/bin/env python3
"""
Diagnostic script to investigate why Milvus hybrid approach is failing
"""

import sys
import json
from pathlib import Path
import numpy as np
import cv2
import yaml
import torch
from pymilvus import MilvusClient

def check_milvus_database(db_path: str = "experiment_results/milvus_retail.db", 
                          collection_name: str = "retail_items_dolg"):
    """Check Milvus database contents and statistics"""
    
    print(f"\n{'='*80}")
    print(f"MILVUS DATABASE DIAGNOSTICS")
    print(f"{'='*80}\n")
    
    # Connect to Milvus
    try:
        client = MilvusClient(uri=db_path)
        print(f"✅ Connected to Milvus database: {db_path}")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        return
    
    # Check collection exists
    collections = client.list_collections()
    print(f"\n📋 Available collections: {collections}")
    
    if collection_name not in collections:
        print(f"❌ Collection '{collection_name}' not found!")
        return
    
    # Get collection stats
    print(f"\n📊 Collection: {collection_name}")
    try:
        stats = client.get_collection_stats(collection_name=collection_name)
        print(f"   Stats: {stats}")
        
        # Count entities
        result = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["count(*)"]
        )
        print(f"   Total entities: {result}")
        
    except Exception as e:
        print(f"   Error getting stats: {e}")
    
    # Sample some embeddings
    print(f"\n🔍 Sampling embeddings from database...")
    try:
        # Get first 10 entities
        results = client.query(
            collection_name=collection_name,
            filter="",
            output_fields=["id", "class_id", "class_name", "upc", "embedding"],
            limit=10
        )
        
        print(f"   Retrieved {len(results)} sample entities")
        
        if results:
            print(f"\n   Sample entity structure:")
            for i, entity in enumerate(results[:3]):
                print(f"\n   Entity {i+1}:")
                print(f"      ID: {entity.get('id', 'N/A')}")
                print(f"      Class ID: {entity.get('class_id', 'N/A')}")
                print(f"      Class Name: {entity.get('class_name', 'N/A')}")
                print(f"      UPC: {entity.get('upc', 'N/A')}")
                
                if 'embedding' in entity:
                    emb = np.array(entity['embedding'])
                    print(f"      Embedding shape: {emb.shape}")
                    print(f"      Embedding norm: {np.linalg.norm(emb):.4f}")
                    print(f"      Embedding stats: min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
            
            # Check class distribution
            print(f"\n   📊 Class distribution in database:")
            class_counts = {}
            all_results = client.query(
                collection_name=collection_name,
                filter="",
                output_fields=["class_name"],
                limit=10000
            )
            
            for entity in all_results:
                cls_name = entity.get('class_name', 'Unknown')
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
            
            print(f"      Total classes: {len(class_counts)}")
            print(f"      Top 10 classes:")
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"         {cls}: {count} embeddings")
                
    except Exception as e:
        print(f"   ❌ Error sampling embeddings: {e}")
    
    # Test similarity search
    print(f"\n🔍 Testing similarity search...")
    try:
        # Create a random query vector (128-d)
        query_vector = np.random.randn(128).astype(np.float32)
        query_vector = query_vector / np.linalg.norm(query_vector)
        
        print(f"   Query vector shape: {query_vector.shape}")
        print(f"   Query vector norm: {np.linalg.norm(query_vector):.4f}")
        
        # Search
        search_results = client.search(
            collection_name=collection_name,
            data=[query_vector.tolist()],
            limit=5,
            output_fields=["class_id", "class_name", "upc"]
        )
        
        print(f"\n   Top 5 search results:")
        if search_results and len(search_results) > 0:
            for i, result in enumerate(search_results[0]):
                print(f"      {i+1}. Class: {result.get('entity', {}).get('class_name', 'N/A')}")
                print(f"         Distance: {result.get('distance', 'N/A'):.4f}")
                print(f"         ID: {result.get('id', 'N/A')}")
        else:
            print(f"      ❌ No results returned!")
            
    except Exception as e:
        print(f"   ❌ Search failed: {e}")
    
    client.close()


def test_actual_inference(dataset_yaml: str = "data/grocery_augmented/grocery_augmented.yaml",
                         dolg_model_path: str = "dolg_model.pth"):
    """Test actual inference with real images to see similarity scores"""
    
    print(f"\n{'='*80}")
    print(f"TESTING ACTUAL INFERENCE")
    print(f"{'='*80}\n")
    
    # Load dataset
    with open(dataset_yaml, 'r') as f:
        dataset = yaml.safe_load(f)
    
    test_dir = Path(dataset.get('test', dataset.get('val', '')))
    if not test_dir.exists():
        test_dir = Path(dataset_yaml).parent / 'test' / 'images'
    
    # Get first few test images
    test_images = list(test_dir.glob('*.jpg'))[:5]
    if not test_images:
        test_images = list(test_dir.glob('*.png'))[:5]
    
    if not test_images:
        print(f"❌ No test images found in {test_dir}")
        return
    
    print(f"📸 Found {len(test_images)} test images")
    
    # Load DOLG model
    print(f"\n🔧 Loading DOLG model from {dolg_model_path}...")
    try:
        from yolo_vs_embeding_malvious.experimental_framework import DOLGEmbeddingExtractor, MilvusRetailDB
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Using device: {device}")
        
        extractor = DOLGEmbeddingExtractor(
            model_path=dolg_model_path,
            device=device,
            batch_size=1
        )
        print(f"✅ DOLG model loaded")
        
        # Connect to Milvus
        milvus_db = MilvusRetailDB(
            db_path="experiment_results/milvus_retail.db",
            collection_name="retail_items_dolg",
            embedding_dim=128
        )
        print(f"✅ Connected to Milvus")
        
        # Test on images
        print(f"\n🧪 Testing on sample images...")
        for img_path in test_images:
            print(f"\n   Image: {img_path.name}")
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"      ❌ Failed to load image")
                continue
            
            # Extract embedding
            embedding = extractor.extract_embedding(img)
            print(f"      Embedding shape: {embedding.shape}")
            print(f"      Embedding norm: {np.linalg.norm(embedding):.4f}")
            
            # Search Milvus
            results = milvus_db.search_similar(embedding, top_k=5)
            
            print(f"      Top 5 matches:")
            if results:
                for i, result in enumerate(results):
                    cls_name = result.get('entity', {}).get('class_name', 'Unknown')
                    distance = result.get('distance', 0.0)
                    print(f"         {i+1}. {cls_name} (distance: {distance:.4f})")
            else:
                print(f"         ❌ No matches found!")
        
        milvus_db.close()
        
    except Exception as e:
        print(f"❌ Error during inference test: {e}")
        import traceback
        traceback.print_exc()


def analyze_experiment_results(results_file: str = "experiment_comparison.json"):
    """Analyze the experiment results to understand the failure"""
    
    print(f"\n{'='*80}")
    print(f"EXPERIMENT RESULTS ANALYSIS")
    print(f"{'='*80}\n")
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Compare baseline vs hybrid
    baseline = results.get("YOLOv8_Baseline_488_Classes", {})
    hybrid = results.get("YOLOv8_DOLG_Milvus_Hybrid", {})
    
    print(f"📊 Baseline YOLOv8 Performance:")
    print(f"   mAP50: {baseline.get('map50', 0):.4f}")
    print(f"   Precision: {baseline.get('precision', 0):.4f}")
    print(f"   Recall: {baseline.get('recall', 0):.4f}")
    
    print(f"\n📊 Hybrid YOLOv8+Milvus Performance:")
    print(f"   mAP50: {hybrid.get('map50', 0):.4f}")
    print(f"   Precision: {hybrid.get('precision', 0):.4f}")
    print(f"   Recall: {hybrid.get('recall', 0):.4f}")
    print(f"   Milvus Hit Rate: {hybrid.get('milvus_hit_rate', 0):.2%}")
    
    # Check which classes have non-zero AP in hybrid
    print(f"\n🔍 Classes with non-zero AP in hybrid:")
    hybrid_per_class = hybrid.get('per_class_ap', {})
    non_zero_classes = {k: v for k, v in hybrid_per_class.items() if v > 0}
    print(f"   Found {len(non_zero_classes)} classes with AP > 0:")
    for cls, ap in non_zero_classes.items():
        print(f"      {cls}: {ap:.4f}")
    
    # Check class name mismatches
    baseline_classes = set(baseline.get('per_class_ap', {}).keys())
    hybrid_classes = set(hybrid_per_class.keys())
    
    print(f"\n📋 Class set comparison:")
    print(f"   Baseline classes: {len(baseline_classes)}")
    print(f"   Hybrid classes: {len(hybrid_classes)}")
    
    only_in_baseline = baseline_classes - hybrid_classes
    only_in_hybrid = hybrid_classes - baseline_classes
    
    if only_in_baseline:
        print(f"\n   Classes only in baseline ({len(only_in_baseline)}):")
        for cls in sorted(only_in_baseline)[:10]:
            print(f"      {cls}")
        if len(only_in_baseline) > 10:
            print(f"      ... and {len(only_in_baseline) - 10} more")
    
    if only_in_hybrid:
        print(f"\n   Classes only in hybrid ({len(only_in_hybrid)}):")
        for cls in sorted(only_in_hybrid)[:10]:
            print(f"      {cls}")
        if len(only_in_hybrid) > 10:
            print(f"      ... and {len(only_in_hybrid) - 10} more")


def main():
    print(f"\n{'='*80}")
    print(f"MILVUS HYBRID FAILURE DIAGNOSTIC TOOL")
    print(f"{'='*80}")
    
    # Run diagnostics
    check_milvus_database()
    analyze_experiment_results()
    test_actual_inference()
    
    print(f"\n{'='*80}")
    print(f"DIAGNOSTIC COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
