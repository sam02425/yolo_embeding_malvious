#!/usr/bin/env python3
"""Check liquor dataset for similar issues"""

import yaml
from pathlib import Path
from pymilvus import MilvusClient

def check_liquor_dataset():
    print(f"\n{'='*80}")
    print("LIQUOR DATASET ANALYSIS")
    print(f"{'='*80}\n")
    
    # Check available liquor data.yaml files
    liquor_yamls = [
        "data/Liquor-data.v4i.yolov11/data.yaml",
        "liquor/data.yaml"
    ]
    
    for yaml_path in liquor_yamls:
        if Path(yaml_path).exists():
            print(f"\n📄 Found: {yaml_path}")
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            nc = config.get('nc', len(config.get('names', [])))
            names = config.get('names', [])
            
            print(f"   Classes (nc): {nc}")
            print(f"   Names count: {len(names)}")
            
            if isinstance(names, list):
                print(f"   First 10 classes:")
                for i, name in enumerate(names[:10]):
                    print(f"      {i}: {name}")
            else:
                print(f"   First 10 classes:")
                for i, (k, v) in enumerate(list(names.items())[:10]):
                    print(f"      {k}: {v}")
    
    # Check if liquor Milvus DB exists
    liquor_db_paths = [
        "experiments/milvus_release/databases/milvus_liquor.db",
        "experiment_results/milvus_liquor.db",
        "milvus_liquor.db"
    ]
    
    found_db = None
    for db_path in liquor_db_paths:
        if Path(db_path).exists():
            found_db = db_path
            break
    
    if found_db:
        print(f"\n🗄️  Found Milvus DB: {found_db}")
        try:
            client = MilvusClient(uri=found_db)
            collections = client.list_collections()
            print(f"   Collections: {collections}")
            
            for collection in collections:
                results = client.query(
                    collection_name=collection,
                    filter='',
                    output_fields=['class_id', 'class_name'],
                    limit=10000
                )
                
                unique_classes = {}
                for entity in results:
                    cid = entity.get('class_id')
                    cname = entity.get('class_name')
                    if cid not in unique_classes:
                        unique_classes[cid] = cname
                
                print(f"\n   Collection '{collection}':")
                print(f"      Total embeddings: {len(results)}")
                print(f"      Unique classes: {len(unique_classes)}")
            
            client.close()
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print(f"\n⚠️  No liquor Milvus database found")
    
    # Check experiment results
    liquor_results = "experiments/milvus_release/metrics/experiment_comparison_liquor.json"
    if Path(liquor_results).exists():
        import json
        print(f"\n📊 Experiment results: {liquor_results}")
        
        with open(liquor_results, 'r') as f:
            results = json.load(f)
        
        for exp_name, metrics in results.items():
            if 'Baseline' in exp_name or 'Hybrid' in exp_name:
                print(f"\n   {exp_name}:")
                print(f"      mAP50: {metrics.get('map50', 0):.4f}")
                print(f"      Precision: {metrics.get('precision', 0):.4f}")
                print(f"      Recall: {metrics.get('recall', 0):.4f}")
                
                if 'per_class_ap' in metrics:
                    per_class = metrics['per_class_ap']
                    print(f"      Total classes in results: {len(per_class)}")

if __name__ == "__main__":
    check_liquor_dataset()
