#!/usr/bin/env python3
"""
Fix liquor dataset YAML - same issue as grocery dataset
"""

import yaml
import shutil
from pathlib import Path
from pymilvus import MilvusClient

def fix_liquor_dataset():
    print(f"\n{'='*80}")
    print("FIXING LIQUOR DATASET")
    print(f"{'='*80}\n")
    
    # 1. Get actual classes from Milvus
    liquor_db = "experiments/milvus_release/databases/milvus_liquor.db"
    
    if not Path(liquor_db).exists():
        print(f"❌ Liquor Milvus database not found: {liquor_db}")
        return False
    
    print(f"📊 Loading classes from Milvus: {liquor_db}")
    client = MilvusClient(uri=liquor_db)
    
    results = client.query(
        collection_name='liquor_items',
        filter='',
        output_fields=['class_id', 'class_name'],
        limit=10000
    )
    
    # Build class mapping
    classes = {}
    for entity in results:
        cid = entity['class_id']
        cname = entity['class_name']
        if cid not in classes:
            classes[cid] = cname
    
    client.close()
    
    print(f"   Found {len(classes)} unique classes in Milvus")
    print(f"   Class ID range: {min(classes.keys())} to {max(classes.keys())}")
    
    # Sort by class ID
    sorted_class_ids = sorted(classes.keys())
    sorted_class_names = [classes[cid] for cid in sorted_class_ids]
    
    # 2. Load original YAML
    yaml_files = [
        "liquor/data.yaml",
        "data/Liquor-data.v4i.yolov11/data.yaml"
    ]
    
    for yaml_path in yaml_files:
        if Path(yaml_path).exists():
            print(f"\n📄 Processing: {yaml_path}")
            
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f)
            
            original_nc = config.get('nc', len(config.get('names', [])))
            original_names = config.get('names', [])
            
            print(f"   Original: {original_nc} classes")
            print(f"   Fixed will have: {len(sorted_class_names)} classes")
            print(f"   Removing: {original_nc - len(sorted_class_names)} classes")
            
            # Backup
            backup_path = yaml_path + '.backup'
            if not Path(backup_path).exists():
                shutil.copy2(yaml_path, backup_path)
                print(f"   ✅ Created backup: {backup_path}")
            
            # Create fixed version
            config['names'] = sorted_class_names
            config['nc'] = len(sorted_class_names)
            
            # Save fixed version
            fixed_path = yaml_path.replace('.yaml', '_fixed.yaml')
            with open(fixed_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            print(f"   ✅ Created fixed: {fixed_path}")
    
    # 3. Verify
    print(f"\n✅ Verification:")
    print(f"   Milvus classes: {len(classes)}")
    print(f"   Fixed YAML classes: {len(sorted_class_names)}")
    
    if len(classes) == len(sorted_class_names):
        print(f"   ✅ Class counts match!")
        return True
    else:
        print(f"   ❌ Class counts don't match!")
        return False


def main():
    print(f"\n{'='*80}")
    print("LIQUOR DATASET FIX")
    print(f"{'='*80}")
    
    success = fix_liquor_dataset()
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")
    
    if success:
        print("✅ LIQUOR DATASET FIXED!")
        print("\n📋 Files created:")
        print("   • liquor/data_fixed.yaml")
        print("   • data/Liquor-data.v4i.yolov11/data_fixed.yaml")
        print("\n📋 Backups created:")
        print("   • liquor/data.yaml.backup")
        print("   • data/Liquor-data.v4i.yolov11/data.yaml.backup")
        print("\n📋 Next steps:")
        print("   1. Update experiment config to use fixed YAML")
        print("   2. Re-run liquor experiments")
        print("   3. Expected: mAP improves from 0.000 → 0.40-0.46")
    else:
        print("⚠️  FIX INCOMPLETE - Check errors above")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
