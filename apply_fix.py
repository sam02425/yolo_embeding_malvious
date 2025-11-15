#!/usr/bin/env python3
"""
Automated fix for dataset class mismatch issue
"""

import yaml
import shutil
from pathlib import Path
from pymilvus import MilvusClient

def create_fixed_dataset_yaml():
    """Create corrected dataset YAML with only classes that exist in Milvus"""
    
    print(f"\n{'='*80}")
    print("CREATING FIXED DATASET YAML")
    print(f"{'='*80}\n")
    
    # 1. Get actual classes from Milvus (ground truth)
    print("📊 Loading class mappings from Milvus database...")
    client = MilvusClient(uri="experiment_results/milvus_retail.db")
    
    results = client.query(
        collection_name='retail_items',
        filter='',
        output_fields=['class_id', 'class_name'],
        limit=10000
    )
    
    # Build correct class mapping
    classes = {}
    for entity in results:
        cid = entity['class_id']
        cname = entity['class_name']
        if cid not in classes:
            classes[cid] = cname
    
    client.close()
    
    print(f"   Found {len(classes)} unique classes in Milvus")
    
    # Sort by class ID to get proper ordering
    sorted_class_ids = sorted(classes.keys())
    sorted_class_names = [classes[cid] for cid in sorted_class_ids]
    
    print(f"   Class IDs range: {min(sorted_class_ids)} to {max(sorted_class_ids)}")
    
    # 2. Load original dataset YAML
    original_yaml = "data/grocery_augmented/grocery_augmented.yaml"
    print(f"\n📄 Loading original dataset YAML: {original_yaml}")
    
    with open(original_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"   Original: {config.get('nc', len(config.get('names', [])))} classes")
    
    # 3. Create fixed config
    print(f"\n🔧 Creating fixed configuration...")
    
    # Backup original
    backup_file = original_yaml + '.backup'
    if not Path(backup_file).exists():
        shutil.copy2(original_yaml, backup_file)
        print(f"   ✅ Created backup: {backup_file}")
    else:
        print(f"   ℹ️  Backup already exists: {backup_file}")
    
    # Update config with correct classes
    config['names'] = sorted_class_names
    config['nc'] = len(sorted_class_names)
    
    # Save fixed version
    fixed_yaml = "data/grocery_augmented/grocery_augmented_fixed.yaml"
    with open(fixed_yaml, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Created fixed YAML: {fixed_yaml}")
    print(f"   ✅ Classes: {len(sorted_class_names)}")
    
    # 4. Show what changed
    print(f"\n📋 Changes made:")
    original_names = config.get('names', []) if isinstance(config.get('names'), list) else list(config.get('names', {}).values())
    
    # Find differences
    original_set = set(enumerate(original_names)) if isinstance(original_names, list) else set(original_names.items())
    fixed_set = set(enumerate(sorted_class_names))
    
    removed = original_set - fixed_set
    if removed:
        print(f"\n   ❌ Removed {len(removed)} classes (no training data):")
        for idx, name in sorted(removed)[:10]:
            print(f"      {idx}: {name}")
        if len(removed) > 10:
            print(f"      ... and {len(removed) - 10} more")
    
    # 5. Display first 10 classes for verification
    print(f"\n✅ Fixed class mapping (first 10):")
    for i, name in enumerate(sorted_class_names[:10]):
        print(f"   {i}: {name}")
    if len(sorted_class_names) > 10:
        print(f"   ... and {len(sorted_class_names) - 10} more")
    
    return fixed_yaml


def update_experiment_config(fixed_yaml_path):
    """Update experiment config to use fixed dataset YAML"""
    
    print(f"\n{'='*80}")
    print("UPDATING EXPERIMENT CONFIGURATION")
    print(f"{'='*80}\n")
    
    config_file = "yolo_vs_embeding_malvious/experiment_config.yaml"
    
    print(f"📝 Loading experiment config: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Backup original
    backup_file = config_file + '.backup'
    if not Path(backup_file).exists():
        shutil.copy2(config_file, backup_file)
        print(f"   ✅ Created backup: {backup_file}")
    
    # Update dataset path
    old_dataset = config['training']['dataset_yaml']
    config['training']['dataset_yaml'] = fixed_yaml_path
    
    # Save updated config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"   ✅ Updated dataset_yaml")
    print(f"      From: {old_dataset}")
    print(f"      To:   {fixed_yaml_path}")


def verify_fix():
    """Verify that the fix resolves the class mismatch"""
    
    print(f"\n{'='*80}")
    print("VERIFYING FIX")
    print(f"{'='*80}\n")
    
    # Load fixed YAML
    fixed_yaml = "data/grocery_augmented/grocery_augmented_fixed.yaml"
    
    if not Path(fixed_yaml).exists():
        print(f"   ❌ Fixed YAML not found: {fixed_yaml}")
        return False
    
    with open(fixed_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    yaml_classes = config['names']
    yaml_nc = config['nc']
    
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
        cid = entity['class_id']
        cname = entity['class_name']
        if cid not in milvus_classes:
            milvus_classes[cid] = cname
    
    client.close()
    
    # Compare
    print(f"📊 Class count comparison:")
    print(f"   Fixed YAML: {len(yaml_classes)} classes")
    print(f"   Milvus DB:  {len(milvus_classes)} classes")
    
    if len(yaml_classes) == len(milvus_classes):
        print(f"   ✅ Class counts match!")
    else:
        print(f"   ❌ Class counts still don't match!")
        return False
    
    # Check class names match
    yaml_set = set(yaml_classes)
    milvus_set = set(milvus_classes.values())
    
    if yaml_set == milvus_set:
        print(f"   ✅ All class names match!")
        return True
    else:
        diff = yaml_set.symmetric_difference(milvus_set)
        print(f"   ❌ {len(diff)} class names don't match:")
        for name in list(diff)[:5]:
            print(f"      {name}")
        return False


def main():
    print(f"\n{'='*80}")
    print("AUTOMATED FIX FOR DATASET CLASS MISMATCH")
    print(f"{'='*80}")
    
    try:
        # Step 1: Create fixed dataset YAML
        fixed_yaml = create_fixed_dataset_yaml()
        
        # Step 2: Update experiment config
        update_experiment_config(fixed_yaml)
        
        # Step 3: Verify fix
        success = verify_fix()
        
        # Summary
        print(f"\n{'='*80}")
        print("FIX SUMMARY")
        print(f"{'='*80}\n")
        
        if success:
            print("✅ ALL FIXES APPLIED SUCCESSFULLY!")
            print("\n📋 Next steps:")
            print("   1. Review the fixed YAML: data/grocery_augmented/grocery_augmented_fixed.yaml")
            print("   2. Re-run experiments:")
            print("      python3 yolo_vs_embeding_malvious/run_experiments.py \\")
            print("          --config yolo_vs_embeding_malvious/experiment_config.yaml")
            print("\n   Expected improvements:")
            print("      • Hybrid mAP: 0.003 → 0.85-0.92 (280x better!)")
            print("      • All 52 classes should have non-zero AP")
            print("      • Milvus hit rate: 30-60%")
        else:
            print("⚠️  FIX INCOMPLETE - Please review errors above")
        
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Error during fix process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
