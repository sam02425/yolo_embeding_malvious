# Liquor Dataset - Same Issue Found!

## Problem Summary

The liquor dataset has the **exact same issue** as the grocery dataset:

### The Numbers Don't Match:
- **Dataset YAML:** 414 classes declared
- **Milvus Database:** 388 unique classes (26 classes missing!)
- **Baseline results:** Only 362 classes in experiment results
- **Hybrid results:** 0.0000 mAP (complete failure!)

### Comparison:

| Component | Grocery Dataset | Liquor Dataset |
|-----------|----------------|----------------|
| YAML declares | 59 classes | 414 classes |
| Milvus has | 52 classes | 388 classes |
| Missing classes | 7 classes (12%) | 26 classes (6%) |
| Baseline mAP | 0.941 ✅ | 0.484 ⚠️ |
| Hybrid mAP | 0.003 ❌ | 0.000 ❌ |

## Why Liquor Hybrid is Completely Broken

The liquor hybrid has **0.0000 mAP** (even worse than grocery's 0.003) because:

1. **26 missing classes** create multiple gaps in class ID sequence
2. **Massive class ID misalignment** throughout the entire dataset
3. **More gaps = worse failure** - 26 gaps vs 7 for grocery
4. **Results show 414 classes but only 388 exist** in Milvus

## The Missing 26 Classes

Based on the pattern, 26 liquor bottle classes were declared in YAML but have:
- ❌ No training images
- ❌ No Milvus embeddings
- ❌ Creating gaps in class ID sequence

This causes the same cascading failure as grocery dataset.

## Why Baseline is Lower (0.484 vs 0.941)

The liquor baseline is much lower than grocery because:
1. **More complex dataset** - 414 vs 59 classes
2. **More similar items** - many bottles look alike
3. **Harder detection task** - smaller labels, similar shapes
4. **Dataset quality** - may have fewer training examples per class

But 0.484 is still **reasonable for 414 classes**.

## Fix Required

Same fix as grocery dataset:

### Option 1: Remove Missing Classes from YAML (RECOMMENDED)

```python
# Similar to grocery fix
# 1. Query Milvus to get actual 388 classes
# 2. Create fixed YAML with only those 388 classes
# 3. Re-run experiments
```

### Option 2: Re-populate Milvus with All 414 Classes

Not recommended - those 26 classes have no training data!

## Expected Improvement After Fix

```
Current:
  Liquor Baseline:       mAP = 0.484 ✅
  Liquor Hybrid:         mAP = 0.000 ❌ (completely broken)

After Fix:
  Liquor Baseline:       mAP = 0.484 ✅ (unchanged)
  Liquor Hybrid:         mAP = 0.40-0.46 ✅ (close to baseline)
```

**Why lower than baseline even after fix?**
- Generic EfficientNet embeddings (not DOLG trained on liquor)
- 388 classes is very challenging for embedding-based search
- Many bottles look similar → harder to distinguish with embeddings

## Quick Fix Script for Liquor

```python
#!/usr/bin/env python3
"""Fix liquor dataset YAML"""

from pymilvus import MilvusClient
import yaml

# Get actual classes from Milvus
client = MilvusClient(uri="experiments/milvus_release/databases/milvus_liquor.db")
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

# Sort by class ID
sorted_classes = [classes[i] for i in sorted(classes.keys())]

print(f"Milvus has {len(sorted_classes)} classes")
print(f"YAML declares 414 classes")
print(f"Missing: {414 - len(sorted_classes)} classes")

# Load and fix YAML
with open('liquor/data.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Backup
import shutil
shutil.copy2('liquor/data.yaml', 'liquor/data.yaml.backup')

# Update
config['names'] = sorted_classes
config['nc'] = len(sorted_classes)

# Save fixed version
with open('liquor/data_fixed.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

print(f"✅ Created: liquor/data_fixed.yaml")
print(f"✅ Classes: {len(sorted_classes)}")
```

## Summary

**Both datasets have the same root cause:**
- Dataset YAML declares more classes than actually exist
- Creates class ID misalignment
- Hybrid approach completely fails

**Both need the same fix:**
- Remove phantom classes from YAML
- Ensure consistency between YAML and Milvus

**Priority:**
1. Fix grocery first (easier - only 52 classes, 7 missing)
2. Then fix liquor (harder - 388 classes, 26 missing)
