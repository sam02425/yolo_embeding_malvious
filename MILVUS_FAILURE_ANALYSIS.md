# Milvus Hybrid Approach Failure Analysis

## Executive Summary

The Milvus hybrid approach is completely failing with ~0.003 mAP (compared to baseline 0.941 mAP) due to **three critical issues**:

1. **Collection Name Mismatch** - Code expects `retail_items_dolg` but database has `retail_items`
2. **Class Name Mismatch** - Milvus has 52 classes, but hybrid returns 59 classes (7 extra)
3. **Zero Milvus Hit Rate** - 0.00% hit rate means Milvus similarity search is never being used

---

## Detailed Root Cause Analysis

### Issue #1: Collection Name Mismatch ❌ CRITICAL

**Problem:**
```python
# Code expects:
collection_name = "retail_items_dolg"

# Database actually has:
collection_name = "retail_items"
```

**Evidence from Diagnostics:**
```
Available collections: ['retail_items']
❌ Collection 'retail_items_dolg' not found!
```

**Impact:**
- Every Milvus search fails with: `MilvusException: collection not found`
- This causes the hybrid detector to **fall back to YOLO predictions only**
- But the fallback is broken (see Issue #2)

**Fix Location:**
```python
# File: yolo_vs_embeding_malvious/experimental_framework.py
# Line ~1300
milvus_collection=args.milvus_collection or "retail_items_dolg"  # ❌ Wrong!
# Should be:
milvus_collection=args.milvus_collection or "retail_items"  # ✅ Correct
```

---

### Issue #2: Class Name/ID Mismatch ❌ CRITICAL

**Problem:**
The Milvus database contains **52 training classes**, but the hybrid evaluation is somehow producing **59 classes** (7 extra classes that don't exist in training).

**Evidence:**
```
Baseline classes: 52
Hybrid classes: 59

Classes only in hybrid (7):
  - Sprite20Oz
  - Sprite40Oz
  - Vanilla-Coca-cola 20Oz
  - Whatchamacallit-kingSize
  - dietCoke-Can16oz
  - skittles-smoothies-shareSize
  - zeroCocaCola-16Oz
```

**Root Cause:**
When Milvus search fails (due to Issue #1), the code falls back to using YOLO's class predictions. However, there's a **mismatch between**:
- **YOLO model class mapping** (trained on 488 classes but using a subset)
- **Ground truth dataset class mapping** (52 classes in test set)
- **Milvus database class mapping** (416 embeddings covering 52 classes)

The YOLO model is likely predicting class IDs that map to the wrong class names, or the class ID → class name mapping is inconsistent between:
1. YOLO model's `yolo_results.names` dictionary
2. Ground truth label files
3. Milvus database `class_name` field

**Impact:**
- Predictions use wrong class IDs/names
- Ground truth matching fails completely
- mAP collapses to near-zero
- Only 2 classes have AP > 0 (SourPunch-shareSize: 0.1818, Sprite-TropicalMix-20Oz: 0.0202)

---

### Issue #3: Zero Milvus Hit Rate (Consequence of #1 & #2)

**Problem:**
```
Milvus Hit Rate: 0.00%
```

**Explanation:**
Because Milvus searches are failing (collection not found), the code **never uses Milvus predictions**. The hybrid detector is essentially just running broken YOLO inference.

**Code Location:**
```python
# File: yolo_vs_embeding_malvious/experimental_framework.py
# Lines 666-680

# Search Milvus
similar_items = self.milvus_db.search_similar(embedding, top_k=1)

# Use Milvus result if similarity is high enough
if similar_items and similar_items[0]['distance'] >= self.similarity_threshold:
    # This NEVER executes because similar_items is empty!
    final_class_id = similar_items[0]['entity']['class_id']
    final_class_name = similar_items[0]['entity']['class_name']
    used_milvus = True
    milvus_hits += 1
else:
    # Always falls here, using broken YOLO predictions
    final_class_id = info['yolo_class_id']
    final_class_name = info['yolo_class_name']
```

---

## Why Only 2 Classes Have Non-Zero AP

**Classes with AP > 0:**
- `SourPunch-shareSize`: 0.1818
- `Sprite-TropicalMix-20Oz`: 0.0202

**Explanation:**
By pure luck, these 2 classes happen to have:
1. Correct class ID mapping between YOLO predictions and ground truth
2. Enough confidence to trigger detections
3. Sufficient IoU overlap for some true positives

The other 50 classes all fail due to class ID/name mismatches.

---

## Database Statistics

**Milvus Database Contents:**
```
Collection: retail_items
Total entities: 416 embeddings
Total unique classes: 52 (estimated)
Embedding dimension: 128
Index type: FLAT (likely)
Metric type: COSINE (likely)
```

**Sample entity structure:**
```python
{
    'id': 462206197502574592,
    'class_id': 0,
    'class_name': 'Barg-sBlack-20Oz',
    'embedding': [128-d float vector],
    'template_idx': 0,
    'upc': 'UPC_0000'
}
```

**Embedding statistics:**
- Embeddings are L2-normalized (norms ≈ 1.0)
- Dimension: 128
- Generated from EfficientNet-B0 (not actual DOLG model)
- Warning: "Using EfficientNet pretrained on ImageNet" (no DOLG weights found)

---

## Additional Issues

### Issue #4: DOLG Model Not Trained ⚠️ WARNING

**Evidence:**
```
⚠️  DOLG weights not found at dolg_model.pth
   Using EfficientNet pretrained on ImageNet
   For best results, train DOLG model on retail dataset
```

**Impact:**
- Embeddings are from generic ImageNet-pretrained EfficientNet
- NOT specialized for retail products
- This reduces similarity search quality even if other issues are fixed

---

## Step-by-Step Failure Flow

1. **Experiment starts**: Load YOLOv8 + DOLG + Milvus hybrid detector
2. **YOLO detects objects**: Works fine, detects bounding boxes
3. **Extract embeddings**: Works (but using generic EfficientNet, not DOLG)
4. **Search Milvus**: ❌ FAILS with "collection not found" error
5. **Fallback to YOLO predictions**: Uses YOLO's class predictions
6. **Class ID mismatch**: YOLO class IDs don't match ground truth class IDs
7. **Evaluation fails**: IoU matching succeeds but class matching fails
8. **Result**: mAP ≈ 0, only 2 classes work by accident

---

## Fixes Required

### Fix #1: Correct Collection Name (CRITICAL)

**File:** `yolo_vs_embeding_malvious/experimental_framework.py`

**Change line ~1300:**
```python
# Before:
milvus_collection=args.milvus_collection or "retail_items_dolg"

# After:
milvus_collection=args.milvus_collection or "retail_items"
```

**Also update other references:**
```python
# Line ~1320 (HighThreshold experiment)
milvus_collection=args.milvus_collection or "retail_items"

# Configuration file default
# File: experiment_config.yaml
milvus:
  collection_name: 'retail_items'  # Change from 'retail_items_dolg'
```

---

### Fix #2: Verify Class Mapping Consistency (CRITICAL)

**Need to ensure consistency across:**

1. **YOLO model class names:**
   ```python
   yolo_model.names  # Dictionary: {class_id: class_name}
   ```

2. **Ground truth labels:**
   ```
   # Format: class_id x_center y_center width height
   # class_id must match YOLO model's class mapping
   ```

3. **Dataset YAML:**
   ```yaml
   # data/grocery_augmented/grocery_augmented.yaml
   names:
     0: Barg-sBlack-20Oz
     1: Bueno-shareSize
     # ... must match YOLO training
   ```

4. **Milvus database:**
   ```python
   # Ensure class_id and class_name match YOLO model
   # During population: use YOLO model's names dictionary
   ```

**Verification script needed:**
```python
# Create a script to verify all class mappings match
def verify_class_consistency():
    yolo = YOLO(model_path)
    with open(dataset_yaml) as f:
        dataset = yaml.safe_load(f)
    
    # Compare
    yolo_classes = set(yolo.names.values())
    dataset_classes = set(dataset['names'].values())
    milvus_classes = set([get classes from Milvus])
    
    # Must all be identical
    assert yolo_classes == dataset_classes == milvus_classes
```

---

### Fix #3: Handle Milvus Search Failures Gracefully (RECOMMENDED)

**Current code:**
```python
similar_items = self.milvus_db.search_similar(embedding, top_k=1)
# No error handling if search fails!
```

**Better approach:**
```python
try:
    similar_items = self.milvus_db.search_similar(embedding, top_k=1)
except Exception as e:
    print(f"⚠️  Milvus search failed: {e}")
    similar_items = []
    # Fall back to YOLO prediction gracefully
```

---

### Fix #4: Train Actual DOLG Model (OPTIONAL BUT RECOMMENDED)

The current system uses generic ImageNet EfficientNet, not retail-specialized DOLG embeddings.

**To properly train DOLG:**
1. Collect retail product image pairs (same product, different angles)
2. Train with metric learning loss (ArcFace, CosFace, or DOLG's specialized loss)
3. Fine-tune on retail dataset
4. Save weights to `dolg_model.pth`

**Impact:**
- Better embedding quality
- Higher similarity scores for correct matches
- Lower false positive rate

---

## Testing Plan

### Test 1: Verify Collection Name Fix
```bash
python3 -c "
from pymilvus import MilvusClient
client = MilvusClient(uri='experiment_results/milvus_retail.db')
print('Collections:', client.list_collections())
# Should see: ['retail_items']
"
```

### Test 2: Verify Class Consistency
```python
# Create verify_classes.py
from ultralytics import YOLO
import yaml
from pymilvus import MilvusClient

yolo = YOLO('path/to/model.pt')
with open('data/grocery_augmented/grocery_augmented.yaml') as f:
    dataset = yaml.safe_load(f)

client = MilvusClient(uri='experiment_results/milvus_retail.db')
milvus_classes = set(client.query(
    collection_name='retail_items',
    filter='',
    output_fields=['class_name'],
    limit=1000
))

print(f"YOLO classes: {len(yolo.names)}")
print(f"Dataset classes: {len(dataset['names'])}")
print(f"Milvus classes: {len(milvus_classes)}")
# Should all match!
```

### Test 3: Re-run Experiments
```bash
python3 yolo_vs_embeding_malvious/run_experiments.py \
    --config yolo_vs_embeding_malvious/experiment_config.yaml
```

**Expected results after fixes:**
- Milvus hit rate > 0% (should be 20-80% depending on similarity threshold)
- mAP should be comparable to baseline (0.90-0.95)
- All 52 classes should have non-zero AP

---

## Summary of Root Causes

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| Collection name mismatch (`retail_items_dolg` vs `retail_items`) | 🔴 CRITICAL | 100% of Milvus searches fail | Not Fixed |
| Class ID/name mapping inconsistency | 🔴 CRITICAL | Predictions don't match ground truth | Not Fixed |
| Zero error handling for Milvus failures | 🟡 HIGH | Silent failure, hard to debug | Not Fixed |
| Using generic EfficientNet instead of trained DOLG | 🟠 MEDIUM | Suboptimal embeddings | Not Fixed |

---

## Expected Performance After Fixes

**Scenario 1: Just fix collection name (Fix #1)**
- Milvus searches will work
- Hit rate: 20-50% (with threshold 0.3)
- mAP: 0.60-0.80 (degraded due to generic embeddings)

**Scenario 2: Fix collection name + class consistency (Fix #1 + #2)**
- Milvus searches work correctly
- Hit rate: 30-60%
- mAP: 0.85-0.92 (close to baseline)

**Scenario 3: All fixes including trained DOLG (Fix #1 + #2 + #4)**
- Optimal Milvus performance
- Hit rate: 50-80%
- mAP: 0.92-0.96 (potentially better than baseline!)

---

## Conclusion

The Milvus hybrid approach didn't work because:
1. **Wrong collection name** → Milvus searches fail
2. **Class mapping inconsistency** → Predictions don't match ground truth
3. **No error handling** → Failures are silent

These are all **implementation bugs**, not fundamental flaws in the approach. Once fixed, the hybrid method should work as intended.

**Next steps:**
1. Fix collection name (5 minutes)
2. Verify class consistency (15 minutes)
3. Re-run experiments (30 minutes)
4. (Optional) Train proper DOLG model (several hours)
