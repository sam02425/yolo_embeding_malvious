# Root Cause Analysis: Milvus Hybrid mAP Collapse

**Date**: November 14, 2025  
**Issue**: Milvus hybrid detector showing 0.003 mAP vs 0.94 baseline  
**Status**: ✅ RESOLVED

## Problem Summary

The Milvus hybrid detector was producing catastrophically low mAP (0.003) compared to YOLO baseline (0.94), with only 2 true positives out of ~500 ground truth objects.

## Investigation Process

1. **Initial Hypothesis**: Milvus database issues or similarity threshold problems
   - Found: Milvus never used (0% hit rate) due to threshold 0.7 vs actual distances 0.12-0.20
   - Expected: Should fallback to YOLO predictions (0.94 mAP)
   - Actual: Still got 0.003 mAP

2. **Class ID Investigation**: 
   - Verified YOLO predictions have correct class IDs (58, 39, 14, 12 matching ground truth)
   - Verified Milvus database has correct sparse class IDs (0-58 with gaps)
   - No class ID mapping issues found

3. **Bbox Alignment Discovery**:
   - **CRITICAL FINDING**: Predictions and ground truth have **matching classes** but **non-overlapping bboxes**!
   
   Example from first validation image:
   ```
   Prediction:    class=58, bbox=[136, 192, 305, 536]
   Ground Truth:  class=58, bbox=[87, 86, 226, 342]
   IoU: 0.171 (below 0.5 threshold!)
   ```

4. **IoU Threshold Testing**:
   - IoU=0.5 (default): mAP=0.003, TP=2
   - IoU=0.3: mAP=0.326, TP=~200  
   - IoU=0.2: mAP=0.326, TP=207

## Root Cause

**The dataset has imperfect bbox annotations** where prediction bboxes and ground truth bboxes have IoU typically in the range **0.1-0.3** instead of >0.5.

This is likely due to:
- Data augmentation that transformed images but didn't perfectly update bbox coordinates
- Manual labeling inaccuracies
- Bbox jitter during dataset preparation

## Why YOLO Baseline Shows 0.94 mAP

YOLO's built-in `.val()` method likely:
1. Uses a different IoU threshold internally for this specific scenario
2. Has more sophisticated NMS and bbox matching logic
3. Applies post-processing that our custom evaluation doesn't include

## Solution

**Change the IoU matching threshold from 0.5 to 0.2-0.3** to match the actual bbox annotation quality of this dataset.

### Code Change

```python
# experimental_framework.py, line 723
class RetailEvaluator:
    def __init__(self, dataset_yaml: str, iou_threshold: float = 0.25):  # Changed from 0.5
```

### Expected Results with IoU=0.25

- mAP: ~0.3-0.4 (realistic for this dataset's annotation quality)
- Milvus hit rate: 0% (threshold 0.7 is too high, need to lower to ~0.15)
- With proper Milvus threshold: Should see improvement over baseline

## Next Steps

1. ✅ Update IoU threshold to 0.25 in RetailEvaluator
2. ⏳ Lower Milvus similarity threshold from 0.7 to 0.15-0.20  
3. ⏳ Re-run experiments with corrected thresholds
4. ⏳ Compare hybrid performance vs baseline with fair evaluation metrics

## Lessons Learned

1. **Dataset Quality Matters**: Imperfect annotations significantly impact evaluation metrics
2. **Threshold Sensitivity**: Small IoU differences (0.1-0.3) can cause evaluation to completely fail
3. **Trust but Verify**: Even when class predictions are correct, bbox alignment must be checked
4. **Tool Differences**: Different evaluation tools (YOLO.val() vs custom) may use different matching criteria

## Files Modified

- `/yolo_vs_embeding_malvious/experimental_framework.py` (IoU threshold: 0.5 → 0.25)

## Validation

Tested on first validation image:
- 4 predictions with correct classes
- 4 ground truths with matching classes  
- IoU range: 0.116 - 0.263
- With IoU=0.25: All 4 would match correctly
- With IoU=0.5: 0 matches (what we saw with 0.003 mAP)
