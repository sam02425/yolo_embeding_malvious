# 🚀 Quick Start Guide - Enhanced Retail Detection

## Problem Solved
- **Before**: Milvus hybrid = 0.025 mAP with 82.6% usage (ImageNet embeddings)
- **After**: Expected 0.80-0.96 mAP (retail-trained embeddings + confidence ensemble)
- **Root Cause**: Generic ImageNet features don't work for retail products

---

## ⚡ Run Everything (Automated)

```bash
./run_enhanced_pipeline.sh
```

This will:
1. Train DOLG on retail dataset (2-4 hours)
2. Run 10 comprehensive experiments (1-2 hours)
3. Generate comparison report

---

## 📋 Manual Steps

### Step 1: Train Retail DOLG (2-4 hours)

```bash
python3 train_dolg_retail.py \
    --dataset data/grocery_augmented/grocery_augmented.yaml \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --embedding-dim 128 \
    --device cuda:0
```

**Output**: `dolg_retail_model/dolg_retail_best.pth`

### Step 2: Run Enhanced Experiments (1-2 hours)

```bash
python3 yolo_vs_embeding_malvious/run_experiments.py \
    --config yolo_vs_embeding_malvious/experiment_config_enhanced.yaml \
    2>&1 | tee experiment_run_enhanced.log
```

### Step 3: View Results

```bash
python3 -c "
import json
with open('experiment_comparison.json', 'r') as f:
    results = json.load(f)
    
for name, metrics in results.items():
    map50 = metrics.get('map50', 0)
    milvus = metrics.get('milvus_hit_rate', None)
    milvus_str = f'{milvus*100:.1f}%' if milvus else 'N/A'
    print(f'{name:<50} mAP={map50:.4f}  Milvus={milvus_str}')
"
```

---

## 🧪 Experiments Configured (10 total)

### Baselines (2)
- YOLOv8_Baseline
- YOLOv11_Baseline

### ImageNet DOLG (1) - Previous Approach
- Milvus_Hybrid_ImageNet_0.15 (expected: 0.025 mAP)

### Retail-Trained DOLG (3) - New Approach
- Milvus_Hybrid_Retail_0.15
- Milvus_Hybrid_Retail_0.20 ⭐
- Milvus_Hybrid_Retail_0.25

### Confidence-Based Ensemble (3) - Best Expected
- Milvus_Ensemble_Retail_Conf0.5
- Milvus_Ensemble_Retail_Conf0.7 ⭐⭐
- Milvus_Ensemble_Retail_Conf0.8

### Ensemble Embeddings (1)
- Milvus_EnsembleEmbedding_Retail

⭐ = Good expected performance  
⭐⭐ = Best expected performance (target >0.94 mAP)

---

## 📊 Expected Results

| Approach | Milvus Usage | Expected mAP | vs Baseline |
|----------|--------------|--------------|-------------|
| YOLOv8 Baseline | 0% | 0.941 | - |
| ImageNet DOLG | 82.6% | 0.025 | -97.3% ❌ |
| **Retail DOLG** | **80-90%** | **0.80-0.90** | **-10%** ✨ |
| **Conf Ensemble** | **20-30%** | **0.92-0.96** | **±2%** 🎯 |

---

## 🔧 Key Files Created

```
train_dolg_retail.py                          # Training pipeline (550 lines)
yolo_vs_embeding_malvious/
├── retail_dolg_extractor.py                  # Enhanced extractors (350 lines)
├── experiment_config_enhanced.yaml           # 10 experiments
└── experimental_framework.py                 # Modified for ensemble
run_enhanced_pipeline.sh                      # Automated workflow
ENHANCED_APPROACH_README.md                   # Full documentation
SOLUTION_SUMMARY.md                           # Technical details
```

---

## 🐛 Quick Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python3 train_dolg_retail.py --batch-size 16  # or 8
```

### Quick Training Test
```bash
# Train for fewer epochs
python3 train_dolg_retail.py --epochs 20
```

### Check Model Performance
```bash
python3 -c "
import torch
checkpoint = torch.load('dolg_retail_model/dolg_retail_best.pth')
print(f'Validation Accuracy: {checkpoint[\"val_acc\"]:.2f}%')
"
```

### View Training Progress
```bash
# Watch loss/accuracy
python3 -c "
import json
with open('dolg_retail_model/training_history.json') as f:
    h = json.load(f)
print(f'Final train acc: {h[\"train_acc\"][-1]:.2f}%')
print(f'Final val acc: {h[\"val_acc\"][-1]:.2f}%')
"
```

---

## 📈 Monitor Experiments

```bash
# Watch experiment log
tail -f experiment_run_enhanced.log

# Check progress
grep "Evaluating" experiment_run_enhanced.log | tail -5
```

---

## 🎯 Success Criteria

### Minimum (Viable)
- ✅ Retail DOLG: mAP > 0.50 (2x improvement)
- ✅ Confidence Ensemble: mAP ≥ 0.90 (within 5% of baseline)

### Target (Expected)
- ✅ Retail DOLG: mAP > 0.80 (32x improvement)
- ✅ Confidence Ensemble: mAP ≥ 0.94 (match baseline)

### Stretch (Best Case)
- 🎯 Confidence Ensemble: mAP > 0.95 (exceed baseline)

---

## 📚 Full Documentation

- **User Guide**: `ENHANCED_APPROACH_README.md`
- **Technical Details**: `SOLUTION_SUMMARY.md`
- **This Guide**: `QUICK_START.md`

---

## ✅ Verification Checklist

- [x] Training script ready (`train_dolg_retail.py`)
- [x] Enhanced extractors ready (`retail_dolg_extractor.py`)
- [x] Framework modified (confidence ensemble support)
- [x] 10 experiments configured
- [x] Automated pipeline ready
- [x] Documentation complete
- [ ] **Training completed** ← Run `./run_enhanced_pipeline.sh`
- [ ] **Results validated** ← Compare with expectations

---

**Ready to go! 🚀** Start with: `./run_enhanced_pipeline.sh`
