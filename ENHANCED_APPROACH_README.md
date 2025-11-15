# Enhanced Retail Detection with Retail-Trained DOLG and Confidence Ensemble

## 🎯 Overview

This enhanced approach addresses the root cause of poor Milvus hybrid performance: **generic ImageNet embeddings lack discrimination for retail products**.

### Previous Issue
- **Problem**: DOLG embeddings pretrained on ImageNet (generic objects) couldn't distinguish similar retail products
- **Result**: With 82.6% Milvus usage → 0.025 mAP (vs baseline 0.94 mAP)
- **Root Cause**: Generic features don't capture fine-grained retail details (logos, packaging, labels)

### Enhanced Solutions

#### 1. **Retail-Trained DOLG Embeddings** ✨
Train DOLG specifically on retail products to learn discriminative features:
- Fine-tune EfficientNet-B0 backbone on grocery dataset
- Use ArcFace loss for better metric learning
- Learn product-specific features (brand logos, packaging colors, label text)

#### 2. **Confidence-Based Ensemble** 🎭
Use Milvus only when YOLO is uncertain:
- High YOLO confidence (>0.7) → Trust YOLO prediction
- Low YOLO confidence (<0.7) → Query Milvus for similar products
- Leverages strengths: YOLO for clear cases, Milvus for ambiguous ones

#### 3. **Ensemble Embeddings** 🤝
Combine multiple embedding models:
- Retail-trained DOLG (retail-specific features)
- ImageNet-pretrained DOLG (general features)
- Concatenate or weighted average for robust representations

---

## 📦 Installation

### Requirements
```bash
pip install torch torchvision timm ultralytics pymilvus pyyaml tqdm opencv-python numpy
```

### File Structure
```
yolo_embeding_malvious_repo/
├── train_dolg_retail.py                    # Train DOLG on retail dataset
├── yolo_vs_embeding_malvious/
│   ├── retail_dolg_extractor.py            # Enhanced embedding extractors
│   ├── experimental_framework.py           # Updated with ensemble support
│   ├── experiment_config_enhanced.yaml     # New experiment configurations
│   └── run_experiments.py                  # Experiment orchestration
├── run_enhanced_pipeline.sh                # Complete automated workflow
└── data/
    └── grocery_augmented/
        └── grocery_augmented.yaml          # Dataset config
```

---

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)

```bash
# Run complete pipeline: Train DOLG → Populate Milvus → Run experiments
./run_enhanced_pipeline.sh
```

This will:
1. Train DOLG on retail dataset (50 epochs, ~2-4 hours on RTX 5080)
2. Create Milvus database with retail embeddings
3. Run all experiments and compare results

### Option 2: Manual Step-by-Step

#### Step 1: Train DOLG on Retail Dataset

```bash
python3 train_dolg_retail.py \
    --dataset data/grocery_augmented/grocery_augmented.yaml \
    --output-dir dolg_retail_model \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-4 \
    --embedding-dim 128 \
    --device cuda:0
```

**Training Details:**
- Extracts crops from all training images using YOLO labels
- Applies data augmentation (flips, color jitter, rotation)
- Uses ArcFace loss for better embedding discrimination
- Saves best model based on validation accuracy
- Training time: ~2-4 hours for 50 epochs (RTX 5080)

**Output:**
```
dolg_retail_model/
├── dolg_retail_best.pth          # Best model (highest val accuracy)
├── dolg_retail_final.pth         # Final model (last epoch)
├── dolg_retail_epoch_10.pth      # Checkpoint every 10 epochs
├── dolg_retail_epoch_20.pth
├── ...
└── training_history.json         # Loss/accuracy curves
```

#### Step 2: Run Enhanced Experiments

```bash
python3 yolo_vs_embeding_malvious/run_experiments.py \
    --config yolo_vs_embeding_malvious/experiment_config_enhanced.yaml \
    2>&1 | tee experiment_run_enhanced.log
```

#### Step 3: View Results

```bash
# Display formatted results
python3 -c "
import json

with open('experiment_comparison.json', 'r') as f:
    results = json.load(f)

print('\\n' + '='*100)
print('🎯 ENHANCED EXPERIMENT RESULTS')
print('='*100)
print(f'{'Experiment':<50} {'mAP@0.5':<12} {'Precision':<12} {'Recall':<12} {'Milvus%':<12}')
print('='*100)

for name, metrics in results.items():
    map50 = metrics.get('map50', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    milvus_rate = metrics.get('milvus_hit_rate', None)
    
    milvus_str = f'{milvus_rate*100:.1f}%' if milvus_rate is not None else 'N/A'
    print(f'{name:<50} {map50:<12.4f} {precision:<12.4f} {recall:<12.4f} {milvus_str:<12}')

print('='*100)
"
```

---

## 🧪 Experiments

### Experiment Groups

#### 1. **Baselines** (Reference Performance)
- `YOLOv8_Baseline`: Pure YOLOv8 detection (0.94 mAP)
- `YOLOv11_Baseline`: Pure YOLOv11 detection (0.93 mAP)

#### 2. **ImageNet DOLG** (Previous Approach)
- `Milvus_Hybrid_ImageNet_0.15`: Generic embeddings with similarity=0.15
  - **Expected**: ~0.025 mAP (confirmed poor performance)

#### 3. **Retail-Trained DOLG** (New Approach)
- `Milvus_Hybrid_Retail_0.15`: Retail embeddings with similarity=0.15
- `Milvus_Hybrid_Retail_0.20`: Retail embeddings with similarity=0.20
- `Milvus_Hybrid_Retail_0.25`: Retail embeddings with similarity=0.25
  - **Expected**: Significant improvement over ImageNet (target: >0.80 mAP)

#### 4. **Confidence-Based Ensemble**
- `Milvus_Ensemble_Retail_Conf0.5`: Use Milvus when YOLO confidence < 0.5
- `Milvus_Ensemble_Retail_Conf0.7`: Use Milvus when YOLO confidence < 0.7
- `Milvus_Ensemble_Retail_Conf0.8`: Use Milvus when YOLO confidence < 0.8
  - **Expected**: Best of both worlds (target: ≥0.94 mAP)

#### 5. **Ensemble Embeddings**
- `Milvus_EnsembleEmbedding_Retail`: Combine retail + ImageNet embeddings
  - **Expected**: More robust similarity matching

---

## 📊 Expected Results

### Hypothesis

| Experiment Type | Milvus Usage | Expected mAP | Reasoning |
|----------------|--------------|--------------|-----------|
| YOLOv8 Baseline | 0% | 0.94 | Known good baseline |
| ImageNet DOLG | 82.6% | 0.025 | Generic embeddings (confirmed) |
| **Retail DOLG** | **80-90%** | **0.80-0.90** | **Retail-specific features** |
| **Confidence Ensemble (0.7)** | **20-30%** | **0.92-0.96** | **Use Milvus only for uncertain cases** |
| Ensemble Embeddings | 80-90% | 0.85-0.92 | More robust matching |

### Key Insights

1. **Retail-trained DOLG should dramatically improve performance**
   - Learn product-specific features (logos, packaging, labels)
   - Better discrimination between similar products
   - Target: 0.80-0.90 mAP (vs 0.025 with ImageNet)

2. **Confidence ensemble should match/exceed baseline**
   - Trust YOLO for high-confidence detections (most cases)
   - Use Milvus only for low-confidence cases (ambiguous products)
   - Target: 0.92-0.96 mAP (potentially better than baseline)

3. **Optimal strategy depends on use case**
   - **High accuracy priority**: Confidence ensemble (conf=0.7)
   - **New product discovery**: Pure retail DOLG hybrid
   - **Robustness**: Ensemble embeddings

---

## 🔧 Configuration

### Experiment Config (`experiment_config_enhanced.yaml`)

```yaml
# Key parameters
advanced:
  iou_threshold: 0.25              # For bbox matching (adjusted for dataset quality)

milvus:
  embedding_dim: 128
  metric_type: "COSINE"

experiments:
  - name: "Milvus_Ensemble_Retail_Conf0.7"
    similarity_threshold: 0.20      # Milvus match threshold
    use_retail_embeddings: true
    retail_model_path: "dolg_retail_model/dolg_retail_best.pth"
    use_confidence_ensemble: true
    confidence_threshold: 0.7       # YOLO confidence threshold
```

### Training Config (`train_dolg_retail.py`)

```python
# Modify these parameters as needed
--epochs 50                         # Training epochs
--batch-size 32                     # Batch size
--lr 1e-4                          # Learning rate
--embedding-dim 128                 # Embedding dimension
--device cuda:0                     # GPU device
```

---

## 📈 Monitoring Training

### View Training Progress

```bash
# Watch training log
tail -f dolg_retail_model/training.log

# Plot training curves
python3 -c "
import json
import matplotlib.pyplot as plt

with open('dolg_retail_model/training_history.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc')
plt.plot(history['val_acc'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Training Accuracy')

plt.tight_layout()
plt.savefig('dolg_retail_model/training_curves.png')
print('✅ Saved training curves to dolg_retail_model/training_curves.png')
"
```

---

## 🎓 Technical Details

### Why Retail Training Helps

1. **Fine-Grained Discrimination**
   - Retail products have subtle differences (similar packaging, colors)
   - ImageNet trained on coarse categories (dog vs cat vs car)
   - Retail training learns product-specific features

2. **ArcFace Loss**
   - Improves embedding separation between classes
   - Enforces larger margins between embeddings
   - Better than standard cross-entropy for metric learning

3. **Data Augmentation**
   - Simulates real-world variations (lighting, rotation, occlusion)
   - Improves model robustness
   - Reduces overfitting

### Confidence Ensemble Strategy

```python
if yolo_confidence >= confidence_threshold:
    # High confidence → Trust YOLO
    final_class = yolo_class
else:
    # Low confidence → Query Milvus
    similar_items = milvus.search(embedding)
    if similar_items[0].distance >= similarity_threshold:
        final_class = similar_items[0].class_id
    else:
        final_class = yolo_class  # Fallback to YOLO
```

**Benefits:**
- Leverages YOLO's strong performance for clear cases
- Uses Milvus for ambiguous detections (occlusion, new products)
- Reduces Milvus queries (faster inference)
- Minimizes negative impact of incorrect Milvus matches

---

## 🐛 Troubleshooting

### Training Issues

**Problem**: Out of memory during training
```bash
# Solution: Reduce batch size
python3 train_dolg_retail.py --batch-size 16  # or 8
```

**Problem**: Training too slow
```bash
# Solution: Use fewer epochs for quick test
python3 train_dolg_retail.py --epochs 20
```

### Experiment Issues

**Problem**: `retail_dolg_extractor not found`
```bash
# Solution: Ensure file exists
ls yolo_vs_embeding_malvious/retail_dolg_extractor.py

# If missing, it will fallback to ImageNet (with warning)
```

**Problem**: Low performance even with retail model
```bash
# Solution: Check model was trained properly
python3 -c "
import torch
checkpoint = torch.load('dolg_retail_model/dolg_retail_best.pth')
print(f'Validation accuracy: {checkpoint[\"val_acc\"]:.2f}%')
"

# If val_acc < 80%, consider retraining with more epochs
```

---

## 📚 References

### DOLG Architecture
- Paper: "Unifying Deep Local and Global Features for Image Search" (Google Research, 2020)
- Uses EfficientNet backbone + global pooling + embedding head
- L2-normalized embeddings for cosine similarity

### ArcFace Loss
- Paper: "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (2019)
- Improves embedding discriminability
- Widely used in face recognition and metric learning

### Milvus Vector Database
- Open-source vector database optimized for similarity search
- Supports GPU acceleration for fast queries
- COSINE metric for embedding similarity

---

## 📝 Next Steps

1. **Run Training**: Train DOLG on retail dataset
2. **Baseline Comparison**: Compare retail vs ImageNet embeddings
3. **Threshold Tuning**: Find optimal similarity and confidence thresholds
4. **Production Deployment**: Deploy best-performing model

---

## 🤝 Contributing

Found an issue or have a suggestion? Please create an issue or submit a PR!

---

## 📄 License

MIT License - See LICENSE file for details
