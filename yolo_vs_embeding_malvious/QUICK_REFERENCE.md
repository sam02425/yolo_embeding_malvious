# Quick Reference Guide - Retail Detection Experiments

## 📦 What You Have

### Core Scripts
1. **experimental_framework.py** - Main experiment runner (all comparisons)
2. **populate_milvus_embeddings.py** - Database population with DOLG embeddings
3. **run_experiments.py** - Complete pipeline orchestrator
4. **setup_experiments.sh** - Automated environment setup

### Configuration
- **experiment_config.yaml** - Complete experiment configuration
- **requirements.txt** - Python dependencies

### Documentation
- **README_EXPERIMENTS.md** - Comprehensive user guide
- **IMPLEMENTATION_GUIDE.md** - Detailed implementation summary

## 🚀 Quick Start

### Option 1: Automated Pipeline (Recommended)
```bash
# 1. Setup environment
chmod +x setup_experiments.sh
./setup_experiments.sh

# 2. Prepare your dataset
# - Place images in data/images/train/ and data/images/val/
# - Place labels in data/labels/train/ and data/labels/val/
# - Update data/retail_488.yaml with your 488 class names

# 3. Run complete pipeline
python run_experiments.py --config experiment_config.yaml

# 4. View results
mlflow ui --backend-store-uri file:./mlruns
# Open: http://localhost:5000
```

### Option 2: Step-by-Step
```bash
# Step 1: Train YOLOv8
python yolo_tuning.py --data data/retail_488.yaml --model yolov8m.pt --epochs 100

# Step 2: Train YOLOv11
python yolo_tuning.py --data data/retail_488.yaml --model yolo11m.pt --epochs 100

# Step 3: Populate Milvus
python populate_milvus_embeddings.py --dataset data/retail_488.yaml --max-templates 10

# Step 4: Run experiments
python experimental_framework.py --dataset data/retail_488.yaml --run-all

# Step 5: View results
mlflow ui
```

### Option 3: Baseline Only (Fastest)
```bash
# Just compare YOLOv8 vs YOLOv11 (no Milvus)
python experimental_framework.py \
  --dataset data/retail_488.yaml \
  --yolov8-model yolov8m.pt \
  --yolov11-model yolo11m.pt \
  --run-baseline-only
```

## 📊 What Gets Measured

### Detection Metrics
- ✅ mAP@0.5 - Mean Average Precision at IoU 0.5
- ✅ mAP@0.5:0.95 - Mean AP across IoU thresholds
- ✅ Precision - True Positives / (TP + False Positives)
- ✅ Recall - True Positives / (TP + False Negatives)
- ✅ F1-Score - Harmonic mean of Precision & Recall
- ✅ Per-class metrics - Individual performance for all 488 classes

### Speed Metrics
- ⚡ FPS - Frames per second throughput
- ⚡ Inference Time - Milliseconds per image
- ⚡ Preprocess Time - Image preprocessing overhead
- ⚡ Postprocess Time - NMS and formatting time

### Hybrid Metrics (for Milvus experiments)
- 🔍 Milvus Hit Rate - % detections using similarity search
- 🔍 Embedding Time - DOLG extraction time per detection
- 🔍 Search Time - Milvus query time per detection

## 🎯 Four Experiments Run

1. **YOLOv8 Baseline (488 classes)**
   - Standard YOLOv8 detection
   - Baseline for comparison

2. **YOLOv11 Baseline (488 classes)**
   - Newer YOLO architecture
   - Compare vs YOLOv8

3. **YOLOv8 + DOLG + Milvus (threshold=0.5)**
   - Hybrid approach with embeddings
   - Standard similarity threshold

4. **YOLOv8 + DOLG + Milvus (threshold=0.7)**
   - Stricter similarity matching
   - Higher precision, potentially lower recall

## 📁 Expected Output Structure

```
experiment_results/
├── training/                       # Trained models
│   ├── yolov8_488/weights/best.pt
│   └── yolov11_488/weights/best.pt
├── visualizations/                 # Charts & graphs
│   ├── detection_performance.png   # mAP, P, R, F1 comparison
│   ├── speed_performance.png       # FPS and time analysis
│   ├── milvus_analysis.png         # Hybrid approach analysis
│   ├── summary_table.png           # Results table
│   └── results_summary.csv         # Exportable data
├── milvus_retail.db               # Vector database
├── embedding_cache.pkl            # Cached DOLG embeddings
├── experiment_comparison.json     # All metrics (JSON)
└── EXPERIMENT_REPORT.md           # Comprehensive report

mlruns/                            # MLflow tracking
└── 0/                            # Experiment
    ├── <run_id_1>/              # Each experiment run
    ├── <run_id_2>/
    ├── <run_id_3>/
    └── <run_id_4>/
```

## ⚙️ Key Configuration Options

### Basic Settings (experiment_config.yaml)
```yaml
training:
  dataset_yaml: 'data/retail_488.yaml'
  
  # Pretrained models (use these if available - SAVES TIME!)
  yolov8_pretrained: 'models/yolov8_488_classes_best.pt'
  yolov11_pretrained: 'models/yolov11_488_classes_best.pt'
  
  # Training settings
  epochs: 100              # Training epochs
  batch_size: 16          # Adjust for GPU memory
  imgsz: 640              # Image size (416/640/1280)
  device: 'cuda:0'        # GPU device
  
  # Control flags
  force_retrain: false    # Set true to ignore pretrained models
  train_yolov8: true      # Set false to skip YOLOv8 completely
  train_yolov11: true     # Set false to skip YOLOv11 completely

milvus:
  max_templates_per_class: 10  # Embeddings per class

experiments:
  similarity_threshold: 0.5     # Milvus matching threshold
```

### 💡 Using Pretrained Models (RECOMMENDED)

If you already have trained models, save 8-12 hours:

```bash
# 1. Place your models in models/ directory
mkdir -p models
cp /path/to/your/trained_yolov8.pt models/yolov8_488_classes_best.pt
cp /path/to/your/trained_yolov11.pt models/yolov11_488_classes_best.pt

# 2. Update experiment_config.yaml
training:
  yolov8_pretrained: 'models/yolov8_488_classes_best.pt'
  yolov11_pretrained: 'models/yolov11_488_classes_best.pt'
  force_retrain: false

# 3. Run experiments (will use pretrained - FAST!)
python run_experiments.py --config experiment_config.yaml
```

**See PRETRAINED_MODELS_GUIDE.md for detailed instructions**

### GPU Memory Issues?
```yaml
training:
  batch_size: 8           # Reduce if OOM
  imgsz: 416             # Smaller images
```

### Want More Accuracy?
```yaml
training:
  epochs: 200            # More training
  imgsz: 1280           # Larger images
  
milvus:
  max_templates_per_class: 20  # More templates
```

## 🎨 Visualizations Generated

### 1. Detection Performance (4-panel chart)
- mAP@0.5 bar chart
- Precision vs Recall comparison
- F1-Score comparison
- Multi-metric radar chart

### 2. Speed Performance (2-panel chart)
- FPS comparison with real-time threshold
- Inference time breakdown

### 3. Milvus Analysis (if applicable)
- Hit rate by experiment
- Accuracy vs Milvus usage scatter plot

### 4. Summary Table
- Publication-ready results table
- All metrics in one view

## 🔧 Common Commands

### Train Models
```bash
# YOLOv8 with default settings
python yolo_tuning.py --data data/retail_488.yaml --model yolov8m.pt

# YOLOv8 with hyperparameter tuning (slower but better)
python yolo_tuning.py --data data/retail_488.yaml --model yolov8m.pt --iterations 30

# YOLOv8 on specific GPU
python yolo_tuning.py --data data/retail_488.yaml --model yolov8m.pt --device cuda:1
```

### Populate Milvus
```bash
# Standard population
python populate_milvus_embeddings.py --dataset data/retail_488.yaml

# With more templates per class
python populate_milvus_embeddings.py --dataset data/retail_488.yaml --max-templates 20

# Use cached embeddings (faster)
python populate_milvus_embeddings.py --dataset data/retail_488.yaml --cache embedding_cache.pkl
```

### Run Experiments
```bash
# All experiments
python experimental_framework.py --dataset data/retail_488.yaml --run-all

# Baseline only
python experimental_framework.py --dataset data/retail_488.yaml --run-baseline-only

# Custom similarity threshold
python experimental_framework.py --dataset data/retail_488.yaml --similarity-threshold 0.6
```

### View Results
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Start on different port
mlflow ui --backend-store-uri file:./mlruns --port 5001

# Access from remote machine
mlflow ui --backend-store-uri file:./mlruns --host 0.0.0.0
```

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
--batch 8  # or even 4

# Use smaller images
--imgsz 416

# Use smaller model
--model yolov8s.pt  # instead of yolov8m.pt
```

### Slow Embedding Extraction
```bash
# Ensure GPU is used
--device cuda:0

# Use embedding cache
--cache embedding_cache.pkl

# Reduce templates per class
--max-templates 5
```

### Milvus Database Locked
```bash
# Kill processes
pkill -f milvus

# Remove lock files
rm milvus_retail.db/*.lock

# Use new database
--milvus-db milvus_new.db
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade

# Check CUDA version
python -c "import torch; print(torch.cuda.is_available())"

# Test imports
python -c "from ultralytics import YOLO; from pymilvus import MilvusClient"
```

## 📈 Expected Performance

### YOLOv8 Baseline
- **mAP@0.5**: ~0.75-0.85 (depends on dataset quality)
- **FPS**: ~40-60 FPS (on modern GPU)
- **Inference**: ~15-25ms per image

### YOLOv11 Baseline
- **mAP@0.5**: ~0.77-0.87 (typically 2-3% better than v8)
- **FPS**: ~35-55 FPS (slightly slower than v8)
- **Inference**: ~18-28ms per image

### Hybrid Approach
- **mAP@0.5**: ~0.78-0.88 (expected 1-5% improvement)
- **FPS**: ~15-25 FPS (slower due to embeddings)
- **Inference**: ~40-70ms per image (including embedding + search)
- **Milvus Hit Rate**: ~30-60% (varies by threshold)

## 📚 Documentation Files

- **README_EXPERIMENTS.md** - Full user guide with examples
- **IMPLEMENTATION_GUIDE.md** - Technical implementation details
- **This file** - Quick reference for common tasks

## 💡 Tips for Best Results

1. **Dataset Quality Matters**
   - Ensure good quality annotations
   - Balance class distribution if possible
   - Remove corrupted images/labels

2. **Start Small, Scale Up**
   - Test with baseline first
   - Add Milvus only if needed
   - Tune hyperparameters last

3. **Monitor Training**
   - Watch MLflow metrics during training
   - Check for overfitting (val loss increases)
   - Stop early if not improving

4. **Compare Fairly**
   - Use same dataset for all experiments
   - Same image size and batch size
   - Same hardware for speed tests

5. **Iterate and Improve**
   - Analyze failure cases
   - Collect more data for hard classes
   - Update embeddings with new data

## 🎯 Success Criteria

**For Production Deployment**, you want:

✅ mAP@0.5 > 0.80 (good detection accuracy)
✅ Precision > 0.85 (few false alarms)
✅ Recall > 0.75 (catch most items)
✅ FPS > 30 (real-time video processing)

**If hybrid approach meets these criteria better than baseline → Use it!**
**If baseline is fast enough and accurate → Keep it simple!**

## 🚀 Ready to Start!

```bash
# Quick 3-step start
./setup_experiments.sh
# ... prepare your dataset ...
python run_experiments.py --config experiment_config.yaml
```

**That's it! The framework handles everything else automatically.**

---

**Questions?** Check README_EXPERIMENTS.md or IMPLEMENTATION_GUIDE.md

**Need help?** All code is well-commented and includes error handling

**Good luck with your experiments! 🎉**
