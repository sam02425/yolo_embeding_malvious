# AI Agent Guidelines: YOLO + Milvus Retail Detection Research

## Project Overview

This is a **research codebase** comparing YOLO object detection with Milvus vector similarity search for retail product recognition. The project explores a **hybrid approach** that combines YOLO detections with DOLG (Deep Orthogonal Local and Global) embeddings stored in Milvus to improve accuracy on visually similar products.

**Key Insight**: Generic ImageNet embeddings failed catastrophically (0.003 mAP vs 0.941 baseline). The solution uses **retail-trained DOLG embeddings** with **confidence-based ensemble** logic.

## Architecture & Major Components

### 1. Three-Part Hybrid Detection Pipeline

```
Image → YOLO Detection → Confidence Check → Milvus Retrieval → Final Prediction
                              ↓ (high conf)        ↓ (low conf)
                         Trust YOLO          Extract embedding
                                             Query Milvus DB
                                             Refine class
```

**Implementation**: `yolo_vs_embeding_malvious/experimental_framework.py`
- `HybridYOLODetector`: Orchestrates YOLO + Milvus pipeline (lines 616+)
- `DOLGEmbeddingExtractor`: Extracts 128-D embeddings from crops (lines 109+)
- `MilvusRetailDB`: Manages vector similarity search (lines 391+)
- `RetailEvaluator`: Computes mAP/precision/recall metrics (lines 834+)

### 2. Embedding Extractor Variants

**Three embedding strategies** (see `retail_dolg_extractor.py`):
- `DOLGEmbeddingExtractor` - Generic ImageNet features (legacy, poor performance)
- `RetailDOLGExtractor` - Retail-trained DOLG (lines 74+, ~0.80-0.90 mAP expected)
- `EnsembleDOLGExtractor` - Combines multiple models (lines 181+, most robust)

**Factory pattern**: `create_embedding_extractor()` in `experimental_framework.py` switches between extractors based on config.

### 3. Experiment Orchestration

**Master script**: `run_experiments.py` (652 lines)
- Loads experiment configs from YAML
- Handles YOLO training/tuning (or uses pretrained models)
- Populates Milvus databases with embeddings
- Runs all experiments in sequence
- Generates comparison reports and visualizations

**Configuration**: `experiment_config_enhanced.yaml` defines 10 experiments:
- 2 baselines (YOLOv8, YOLOv11)
- 1 ImageNet DOLG (failed approach, kept for comparison)
- 3 retail DOLG variants (similarity thresholds 0.15/0.20/0.25)
- 3 confidence ensemble variants (confidence thresholds 0.5/0.7/0.8)
- 1 ensemble embedding variant

## Critical Developer Workflows

### Running Experiments (Production)

```bash
# Full pipeline: Train retail DOLG → Run experiments → Generate report
./run_enhanced_pipeline.sh

# Manual control: Skip training if model exists
python3 yolo_vs_embeding_malvious/run_experiments.py \
    --config yolo_vs_embeding_malvious/experiment_config_enhanced.yaml \
    2>&1 | tee experiment_run.log
```

**Key files created**: `experiment_comparison.json`, `experiment_run*.log`, `metrics_*.json`

### Training Retail DOLG Embeddings

```bash
python3 train_dolg_retail.py \
    --dataset data/grocery_augmented/grocery_augmented.yaml \
    --epochs 50 --batch-size 32 --lr 1e-4 --device cuda:0
```

**Output**: `dolg_retail_model/dolg_retail_best.pth` (2-4 hours on RTX 5080)

**Dataset requirements**: 
- Expects YOLO format: `images/` and `labels/` directories with paired files
- Crops bounding boxes from images to train embedding model
- See `RetailProductDataset` class in `train_dolg_retail.py` (lines 24+)

### Populating Milvus Databases

```bash
python3 yolo_vs_embeding_malvious/populate_milvus_embeddings.py \
    --dataset data/grocery.v3i.yolov11/data.yaml \
    --milvus-db experiments/milvus_release/databases/milvus_grocery.db \
    --collection grocery_items --max-templates 10 --device cuda:0
```

**Critical options**:
- `--embedding-dim 128` or `256` (128 = speed, 256 = accuracy)
- `--min-box-area` - Skip small/occluded products
- `--template-augmentations` - Generate lighting/viewpoint variants
- `--retail-model-path` - Use retail-trained DOLG instead of ImageNet

**Database format**: Milvus Lite (embedded, no external server needed)

### Training Augmented Grocery Baselines

```bash
python3 scripts/train_grocery_baselines.py \
    --base-dataset data/grocery.v3i.yolov11 \
    --background-dataset data/empty-shelf-detection.v1i.yolov11 \
    --output-dataset data/grocery_augmented \
    --train-yolov8 --train-yolov11 \
    --epochs 100 --device cuda:0
```

**What it does**: Mixes grocery products with empty shelf backgrounds to improve generalization

## Project-Specific Conventions

### Dataset Configuration Pitfalls

**CRITICAL**: Class counts in `data.yaml` **must match** actual training data. Mismatch causes catastrophic failure.

**Fixed in**: `FINAL_REPORT.md` documents the 7 missing classes issue that broke initial experiments:
- Dataset YAML declared 59 classes, but only 52 had training data
- Milvus correctly skipped missing classes → class ID misalignment
- Fix: `grocery_augmented_fixed.yaml` with only 52 classes

**When modifying datasets**: 
1. Verify class counts: `python3 check_missing_classes.py --dataset <path>`
2. Always backup original YAML before changes
3. Update Milvus databases after dataset changes

### Confidence-Based Ensemble Logic

**Key pattern** in `HybridYOLODetector.detect()` (~line 680):
```python
if config.use_confidence_ensemble:
    if yolo_confidence >= config.confidence_threshold:
        # High confidence → trust YOLO
        return yolo_prediction
    else:
        # Low confidence → query Milvus for refinement
        embedding = extractor.extract(crop)
        similar_items = milvus.search(embedding)
        return refined_prediction
```

**Tuning thresholds**:
- `confidence_threshold: 0.7` - Sweet spot for most experiments
- `similarity_threshold: 0.15-0.25` - Lower = more permissive Milvus matching
- `iou_threshold: 0.25` - Bbox overlap for evaluation (NOT inference)

### MLflow Experiment Tracking

**Auto-enabled** in all training/evaluation scripts:
- Experiment name: `retail_detection_enhanced`
- URI: `file:./mlruns` (local filesystem)
- Metrics logged: mAP, precision, recall, F1, inference time, Milvus hit rate

**View results**: `mlflow ui --backend-store-uri ./mlruns`

### GPU Optimizations (CUDA-Only)

**All scripts require CUDA** - CPU execution is blocked:
```python
if not torch.cuda.is_available():
    raise RuntimeError("CUDA required. Please check GPU setup.")
```

**Optimizations enabled**:
- AMP (Automatic Mixed Precision): `use_amp: true`
- TF32 matmul on Ampere GPUs
- CuDNN benchmark mode
- Batch processing for embeddings (default 32)

**Memory management**: Auto-scales batch size down on CUDA OOM

## External Dependencies & Data Flow

### Ultralytics YOLO Integration

**Models supported**: YOLOv8m, YOLOv11m/n (loaded via `ultralytics.YOLO`)
- Pretrained weights: `yolo11m.pt`, `yolov8m.pt` in repo root
- Fine-tuned models: `grocery/runs/yolov8_grocery_baseline_*/weights/best.pt`

**Inference**: Uses `model.predict()` for batch processing, `model.val()` for evaluation

### Milvus Lite Vector DB

**No external server needed** - databases are SQLite-like files:
- `milvus_grocery.db`, `milvus_liquor.db` in `experiments/milvus_release/databases/`
- Schema: `[id, embedding, class_id, template_id, class_name]`
- Index: FLAT (exact search, no quantization)
- Metric: COSINE similarity

**API pattern**:
```python
client = MilvusClient(db_path)
results = client.search(
    collection_name="retail_items",
    data=[embedding],
    limit=5,
    output_fields=["class_id", "class_name"]
)
```

### TIMM (Torch Image Models)

**Backbone**: EfficientNet-B0 for DOLG embeddings
- Loaded via `timm.create_model('efficientnet_b0', pretrained=True)`
- Feature extraction: `features_only=True, out_indices=[3]`
- Global pooling → 512-D → 128-D embeddings (L2 normalized)

## Common Debugging Scenarios

### "Milvus hit rate high but mAP still low"

**Root cause**: Class ID mismatch between YOLO and Milvus
**Fix**: Verify dataset YAML matches Milvus collection (see `FINAL_REPORT.md`)
**Debug**: `python3 diagnose_milvus_issue.py --milvus-db <path> --dataset <yaml>`

### "Training diverges / NaN loss"

**Check**:
1. Learning rate too high? Default 1e-4 works for retail DOLG
2. ArcFace margin too aggressive? Default s=30, m=0.5
3. Empty crops in dataset? Increase `--min-crop-size 64`

**Debug**: Monitor `dolg_training.log`, check `training_history.json`

### "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `--batch-size 16` → `8`
2. Reduce embedding dim: `--embedding-dim 128` (not 256)
3. Disable AMP if unstable: `use_amp: false` in config

### "Experiments incomplete / crashes"

**Check**:
- `experiment_run*.log` files for tracebacks
- Milvus DB corruption: Delete `.db` file and repopulate
- Pretrained model paths in config are valid
- MLflow tracking server not running? Ignore, uses local file store

## Key Documentation to Read First

1. **QUICK_START.md** - 10-minute setup guide, expected results table
2. **SOLUTION_SUMMARY.md** - Technical deep-dive on retail DOLG approach
3. **FINAL_REPORT.md** - Class mismatch bug that broke initial experiments
4. **README.md** - Dataset info, Milvus DB specs, reproduction steps

## Testing & Validation

**No formal test suite** - this is research code. Validation via:
- Baseline experiments (YOLOv8/YOLOv11 without Milvus) for sanity checks
- mAP comparison: Hybrid should be ±10% of baseline, not -97%
- Milvus hit rate: Should be 70-90% for successful retrieval

**Diagnostic scripts**:
- `test_first_image_iou.py` - Debug single-image inference
- `diagnose_hybrid.py` - Trace hybrid pipeline step-by-step
- `check_missing_classes.py` - Validate dataset integrity

## Version Pinning

**Critical versions** (see `requirements.txt`):
- `torch>=2.0.0` - PyTorch 2.x for `torch.compile()` support
- `ultralytics>=8.0.0` - YOLO11 requires latest
- `pymilvus>=2.5.0` - Milvus Lite embedded database
- `timm` - No version pin, latest stable works

**Python**: 3.10+ recommended (tested on 3.11)

---

**When extending this codebase**: Focus on the `experimental_framework.py` → `HybridYOLODetector` class for detection logic, and `experiment_config_enhanced.yaml` for adding new experiment variants. Always backup Milvus DBs before changes - repopulation takes 30+ minutes.
