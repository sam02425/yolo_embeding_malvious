#!/bin/bash
#
# Complete workflow to train retail-specific DOLG and run enhanced experiments
#

set -e  # Exit on error

echo "========================================"
echo "🚀 RETAIL DOLG TRAINING AND EXPERIMENTS"
echo "========================================"
echo ""

# Configuration
DATASET_YAML="data/grocery_augmented/grocery_augmented.yaml"
DOLG_OUTPUT_DIR="dolg_retail_model"
EXPERIMENT_CONFIG="yolo_vs_embeding_malvious/experiment_config_enhanced.yaml"
EPOCHS=50
BATCH_SIZE=32
LEARNING_RATE=1e-4
EMBEDDING_DIM=128

echo "📋 Configuration:"
echo "   Dataset: $DATASET_YAML"
echo "   Output Dir: $DOLG_OUTPUT_DIR"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Learning Rate: $LEARNING_RATE"
echo "   Embedding Dim: $EMBEDDING_DIM"
echo ""

# Step 1: Train DOLG on retail dataset
echo "========================================"
echo "📚 STEP 1: Training DOLG on Retail Dataset"
echo "========================================"
echo ""

if [ -f "$DOLG_OUTPUT_DIR/dolg_retail_best.pth" ]; then
    echo "⚠️  Retail DOLG model already exists at $DOLG_OUTPUT_DIR/dolg_retail_best.pth"
    read -p "   Do you want to retrain? (y/n): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "✅ Skipping training, using existing model"
    else
        echo "🔄 Retraining DOLG model..."
        python3 train_dolg_retail.py \
            --dataset "$DATASET_YAML" \
            --output-dir "$DOLG_OUTPUT_DIR" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LEARNING_RATE \
            --embedding-dim $EMBEDDING_DIM \
            --device cuda:0
    fi
else
    echo "🏋️  Training new DOLG model..."
    python3 train_dolg_retail.py \
        --dataset "$DATASET_YAML" \
        --output-dir "$DOLG_OUTPUT_DIR" \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LEARNING_RATE \
        --embedding-dim $EMBEDDING_DIM \
        --device cuda:0
fi

echo ""
echo "✅ DOLG training complete!"
echo ""

# Step 2: Populate Milvus with retail-trained embeddings
echo "========================================"
echo "🗄️  STEP 2: Populating Milvus with Retail Embeddings"
echo "========================================"
echo ""

MILVUS_DB_PATH="experiment_results/milvus_retail_trained.db"

if [ -f "$MILVUS_DB_PATH" ]; then
    echo "⚠️  Milvus database already exists at $MILVUS_DB_PATH"
    read -p "   Do you want to recreate it? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Removing old database..."
        rm -f "$MILVUS_DB_PATH"
        echo "📦 Creating new Milvus database with retail embeddings..."
        # TODO: Create script to populate with retail embeddings
        # For now, experiments will create it automatically
    else
        echo "✅ Using existing Milvus database"
    fi
else
    echo "📦 Milvus database will be created automatically during experiments"
fi

echo ""

# Step 3: Run enhanced experiments
echo "========================================"
echo "🧪 STEP 3: Running Enhanced Experiments"
echo "========================================"
echo ""

echo "📊 Experiments to run:"
echo "   1. Baseline (YOLOv8, YOLOv11)"
echo "   2. ImageNet DOLG (previous approach)"
echo "   3. Retail-trained DOLG (various thresholds)"
echo "   4. Confidence-based ensemble (various thresholds)"
echo "   5. Ensemble embeddings (ImageNet + Retail)"
echo ""

read -p "Start experiments now? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  Experiments postponed. Run manually with:"
    echo "   python3 yolo_vs_embeding_malvious/run_experiments.py --config $EXPERIMENT_CONFIG"
    exit 0
fi

echo "🚀 Starting experiments..."
echo ""

python3 yolo_vs_embeding_malvious/run_experiments.py \
    --config "$EXPERIMENT_CONFIG" \
    2>&1 | tee experiment_run_enhanced.log

echo ""
echo "========================================"
echo "✅ ALL STEPS COMPLETE!"
echo "========================================"
echo ""
echo "📊 Results:"
echo "   - DOLG model: $DOLG_OUTPUT_DIR/dolg_retail_best.pth"
echo "   - Training history: $DOLG_OUTPUT_DIR/training_history.json"
echo "   - Experiment results: experiment_comparison.json"
echo "   - Experiment log: experiment_run_enhanced.log"
echo ""
echo "📈 View results with:"
echo "   python3 -c 'import json; print(json.dumps(json.load(open(\"experiment_comparison.json\")), indent=2))'"
echo ""
