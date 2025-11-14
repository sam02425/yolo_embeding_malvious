#!/bin/bash

# Retail Detection Experiments - Quick Start Setup Script
# This script helps you set up and run the experimental framework

set -e  # Exit on error

echo "=================================================="
echo "Retail Item Detection - Experimental Setup"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

if ! python3 -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
    echo -e "${RED}Error: Python 3.9+ is required${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python version OK${NC}"
echo ""

# Check CUDA availability
echo "Checking CUDA availability..."
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    cuda_version=$(python3 -c "import torch; print(torch.version.cuda)")
    echo -e "${GREEN}✓ CUDA available: $cuda_version${NC}"
    gpu_count=$(python3 -c "import torch; print(torch.cuda.device_count())")
    echo "  GPUs available: $gpu_count"
else
    echo -e "${YELLOW}⚠ CUDA not available - will run on CPU (much slower)${NC}"
fi
echo ""

# Install dependencies
echo "Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi
echo ""

# Create directory structure
echo "Creating directory structure..."
mkdir -p data/{images,labels}/{train,val}
mkdir -p models
mkdir -p experiment_results/{training,tuning,visualizations}
echo -e "${GREEN}✓ Directories created${NC}"
echo ""

# Check for dataset
echo "Checking dataset configuration..."
if [ -f "data/retail_488.yaml" ]; then
    echo -e "${GREEN}✓ Dataset configuration found${NC}"
else
    echo -e "${YELLOW}⚠ Dataset configuration not found${NC}"
    echo "  Creating example dataset configuration..."
    
    cat > data/retail_488.yaml << 'EOF'
# Retail Item Detection Dataset - 488 Classes
path: ../data  # dataset root dir
train: images/train  # train images (relative to 'path')
val: images/val  # val images (relative to 'path')

# Number of classes
nc: 488

# Class names (example - update with your actual classes)
names:
  0: Item_001
  1: Item_002
  2: Item_003
  # ... add all 488 classes ...
  487: Item_488
EOF
    
    echo -e "${YELLOW}  Please update data/retail_488.yaml with your actual classes${NC}"
fi
echo ""

# Check for YOLO weights
echo "Checking YOLO model weights..."
if [ ! -f "models/yolov8m.pt" ]; then
    echo "  Downloading YOLOv8m weights..."
    python3 -c "from ultralytics import YOLO; YOLO('yolov8m.pt')" 2>/dev/null
    mv yolov8m.pt models/ 2>/dev/null || true
fi

if [ ! -f "models/yolo11m.pt" ]; then
    echo "  Downloading YOLOv11m weights..."
    python3 -c "from ultralytics import YOLO; YOLO('yolo11m.pt')" 2>/dev/null
    mv yolo11m.pt models/ 2>/dev/null || true
fi
echo -e "${GREEN}✓ YOLO weights ready${NC}"
echo ""

# Check configuration file
echo "Checking experiment configuration..."
if [ -f "experiment_config.yaml" ]; then
    echo -e "${GREEN}✓ Configuration file found${NC}"
else
    echo -e "${YELLOW}⚠ Configuration file not found - creating default${NC}"
    python3 run_experiments.py --config experiment_config.yaml 2>/dev/null || true
fi
echo ""

# Display usage instructions
echo "=================================================="
echo "Setup Complete!"
echo "=================================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Prepare your dataset:"
echo "   - Place training images in: data/images/train/"
echo "   - Place training labels in: data/labels/train/"
echo "   - Place validation images in: data/images/val/"
echo "   - Place validation labels in: data/labels/val/"
echo "   - Update data/retail_488.yaml with your class names"
echo ""
echo "2. (Optional) Update experiment configuration:"
echo "   - Edit experiment_config.yaml"
echo "   - Adjust batch size, image size, epochs, etc."
echo ""
echo "3. Run experiments:"
echo ""
echo "   Option A - Complete Pipeline (Recommended):"
echo "   $ python run_experiments.py --config experiment_config.yaml"
echo ""
echo "   Option B - Baseline Only (Faster):"
echo "   $ python experimental_framework.py --dataset data/retail_488.yaml --run-baseline-only"
echo ""
echo "   Option C - Step by Step:"
echo "   $ python yolo_tuning.py --data data/retail_488.yaml --model yolov8m.pt --epochs 100"
echo "   $ python populate_milvus_embeddings.py --dataset data/retail_488.yaml"
echo "   $ python experimental_framework.py --dataset data/retail_488.yaml --run-all"
echo ""
echo "4. View results:"
echo "   $ mlflow ui --backend-store-uri file:./mlruns"
echo "   Then open: http://localhost:5000"
echo ""
echo "=================================================="
echo ""
echo -e "${GREEN}Ready to start! 🚀${NC}"
echo ""

# Ask if user wants to run a test
read -p "Would you like to run a quick test? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running quick test..."
    
    # Test imports
    python3 << 'PYTHON_TEST'
import sys
print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  GPUs: {torch.cuda.device_count()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("✓ Ultralytics YOLO")
except ImportError as e:
    print(f"✗ Ultralytics import failed: {e}")
    sys.exit(1)

try:
    from pymilvus import MilvusClient
    print("✓ Pymilvus")
except ImportError as e:
    print(f"✗ Pymilvus import failed: {e}")
    sys.exit(1)

try:
    import mlflow
    print(f"✓ MLflow {mlflow.__version__}")
except ImportError as e:
    print(f"✗ MLflow import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    print("✓ Data science libraries")
except ImportError as e:
    print(f"✗ Data library import failed: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
print("\nYou're ready to run experiments!")
PYTHON_TEST
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}All tests passed! 🎉${NC}"
    else
        echo ""
        echo -e "${RED}Some tests failed. Please check the errors above.${NC}"
    fi
fi

echo ""
echo "For detailed documentation, see: README_EXPERIMENTS.md"
echo ""
