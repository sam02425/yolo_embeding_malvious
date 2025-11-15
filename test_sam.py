#!/usr/bin/env python3
"""
Test SAM augmentation with a small dataset
"""

import sys
import os
sys.path.append('.')

from sam_augmentation import SAMAugmentor

def test_sam_augmentation():
    """Test SAM augmentation with small number of samples"""

    print("🧪 Testing SAM-based augmentation...")

    # Create test output directory
    test_output = "data/sam_test"

    # Create augmentor
    augmentor = SAMAugmentor(device="cuda:0")

    # Create small test dataset (10 samples)
    augmentor.create_augmented_dataset(
        output_dir=test_output,
        num_samples=10,
        grocery_path="data/grocery.v3i.yolov11",
        liquor_path="data/Liquor-data.v4i.yolov11",
        background_path="data/empty-shelf-detection.v1i.yolov11"
    )

    print("✅ Test completed successfully!")

if __name__ == '__main__':
    test_sam_augmentation()