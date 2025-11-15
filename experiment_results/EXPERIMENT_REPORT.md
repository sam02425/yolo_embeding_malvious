# Retail Item Detection - Experiment Comparison Report

Generated: 2025-11-14 23:08:27

## Executive Summary

- **Best mAP@0.5**: YOLOv8_Baseline_488_Classes (0.9414)
- **Best F1-Score**: YOLOv11_Baseline_488_Classes (0.9175)
- **Fastest**: YOLOv8_Baseline_488_Classes (276.03 FPS)

## Experiment Configuration

- **Dataset**: data/grocery_augmented/grocery_augmented.yaml
- **Number of Classes**: 488
- **Image Size**: 640px
- **Batch Size**: 16
- **Device**: cuda:0

## Detailed Results

### YOLOv8_Baseline_488_Classes

**Detection Metrics:**
- mAP@0.5: 0.9414
- mAP@0.5:0.95: 0.8613
- Precision: 0.8720
- Recall: 0.9327
- F1-Score: 0.9013

**Speed Metrics:**
- Inference Time: 3.62 ms
- FPS: 276.03
- Preprocess Time: 0.84 ms
- Postprocess Time: 0.20 ms

---

### YOLOv11_Baseline_488_Classes

**Detection Metrics:**
- mAP@0.5: 0.9352
- mAP@0.5:0.95: 0.8692
- Precision: 0.8948
- Recall: 0.9413
- F1-Score: 0.9175

**Speed Metrics:**
- Inference Time: 3.71 ms
- FPS: 269.25
- Preprocess Time: 0.90 ms
- Postprocess Time: 0.21 ms

---

### YOLOv8_DOLG_Milvus_Hybrid

**Detection Metrics:**
- mAP@0.5: 0.0253
- mAP@0.5:0.95: 0.0253
- Precision: 0.0454
- Recall: 0.0461
- F1-Score: 0.0457

**Speed Metrics:**
- Inference Time: 5.42 ms
- FPS: 184.43
- Preprocess Time: 0.00 ms
- Postprocess Time: 0.00 ms

**Milvus Integration:**
- Hit Rate: 82.64%
- Embedding Time: 54.72 ms
- Search Time: 1.10 ms

---

### YOLOv8_DOLG_Milvus_HighThreshold

**Detection Metrics:**
- mAP@0.5: 0.1124
- mAP@0.5:0.95: 0.1124
- Precision: 0.1874
- Recall: 0.1904
- F1-Score: 0.1889

**Speed Metrics:**
- Inference Time: 5.39 ms
- FPS: 185.58
- Preprocess Time: 0.00 ms
- Postprocess Time: 0.00 ms

---

## Visualizations

See the `visualizations/` directory for:
- Detection performance comparison charts
- Speed analysis graphs
- Milvus integration analysis
- Summary tables and CSV exports

## Recommendations

⚠️ **Hybrid approach did not outperform baseline**

- Consider adjusting similarity thresholds
- May need more diverse training templates
- Baseline YOLO may be sufficient for current use case

## MLflow Tracking

All experiments are logged to MLflow. View results:
```bash
mlflow ui --backend-store-uri file:./mlruns
```

