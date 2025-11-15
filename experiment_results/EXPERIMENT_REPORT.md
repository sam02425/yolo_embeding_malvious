# Retail Item Detection - Experiment Comparison Report

Generated: 2025-11-15 02:40:54

## Executive Summary

- **Best mAP@0.5**: YOLOv8_Baseline_488_Classes (0.9414)
- **Best F1-Score**: YOLOv11_Baseline_488_Classes (0.9175)
- **Fastest**: YOLOv8_Baseline_488_Classes (264.31 FPS)

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
- Inference Time: 3.78 ms
- FPS: 264.31
- Preprocess Time: 0.86 ms
- Postprocess Time: 0.19 ms

---

### YOLOv11_Baseline_488_Classes

**Detection Metrics:**
- mAP@0.5: 0.9352
- mAP@0.5:0.95: 0.8692
- Precision: 0.8948
- Recall: 0.9413
- F1-Score: 0.9175

**Speed Metrics:**
- Inference Time: 3.91 ms
- FPS: 255.85
- Preprocess Time: 0.87 ms
- Postprocess Time: 0.21 ms

---

### YOLOv8_DOLG_Milvus_Hybrid

**Detection Metrics:**
- mAP@0.5: 0.1138
- mAP@0.5:0.95: 0.1138
- Precision: 0.1893
- Recall: 0.1924
- F1-Score: 0.1909

**Speed Metrics:**
- Inference Time: 5.32 ms
- FPS: 187.82
- Preprocess Time: 0.00 ms
- Postprocess Time: 0.00 ms

**Milvus Integration:**
- Hit Rate: 12.23%
- Embedding Time: 54.78 ms
- Search Time: 1.06 ms

---

### YOLOv8_DOLG_Milvus_HighThreshold

**Detection Metrics:**
- mAP@0.5: 0.1138
- mAP@0.5:0.95: 0.1138
- Precision: 0.1893
- Recall: 0.1924
- F1-Score: 0.1909

**Speed Metrics:**
- Inference Time: 5.31 ms
- FPS: 188.23
- Preprocess Time: 0.00 ms
- Postprocess Time: 0.00 ms

**Milvus Integration:**
- Hit Rate: 12.23%
- Embedding Time: 4.84 ms
- Search Time: 1.11 ms

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

