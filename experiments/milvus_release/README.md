# YOLO + Milvus Hybrid Release Package

This folder contains everything you need to reproduce and publish the YOLO vs. Milvus embedding comparison.

```
experiments/milvus_release/
├── databases/
│   ├── milvus_grocery.db         # 128-D FLAT baseline (fast)
│   ├── milvus_grocery_dim256.db  # 256-D FLAT embeddings (richer descriptors)
│   ├── milvus_liquor.db          # 128-D FLAT baseline (fast)
│   └── milvus_liquor_dim256.db   # 256-D FLAT embeddings (richer descriptors)
├── metrics/
│   ├── experiment_comparison_grocery.json
│   ├── experiment_comparison_liquor.json
│   ├── metrics_YOLOv8_Baseline_488_Classes.json
│   ├── metrics_YOLOv8_DOLG_Milvus_HighThreshold.json
│   ├── metrics_YOLOv8_DOLG_Milvus_Hybrid.json
│   └── metrics_YOLOv11_Baseline_488_Classes.json
├── logs/
│   ├── grocery_experiment.log
│   └── liquor_experiment.log
└── README.md (this file)
```

## Reproducing the Databases
1. Ensure your environment has CUDA + PyTorch + `pymilvus[milvus_lite]` installed (see repo `requirements.txt`).
2. Download/prepare the Roboflow datasets referenced in `data/roboflow/grocery-rfn8l-v1/data.yaml` and `data/roboflow/liquor-data-v4/data.yaml`.
3. Run the population script from the repo root. Example commands (adjust paths as needed):
   ```bash
   source .venv/bin/activate  # or your environment
   python "yolo vs embeding_malvious/populate_milvus_embeddings.py" \
       --dataset data/roboflow/grocery-rfn8l-v1/data.yaml \
       --milvus-db "experiments/milvus_release/databases/milvus_grocery.db" \
       --collection grocery_items --max-templates 10 --batch-size 32 --device cuda:0

   # Higher-dimensional variant for accuracy-focused experiments
   python "yolo vs embeding_malvious/populate_milvus_embeddings.py" \
       --dataset data/roboflow/grocery-rfn8l-v1/data.yaml \
       --milvus-db "experiments/milvus_release/databases/milvus_grocery_dim256.db" \
       --collection grocery_items_dim256 --max-templates 10 --batch-size 32 \
       --device cuda:0 --embedding-dim 256 --index-type FLAT

   python "yolo vs embeding_malvious/populate_milvus_embeddings.py" \
       --dataset data/roboflow/liquor-data-v4/data.yaml \
       --milvus-db "experiments/milvus_release/databases/milvus_liquor.db" \
       --collection liquor_items --max-templates 10 --batch-size 32 --device cuda:0

   python "yolo vs embeding_malvious/populate_milvus_embeddings.py" \
       --dataset data/roboflow/liquor-data-v4/data.yaml \
       --milvus-db "experiments/milvus_release/databases/milvus_liquor_dim256.db" \
       --collection liquor_items_dim256 --max-templates 10 --batch-size 32 \
       --device cuda:0 --embedding-dim 256 --index-type FLAT
   ```
   The script extracts DOLG embeddings (EfficientNet-B0 backbone) and populates Milvus Lite files that can be committed to Git.

## Re-running the Comparative Experiments
1. Train/obtain YOLOv8 and YOLOv11 checkpoints for each dataset (see `experiments/train_yolov8.py` & `train_yolov11.py`).
2. Populate Milvus as above (or reuse the tracked DB files).
3. Execute the orchestrator from `yolo vs embeding_malvious/run_experiments.py`, pointing to the appropriate dataset YAML and pretrained weights. Example:
   ```bash
   python "yolo vs embeding_malvious/run_experiments.py" \
       --config "yolo vs embeding_malvious/experiment_config.yaml"
   ```
   Results are stored in `experiment_comparison.json`; rename to `experiment_comparison_grocery.json` or `..._liquor.json` and move into `metrics/` to keep the package updated.
4. Archive the console output into `logs/` (see `grocery_experiment.log` for reference) and refresh the plots under `figures/` when metrics change. Keep both 128-D and 256-D databases synced so latency/accuracy comparisons remain reproducible.

## Publishing
1. Add this directory and the updated reports to git:
   ```bash
   git add experiments/milvus_release README.md "Mechatronic Integration in Smart Point-of-Sale Systems_ A Comparative Study of YOLO and Effici.pdf"
   git commit -m "Publish YOLO + Milvus release bundle"
   git remote add origin https://github.com/sam02425/yolo_embeding_malvious.git  # once per clone
   git push origin <branch>
   ```
2. Tag the release if desired: `git tag v1.0-milvus && git push origin v1.0-milvus`.

## Notes
- The Milvus Lite files are portable and free to use; no external server is required.
- 128-D FLAT collections emphasize query speed; 256-D FLAT collections trade extra latency/storage for improved matching on visually similar SKUs. Use the CLI `--embedding-dim` flag to regenerate either variant.
- Grocery accuracy is low with off-the-shelf YOLO weights—plan to retrain before claiming production readiness.
- Liquor experiments achieve ~0.48 mAP@0.5 with YOLOv8m and form the baseline for Milvus-assisted retrieval.
- The accompanying PDF (`Mechatronic Integration ...`) has been updated to summarize these findings; include it in the GitHub release.
