# YOLO + Milvus Embedding Experiments

This export contains everything needed to reproduce and publish the YOLO-versus-Milvus comparisons described in the “Mechatronic Integration in Smart Point-of-Sale Systems” addendum.

## Contents

- `yolo_vs_embeding_malvious/` – Complete orchestration, evaluation, and Milvus-population scripts.
- `experiments/milvus_release/` – Git-ready artifacts:
  - `databases/milvus_grocery.db` and `databases/milvus_liquor.db` (Milvus Lite binaries populated with DOLG embeddings).
  - `metrics/` – JSON outputs from grocery/liquor experiments plus per-experiment dumps.
  - `logs/` – Console logs for grocery and liquor runs.
  - `README.md` – Detailed reproduction + publishing instructions.
- `Mechatronic Integration in Smart Point-of-Sale Systems_ A Comparative Study of YOLO and Effici.pdf` – Updated paper section referencing the Milvus comparison.

## Getting Started

```bash
cd yolo_embeding_malvious_repo
python -m venv .venv && source .venv/bin/activate
pip install -r yolo_vs_embeding_malvious/requirements.txt  # if needed
```

Rebuild the Milvus databases (optional) with:
```bash
python yolo_vs_embeding_malvious/populate_milvus_embeddings.py \
  --dataset data/roboflow/grocery-rfn8l-v1/data.yaml \
  --milvus-db experiments/milvus_release/databases/milvus_grocery.db \
  --collection grocery_items --max-templates 10 --batch-size 32 --device cuda:0
```
Repeat for the liquor dataset to refresh `milvus_liquor.db`.

## Publishing Workflow

Inside this folder you can initialize a clean git repo and push only the experiment artifacts:
```bash
cd yolo_embeding_malvious_repo
rm -rf .git
git init
git add .
git commit -m "Initial YOLO + Milvus release"
git remote add origin https://github.com/sam02425/yolo_embeding_malvious.git
git push -u origin main
```

This keeps the Milvus study separate from the original `360Image` project while preserving all required scripts, data, metrics, and documentation.
