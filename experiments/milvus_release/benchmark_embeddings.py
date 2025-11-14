#!/usr/bin/env python3
"""
Benchmark YOLO + Milvus embedding configurations.

The script compares two Milvus collections by running YOLO detections on a
validation directory, extracting embeddings (placeholder random vectors unless
you plug in the actual DOLG extractor), and measuring latency and hit rate.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from pymilvus import MilvusClient
import yaml
from ultralytics import YOLO


def load_dataset_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def run_yolo_inference(model_path: str, image_dir: Path, max_images: int = 200) -> List[Dict]:
    model = YOLO(model_path)
    results = []
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    if max_images:
        images = images[:max_images]
    for img_path in images:
        yolo_res = model(img_path, verbose=False)[0]
        for box in yolo_res.boxes:
            results.append({
                "image": str(img_path),
                "cls": int(box.cls[0]),
            })
    return results


def generate_placeholder_embeddings(num: int, dim: int = 128) -> List[np.ndarray]:
    rng = np.random.default_rng(seed=42)
    return [rng.standard_normal(dim).astype(np.float32) for _ in range(num)]


def benchmark_collection(milvus_cfg: Dict, embeddings: List[np.ndarray]) -> Dict:
    client = MilvusClient(milvus_cfg["milvus_db"])
    latencies = []
    hits = 0
    for emb in embeddings:
        start = time.perf_counter()
        search_res = client.search(
            collection_name=milvus_cfg["collection"],
            data=[emb.tolist()],
            limit=milvus_cfg.get("top_k", 5),
            metric_type=milvus_cfg.get("metric_type", "COSINE")
        )
        latencies.append((time.perf_counter() - start) * 1000)
        if search_res and search_res[0]:
            hits += 1  # Placeholder hit logic
    client.close()
    return {
        "name": milvus_cfg["name"],
        "mean_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "queries": len(latencies),
        "hit_rate": hits / max(len(latencies), 1)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Milvus embedding variants.")
    parser.add_argument("--dataset", required=True, help="Dataset YAML path.")
    parser.add_argument("--yolo-weights", required=True, help="YOLO weights for detection.")
    parser.add_argument("--collection-a-config", required=True, help="JSON config for first Milvus collection.")
    parser.add_argument("--collection-b-config", required=True, help="JSON config for second Milvus collection.")
    parser.add_argument("--max-images", type=int, default=200, help="Max validation images to sample.")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Placeholder embedding dimension.")
    args = parser.parse_args()

    dataset_info = load_dataset_yaml(args.dataset)
    val_dir = Path(dataset_info["val"])

    detections = run_yolo_inference(args.yolo_weights, val_dir, max_images=args.max_images)
    embeddings = generate_placeholder_embeddings(len(detections), dim=args.embedding_dim)

    cfg_a = json.loads(Path(args.collection_a_config).read_text())
    cfg_b = json.loads(Path(args.collection_b_config).read_text())

    report = [
        benchmark_collection(cfg_a, embeddings),
        benchmark_collection(cfg_b, embeddings)
    ]

    print("\nBenchmark Summary")
    print("-----------------")
    for entry in report:
        print(f"{entry['name']}: queries={entry['queries']}, hit_rate={entry['hit_rate']:.2%}, "
              f"mean_latency={entry['mean_latency_ms']:.2f}ms, p95_latency={entry['p95_latency_ms']:.2f}ms")


if __name__ == "__main__":
    main()
