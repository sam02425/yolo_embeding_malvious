#!/usr/bin/env python3
"""
Train baseline YOLOv8/YOLOv11 detectors for the grocery dataset with extra
store-context augmentation images pulled from the empty shelf dataset.

The script:
1. Builds an augmented dataset that adds background-only images to the grocery set.
2. Generates a YOLO-ready YAML that points to the augmented assets.
3. Trains YOLOv8 and/or YOLOv11 models and stores the checkpoints under grocery/runs.
4. Optionally validates that the liquor dataset required for experiments is present.
5. Enforces CUDA-only (RTX 5080) half-precision training for speed/accuracy.
6. Applies YOLO early stopping after 5-6 non-improving epochs (configurable patience).
7. Automatically shrinks batch size if CUDA runs out of memory during training.
8. Forces MLflow logging with descriptive experiment/run names for every attempt.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

# Stabilize CUDA allocator to avoid fragmentation on long runs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# YOLO imports are intentionally deferred until training is requested to avoid
# importing torch/ultralytics in dry-run or dataset-only scenarios.


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ERROR_SIGNATURES = (
    "out of memory",
    "cublas_status_alloc_failed",
    "cuda error: an illegal memory access was encountered",
    "cuda error: unspecified launch failure",
    "cuda error: cublas_status_not_initialized",
)


def resolve_repo_path(path: Path) -> Path:
    """Resolve relative paths against the repository root."""
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def format_available_datasets() -> str:
    data_root = REPO_ROOT / "data"
    if not data_root.exists():
        return ""
    candidates = sorted(p.name for p in data_root.iterdir() if p.is_dir())
    if not candidates:
        return ""
    formatted = "\n".join(f"  - {name}" for name in candidates)
    return f"Available datasets under {data_root}:\n{formatted}"


def is_cuda_memory_error(error: BaseException) -> bool:
    message = str(error).lower()
    return any(signature in message for signature in MEMORY_ERROR_SIGNATURES)


def configure_mlflow_logging(
    tracking_uri: str | Path, experiment_name: str, run_label: str, enable: bool = True
) -> None:
    """Set Ultralytics + MLflow env vars to ensure descriptive tracking."""
    if enable:
        try:
            from ultralytics import settings as yolo_settings

            yolo_settings.update({"mlflow": True})
        except Exception:
            pass
    os.environ["MLFLOW_TRACKING_URI"] = str(tracking_uri)
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment_name
    os.environ["MLFLOW_RUN"] = run_label


def close_active_mlflow_run() -> None:
    """End any dangling MLflow run to avoid parameter conflicts when retrying."""
    try:
        import mlflow
    except Exception:
        return
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass


def build_run_name(prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def format_mlflow_run_label(base_name: str, batch_size: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_bs{batch_size}_{timestamp}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO grocery baselines with store-context augmentation."
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=Path("data/grocery.v3i.yolov11"),
        help="Path to the baseline grocery dataset folder (expects data.yaml + splits).",
    )
    parser.add_argument(
        "--background-dataset",
        type=Path,
        default=Path("data/empty-shelf-detection.v1i.yolov11"),
        help="Path to dataset that provides store-context background images.",
    )
    parser.add_argument(
        "--output-dataset",
        type=Path,
        default=Path("data/grocery_augmented"),
        help="Destination directory for the augmented dataset.",
    )
    parser.add_argument(
        "--background-ratio",
        type=float,
        default=0.35,
        help="Max ratio of background-only images to append relative to base split size.",
    )
    parser.add_argument(
        "--background-limit",
        type=int,
        default=None,
        help="Optional hard cap on the number of background images per split.",
    )
    parser.add_argument(
        "--refresh-dataset",
        action="store_true",
        help="Remove the output dataset folder before rebuilding it.",
    )
    parser.add_argument(
        "--train-yolov8",
        action="store_true",
        help="Train the YOLOv8 baseline after preparing the dataset.",
    )
    parser.add_argument(
        "--train-yolov11",
        action="store_true",
        help="Train the YOLOv11 baseline after preparing the dataset.",
    )
    parser.add_argument(
        "--yolov8-weights",
        type=str,
        default="yolov8m.pt",
        help="Path to YOLOv8 base weights or checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--yolov11-weights",
        type=str,
        default="yolo11m.pt",
        help="Path to YOLOv11 base weights or checkpoint to fine-tune.",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training.")
    parser.add_argument(
        "--batch", type=int, default=32, help="Batch size used for both trainings."
    )
    parser.add_argument(
        "--min-batch",
        type=int,
        default=4,
        help="Smallest batch size allowed when auto-scaling after CUDA OOM.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device string passed to Ultralytics (GPU-only, e.g., cuda:0).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path("grocery/runs"),
        help="Where to store YOLO training runs.",
    )
    parser.add_argument(
        "--liquor-dataset",
        type=Path,
        default=Path("data/Liquor-data.v4i.yolov11"),
        help="Liquor dataset root used for experiment sanity checks.",
    )
    parser.add_argument(
        "--check-liquor-data",
        action="store_true",
        help="Only verify the liquor dataset structure without training.",
    )
    parser.add_argument(
        "--dataset-only",
        action="store_true",
        help="Prepare the augmented dataset/YAML but skip all training jobs.",
    )
    parser.add_argument(
        "--background-splits",
        nargs="+",
        default=["train", "valid"],
        choices=["train", "valid", "test"],
        help="Dataset splits that should receive background-only augmentation.",
    )
    parser.add_argument(
        "--yaml-name",
        type=str,
        default="grocery_augmented.yaml",
        help="Name for the generated dataset YAML inside the output folder.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=6,
        help="Non-improving epochs before early stopping (use 5-6 for tighter cutoff).",
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=None,
        help="MLflow tracking URI or folder. Defaults to <project-dir>/mlflow for local logging.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="Retail Grocery Baselines",
        help="MLflow experiment name used for tracking runs.",
    )
    return parser.parse_args()


def find_images(directory: Path) -> List[Path]:
    """Return all image files underneath directory."""
    return sorted(
        [p for p in directory.glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def symlink_or_copy(src: Path, dst: Path) -> None:
    """Symlink src into dst; fall back to copying if symlinks are not allowed."""
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def summarize_split(images_dir: Path) -> int:
    if not images_dir.exists():
        return 0
    return len(find_images(images_dir))


def clone_split(
    src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path
) -> Tuple[int, int]:
    """Link/copy the base grocery dataset into the augmented folder."""
    ensure_dir(dst_images)
    ensure_dir(dst_labels)
    image_count = 0
    label_count = 0

    for img_path in find_images(src_images):
        dst_path = dst_images / img_path.name
        symlink_or_copy(img_path, dst_path)
        image_count += 1
        label_name = img_path.with_suffix(".txt").name
        src_label = src_labels / label_name
        dst_label = dst_labels / label_name
        if src_label.exists():
            symlink_or_copy(src_label, dst_label)
            label_count += 1
        else:
            dst_label.touch(exist_ok=True)
    return image_count, label_count


def add_backgrounds(
    background_images_dir: Path,
    dst_images: Path,
    dst_labels: Path,
    limit: int,
    split_name: str,
) -> int:
    """Append empty annotations for background-only store images."""
    ensure_dir(dst_images)
    ensure_dir(dst_labels)
    added = 0
    for idx, img in enumerate(find_images(background_images_dir)):
        if idx >= limit:
            break
        new_name = f"{split_name}_background_{idx:05d}{img.suffix.lower()}"
        dst_image = dst_images / new_name
        symlink_or_copy(img, dst_image)
        (dst_labels / f"{split_name}_background_{idx:05d}.txt").write_text("")
        added += 1
    return added


def prepare_augmented_dataset(
    base_dataset: Path,
    background_dataset: Path,
    output_dataset: Path,
    background_ratio: float,
    background_limit: int | None,
    refresh: bool,
    background_splits: Iterable[str],
) -> Dict:
    """Build augmented dataset structure and capture metadata summary."""
    if refresh and output_dataset.exists():
        shutil.rmtree(output_dataset)
    ensure_dir(output_dataset)

    summary: Dict[str, Dict[str, int]] = {}
    for split in ("train", "valid", "test"):
        base_image_dir = base_dataset / split / "images"
        base_label_dir = base_dataset / split / "labels"
        dst_image_dir = output_dataset / split / "images"
        dst_label_dir = output_dataset / split / "labels"

        base_images, base_labels = 0, 0
        if base_image_dir.exists():
            base_images, base_labels = clone_split(
                base_image_dir, base_label_dir, dst_image_dir, dst_label_dir
            )

        background_added = 0
        if split in background_splits:
            bg_images_dir = background_dataset / split / "images"
            if bg_images_dir.exists():
                target = int(math.ceil(base_images * background_ratio))
                if background_limit is not None:
                    target = min(target, background_limit)
                if target > 0:
                    background_added = add_backgrounds(
                        bg_images_dir, dst_image_dir, dst_label_dir, target, split
                    )

        summary[split] = {
            "base_images": base_images,
            "base_labels": base_labels,
            "background_images": background_added,
            "total_images": summarize_split(dst_image_dir),
        }
    metadata_path = output_dataset / "metadata.json"
    metadata_path.write_text(json.dumps(summary, indent=2))
    return summary


def generate_dataset_yaml(
    base_dataset: Path, output_dataset: Path, yaml_name: str
) -> Path:
    base_yaml = base_dataset / "data.yaml"
    if not base_yaml.exists():
        raise FileNotFoundError(f"Unable to locate base data.yaml at {base_yaml}")
    config = yaml.safe_load(base_yaml.read_text())
    config["train"] = str((output_dataset / "train" / "images").resolve())
    config["val"] = str((output_dataset / "valid" / "images").resolve())
    config["test"] = str((output_dataset / "test" / "images").resolve())
    augmented_yaml = output_dataset / yaml_name
    augmented_yaml.write_text(yaml.safe_dump(config, sort_keys=False))
    return augmented_yaml


def enforce_gpu_device(device: str) -> str:
    """Ensure CUDA is available and the requested device targets a GPU."""
    if "cuda" not in device.lower():
        raise ValueError(
            "GPU-only training enforced. Please provide a CUDA device via --device, e.g., cuda:0."
        )
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Torch must be installed to verify CUDA availability.") from exc

    if not torch.cuda.is_available():
        raise EnvironmentError(
            "CUDA device requested but torch.cuda.is_available() is False. "
            "Ensure NVIDIA drivers and CUDA are properly installed for the RTX 5080."
        )
    return device


def train_model(
    weights: str,
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    project: Path,
    run_name: str,
    patience: int,
    min_batch: int,
    mlflow_tracking_uri: str | Path,
    mlflow_experiment: str,
) -> Path:
    from ultralytics import YOLO
    import torch

    ensure_dir(project)
    current_batch = batch

    while current_batch >= min_batch:
        model = None
        try:
            close_active_mlflow_run()
            configure_mlflow_logging(
                tracking_uri=mlflow_tracking_uri,
                experiment_name=mlflow_experiment,
                run_label=format_mlflow_run_label(run_name, current_batch),
            )
            model = YOLO(weights)
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=current_batch,
                device=device,
                project=str(project),
                name=run_name,
                save=True,
                plots=True,
                patience=patience,
                half=True,
            )
            save_dir = Path(results.save_dir)
            best_weights = save_dir / "weights" / "best.pt"
            if not best_weights.exists():
                fallback = save_dir / "weights" / "last.pt"
                if fallback.exists():
                    return fallback
                raise FileNotFoundError(f"Cannot find weights for run {save_dir}")
            return best_weights
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            if not is_cuda_memory_error(exc):
                raise
            next_batch = max(min_batch, current_batch // 2)
            if next_batch == current_batch:
                raise
            close_active_mlflow_run()
            print(
                f"\n⚠️  CUDA OOM with batch size {current_batch}. "
                f"Reducing to {next_batch} and retrying..."
            )
            torch.cuda.empty_cache()
            gc.collect()
            current_batch = next_batch
        finally:
            if model is not None:
                del model

    raise RuntimeError(
        f"Unable to train {run_name}: CUDA OOM persists even at minimum batch size {min_batch}."
    )


def verify_liquor_dataset(liquor_dataset: Path) -> Dict[str, int]:
    splits = {}
    for split in ("train", "valid", "test"):
        splits[split] = summarize_split(liquor_dataset / split / "images")
    return splits


def main() -> None:
    args = parse_args()

    # Resolve relative paths against the repo root so the script can run from any cwd.
    args.base_dataset = resolve_repo_path(args.base_dataset)
    args.background_dataset = resolve_repo_path(args.background_dataset)
    args.output_dataset = resolve_repo_path(args.output_dataset)
    args.project_dir = resolve_repo_path(args.project_dir)
    args.liquor_dataset = resolve_repo_path(args.liquor_dataset)
    if args.mlflow_uri:
        if "://" in args.mlflow_uri:
            args.mlflow_uri = args.mlflow_uri
        else:
            args.mlflow_uri = str(resolve_repo_path(Path(args.mlflow_uri)))
    else:
        args.mlflow_uri = str(args.project_dir / "mlflow")
    base_yaml_path = args.base_dataset / "data.yaml"

    if not args.base_dataset.exists():
        hint = format_available_datasets()
        hint_msg = f"\n{hint}" if hint else ""
        raise FileNotFoundError(
            f"Base dataset directory not found at {args.base_dataset}.\n"
            f"Use --base-dataset to point to your grocery dataset root.{hint_msg}"
        )
    if not base_yaml_path.exists():
        hint = format_available_datasets()
        hint_msg = f"\n{hint}" if hint else ""
        raise FileNotFoundError(
            f"Unable to locate base data.yaml at {base_yaml_path}.\n"
            f"Verify the grocery dataset is downloaded and contains train/valid/test splits.{hint_msg}"
        )
    if not args.background_dataset.exists():
        hint = format_available_datasets()
        hint_msg = f"\n{hint}" if hint else ""
        raise FileNotFoundError(
            f"Background dataset not found at {args.background_dataset}.\n"
            f"Use --background-dataset to point to the empty shelf dataset.{hint_msg}"
        )

    if args.check_liquor_data and not args.train_yolov8 and not args.train_yolov11:
        liquor_summary = verify_liquor_dataset(args.liquor_dataset)
        print("Liquor dataset summary:")
        for split, count in liquor_summary.items():
            print(f"  {split}: {count} images ready for experiments")
        return

    summary = prepare_augmented_dataset(
        base_dataset=args.base_dataset,
        background_dataset=args.background_dataset,
        output_dataset=args.output_dataset,
        background_ratio=args.background_ratio,
        background_limit=args.background_limit,
        refresh=args.refresh_dataset,
        background_splits=args.background_splits,
    )

    print("Augmented grocery dataset prepared:")
    for split, stats in summary.items():
        print(
            f"  {split}: base={stats['base_images']}, "
            f"background={stats['background_images']}, total={stats['total_images']}"
        )

    dataset_yaml = generate_dataset_yaml(
        base_dataset=args.base_dataset,
        output_dataset=args.output_dataset,
        yaml_name=args.yaml_name,
    )
    print(f"Dataset YAML: {dataset_yaml}")

    training_requested = args.train_yolov8 or args.train_yolov11
    if args.dataset_only or not training_requested:
        print("Skipping training (dataset-only request or no training flags set).")
        return

    gpu_device = enforce_gpu_device(args.device)

    training_results = {}
    if args.train_yolov8:
        print("\n=== Training YOLOv8 baseline for grocery ===")
        yolov8_run_name = build_run_name("yolov8_grocery_baseline")
        weights = train_model(
            weights=args.yolov8_weights,
            data_yaml=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=gpu_device,
            project=args.project_dir,
            run_name=yolov8_run_name,
            patience=args.patience,
            min_batch=args.min_batch,
            mlflow_tracking_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_experiment,
        )
        training_results["yolov8"] = str(weights)
        print(f"YOLOv8 best weights saved to: {weights}")

    if args.train_yolov11:
        print("\n=== Training YOLOv11 baseline for grocery ===")
        yolo11_run_name = build_run_name("yolo11_grocery_baseline")
        weights = train_model(
            weights=args.yolov11_weights,
            data_yaml=dataset_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=gpu_device,
            project=args.project_dir,
            run_name=yolo11_run_name,
            patience=args.patience,
            min_batch=args.min_batch,
            mlflow_tracking_uri=args.mlflow_uri,
            mlflow_experiment=args.mlflow_experiment,
        )
        training_results["yolov11"] = str(weights)
        print(f"YOLOv11 best weights saved to: {weights}")

    (args.output_dataset / "training_results.json").write_text(
        json.dumps(training_results, indent=2)
    )


if __name__ == "__main__":
    main()
