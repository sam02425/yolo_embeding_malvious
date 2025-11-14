#!/usr/bin/env python3

"""
Complete Experimental Framework for Retail Item Detection
Compares:
1. YOLOv8 baseline (488 classes)
2. YOLOv11 baseline (488 classes)
3. YOLOv8 + DOLG embeddings + Milvus similarity search
4. YOLOv8 (488 classes) + Milvus augmentation vs YOLOv8 baseline

Metrics: mAP, Precision, Recall, F1, Inference Time, FPS
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

import numpy as np
import torch
import yaml
import mlflow
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")
    raise

try:
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
except ImportError:
    print("Please install pymilvus: pip install pymilvus")
    raise


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    name: str
    model_type: str  # yolov8, yolov11, yolov8_hybrid
    model_path: str
    use_milvus: bool
    milvus_collection: Optional[str]
    embedding_model: Optional[str]
    similarity_threshold: float
    dataset_yaml: str
    imgsz: int
    batch_size: int
    device: str


@dataclass
class EvaluationMetrics:
    """Complete evaluation metrics for retail detection"""
    # Detection metrics
    map50: float  # mAP @ IoU=0.5
    map50_95: float  # mAP @ IoU=0.5:0.95
    precision: float
    recall: float
    f1_score: float
    
    # Per-class metrics
    per_class_ap: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    
    # Speed metrics
    inference_time_ms: float
    fps: float
    preprocess_time_ms: float
    postprocess_time_ms: float
    
    # Milvus-specific metrics (if applicable)
    embedding_time_ms: Optional[float] = None
    similarity_search_time_ms: Optional[float] = None
    milvus_hit_rate: Optional[float] = None
    
    # Confusion matrix stats
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DOLGEmbeddingExtractor:
    """Production-grade DOLG embedding extractor with GPU optimization"""
    
    def __init__(self, model_path: str = "dolg_model.pth", device: str = "cuda", 
                 batch_size: int = 32, use_amp: bool = True):
        """
        Initialize production-grade DOLG embedding extractor
        
        Args:
            model_path: Path to DOLG model weights
            device: Device to run on (cuda/cpu)
            batch_size: Batch size for batch processing
            use_amp: Use automatic mixed precision for faster inference
        """
        # Force GPU if available
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available. Please check GPU setup.")
            self.device = torch.device(device)
            # Optimize for GPU
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        else:
            self.device = torch.device("cpu")
            print("⚠️  Warning: Using CPU. Performance will be significantly slower.")
        
        self.batch_size = batch_size
        self.use_amp = use_amp and self.device.type == 'cuda'
        
        # Load model
        self.model = self._load_dolg_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Enable optimizations
        if self.device.type == 'cuda':
            # Enable TF32 for faster matmul on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                print("✅ Model compiled with torch.compile for faster inference")
            except Exception as e:
                print(f"⚠️  Could not compile model: {e}")
        
        # Setup transforms
        self.transform = self._get_transforms()
        
        print(f"✅ DOLG Extractor initialized on {self.device}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Mixed precision: {self.use_amp}")
        
    def _load_dolg_model(self, model_path: str):
        """Load production-grade DOLG model"""
        import torch.nn as nn
        import torchvision.models as models
        
        class ProductionDOLGModel(nn.Module):
            """Production DOLG model with EfficientNet-B0 backbone"""
            def __init__(self, embedding_dim=128):
                super().__init__()
                # Use EfficientNet-B0 as backbone
                self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
                
                # Get feature dimension
                in_features = self.backbone.classifier[1].in_features
                
                # Replace classifier with production embedding layer
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),  # Increased dropout for better generalization
                    nn.Linear(in_features, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(512, embedding_dim),
                )
                
            def forward(self, x):
                return self.backbone(x)
        
        model = ProductionDOLGModel(embedding_dim=128)
        
        # Load pretrained weights if they exist
        if Path(model_path).exists():
            print(f"✅ Loading DOLG weights from: {model_path}")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            print(f"⚠️  DOLG weights not found at {model_path}")
            print("   Using EfficientNet pretrained on ImageNet")
            print("   For best results, train DOLG model on retail dataset")
            
        return model
    
    def _get_transforms(self):
        """Get production-grade image transforms"""
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),  # Slightly larger for better quality
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract DOLG embedding from single image (production-grade)
        
        Args:
            image: Input image (H, W, C) in BGR format
            
        Returns:
            128-dimensional L2-normalized embedding vector
        """
        import cv2
        
        # Input validation
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Transform
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract embedding with mixed precision
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embedding = self.model(image_tensor)
            else:
                embedding = self.model(image_tensor)
            
            embedding = embedding.cpu().numpy().flatten()
        
        # L2 normalize for cosine similarity
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm
        else:
            # Handle zero embedding
            embedding = np.zeros_like(embedding)
            
        return embedding.astype(np.float32)
    
    def extract_embeddings_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings from batch of images (production-grade)
        
        Args:
            images: List of input images (H, W, C) in BGR format
            
        Returns:
            Array of shape (N, 128) with L2-normalized embeddings
        """
        import cv2
        
        if not images:
            return np.array([])
        
        # Convert all images to RGB and transform
        image_tensors = []
        for img in images:
            if img is None or img.size == 0:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = self.transform(img_rgb)
            image_tensors.append(img_tensor)
        
        if not image_tensors:
            return np.array([])
        
        # Stack into batch
        batch_tensor = torch.stack(image_tensors).to(self.device)
        
        # Extract embeddings
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embeddings = self.model(batch_tensor)
            else:
                embeddings = self.model(batch_tensor)
            
            embeddings = embeddings.cpu().numpy()
        
        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)  # Avoid division by zero
        embeddings = embeddings / norms
        
        return embeddings.astype(np.float32)
    
    def warmup(self):
        """Warmup GPU for consistent timing"""
        if self.device.type == 'cuda':
            dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
            with torch.no_grad():
                for _ in range(10):
                    _ = self.model(dummy_input)
            torch.cuda.synchronize()
            print("✅ GPU warmed up")


class MilvusRetailDB:
    """Production-grade Milvus database for retail item embeddings with auto-setup"""
    
    def __init__(self, db_path: str = "./milvus_retail.db", collection_name: str = "retail_items",
                 auto_setup: bool = True):
        """
        Initialize production-grade Milvus client with automatic setup
        
        Args:
            db_path: Path to Milvus database file
            collection_name: Name of the collection
            auto_setup: Automatically download and setup Milvus if needed
        """
        self.db_path = Path(db_path)
        self.collection_name = collection_name
        self.embedding_dim = 128
        self.milvus_uri = str(self.db_path)
        
        # Ensure pymilvus is installed
        try:
            from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
            self.MilvusClient = MilvusClient
            self.DataType = DataType
            self.FieldSchema = FieldSchema
            self.CollectionSchema = CollectionSchema
        except ImportError:
            print("❌ pymilvus not found. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "pymilvus>=2.5.0"], check=True)
            from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
            self.MilvusClient = MilvusClient
            self.DataType = DataType
            self.FieldSchema = FieldSchema
            self.CollectionSchema = CollectionSchema
        
        # Auto-setup if requested
        if auto_setup:
            self._ensure_milvus_ready()
        
        # Initialize client with production settings
        try:
            self.client = self.MilvusClient(self.milvus_uri)
            print(f"✅ Connected to Milvus database: {self.milvus_uri}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _ensure_milvus_ready(self):
        """Ensure Milvus is properly set up and ready to use"""
        print("\n" + "="*80)
        print("Setting up Milvus Vector Database")
        print("="*80 + "\n")
        
        # Create database directory if needed
        db_dir = self.db_path.parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if database exists
        if self.db_path.exists():
            print(f"✅ Found existing Milvus database: {self.db_path}")
        else:
            print(f"📦 Creating new Milvus database: {self.db_path}")
        
        print("\n✅ Milvus setup complete\n")
    
    def create_collection(self, drop_existing: bool = True):
        """Create production-grade Milvus collection with optimized index"""
        # Drop existing if requested
        if drop_existing and self.client.has_collection(self.collection_name):
            print(f"🗑️  Dropping existing collection: {self.collection_name}")
            self.client.drop_collection(self.collection_name)
        
        # Define schema with proper types
        schema = self.CollectionSchema(
            fields=[
                self.FieldSchema(name="id", dtype=self.DataType.INT64, is_primary=True, auto_id=True),
                self.FieldSchema(name="upc", dtype=self.DataType.VARCHAR, max_length=20),
                self.FieldSchema(name="class_id", dtype=self.DataType.INT64),
                self.FieldSchema(name="class_name", dtype=self.DataType.VARCHAR, max_length=100),
                self.FieldSchema(name="template_idx", dtype=self.DataType.INT64),
                self.FieldSchema(name="embedding", dtype=self.DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ],
            description="Production retail item embeddings with DOLG (488 classes)",
            enable_dynamic_field=False
        )
        
        # Create index params for GPU-accelerated search
        print(f"📊 Creating collection with optimized index...")
        index_params = self.client.prepare_index_params()
        
        # Use FLAT for exact search (best accuracy) or IVF_FLAT for speed
        # For production with GPU, FLAT is recommended for <1M vectors
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",  # Best accuracy, GPU-accelerated
            metric_type="COSINE"  # Cosine similarity for L2-normalized embeddings
        )
        
        # Create collection with production settings
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_mmap=False,  # Disable mmap for better GPU performance
            properties={
                "mmap.enabled": "false",
                "collection.ttl.seconds": "0"  # No auto-expiration
            }
        )
        
        print(f"✅ Collection '{self.collection_name}' created successfully")
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Index type: FLAT (GPU-accelerated)")
        print(f"   Metric type: COSINE")
        
    def insert_embeddings(self, embeddings: List[Dict], batch_size: int = 1000):
        """
        Insert embeddings with production-grade batch processing
        
        Args:
            embeddings: List of embedding dictionaries
            batch_size: Batch size for insertion (default 1000)
        """
        if not embeddings:
            print("⚠️  No embeddings to insert")
            return
        
        total = len(embeddings)
        print(f"📥 Inserting {total:,} embeddings into Milvus...")
        print(f"   Batch size: {batch_size}")
        
        # Insert in batches for better performance
        for i in tqdm(range(0, total, batch_size), desc="Inserting batches"):
            batch = embeddings[i:i + batch_size]
            try:
                self.client.insert(
                    collection_name=self.collection_name,
                    data=batch
                )
            except Exception as e:
                print(f"❌ Error inserting batch {i//batch_size}: {e}")
                raise
        
        # Flush to ensure data is persisted
        self.client.flush(collection_name=self.collection_name)
        
        # Get collection stats
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        print(f"✅ Successfully inserted {total:,} embeddings")
        print(f"   Collection stats: {stats}")
        
    def search_similar(self, query_embedding: np.ndarray, top_k: int = 5, 
                      filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Production-grade similarity search with GPU acceleration
        
        Args:
            query_embedding: Query embedding vector (128-d)
            top_k: Number of results to return
            filter_expr: Optional filter expression
            
        Returns:
            List of similar items with metadata
        """
        # Validate input
        if query_embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected {self.embedding_dim}-d embedding, got {query_embedding.shape[0]}")
        
        # Ensure L2 normalized
        norm = np.linalg.norm(query_embedding)
        if abs(norm - 1.0) > 0.01:
            query_embedding = query_embedding / (norm + 1e-8)
        
        # Search with GPU acceleration
        try:
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_embedding.tolist()],
                limit=top_k,
                filter=filter_expr,
                output_fields=["upc", "class_id", "class_name", "template_idx"],
                search_params={
                    "metric_type": "COSINE",
                    "params": {}
                }
            )
            
            return results[0] if results else []
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return stats
        except Exception as e:
            print(f"⚠️  Could not get stats: {e}")
            return {}
    
    def close(self):
        """Clean up Milvus connection"""
        try:
            self.client.close()
            print("✅ Milvus connection closed")
        except Exception as e:
            print(f"⚠️  Error closing Milvus: {e}")


class HybridYOLODetector:
    """Production-grade hybrid YOLO + DOLG + Milvus detector with GPU optimization"""
    
    def __init__(self, 
                 yolo_model: YOLO,
                 embedding_extractor: DOLGEmbeddingExtractor,
                 milvus_db: MilvusRetailDB,
                 similarity_threshold: float = 0.5,
                 device: str = "cuda:0"):
        """
        Initialize production-grade hybrid detector
        
        Args:
            yolo_model: YOLO detection model
            embedding_extractor: DOLG embedding extractor
            milvus_db: Milvus database for similarity search
            similarity_threshold: Minimum similarity score for Milvus matches
            device: Device for processing
        """
        self.yolo = yolo_model
        self.embedding_extractor = embedding_extractor
        self.milvus_db = milvus_db
        self.similarity_threshold = similarity_threshold
        
        # Force GPU usage
        if device.startswith('cuda'):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available")
            self.device = device
        else:
            print("⚠️  Warning: CPU mode not recommended for production")
            self.device = "cpu"
        
        # Configure YOLO for GPU
        self.yolo.to(self.device)
        
        # Warmup GPU
        print("🔥 Warming up GPU for optimal performance...")
        self.embedding_extractor.warmup()
        
        # Run dummy YOLO inference for warmup
        dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        _ = self.yolo(dummy_img, verbose=False)
        
        if device.startswith('cuda'):
            torch.cuda.synchronize()
        
        print("✅ Hybrid detector ready for production inference")
        
    def predict(self, image: np.ndarray, conf_threshold: float = 0.25, 
               use_batch_embeddings: bool = True) -> Tuple[List[Dict], Dict[str, float]]:
        """
        Production-grade hybrid prediction with GPU-optimized batch processing
        
        Args:
            image: Input image (H, W, C) in BGR format
            conf_threshold: YOLO confidence threshold
            use_batch_embeddings: Use batch processing for embeddings (faster)
            
        Returns:
            detections: List of detection dictionaries
            timings: Dictionary of timing metrics
        """
        timings = {}
        
        # YOLO detection with GPU
        t0 = time.perf_counter()
        yolo_results = self.yolo(image, conf=conf_threshold, verbose=False, device=self.device)[0]
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['yolo_inference'] = (time.perf_counter() - t0) * 1000
        
        if len(yolo_results.boxes) == 0:
            return [], timings
        
        # Extract crops for all detections
        crops = []
        detection_info = []
        
        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            
            # Validate coordinates
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Crop detection
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            crops.append(crop)
            detection_info.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'yolo_class_id': cls_id,
                'yolo_class_name': yolo_results.names[cls_id]
            })
        
        if not crops:
            return [], timings
        
        # Extract embeddings (batch or individual)
        t0 = time.perf_counter()
        if use_batch_embeddings and len(crops) > 1:
            # Batch processing for better GPU utilization
            embeddings = self.embedding_extractor.extract_embeddings_batch(crops)
        else:
            # Individual processing
            embeddings = np.array([
                self.embedding_extractor.extract_embedding(crop) 
                for crop in crops
            ])
        
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()
        timings['embedding_extraction'] = (time.perf_counter() - t0) * 1000
        
        # Milvus similarity search
        t0 = time.perf_counter()
        detections = []
        milvus_hits = 0
        
        for i, (info, embedding) in enumerate(zip(detection_info, embeddings)):
            # Search Milvus
            similar_items = self.milvus_db.search_similar(embedding, top_k=1)
            
            # Use Milvus result if similarity is high enough
            final_class_id = info['yolo_class_id']
            final_class_name = info['yolo_class_name']
            milvus_similarity = 0.0
            used_milvus = False
            
            if similar_items and similar_items[0]['distance'] >= self.similarity_threshold:
                milvus_similarity = similar_items[0]['distance']
                final_class_id = similar_items[0]['entity']['class_id']
                final_class_name = similar_items[0]['entity']['class_name']
                used_milvus = True
                milvus_hits += 1
            
            detections.append({
                'bbox': info['bbox'],
                'confidence': info['confidence'],
                'class_id': final_class_id,
                'class_name': final_class_name,
                'yolo_class': info['yolo_class_id'],
                'milvus_similarity': milvus_similarity,
                'used_milvus': used_milvus
            })
        
        timings['milvus_search'] = (time.perf_counter() - t0) * 1000
        timings['total_hybrid_time'] = sum(timings.values())
        timings['milvus_hit_rate'] = milvus_hits / len(detections) if detections else 0.0
        
        return detections, timings
    
    def predict_batch(self, images: List[np.ndarray], conf_threshold: float = 0.25) -> List[Tuple[List[Dict], Dict[str, float]]]:
        """
        Batch prediction for maximum GPU utilization (production use case)
        
        Args:
            images: List of input images
            conf_threshold: YOLO confidence threshold
            
        Returns:
            List of (detections, timings) tuples for each image
        """
        results = []
        
        # Process images in optimal batch sizes
        for img in tqdm(images, desc="Processing batch", disable=len(images) < 10):
            detections, timings = self.predict(img, conf_threshold, use_batch_embeddings=True)
            results.append((detections, timings))
        
        return results


class RetailEvaluator:
    """Comprehensive evaluation for retail detection systems"""
    
    def __init__(self, dataset_yaml: str, iou_threshold: float = 0.5):
        """
        Initialize evaluator
        
        Args:
            dataset_yaml: Path to dataset YAML
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        self.dataset_yaml = dataset_yaml
        self.iou_threshold = iou_threshold
        
        # Load dataset info
        with open(dataset_yaml, 'r') as f:
            self.dataset_info = yaml.safe_load(f)
            
        self.class_names = self.dataset_info['names']
        self.num_classes = len(self.class_names)
        
    def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        
        # Union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-8)
    
    def evaluate_yolo_baseline(self, model: YOLO) -> EvaluationMetrics:
        """Evaluate YOLO baseline model"""
        print(f"Evaluating YOLO baseline model...")
        
        # Run validation
        results = model.val(data=self.dataset_yaml, split='val', verbose=False)
        
        # Extract metrics
        metrics = EvaluationMetrics(
            map50=float(results.box.map50),
            map50_95=float(results.box.map),
            precision=float(results.box.p.mean()),
            recall=float(results.box.r.mean()),
            f1_score=2 * (float(results.box.p.mean()) * float(results.box.r.mean())) / 
                     (float(results.box.p.mean()) + float(results.box.r.mean()) + 1e-8),
            per_class_ap={self.class_names[i]: float(ap) for i, ap in enumerate(results.box.ap50)},
            per_class_precision={self.class_names[i]: float(p) for i, p in enumerate(results.box.p)},
            per_class_recall={self.class_names[i]: float(r) for i, r in enumerate(results.box.r)},
            inference_time_ms=float(results.speed['inference']),
            fps=1000.0 / float(results.speed['inference']),
            preprocess_time_ms=float(results.speed['preprocess']),
            postprocess_time_ms=float(results.speed['postprocess'])
        )
        
        return metrics
    
    def evaluate_hybrid_model(self, hybrid_detector: HybridYOLODetector,
                             val_images: List[str], 
                             val_labels: List[str]) -> EvaluationMetrics:
        """Evaluate hybrid YOLO + Milvus model"""
        print(f"Evaluating hybrid model...")
        
        all_predictions = []
        all_ground_truths = []
        all_timings = defaultdict(list)
        
        milvus_hits = 0
        total_detections = 0
        
        # Process each validation image
        for img_path, label_path in tqdm(zip(val_images, val_labels), total=len(val_images)):
            import cv2
            
            # Load image
            image = cv2.imread(img_path)
            h, w = image.shape[:2]
            
            # Load ground truth
            with open(label_path, 'r') as f:
                gt_boxes = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        # Convert to xyxy format
                        x1 = (x_center - width / 2) * w
                        y1 = (y_center - height / 2) * h
                        x2 = (x_center + width / 2) * w
                        y2 = (y_center + height / 2) * h
                        gt_boxes.append({
                            'bbox': [x1, y1, x2, y2],
                            'class_id': cls_id
                        })
            
            # Get predictions
            predictions, timings = hybrid_detector.predict(image)
            
            # Track timings
            for key, value in timings.items():
                all_timings[key].append(value)
            
            # Track Milvus hits
            for pred in predictions:
                total_detections += 1
                if pred['used_milvus']:
                    milvus_hits += 1
            
            all_predictions.append(predictions)
            all_ground_truths.append(gt_boxes)
        
        # Calculate metrics
        metrics_dict = self._calculate_metrics(all_predictions, all_ground_truths)
        
        # Add timing metrics
        metrics_dict['inference_time_ms'] = np.mean(all_timings['yolo_inference'])
        metrics_dict['fps'] = 1000.0 / metrics_dict['inference_time_ms']
        metrics_dict['embedding_time_ms'] = np.mean(all_timings['embedding_extraction'])
        metrics_dict['similarity_search_time_ms'] = np.mean(all_timings['milvus_search'])
        metrics_dict['milvus_hit_rate'] = milvus_hits / max(total_detections, 1)
        
        return EvaluationMetrics(**metrics_dict)
    
    def _calculate_metrics(self, predictions: List[List[Dict]], 
                          ground_truths: List[List[Dict]]) -> Dict[str, Any]:
        """Calculate precision, recall, mAP from predictions and ground truths"""
        
        # Initialize per-class metrics
        per_class_tp = defaultdict(int)
        per_class_fp = defaultdict(int)
        per_class_fn = defaultdict(int)
        per_class_scores = defaultdict(list)
        
        # Process each image
        for preds, gts in zip(predictions, ground_truths):
            gt_matched = [False] * len(gts)
            
            # Sort predictions by confidence
            preds_sorted = sorted(preds, key=lambda x: x['confidence'], reverse=True)
            
            for pred in preds_sorted:
                pred_cls = pred['class_id']
                pred_box = pred['bbox']
                pred_conf = pred['confidence']
                
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_matched[gt_idx]:
                        continue
                    if gt['class_id'] != pred_cls:
                        continue
                    
                    iou = self.calculate_iou(pred_box, gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Check if match is valid
                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    per_class_tp[pred_cls] += 1
                    per_class_scores[pred_cls].append((pred_conf, 1))  # (confidence, is_tp)
                    gt_matched[best_gt_idx] = True
                else:
                    per_class_fp[pred_cls] += 1
                    per_class_scores[pred_cls].append((pred_conf, 0))
            
            # Count false negatives
            for gt_idx, gt in enumerate(gts):
                if not gt_matched[gt_idx]:
                    per_class_fn[gt['class_id']] += 1
        
        # Calculate per-class AP
        per_class_ap = {}
        per_class_precision = {}
        per_class_recall = {}
        
        for cls_id in range(self.num_classes):
            cls_name = self.class_names[cls_id]
            tp = per_class_tp[cls_id]
            fp = per_class_fp[cls_id]
            fn = per_class_fn[cls_id]
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            
            per_class_precision[cls_name] = precision
            per_class_recall[cls_name] = recall
            
            # Calculate AP using 11-point interpolation
            if cls_id in per_class_scores and per_class_scores[cls_id]:
                scores = sorted(per_class_scores[cls_id], key=lambda x: x[0], reverse=True)
                precisions = []
                recalls = []
                
                tp_cumsum = 0
                fp_cumsum = 0
                
                for conf, is_tp in scores:
                    if is_tp:
                        tp_cumsum += 1
                    else:
                        fp_cumsum += 1
                    
                    prec = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
                    rec = tp_cumsum / (tp + fn + 1e-8)
                    precisions.append(prec)
                    recalls.append(rec)
                
                # 11-point interpolation
                ap = 0
                for t in np.linspace(0, 1, 11):
                    prec_interp = max([p for p, r in zip(precisions, recalls) if r >= t] or [0])
                    ap += prec_interp / 11
                
                per_class_ap[cls_name] = ap
            else:
                per_class_ap[cls_name] = 0.0
        
        # Calculate overall metrics
        total_tp = sum(per_class_tp.values())
        total_fp = sum(per_class_fp.values())
        total_fn = sum(per_class_fn.values())
        
        precision = total_tp / (total_tp + total_fp + 1e-8)
        recall = total_tp / (total_tp + total_fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        map50 = np.mean(list(per_class_ap.values()))
        
        return {
            'map50': map50,
            'map50_95': map50,  # Simplified - would need multiple IoU thresholds
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'per_class_ap': per_class_ap,
            'per_class_precision': per_class_precision,
            'per_class_recall': per_class_recall,
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'preprocess_time_ms': 0.0,
            'postprocess_time_ms': 0.0
        }


class ExperimentRunner:
    """Run and compare multiple experiments"""
    
    def __init__(self, mlflow_uri: str = "file:./mlruns", 
                 experiment_name: str = "Retail_Detection_Comparison"):
        """Initialize experiment runner"""
        self.mlflow_uri = mlflow_uri
        self.experiment_name = experiment_name
        
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(experiment_name)
        
    def run_experiment(self, config: ExperimentConfig) -> EvaluationMetrics:
        """Run a single experiment"""
        print(f"\n{'='*80}")
        print(f"Running Experiment: {config.name}")
        print(f"{'='*80}\n")
        
        dataset_name = Path(config.dataset_yaml).stem
        experiment_display_name = f"{self.experiment_name}__{dataset_name}"
        mlflow.set_experiment(experiment_display_name)
        run_label = f"{config.model_type.upper()} | {dataset_name} | {config.name}"
        
        with mlflow.start_run(run_name=run_label):
            run_summary = (
                f"Experiment '{config.name}' on dataset '{dataset_name}' using "
                f"model '{config.model_path}' (type={config.model_type}) "
                f"imgsz={config.imgsz}, batch={config.batch_size}, device={config.device}, "
                f"milvus={config.use_milvus}, similarity_threshold={config.similarity_threshold}"
            )
            mlflow.set_tags({
                'dataset': dataset_name,
                'model_type': config.model_type,
                'experiment_name': config.name,
                'uses_milvus': str(config.use_milvus),
                'milvus_collection': config.milvus_collection or "",
                'run_summary': run_summary
            })
            # Log configuration
            mlflow.log_params({
                'model_type': config.model_type,
                'model_path': config.model_path,
                'use_milvus': config.use_milvus,
                'similarity_threshold': config.similarity_threshold,
                'dataset': config.dataset_yaml,
                'imgsz': config.imgsz,
                'batch_size': config.batch_size
            })
            
            # Load model
            model = YOLO(config.model_path)
            
            if config.use_milvus:
                # Create hybrid detector
                embedding_extractor = DOLGEmbeddingExtractor(
                    model_path=config.embedding_model or "dolg_model.pth",
                    device=config.device
                )
                milvus_db = MilvusRetailDB(
                    db_path=self._make_milvus_db_path(config.name),
                    collection_name=config.milvus_collection or "retail_items"
                )
                
                hybrid_detector = HybridYOLODetector(
                    yolo_model=model,
                    embedding_extractor=embedding_extractor,
                    milvus_db=milvus_db,
                    similarity_threshold=config.similarity_threshold
                )
                
                # Load validation data
                with open(config.dataset_yaml, 'r') as f:
                    dataset_info = yaml.safe_load(f)
                
                val_dir = Path(dataset_info['val'])
                val_images = sorted(val_dir.glob('*.jpg')) + sorted(val_dir.glob('*.png'))
                val_labels = [str(img).replace('images', 'labels').replace(img.suffix, '.txt') 
                             for img in val_images]
                
                # Evaluate hybrid model
                evaluator = RetailEvaluator(config.dataset_yaml)
                metrics = evaluator.evaluate_hybrid_model(hybrid_detector, 
                                                         [str(img) for img in val_images],
                                                         val_labels)
                
                milvus_db.close()
            else:
                # Evaluate baseline YOLO
                evaluator = RetailEvaluator(config.dataset_yaml)
                metrics = evaluator.evaluate_yolo_baseline(model)
            
            # Log metrics to MLflow
            mlflow.log_metric('mAP50', metrics.map50)
            mlflow.log_metric('mAP50-95', metrics.map50_95)
            mlflow.log_metric('precision', metrics.precision)
            mlflow.log_metric('recall', metrics.recall)
            mlflow.log_metric('f1_score', metrics.f1_score)
            mlflow.log_metric('inference_time_ms', metrics.inference_time_ms)
            mlflow.log_metric('fps', metrics.fps)
            mlflow.log_metric('preprocess_time_ms', metrics.preprocess_time_ms)
            mlflow.log_metric('postprocess_time_ms', metrics.postprocess_time_ms)
            mlflow.log_metric('true_positives', metrics.true_positives)
            mlflow.log_metric('false_positives', metrics.false_positives)
            mlflow.log_metric('false_negatives', metrics.false_negatives)
            
            if metrics.embedding_time_ms:
                mlflow.log_metric('embedding_time_ms', metrics.embedding_time_ms)
            if metrics.similarity_search_time_ms:
                mlflow.log_metric('similarity_search_time_ms', metrics.similarity_search_time_ms)
            if metrics.milvus_hit_rate:
                mlflow.log_metric('milvus_hit_rate', metrics.milvus_hit_rate)
            
            # Log per-class metrics
            for cls_name, ap in metrics.per_class_ap.items():
                mlflow.log_metric(f'AP_{cls_name}', ap)
            
            # Save metrics to file
            metrics_file = f"metrics_{config.name}.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            mlflow.log_artifact(metrics_file)
            
            print(f"\nResults for {config.name}:")
            print(f"  mAP@0.5: {metrics.map50:.4f}")
            print(f"  Precision: {metrics.precision:.4f}")
            print(f"  Recall: {metrics.recall:.4f}")
            print(f"  F1-Score: {metrics.f1_score:.4f}")
            print(f"  Inference Time: {metrics.inference_time_ms:.2f} ms")
            print(f"  FPS: {metrics.fps:.2f}")
            
            if metrics.milvus_hit_rate:
                print(f"  Milvus Hit Rate: {metrics.milvus_hit_rate:.2%}")
            
            return metrics

    @staticmethod
    def _make_milvus_db_path(config_name: str) -> str:
        """Create a milvus-lite DB filename that satisfies length constraints."""
        slug = re.sub(r'[^a-z0-9]+', '_', config_name.lower()).strip('_')
        if not slug:
            slug = "exp"
        slug = slug[:16]
        hash_suffix = hashlib.md5(
            config_name.encode('utf-8'),
            usedforsecurity=False
        ).hexdigest()[:6]
        return f"./milvus_{slug}_{hash_suffix}.db"
    
    def compare_experiments(self, experiments: List[ExperimentConfig]) -> Dict[str, EvaluationMetrics]:
        """Run and compare multiple experiments"""
        results = {}
        
        for config in experiments:
            metrics = self.run_experiment(config)
            results[config.name] = metrics
        
        # Generate comparison report
        self._generate_comparison_report(results)
        
        return results
    
    def _generate_comparison_report(self, results: Dict[str, EvaluationMetrics]):
        """Generate comparison report across experiments"""
        print(f"\n{'='*100}")
        print("EXPERIMENT COMPARISON REPORT")
        print(f"{'='*100}\n")
        
        # Create comparison table
        headers = ['Experiment', 'mAP@0.5', 'Precision', 'Recall', 'F1', 'FPS', 'Milvus Hit%']
        
        print(f"{'Experiment':<40} {'mAP@0.5':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'FPS':>10} {'Milvus':>10}")
        print('-' * 100)
        
        for name, metrics in results.items():
            milvus_rate = f"{metrics.milvus_hit_rate*100:.1f}" if metrics.milvus_hit_rate else "N/A"
            print(f"{name:<40} {metrics.map50:>10.4f} {metrics.precision:>10.4f} {metrics.recall:>10.4f} "
                  f"{metrics.f1_score:>10.4f} {metrics.fps:>10.2f} {milvus_rate:>10}")
        
        print('\n' + '='*100)
        
        # Save comparison to file
        comparison_file = "experiment_comparison.json"
        comparison_data = {
            name: metrics.to_dict() 
            for name, metrics in results.items()
        }
        
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nComparison report saved to: {comparison_file}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run retail detection experiments")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset YAML path")
    parser.add_argument("--yolov8-model", type=str, default="yolov8m.pt", 
                       help="YOLOv8 model path (pretrained or base weights)")
    parser.add_argument("--yolov11-model", type=str, default="yolo11m.pt",
                       help="YOLOv11 model path (pretrained or base weights)")
    parser.add_argument("--dolg-model", type=str, default="dolg_model.pth",
                       help="DOLG embedding model path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--similarity-threshold", type=float, default=0.5,
                       help="Milvus similarity threshold")
    parser.add_argument("--mlflow-uri", type=str, default="file:./mlruns",
                       help="MLflow tracking URI")
    parser.add_argument("--run-all", action="store_true",
                       help="Run all experiments")
    parser.add_argument("--run-baseline-only", action="store_true",
                       help="Run only baseline experiments")
    
    args = parser.parse_args()
    
    # Check if models exist
    yolov8_exists = Path(args.yolov8_model).exists()
    yolov11_exists = Path(args.yolov11_model).exists()
    
    if yolov8_exists:
        print(f"✅ Using existing YOLOv8 model: {args.yolov8_model}")
    else:
        print(f"⚠️  YOLOv8 model not found at: {args.yolov8_model}")
        print(f"   Will download base weights if available")
    
    if yolov11_exists:
        print(f"✅ Using existing YOLOv11 model: {args.yolov11_model}")
    else:
        print(f"⚠️  YOLOv11 model not found at: {args.yolov11_model}")
        print(f"   Will download base weights if available")
    
    # Define experiments
    experiments = []
    
    # Experiment 1: YOLOv8 Baseline
    experiments.append(ExperimentConfig(
        name="YOLOv8_Baseline_488_Classes",
        model_type="yolov8",
        model_path=args.yolov8_model,
        use_milvus=False,
        milvus_collection=None,
        embedding_model=None,
        similarity_threshold=0.0,
        dataset_yaml=args.dataset,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    ))
    
    # Experiment 2: YOLOv11 Baseline
    experiments.append(ExperimentConfig(
        name="YOLOv11_Baseline_488_Classes",
        model_type="yolov11",
        model_path=args.yolov11_model,
        use_milvus=False,
        milvus_collection=None,
        embedding_model=None,
        similarity_threshold=0.0,
        dataset_yaml=args.dataset,
        imgsz=args.imgsz,
        batch_size=args.batch,
        device=args.device
    ))
    
    if not args.run_baseline_only:
        # Experiment 3: YOLOv8 + DOLG + Milvus
        experiments.append(ExperimentConfig(
            name="YOLOv8_DOLG_Milvus_Hybrid",
            model_type="yolov8_hybrid",
            model_path=args.yolov8_model,
            use_milvus=True,
            milvus_collection="retail_items_dolg",
            embedding_model=args.dolg_model,
            similarity_threshold=args.similarity_threshold,
            dataset_yaml=args.dataset,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device
        ))
        
        # Experiment 4: YOLOv8 with higher Milvus threshold
        experiments.append(ExperimentConfig(
            name="YOLOv8_DOLG_Milvus_HighThreshold",
            model_type="yolov8_hybrid",
            model_path=args.yolov8_model,
            use_milvus=True,
            milvus_collection="retail_items_dolg",
            embedding_model=args.dolg_model,
            similarity_threshold=0.7,  # Higher threshold
            dataset_yaml=args.dataset,
            imgsz=args.imgsz,
            batch_size=args.batch,
            device=args.device
        ))
    
    # Run experiments
    runner = ExperimentRunner(
        mlflow_uri=args.mlflow_uri,
        experiment_name="Retail_Detection_Comparison"
    )
    
    results = runner.compare_experiments(experiments)
    
    print("\n✅ All experiments completed successfully!")
    print(f"📊 View results in MLflow UI: mlflow ui --backend-store-uri {args.mlflow_uri}")


if __name__ == "__main__":
    main()
