#!/usr/bin/env python3

"""
Populate Milvus database with DOLG embeddings from retail item training dataset
Generates embeddings for all 488 classes with multiple templates per class
"""

from __future__ import annotations

import argparse
import pickle
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import yaml

import cv2
import numpy as np
import torch
from tqdm import tqdm

# Ensure required packages
try:
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
except ImportError:
    print("📦 Installing pymilvus...")
    subprocess.run([sys.executable, "-m", "pip", "install", "pymilvus>=2.5.0"], check=True)
    from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema


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


class MilvusPopulator:
    """Production-grade Milvus database populator with auto-setup"""
    
    def __init__(self, db_path: str = "./milvus_retail.db", 
                 collection_name: str = "retail_items"):
        """Initialize production-grade Milvus populator with auto-setup"""
        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_dim = 128
        
        # Auto-setup Milvus
        self._ensure_milvus_ready()
        
        # Initialize client
        try:
            self.client = MilvusClient(db_path)
            print(f"✅ Connected to Milvus database: {db_path}")
        except Exception as e:
            print(f"❌ Failed to connect to Milvus: {e}")
            raise
    
    def _ensure_milvus_ready(self):
        """Ensure Milvus is properly set up and ready to use"""
        print("\n" + "="*80)
        print("Setting up Milvus Vector Database for Production")
        print("="*80 + "\n")
        
        # Create database directory if needed
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if database exists
        if Path(self.db_path).exists():
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
        
        # Define schema
        print(f"📊 Creating production collection: {self.collection_name}")
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="upc", dtype=DataType.VARCHAR, max_length=20),
                FieldSchema(name="class_id", dtype=DataType.INT64),
                FieldSchema(name="class_name", dtype=DataType.VARCHAR, max_length=100),
                FieldSchema(name="template_idx", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ],
            description="Production retail item embeddings with DOLG (488 classes)",
            enable_dynamic_field=False
        )
        
        # Create index params for GPU-accelerated search
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="embedding",
            index_type="FLAT",  # Best accuracy, GPU-accelerated
            metric_type="COSINE"
        )
        
        # Create collection with production settings
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
            enable_mmap=False,  # Disable mmap for better GPU performance
            properties={
                "mmap.enabled": "false",
                "collection.ttl.seconds": "0"
            }
        )
        
        print(f"✅ Collection '{self.collection_name}' created successfully")
        print(f"   Embedding dimension: {self.embedding_dim}")
        print(f"   Index type: FLAT (GPU-accelerated)")
        print(f"   Metric type: COSINE")
        
    def insert_embeddings(self, embeddings: List[Dict], batch_size: int = 1000):
        """Insert embeddings with production-grade batch processing"""
        if not embeddings:
            print("⚠️  No embeddings to insert")
            return
        
        total = len(embeddings)
        print(f"\n📥 Inserting {total:,} embeddings into Milvus...")
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
        print("💾 Flushing data to disk...")
        self.client.flush(collection_name=self.collection_name)
        
        # Get collection stats
        stats = self.client.get_collection_stats(collection_name=self.collection_name)
        print(f"\n✅ Successfully inserted {total:,} embeddings")
        print(f"   Collection stats: {stats}\n")
        
    def save_embedding_cache(self, embeddings: Dict, cache_path: str):
        """Save embeddings to pickle file for faster loading"""
        print(f"💾 Saving embedding cache to: {cache_path}")
        cache_dir = Path(cache_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✅ Embedding cache saved ({Path(cache_path).stat().st_size / 1024 / 1024:.2f} MB)")
        
    def load_embedding_cache(self, cache_path: str) -> Dict:
        """Load embeddings from pickle file"""
        print(f"📦 Loading embedding cache from: {cache_path}")
        with open(cache_path, 'rb') as f:
            embeddings = pickle.load(f)
        cache_size = Path(cache_path).stat().st_size / 1024 / 1024
        print(f"✅ Loaded {len(embeddings)} cached classes ({cache_size:.2f} MB)")
        return embeddings
    
    def close(self):
        """Close Milvus connection"""
        try:
            self.client.close()
            print("✅ Milvus connection closed")
        except Exception as e:
            print(f"⚠️  Error closing Milvus: {e}")


def extract_class_embeddings(dataset_yaml: str, 
                            embedding_extractor: DOLGEmbeddingExtractor,
                            max_templates_per_class: int = 10,
                            batch_size: int = 32) -> Dict[int, List[np.ndarray]]:
    """
    Extract embeddings for each class from training dataset with GPU batch processing
    
    Args:
        dataset_yaml: Path to dataset YAML file
        embedding_extractor: DOLG embedding extractor
        max_templates_per_class: Maximum number of template embeddings per class
        batch_size: Batch size for GPU processing
        
    Returns:
        Dictionary mapping class_id to list of embedding vectors
    """
    # Load dataset info
    with open(dataset_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    
    class_names = dataset_info['names']
    dataset_yaml_path = Path(dataset_yaml).resolve()
    base_dir = dataset_yaml_path.parent
    
    train_path = Path(dataset_info['train'])
    if not train_path.is_absolute():
        train_path = (base_dir / train_path).resolve()
    train_path = train_path.resolve()
    if not train_path.exists():
        fallback = base_dir / "train" / "images"
        if fallback.exists():
            train_path = fallback.resolve()
        elif (base_dir / "train").exists():
            train_path = (base_dir / "train").resolve()
        else:
            raise FileNotFoundError(f"Could not resolve training images directory from '{dataset_info['train']}'")
    
    if train_path.name.lower() == "images":
        train_dir = train_path.parent
    else:
        train_dir = train_path
   
    # Dictionary to store embeddings per class
    class_embeddings = {i: [] for i in range(len(class_names))}
    class_counts = {i: 0 for i in range(len(class_names))}
    
    print(f"\n{'='*80}")
    print("Extracting Production-Grade DOLG Embeddings from Training Dataset")
    print(f"{'='*80}\n")
    print(f"Number of classes: {len(class_names)}")
    print(f"Max templates per class: {max_templates_per_class}")
    print(f"Batch size for GPU: {batch_size}\n")
    
    # Find all label files
    labels_dir = train_dir / 'labels'
    if not labels_dir.exists():
        labels_dir = train_dir.parent / 'labels' / 'train'
    
    images_dir = train_dir / 'images'
    if not images_dir.exists():
        images_dir = train_dir.parent / 'images' / 'train'
    
    label_files = sorted(labels_dir.glob('*.txt'))
    
    print(f"Found {len(label_files):,} label files")
    print(f"Processing images from: {images_dir}\n")
    
    # Collect all crops first for batch processing
    all_crops = []
    crop_metadata = []
    
    print("📦 Collecting image crops...")
    for label_file in tqdm(label_files, desc="Scanning images"):
        # Get corresponding image file
        img_file = images_dir / label_file.with_suffix('.jpg').name
        if not img_file.exists():
            img_file = images_dir / label_file.with_suffix('.png').name
        if not img_file.exists():
            continue
        
        # Load image
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Read labels
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                
                # Skip if we already have enough templates for this class
                if class_counts[class_id] >= max_templates_per_class:
                    continue
                
                # Parse bounding box (YOLO format: class x_center y_center width height)
                x_center, y_center, width, height = map(float, parts[1:5])
                
                # Convert to pixel coordinates
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Crop item from image
                crop = image[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Store crop and metadata
                all_crops.append(crop)
                crop_metadata.append({'class_id': class_id})
                class_counts[class_id] += 1
        
        # Check if we have enough templates for all classes
        if all(count >= max_templates_per_class for count in class_counts.values()):
            print("\n✅ Reached maximum templates for all classes")
            break
    
    print(f"\n📊 Collected {len(all_crops):,} crops for embedding extraction")
    print(f"⚡ Extracting embeddings with GPU batch processing (batch_size={batch_size})...")
    
    # Extract embeddings in batches for optimal GPU utilization
    all_embeddings = []
    for i in tqdm(range(0, len(all_crops), batch_size), desc="Extracting embeddings"):
        batch_crops = all_crops[i:i + batch_size]
        try:
            batch_embeddings = embedding_extractor.extract_embeddings_batch(batch_crops)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"❌ Error extracting batch {i//batch_size}: {e}")
            # Fallback to individual processing for this batch
            for crop in batch_crops:
                try:
                    emb = embedding_extractor.extract_embedding(crop)
                    all_embeddings.append(emb)
                except Exception:
                    continue
    
    # Synchronize GPU
    if embedding_extractor.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Organize embeddings by class
    print(f"\n📂 Organizing {len(all_embeddings):,} embeddings by class...")
    for embedding, metadata in zip(all_embeddings, crop_metadata):
        class_id = metadata['class_id']
        class_embeddings[class_id].append(embedding)
    
    # Print statistics
    print("\n" + "="*80)
    print("📊 Embedding Extraction Statistics (Production)")
    print("="*80)
    print(f"{'Class ID':<10} {'Class Name':<40} {'Templates':<12} {'Status':<10}")
    print("-" * 72)
    
    for class_id, embeddings in class_embeddings.items():
        class_name = class_names[class_id]
        num_templates = len(embeddings)
        status = "✅" if num_templates >= max_templates_per_class else "⚠️ "
        print(f"{class_id:<10} {class_name:<40} {num_templates:<12} {status:<10}")
    
    total_embeddings = sum(len(embs) for embs in class_embeddings.values())
    print("-" * 72)
    print(f"{'TOTAL':<51} {total_embeddings:<12}")
    print("="*80 + "\n")
    
    # GPU memory cleanup
    if embedding_extractor.device.type == 'cuda':
        torch.cuda.empty_cache()
    
    return class_embeddings


def populate_milvus_from_dataset(dataset_yaml: str,
                                dolg_model_path: str,
                                milvus_db_path: str,
                                collection_name: str = "retail_items",
                                max_templates_per_class: int = 10,
                                batch_size: int = 32,
                                cache_path: Optional[str] = None,
                                use_cache: bool = True,
                                device: str = "cuda:0"):
    """
    Production-grade function to populate Milvus database with embeddings
    
    Args:
        dataset_yaml: Path to dataset YAML file
        dolg_model_path: Path to DOLG model weights
        milvus_db_path: Path to Milvus database file
        collection_name: Name of Milvus collection
        max_templates_per_class: Maximum templates per class
        batch_size: Batch size for GPU processing
        cache_path: Path to save/load embedding cache
        use_cache: Whether to use cached embeddings
        device: Device to run on (enforce GPU for production)
    """
    print("\n" + "🚀"*40)
    print("PRODUCTION-GRADE MILVUS POPULATION PIPELINE")
    print("🚀"*40 + "\n")
    
    # Force GPU for production
    if not device.startswith('cuda'):
        print("⚠️  WARNING: CPU mode specified. This is NOT recommended for production.")
        print("   Forcing GPU usage for optimal performance...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if not torch.cuda.is_available():
        print("❌ ERROR: No GPU available. Production pipeline requires GPU.")
        print("   Please ensure CUDA is properly installed and GPU is accessible.")
        raise RuntimeError("GPU required for production pipeline")
    
    # Show GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"🎮 GPU Device: {gpu_name}")
    print(f"💾 GPU Memory: {gpu_memory:.2f} GB\n")
    
    # Load dataset info for class names
    with open(dataset_yaml, 'r') as f:
        dataset_info = yaml.safe_load(f)
    class_names = dataset_info['names']
    
    # Check for cached embeddings
    if use_cache and cache_path and Path(cache_path).exists():
        print(f"📦 Loading cached embeddings from: {cache_path}")
        with open(cache_path, 'rb') as f:
            class_embeddings = pickle.load(f)
        cache_size = Path(cache_path).stat().st_size / 1024 / 1024
        print(f"✅ Loaded cached embeddings ({cache_size:.2f} MB)\n")
    else:
        # Extract embeddings with GPU acceleration
        print("🔬 Initializing Production DOLG Embedding Extractor...")
        embedding_extractor = DOLGEmbeddingExtractor(
            model_path=dolg_model_path,
            device=device,
            batch_size=batch_size,
            use_amp=True  # Enable mixed precision for speed
        )
        
        class_embeddings = extract_class_embeddings(
            dataset_yaml=dataset_yaml,
            embedding_extractor=embedding_extractor,
            max_templates_per_class=max_templates_per_class,
            batch_size=batch_size
        )
        
        # Save cache if requested
        if cache_path:
            print(f"💾 Saving embedding cache to: {cache_path}")
            cache_dir = Path(cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(class_embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            cache_size = Path(cache_path).stat().st_size / 1024 / 1024
            print(f"✅ Embedding cache saved ({cache_size:.2f} MB)\n")
    
    # Initialize Milvus with auto-setup
    print(f"🗄️  Initializing Production Milvus Database...")
    populator = MilvusPopulator(
        db_path=milvus_db_path,
        collection_name=collection_name
    )
    
    # Create collection with GPU-optimized index
    populator.create_collection(drop_existing=True)
    
    # Prepare embeddings for insertion
    print("\n📋 Preparing embeddings for batch insertion...")
    embeddings_to_insert = []
    
    for class_id, embeddings in class_embeddings.items():
        class_name = class_names[class_id]
        
        for template_idx, embedding in enumerate(embeddings):
            embeddings_to_insert.append({
                "upc": f"UPC_{class_id:04d}",  # Generate synthetic UPC
                "class_id": int(class_id),  # Ensure int type
                "class_name": str(class_name),
                "template_idx": int(template_idx),
                "embedding": embedding.tolist()
            })
    
    # Insert embeddings with production batch processing
    if embeddings_to_insert:
        populator.insert_embeddings(embeddings_to_insert, batch_size=1000)
    else:
        print("❌ ERROR: No embeddings to insert!")
        raise ValueError("No embeddings were extracted from dataset")
    
    # Verify insertion
    stats = populator.client.get_collection_stats(collection_name=collection_name)
    print("\n" + "="*80)
    print("✅ MILVUS DATABASE POPULATION COMPLETE")
    print("="*80)
    print(f"📊 Database: {milvus_db_path}")
    print(f"📦 Collection: {collection_name}")
    print(f"🔢 Total embeddings: {len(embeddings_to_insert):,}")
    print(f"🏷️  Total classes: {len(class_names)}")
    print(f"📈 Collection stats: {stats}")
    print(f"🎯 Ready for production similarity search!")
    print("="*80 + "\n")
    
    # Close connection
    populator.close()
    
    # GPU memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 GPU memory cleaned up\n")


def main():
    """Main execution function with production-grade GPU enforcement"""
    parser = argparse.ArgumentParser(
        description="Production-grade Milvus population with DOLG embeddings (GPU-accelerated)"
    )
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to dataset YAML file")
    parser.add_argument("--dolg-model", type=str, default="dolg_model.pth",
                       help="Path to DOLG model weights")
    parser.add_argument("--milvus-db", type=str, default="./milvus_retail.db",
                       help="Path to Milvus database file")
    parser.add_argument("--collection", type=str, default="retail_items",
                       help="Name of Milvus collection")
    parser.add_argument("--max-templates", type=int, default=10,
                       help="Maximum templates per class")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for GPU processing (default: 32)")
    parser.add_argument("--cache", type=str, default="embedding_cache.pkl",
                       help="Path to embedding cache file")
    parser.add_argument("--no-cache", action="store_true",
                       help="Don't use cached embeddings")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run on (cuda:0, cuda:1, cpu) - GPU strongly recommended")
    
    args = parser.parse_args()
    
    # Validate GPU availability if CUDA requested
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("\n" + "❌"*40)
            print("ERROR: CUDA device requested but not available!")
            print("❌"*40 + "\n")
            print("Options:")
            print("1. Install CUDA and PyTorch with CUDA support")
            print("2. Use --device cpu (NOT recommended for production)")
            print("\nFor installation help:")
            print("  https://pytorch.org/get-started/locally/\n")
            sys.exit(1)
    
    populate_milvus_from_dataset(
        dataset_yaml=args.dataset,
        dolg_model_path=args.dolg_model,
        milvus_db_path=args.milvus_db,
        collection_name=args.collection,
        max_templates_per_class=args.max_templates,
        batch_size=args.batch_size,
        cache_path=args.cache if not args.no_cache else None,
        use_cache=not args.no_cache,
        device=args.device
    )


if __name__ == "__main__":
    main()
