#!/usr/bin/env python3

"""
PRODUCTION-GRADE Milvus Population Script
- GPU-optimized DOLG embedding extraction
- Automatic Milvus setup and configuration
- Batch processing for maximum throughput
- Progress tracking and error handling
- Production logging and monitoring
"""

import argparse
import logging
import pickle
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

# Setup production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('milvus_population.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def enforce_gpu():
    """Enforce GPU for production"""
    if not torch.cuda.is_available():
        logger.error("=" * 80)
        logger.error("PRODUCTION ERROR: GPU Required")
        logger.error("=" * 80)
        logger.error("This production script requires GPU for:")
        logger.error("  - 20-30x faster embedding extraction")
        logger.error("  - Processing 488 classes with 10+ templates each")
        logger.error("  - Completing in minutes instead of hours")
        logger.error("=" * 80)
        raise RuntimeError("GPU required for production")
    
    logger.info("=" * 80)
    logger.info("GPU VERIFIED")
    logger.info("=" * 80)
    logger.info(f"Device: {torch.cuda.get_device_name(0)}")
    logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    logger.info("=" * 80)


def install_milvus():
    """Install Milvus Lite automatically"""
    logger.info("📥 Installing Milvus Lite...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'pymilvus', '--upgrade', '-q'
        ])
        logger.info("✅ Milvus installed successfully\n")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to install Milvus: {e}")
        return False


def verify_milvus():
    """Verify Milvus is working"""
    try:
        from pymilvus import MilvusClient
        test = MilvusClient(":memory:")
        test.close()
        logger.info("✅ Milvus verified and working\n")
        return True
    except Exception as e:
        logger.warning(f"Milvus verification failed: {e}")
        return install_milvus()


class ProductionDOLG:
    """Production DOLG with GPU optimization"""
    
    def __init__(self, model_path: str, device: str = "cuda:0"):
        enforce_gpu()
        
        self.device = torch.device(device)
        logger.info(f"🎯 Initializing Production DOLG")
        logger.info(f"   Device: {device}")
        
        # Load model
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # GPU optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Compile for faster inference
        if hasattr(torch, 'compile'):
            try:
                self.model = torch.compile(self.model, mode='max-autotune')
                logger.info("   ✅ Compiled with torch.compile")
            except:
                pass
        
        # Setup preprocessing
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        logger.info("   ✅ DOLG ready\n")
    
    def _load_model(self, path):
        import torch.nn as nn
        import torchvision.models as models
        
        class DOLG(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
                in_feat = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(0.2),
                    nn.Linear(in_feat, 128)
                )
            
            def forward(self, x):
                return self.backbone(x)
        
        model = DOLG()
        if Path(path).exists():
            logger.info(f"   Loading weights: {path}")
            model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            logger.info(f"   Using ImageNet pretrained")
        return model
    
    def extract_batch(self, images: List[np.ndarray], batch_size: int = 64) -> np.ndarray:
        """Extract embeddings in batches on GPU"""
        if not images:
            return np.array([])
        
        all_embs = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = [self.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) for img in batch]
            batch_tensor = torch.stack(tensors).to(self.device)
            
            with torch.no_grad(), torch.cuda.amp.autocast():
                embs = self.model(batch_tensor).float().cpu().numpy()
            
            # L2 normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            embs = embs / (norms + 1e-8)
            all_embs.append(embs)
        
        return np.vstack(all_embs)


class ProductionMilvus:
    """Production Milvus manager"""
    
    def __init__(self, db_path: str, collection: str):
        verify_milvus()
        
        from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema
        
        logger.info(f"🗄️  Initializing Milvus Database")
        logger.info(f"   Path: {db_path}")
        logger.info(f"   Collection: {collection}\n")
        
        self.client = MilvusClient(db_path)
        self.collection = collection
        self.dim = 128
    
    def create_collection(self, drop: bool = True):
        """Create optimized collection"""
        from pymilvus import DataType, FieldSchema, CollectionSchema
        
        if drop and self.client.has_collection(self.collection):
            logger.info(f"   Dropping existing: {self.collection}")
            self.client.drop_collection(self.collection)
        
        schema = CollectionSchema([
            FieldSchema("id", DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema("upc", DataType.VARCHAR, max_length=20),
            FieldSchema("class_id", DataType.INT64),
            FieldSchema("class_name", DataType.VARCHAR, max_length=100),
            FieldSchema("template_idx", DataType.INT64),
            FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=self.dim),
        ], description="Production retail embeddings (488 classes)")
        
        index = self.client.prepare_index_params()
        index.add_index("embedding", "IVF_FLAT", "COSINE", {"nlist": 128})
        
        self.client.create_collection(
            self.collection, schema, index, enable_mmap=False
        )
        logger.info(f"   ✅ Collection created: {self.collection}\n")
    
    def insert_batch(self, embeddings: List[Dict]):
        """Insert in optimized batches"""
        logger.info(f"📤 Inserting {len(embeddings)} embeddings...")
        batch_size = 1000
        for i in range(0, len(embeddings), batch_size):
            self.client.insert(self.collection, embeddings[i:i + batch_size])
        logger.info(f"   ✅ Inserted {len(embeddings)} embeddings\n")
    
    def close(self):
        self.client.close()


def extract_class_embeddings(dataset_yaml: str, dolg: ProductionDOLG, max_per_class: int = 10):
    """Extract embeddings from dataset with GPU acceleration"""
    with open(dataset_yaml) as f:
        data = yaml.safe_load(f)
    
    class_names = data['names']
    train_dir = Path(data['train']).parent
    labels_dir = train_dir / 'labels'
    images_dir = train_dir / 'images'
    
    if not labels_dir.exists():
        labels_dir = train_dir.parent / 'labels' / 'train'
        images_dir = train_dir.parent / 'images' / 'train'
    
    logger.info(f"📊 Extracting Embeddings")
    logger.info(f"   Classes: {len(class_names)}")
    logger.info(f"   Templates per class: {max_per_class}")
    logger.info(f"   Images dir: {images_dir}")
    logger.info(f"   Labels dir: {labels_dir}\n")
    
    class_embeddings = {i: [] for i in range(len(class_names))}
    class_counts = {i: 0 for i in range(len(class_names))}
    
    label_files = list(labels_dir.glob('*.txt'))
    logger.info(f"Processing {len(label_files)} label files...\n")
    
    for label_file in tqdm(label_files, desc="Extracting"):
        img_file = images_dir / label_file.with_suffix('.jpg').name
        if not img_file.exists():
            img_file = images_dir / label_file.with_suffix('.png').name
        if not img_file.exists():
            continue
        
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        h, w = image.shape[:2]
        
        # Collect all crops from this image
        crops_for_batch = []
        class_ids_for_batch = []
        
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_counts[class_id] >= max_per_class:
                    continue
                
                x_c, y_c, width, height = map(float, parts[1:5])
                x1 = int((x_c - width / 2) * w)
                y1 = int((y_c - height / 2) * h)
                x2 = int((x_c + width / 2) * w)
                y2 = int((y_c + height / 2) * h)
                
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                
                crops_for_batch.append(crop)
                class_ids_for_batch.append(class_id)
        
        # Batch extract embeddings for this image
        if crops_for_batch:
            embeddings = dolg.extract_batch(crops_for_batch, batch_size=64)
            for emb, cls_id in zip(embeddings, class_ids_for_batch):
                if class_counts[cls_id] < max_per_class:
                    class_embeddings[cls_id].append(emb)
                    class_counts[cls_id] += 1
        
        # Check if done
        if all(cnt >= max_per_class for cnt in class_counts.values()):
            break
    
    logger.info("\n📊 Extraction Statistics:")
    logger.info(f"{'Class ID':<10} {'Class Name':<40} {'Templates':<12}")
    logger.info('-' * 62)
    for cls_id, embs in class_embeddings.items():
        logger.info(f"{cls_id:<10} {class_names[cls_id]:<40} {len(embs):<12}")
    
    total = sum(len(embs) for embs in class_embeddings.values())
    logger.info('-' * 62)
    logger.info(f"Total embeddings: {total}\n")
    
    return class_embeddings, class_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dolg-model", default="dolg_model.pth")
    parser.add_argument("--milvus-db", default="./milvus_production.db")
    parser.add_argument("--collection", default="retail_items")
    parser.add_argument("--max-templates", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache", default="embedding_cache.pkl")
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("PRODUCTION MILVUS POPULATION")
    logger.info("=" * 80)
    
    # Check cache
    cache_path = Path(args.cache)
    if cache_path.exists():
        logger.info(f"📦 Loading cached embeddings: {cache_path}")
        with open(cache_path, 'rb') as f:
            class_embeddings = pickle.load(f)
        with open(args.dataset) as f:
            class_names = yaml.safe_load(f)['names']
    else:
        # Extract embeddings
        dolg = ProductionDOLG(args.dolg_model, args.device)
        class_embeddings, class_names = extract_class_embeddings(
            args.dataset, dolg, args.max_templates
        )
        
        # Cache results
        logger.info(f"💾 Saving cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(class_embeddings, f)
    
    # Setup Milvus
    milvus = ProductionMilvus(args.milvus_db, args.collection)
    milvus.create_collection(drop=True)
    
    # Prepare data
    embeddings_to_insert = []
    for cls_id, embs in class_embeddings.items():
        for idx, emb in enumerate(embs):
            embeddings_to_insert.append({
                "upc": f"UPC_{cls_id:04d}",
                "class_id": cls_id,
                "class_name": class_names[cls_id],
                "template_idx": idx,
                "embedding": emb.tolist()
            })
    
    # Insert
    milvus.insert_batch(embeddings_to_insert)
    milvus.close()
    
    logger.info("=" * 80)
    logger.info("✅ PRODUCTION MILVUS POPULATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Database: {args.milvus_db}")
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Total embeddings: {len(embeddings_to_insert)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
