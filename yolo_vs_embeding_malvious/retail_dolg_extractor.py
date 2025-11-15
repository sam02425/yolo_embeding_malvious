"""
Retail-specific DOLG embedding extractor with support for:
1. Retail-trained DOLG models
2. Confidence-based embedding extraction
3. Multiple embedding strategies
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import timm
from pathlib import Path


class DOLGModel(nn.Module):
    """DOLG model architecture (must match training script)"""
    
    def __init__(self, num_classes: int, embedding_dim: int = 128, 
                 backbone: str = 'efficientnet_b0', pretrained: bool = True):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        
        # Backbone
        self.backbone = timm.create_model(backbone, pretrained=pretrained, 
                                         features_only=True, out_indices=[3])
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy_input)[0]
            feat_dim = features.shape[1]
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )
        
        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
        # L2 normalization for embeddings
        self.l2_norm = lambda x: nn.functional.normalize(x, p=2, dim=1)
    
    def forward(self, x, return_embeddings=False):
        # Extract features
        features = self.backbone(x)[0]  # [B, C, H, W]
        
        # Global pooling
        global_feat = self.global_pool(features).flatten(1)  # [B, C]
        
        # Get embeddings
        embeddings = self.embedding_head(global_feat)
        embeddings = self.l2_norm(embeddings)
        
        if return_embeddings:
            return embeddings
        
        # Get logits
        logits = self.classifier(embeddings)
        
        return logits, embeddings


class RetailDOLGExtractor:
    """
    Enhanced DOLG embedding extractor for retail products
    Supports both ImageNet pretrained and retail fine-tuned models
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        embedding_dim: int = 128,
        device: str = 'cuda:0',
        use_retail_model: bool = True
    ):
        """
        Args:
            model_path: Path to retail-trained DOLG model checkpoint
            embedding_dim: Embedding dimension
            device: Device to run on
            use_retail_model: If True, use retail-trained model; else use ImageNet pretrained
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self.use_retail_model = use_retail_model
        
        # Load model
        if use_retail_model and model_path and Path(model_path).exists():
            print(f"🎯 Loading retail-trained DOLG model from {model_path}")
            checkpoint = torch.load(model_path, map_location=device)
            
            num_classes = checkpoint.get('num_classes', 59)  # Default to grocery dataset
            embedding_dim = checkpoint.get('embedding_dim', 128)
            
            self.model = DOLGModel(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                backbone='efficientnet_b0',
                pretrained=False
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(device)
            self.model.eval()
            
            print(f"✅ Retail DOLG model loaded (val_acc: {checkpoint.get('val_acc', 'N/A'):.2f}%)")
        else:
            # Fallback to ImageNet pretrained
            print("⚠️  Using ImageNet pretrained DOLG (may have lower performance)")
            print("   Consider training retail-specific model with train_dolg_retail.py")
            
            self.model = DOLGModel(
                num_classes=1000,  # ImageNet classes
                embedding_dim=embedding_dim,
                backbone='efficientnet_b0',
                pretrained=True
            )
            self.model.to(device)
            self.model.eval()
        
        # Preprocessing
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    def preprocess_crop(self, crop: np.ndarray) -> torch.Tensor:
        """Preprocess crop for embedding extraction"""
        # Resize to 224x224
        crop_resized = cv2.resize(crop, (224, 224))
        
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        crop_tensor = torch.from_numpy(crop_rgb).float().permute(2, 0, 1).unsqueeze(0)
        crop_tensor = crop_tensor.to(self.device) / 255.0
        crop_tensor = (crop_tensor - self.mean) / self.std
        
        return crop_tensor
    
    @torch.no_grad()
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract embedding from single crop"""
        crop_tensor = self.preprocess_crop(crop)
        embedding = self.model(crop_tensor, return_embeddings=True)
        return embedding.cpu().numpy().flatten()
    
    @torch.no_grad()
    def extract_batch_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract embeddings from batch of crops"""
        if not crops:
            return np.array([])
        
        # Preprocess all crops
        crop_tensors = [self.preprocess_crop(crop) for crop in crops]
        batch_tensor = torch.cat(crop_tensors, dim=0)
        
        # Extract embeddings
        embeddings = self.model(batch_tensor, return_embeddings=True)
        return embeddings.cpu().numpy()
    
    def get_embedding_stats(self) -> Dict[str, any]:
        """Get statistics about the embedding extractor"""
        return {
            'model_type': 'retail_trained' if self.use_retail_model else 'imagenet_pretrained',
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'backbone': 'efficientnet_b0'
        }


class EnsembleDOLGExtractor:
    """
    Ensemble embedding extractor that combines multiple models
    for more robust embeddings
    """
    
    def __init__(
        self,
        retail_model_path: str,
        embedding_dim: int = 128,
        device: str = 'cuda:0',
        ensemble_mode: str = 'concat'  # 'concat', 'average', 'weighted'
    ):
        """
        Args:
            retail_model_path: Path to retail-trained model
            embedding_dim: Base embedding dimension
            device: Device to run on
            ensemble_mode: How to combine embeddings
        """
        self.device = device
        self.ensemble_mode = ensemble_mode
        
        # Load retail-trained model
        self.retail_extractor = RetailDOLGExtractor(
            model_path=retail_model_path,
            embedding_dim=embedding_dim,
            device=device,
            use_retail_model=True
        )
        
        # Load ImageNet pretrained model for comparison
        self.imagenet_extractor = RetailDOLGExtractor(
            model_path=None,
            embedding_dim=embedding_dim,
            device=device,
            use_retail_model=False
        )
        
        if ensemble_mode == 'concat':
            self.final_embedding_dim = embedding_dim * 2
        else:
            self.final_embedding_dim = embedding_dim
        
        print(f"🎯 Ensemble extractor initialized (mode={ensemble_mode}, dim={self.final_embedding_dim})")
    
    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract ensemble embedding from crop"""
        # Get embeddings from both models
        retail_emb = self.retail_extractor.extract_embedding(crop)
        imagenet_emb = self.imagenet_extractor.extract_embedding(crop)
        
        # Combine based on mode
        if self.ensemble_mode == 'concat':
            return np.concatenate([retail_emb, imagenet_emb])
        elif self.ensemble_mode == 'average':
            return (retail_emb + imagenet_emb) / 2
        elif self.ensemble_mode == 'weighted':
            # Weight retail model higher (0.7 vs 0.3)
            return 0.7 * retail_emb + 0.3 * imagenet_emb
        else:
            return retail_emb
    
    def extract_batch_embeddings(self, crops: List[np.ndarray]) -> np.ndarray:
        """Extract ensemble embeddings from batch"""
        if not crops:
            return np.array([])
        
        retail_embs = self.retail_extractor.extract_batch_embeddings(crops)
        imagenet_embs = self.imagenet_extractor.extract_batch_embeddings(crops)
        
        if self.ensemble_mode == 'concat':
            return np.concatenate([retail_embs, imagenet_embs], axis=1)
        elif self.ensemble_mode == 'average':
            return (retail_embs + imagenet_embs) / 2
        elif self.ensemble_mode == 'weighted':
            return 0.7 * retail_embs + 0.3 * imagenet_embs
        else:
            return retail_embs


def create_embedding_extractor(
    extractor_type: str = 'retail',
    model_path: Optional[str] = None,
    embedding_dim: int = 128,
    device: str = 'cuda:0',
    **kwargs
) -> RetailDOLGExtractor:
    """
    Factory function to create appropriate embedding extractor
    
    Args:
        extractor_type: 'retail', 'imagenet', or 'ensemble'
        model_path: Path to retail model checkpoint
        embedding_dim: Embedding dimension
        device: Device to run on
        **kwargs: Additional arguments for specific extractors
    """
    if extractor_type == 'retail':
        return RetailDOLGExtractor(
            model_path=model_path,
            embedding_dim=embedding_dim,
            device=device,
            use_retail_model=True
        )
    elif extractor_type == 'imagenet':
        return RetailDOLGExtractor(
            model_path=None,
            embedding_dim=embedding_dim,
            device=device,
            use_retail_model=False
        )
    elif extractor_type == 'ensemble':
        ensemble_mode = kwargs.get('ensemble_mode', 'concat')
        return EnsembleDOLGExtractor(
            retail_model_path=model_path,
            embedding_dim=embedding_dim,
            device=device,
            ensemble_mode=ensemble_mode
        )
    else:
        raise ValueError(f"Unknown extractor type: {extractor_type}")


if __name__ == "__main__":
    # Test embedding extraction
    print("Testing retail DOLG extractor...")
    
    # Create dummy crop
    dummy_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    # Test retail extractor
    extractor = create_embedding_extractor(
        extractor_type='retail',
        model_path='dolg_retail_model/dolg_retail_best.pth',
        device='cuda:0'
    )
    
    embedding = extractor.extract_embedding(dummy_crop)
    print(f"✅ Embedding shape: {embedding.shape}")
    print(f"✅ Embedding L2 norm: {np.linalg.norm(embedding):.4f}")
