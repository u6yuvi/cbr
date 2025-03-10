import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from torchvision.models import resnet18, ResNet18_Weights

class ClassificationByRetrieval(nn.Module):
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        embedding_dim: int = 512,  # ResNet18's output dimension is 512
        normalize_embeddings: bool = True
    ):
        """
        Classification by Retrieval (CbR) model implementation.
        
        Args:
            backbone: Optional pre-trained backbone model. If None, uses ResNet18
            embedding_dim: Dimension of the embedding space
            normalize_embeddings: Whether to L2 normalize embeddings
        """
        super().__init__()
        
        # Initialize backbone
        if backbone is None:
            # Use ResNet18 by default, removing the classification head
            self.backbone = nn.Sequential(
                *list(resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).children())[:-1]
            )
        else:
            self.backbone = backbone
            
        self.embedding_dim = embedding_dim
        self.normalize_embeddings = normalize_embeddings
        
        # These will be set when index data is added
        self.register_buffer('index_embeddings', None)
        self.register_buffer('class_labels', None)
        self.num_classes = 0
        self.classes_to_idx = {}
        self.idx_to_classes = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CbR model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Get embeddings from backbone
        embeddings = self.backbone(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten to (batch_size, embedding_dim)
        
        # Normalize embeddings if specified
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # If no index data is set, return embeddings
        if self.index_embeddings is None:
            return embeddings
            
        # Compute similarities with index embeddings (nearest neighbor matching)
        similarities = torch.mm(embeddings, self.index_embeddings.t())
        
        # Aggregate results per class
        logits = torch.zeros(embeddings.size(0), self.num_classes, device=x.device)
        
        for class_idx in range(self.num_classes):
            # Get mask for current class
            class_mask = (self.class_labels == class_idx)
            
            # Get similarities for current class
            class_similarities = similarities[:, class_mask]
            
            # Aggregate using max (can be modified to use other aggregation functions)
            class_logits = torch.max(class_similarities, dim=1)[0]
            logits[:, class_idx] = class_logits
            
        return logits

    def add_index_data(self, embeddings: torch.Tensor, labels: List[str]):
        """
        Add index data to the model.
        
        Args:
            embeddings: Tensor of shape (num_samples, embedding_dim)
            labels: List of class labels corresponding to the embeddings
        """
        # Create class mapping if not exists
        unique_labels = sorted(set(labels))
        if not self.classes_to_idx:
            self.classes_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            self.idx_to_classes = {idx: label for label, idx in self.classes_to_idx.items()}
            self.num_classes = len(unique_labels)
        
        # Convert labels to indices
        label_indices = torch.tensor([self.classes_to_idx[label] for label in labels])
        
        # Normalize embeddings if specified
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Register embeddings and labels as buffers
        self.register_buffer('index_embeddings', embeddings)
        self.register_buffer('class_labels', label_indices)

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embeddings for input images.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embeddings of shape (batch_size, embedding_dim)
        """
        embeddings = self.backbone(x)
        embeddings = embeddings.view(embeddings.size(0), -1)  # Flatten to (batch_size, embedding_dim)
        if self.normalize_embeddings:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings 