import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
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

    def update_class_embeddings(self, class_name: str, new_embeddings: torch.Tensor, append: bool = False):
        """
        Update embeddings for a specific class.
        
        Args:
            class_name: Name of the class to update
            new_embeddings: New embeddings for the class
            append: If True, append to existing embeddings; if False, replace them
        """
        if class_name not in self.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found in index")
            
        class_idx = self.classes_to_idx[class_name]
        class_mask = (self.class_labels == class_idx)
        
        if append:
            # Append new embeddings to existing ones
            combined_embeddings = torch.cat([self.index_embeddings, new_embeddings])
            combined_labels = torch.cat([
                self.class_labels,
                torch.full((len(new_embeddings),), class_idx, dtype=torch.long)
            ])
        else:
            # Replace existing embeddings
            non_class_mask = ~class_mask
            combined_embeddings = torch.cat([
                self.index_embeddings[non_class_mask],
                new_embeddings
            ])
            combined_labels = torch.cat([
                self.class_labels[non_class_mask],
                torch.full((len(new_embeddings),), class_idx, dtype=torch.long)
            ])
            
        # Normalize if needed
        if self.normalize_embeddings:
            combined_embeddings = F.normalize(combined_embeddings, p=2, dim=-1)
            
        # Update buffers
        self.register_buffer('index_embeddings', combined_embeddings)
        self.register_buffer('class_labels', combined_labels)

    def remove_class(self, class_name: str):
        """
        Remove a class and its embeddings from the index.
        
        Args:
            class_name: Name of the class to remove
        """
        if class_name not in self.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found in index")
            
        class_idx = self.classes_to_idx[class_name]
        keep_mask = (self.class_labels != class_idx)
        
        # Update embeddings and labels
        self.register_buffer('index_embeddings', self.index_embeddings[keep_mask])
        self.register_buffer('class_labels', self.class_labels[keep_mask])
        
        # Update class mappings
        del self.classes_to_idx[class_name]
        del self.idx_to_classes[class_idx]
        
        # Remap remaining class indices
        new_idx = 0
        new_classes_to_idx = {}
        new_idx_to_classes = {}
        
        for old_class, old_idx in self.classes_to_idx.items():
            new_classes_to_idx[old_class] = new_idx
            new_idx_to_classes[new_idx] = old_class
            # Update labels to new indices
            self.class_labels[self.class_labels == old_idx] = new_idx
            new_idx += 1
            
        self.classes_to_idx = new_classes_to_idx
        self.idx_to_classes = new_idx_to_classes
        self.num_classes = len(self.classes_to_idx)

    def remove_examples(self, indices: Union[List[int], torch.Tensor]):
        """
        Remove specific examples from the index.
        
        Args:
            indices: List or tensor of indices to remove
        """
        if isinstance(indices, list):
            indices = torch.tensor(indices)
            
        # Create mask for keeping examples
        keep_mask = torch.ones(len(self.index_embeddings), dtype=torch.bool)
        keep_mask[indices] = False
        
        # Update embeddings and labels
        self.register_buffer('index_embeddings', self.index_embeddings[keep_mask])
        self.register_buffer('class_labels', self.class_labels[keep_mask])
        
        # Check if any classes are now empty
        unique_labels = torch.unique(self.class_labels)
        empty_classes = []
        
        for idx in range(self.num_classes):
            if idx not in unique_labels:
                class_name = self.idx_to_classes[idx]
                empty_classes.append(class_name)
                
        # Remove empty classes
        for class_name in empty_classes:
            self.remove_class(class_name) 