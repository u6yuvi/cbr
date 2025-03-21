import torch
from torchvision import transforms
from PIL import Image
import io
from cbr_model import ClassificationByRetrieval
from typing import List, Dict, Optional
import numpy as np
import logging
import traceback
import uuid
from datetime import datetime, UTC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        logger.info("Initializing ModelService...")
        self.model = ClassificationByRetrieval()
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Add storage for original images
        self.model.original_images = []
        logger.info("ModelService initialized successfully")
        
    def process_image(self, image_bytes: bytes) -> Dict:
        """Process a single image and return predictions."""
        # Load and transform image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predictions
            class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = self.model.idx_to_classes.get(class_idx, "unknown")
            confidence = probabilities[0, class_idx].item()
            
            # Get all class probabilities
            class_probs = {
                self.model.idx_to_classes[idx]: prob.item()
                for idx, prob in enumerate(probabilities[0])
            }
            
            return {
                "predicted_class": predicted_class,
                "confidence": float(confidence),
                "class_probabilities": class_probs
            }
            
    def get_embedding(self, image_bytes: bytes) -> torch.Tensor:
        """Get embedding for a single image."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            
            with torch.no_grad():
                embedding = self.model.get_embedding(image_tensor)
                return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise ValueError(f"Error processing image: {str(e)}")
            
    def add_class(self, class_name: str, image_bytes_list: List[bytes]) -> Dict:
        """Add a new class with example images."""
        embeddings_list = []
        for image_bytes in image_bytes_list:
            # Store original image
            self.model.original_images.append(image_bytes)
            # Process image for embedding
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model.get_embedding(image_tensor)
                embeddings_list.append(embedding)
                
        embeddings = torch.cat(embeddings_list)
        
        # If this is the first class, initialize the model
        if self.model.index_embeddings is None:
            self.model.add_index_data(embeddings, [class_name] * len(image_bytes_list))
        else:
            # Combine with existing embeddings
            combined_embeddings = torch.cat([self.model.index_embeddings, embeddings])
            
            # Get current labels and add new ones
            current_labels = [self.model.idx_to_classes[idx.item()] for idx in self.model.class_labels]
            combined_labels = current_labels + [class_name] * len(image_bytes_list)
            
            # Update model with all data
            self.model.add_index_data(combined_embeddings, combined_labels)
        
        return {
            "status": "success",
            "message": f"Added class '{class_name}' with {len(image_bytes_list)} examples",
            "num_classes": self.model.num_classes,
            "available_classes": list(self.model.classes_to_idx.keys())
        }
        
    def update_class(self, class_name: str, image_bytes_list: List[bytes], append: bool = False) -> Dict:
        """Update examples for an existing class."""
        if class_name not in self.model.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found")
            
        embeddings_list = []
        for image_bytes in image_bytes_list:
            # Store original image
            self.model.original_images.append(image_bytes)
            # Process image for embedding
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                embedding = self.model.get_embedding(image_tensor)
                embeddings_list.append(embedding)
                
        new_embeddings = torch.cat(embeddings_list)
        
        if append:
            # Combine with existing embeddings
            combined_embeddings = torch.cat([self.model.index_embeddings, new_embeddings])
            
            # Get current labels and add new ones
            current_labels = [self.model.idx_to_classes[idx.item()] for idx in self.model.class_labels]
            combined_labels = current_labels + [class_name] * len(image_bytes_list)
            
            # Update model with all data
            self.model.add_index_data(combined_embeddings, combined_labels)
        else:
            # Get indices for this class
            class_idx = self.model.classes_to_idx[class_name]
            class_mask = (self.model.class_labels == class_idx)
            other_mask = ~class_mask
            
            # Keep embeddings from other classes
            other_embeddings = self.model.index_embeddings[other_mask]
            other_labels = [self.model.idx_to_classes[idx.item()] for idx in self.model.class_labels[other_mask]]
            
            # Combine with new embeddings
            combined_embeddings = torch.cat([other_embeddings, new_embeddings])
            combined_labels = other_labels + [class_name] * len(image_bytes_list)
            
            # Update model with all data
            self.model.add_index_data(combined_embeddings, combined_labels)
            
            # Remove old images for this class
            old_indices = [i for i, is_class in enumerate(class_mask) if is_class]
            for idx in sorted(old_indices, reverse=True):
                self.model.original_images.pop(idx)
            
        return {
            "status": "success",
            "message": f"{'Added' if append else 'Updated'} {len(image_bytes_list)} examples for class '{class_name}'",
            "num_examples": len(self.model.index_embeddings)
        }
        
    def remove_class(self, class_name: str) -> Dict:
        """Remove a class and all its examples."""
        if class_name not in self.model.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found")
            
        # Get indices for this class
        class_idx = self.model.classes_to_idx[class_name]
        class_mask = (self.model.class_labels == class_idx)
        other_mask = ~class_mask
        
        # Keep embeddings from other classes
        if torch.any(other_mask):
            other_embeddings = self.model.index_embeddings[other_mask]
            other_labels = [self.model.idx_to_classes[idx.item()] for idx in self.model.class_labels[other_mask]]
            
            # Update model with remaining data
            self.model.add_index_data(other_embeddings, other_labels)
        else:
            # If no other classes exist, reset the model
            self.model.index_embeddings = None
            self.model.class_labels = None
            self.model.classes_to_idx = {}
            self.model.idx_to_classes = {}
        
        # Remove original images for this class
        indices_to_remove = [i for i, is_class in enumerate(class_mask) if is_class]
        for idx in sorted(indices_to_remove, reverse=True):
            self.model.original_images.pop(idx)
        
        return {
            "status": "success",
            "message": f"Removed class '{class_name}'",
            "num_classes": len(self.model.classes_to_idx),
            "available_classes": list(self.model.classes_to_idx.keys())
        }
        
    def remove_examples(self, indices: List[int]) -> Dict:
        """Remove specific examples by their indices."""
        if not self.model.index_embeddings is None:
            if max(indices) >= len(self.model.index_embeddings):
                raise ValueError("Invalid index provided")
                
        # Remove original images
        for idx in sorted(indices, reverse=True):
            self.model.original_images.pop(idx)
            
        self.model.remove_examples(indices)
        
        return {
            "status": "success",
            "message": f"Removed {len(indices)} examples",
            "num_examples": len(self.model.index_embeddings)
        }
        
    def get_model_info(self) -> Dict:
        """Get current model state information."""
        return {
            "num_classes": self.model.num_classes,
            "num_examples": len(self.model.index_embeddings) if self.model.index_embeddings is not None else 0,
            "available_classes": list(self.model.classes_to_idx.keys()),
            "examples_per_class": {
                class_name: int((self.model.class_labels == class_idx).sum().item())
                for class_name, class_idx in self.model.classes_to_idx.items()
            } if self.model.index_embeddings is not None else {}
        }

class TenantModelManager:
    def __init__(self):
        self.tenant_models: Dict[str, ModelService] = {}
        self.tenant_metadata: Dict[str, Dict] = {}
        logger.info("Initialized TenantModelManager")
    
    def create_tenant(self, name: Optional[str] = None) -> Dict[str, str]:
        """Create a new tenant with a unique ID."""
        tenant_id = str(uuid.uuid4())
        creation_time = datetime.now(UTC).isoformat()
        
        # Store tenant metadata
        self.tenant_metadata[tenant_id] = {
            "name": name,
            "created_at": creation_time,
            "last_accessed": creation_time
        }
        
        logger.info(f"Created new tenant: {tenant_id} (name: {name})")
        return {
            "tenant_id": tenant_id,
            "name": name,
            "created_at": creation_time
        }
    
    def get_tenant_metadata(self, tenant_id: str) -> Optional[Dict]:
        """Get metadata for a specific tenant."""
        return self.tenant_metadata.get(tenant_id)
    
    def get_or_create_model(self, tenant_id: str) -> ModelService:
        """Get an existing model instance for a tenant or create a new one."""
        if tenant_id not in self.tenant_models:
            if tenant_id not in self.tenant_metadata:
                raise ValueError(f"Tenant {tenant_id} does not exist. Please create a tenant first.")
            logger.info(f"Creating new model instance for tenant {tenant_id}")
            self.tenant_models[tenant_id] = ModelService()
            
        # Update last accessed time
        self.tenant_metadata[tenant_id]["last_accessed"] = datetime.now(UTC).isoformat()
        return self.tenant_models[tenant_id]
    
    def remove_tenant(self, tenant_id: str) -> bool:
        """Remove a tenant's model instance and metadata."""
        if tenant_id in self.tenant_models:
            del self.tenant_models[tenant_id]
            del self.tenant_metadata[tenant_id]
            logger.info(f"Removed tenant {tenant_id} and their model instance")
            return True
        return False
    
    def get_tenant_ids(self) -> List[str]:
        """Get list of all tenant IDs."""
        return list(self.tenant_metadata.keys())
    
    def get_tenant_info(self, tenant_id: str) -> Optional[Dict]:
        """Get model info for a specific tenant."""
        if tenant_id in self.tenant_models:
            model_info = self.tenant_models[tenant_id].get_model_info()
            metadata = self.tenant_metadata[tenant_id]
            return {
                **model_info,
                "tenant_metadata": metadata
            }
        return None 