import torch
from torchvision import transforms
from PIL import Image
import io
from cbr_model import ClassificationByRetrieval
from typing import List, Dict, Optional
import numpy as np
import logging
import traceback

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
        """Add a new class with one or more example images."""
        logger.info(f"Adding new class '{class_name}' with {len(image_bytes_list)} examples")
        
        if not image_bytes_list:
            raise ValueError("No images provided")
            
        embeddings_list = []
        
        # Get embeddings for all images
        for i, image_bytes in enumerate(image_bytes_list):
            try:
                logger.debug(f"Processing image {i+1}/{len(image_bytes_list)}")
                if not image_bytes:
                    raise ValueError(f"Empty image data for image {i+1}")
                    
                embedding = self.get_embedding(image_bytes)
                embeddings_list.append(embedding)
            except Exception as e:
                logger.error(f"Error processing image {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
                raise ValueError(f"Error processing image {i+1}: {str(e)}")
            
        try:
            # Combine embeddings
            new_embeddings = torch.cat(embeddings_list)
            logger.debug(f"Created embeddings tensor of shape: {new_embeddings.shape}")
            
            # If this is the first class, initialize the model
            if self.model.index_embeddings is None:
                logger.info("Initializing first class")
                self.model.add_index_data(new_embeddings, [class_name] * len(image_bytes_list))
            else:
                logger.info("Adding to existing classes")
                # Get current state for debugging
                logger.debug(f"Current index embeddings shape: {self.model.index_embeddings.shape}")
                logger.debug(f"Current class labels: {self.model.class_labels}")
                
                # Combine with existing embeddings
                combined_embeddings = torch.cat([self.model.index_embeddings, new_embeddings])
                logger.debug(f"Combined embeddings shape: {combined_embeddings.shape}")
                
                # Get current labels and add new ones
                current_labels = [self.model.idx_to_classes[idx.item()] for idx in self.model.class_labels]
                combined_labels = current_labels + [class_name] * len(image_bytes_list)
                logger.debug(f"Combined labels: {combined_labels}")
                
                # Update model with all data
                try:
                    self.model.add_index_data(combined_embeddings, combined_labels)
                except Exception as e:
                    logger.error(f"Error in add_index_data: {str(e)}")
                    logger.error(traceback.format_exc())
                    raise ValueError(f"Error updating model: {str(e)}")
                
            logger.info(f"Successfully added class '{class_name}'")
            return {
                "status": "success",
                "message": f"Added class '{class_name}' with {len(image_bytes_list)} examples",
                "num_classes": self.model.num_classes,
                "available_classes": list(self.model.classes_to_idx.keys())
            }
        except Exception as e:
            logger.error(f"Error adding class: {str(e)}")
            logger.error(traceback.format_exc())
            raise ValueError(f"Error adding class '{class_name}': {str(e)}")
            
    def update_class(self, class_name: str, image_bytes_list: List[bytes], append: bool = False) -> Dict:
        """Update examples for an existing class."""
        if class_name not in self.model.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found")
            
        embeddings_list = []
        for image_bytes in image_bytes_list:
            embedding = self.get_embedding(image_bytes)
            embeddings_list.append(embedding)
            
        new_embeddings = torch.cat(embeddings_list)
        self.model.update_class_embeddings(class_name, new_embeddings, append=append)
        
        return {
            "status": "success",
            "message": f"{'Added' if append else 'Updated'} {len(image_bytes_list)} examples for class '{class_name}'",
            "num_examples": len(self.model.index_embeddings)
        }
        
    def remove_class(self, class_name: str) -> Dict:
        """Remove a class and all its examples."""
        if class_name not in self.model.classes_to_idx:
            raise ValueError(f"Class '{class_name}' not found")
            
        self.model.remove_class(class_name)
        
        return {
            "status": "success",
            "message": f"Removed class '{class_name}'",
            "num_classes": self.model.num_classes,
            "available_classes": list(self.model.classes_to_idx.keys())
        }
        
    def remove_examples(self, indices: List[int]) -> Dict:
        """Remove specific examples by their indices."""
        if not self.model.index_embeddings is None:
            if max(indices) >= len(self.model.index_embeddings):
                raise ValueError("Invalid index provided")
                
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