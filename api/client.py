"""
Classification by Retrieval (CbR) API Client

This module provides a Python client for interacting with the CbR FastAPI service.
It includes methods for managing classes, updating examples, and making predictions.
"""

import requests
from pathlib import Path
from typing import List, Dict, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CbRClient:
    """Client for interacting with the Classification by Retrieval (CbR) API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the CbR client.
        
        Args:
            base_url: Base URL of the CbR API service
        """
        self.base_url = base_url.rstrip('/')
        
    def _handle_response(self, response: requests.Response) -> Dict:
        """
        Handle API response and potential errors.
        
        Args:
            response: Response object from requests
            
        Returns:
            Dict containing the response data
            
        Raises:
            requests.exceptions.HTTPError: If the response indicates an error
        """
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_msg = f"API request failed: {str(e)}"
            try:
                error_detail = response.json().get('detail', '')
                if error_detail:
                    error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(error_msg)
            raise
            
    def get_model_info(self) -> Dict:
        """
        Get current model state information.
        
        Returns:
            Dict containing model information including:
            - num_classes: Number of classes in the model
            - num_examples: Total number of examples
            - available_classes: List of class names
            - examples_per_class: Dict mapping class names to number of examples
        """
        url = f"{self.base_url}/model/info"
        response = requests.get(url)
        return self._handle_response(response)
        
    def add_class(self, class_name: str, image_paths: List[Union[str, Path]]) -> Dict:
        """
        Add a new class with example images.
        
        Args:
            class_name: Name of the class to add
            image_paths: List of paths to example images
            
        Returns:
            Dict containing operation status and updated model information
            
        Raises:
            FileNotFoundError: If any image file cannot be found
            ValueError: If no image paths are provided
        """
        if not image_paths:
            raise ValueError("At least one image path must be provided")
            
        url = f"{self.base_url}/class/add/{class_name}"
        files = []
        
        try:
            for path in image_paths:
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                files.append(("files", open(path, "rb")))
                
            response = requests.post(url, files=files)
            return self._handle_response(response)
            
        finally:
            # Ensure all opened files are closed
            for _, file in files:
                file.close()
                
    def update_class(self, class_name: str, image_paths: List[Union[str, Path]], append: bool = True) -> Dict:
        """
        Update or append examples to an existing class.
        
        Args:
            class_name: Name of the class to update
            image_paths: List of paths to example images
            append: If True, append new examples; if False, replace existing ones
            
        Returns:
            Dict containing operation status and number of examples
        """
        url = f"{self.base_url}/class/update/{class_name}"
        files = []
        
        try:
            for path in image_paths:
                path = Path(path)
                if not path.exists():
                    raise FileNotFoundError(f"Image file not found: {path}")
                files.append(("files", open(path, "rb")))
                
            data = {"append": "true" if append else "false"}
            response = requests.post(url, files=files, data=data)
            return self._handle_response(response)
            
        finally:
            for _, file in files:
                file.close()
                
    def predict(self, image_path: Union[str, Path]) -> Dict:
        """
        Classify an image.
        
        Args:
            image_path: Path to the image file to classify
            
        Returns:
            Dict containing:
            - predicted_class: Name of the predicted class
            - confidence: Confidence score for the prediction
            - class_probabilities: Dict mapping class names to probabilities
        """
        url = f"{self.base_url}/predict"
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
            
        with open(path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            return self._handle_response(response)
            
    def remove_class(self, class_name: str) -> Dict:
        """
        Remove a class and all its examples.
        
        Args:
            class_name: Name of the class to remove
            
        Returns:
            Dict containing operation status and updated model information
        """
        url = f"{self.base_url}/class/{class_name}"
        response = requests.delete(url)
        return self._handle_response(response)
        
    def remove_examples(self, indices: List[int]) -> Dict:
        """
        Remove specific examples by their indices.
        
        Args:
            indices: List of indices to remove
            
        Returns:
            Dict containing operation status and number of remaining examples
        """
        url = f"{self.base_url}/examples"
        response = requests.delete(url, json=indices)
        return self._handle_response(response)


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = CbRClient()
    
    try:
        # Print initial model info
        print("Initial model state:")
        print(client.get_model_info())
        
        # Add cat class
        print("\nAdding cat class:")
        cat_images = ["index_images/cat/cat1.jpg", "index_images/cat/cat2.jpg"]
        print(client.add_class("cat", cat_images))
        
        # Add dog class
        print("\nAdding dog class:")
        dog_images = ["index_images/dog/dog1.jpg"]
        print(client.add_class("dog", dog_images))
        
        # Update dog class with more examples
        print("\nAdding more dog examples:")
        more_dogs = ["index_images/dog/dog2.jpg", "index_images/dog/dog3.jpg"]
        print(client.update_class("dog", more_dogs, append=True))
        
        # Make a prediction
        print("\nMaking prediction:")
        print(client.predict("test_images/test_dog.jpg"))
        
        # Get final model info
        print("\nFinal model state:")
        print(client.get_model_info())
        
    except requests.exceptions.ConnectionError:
        logger.error("Failed to connect to the API server. Make sure it's running at %s", client.base_url)
    except Exception as e:
        logger.error("An error occurred: %s", str(e)) 