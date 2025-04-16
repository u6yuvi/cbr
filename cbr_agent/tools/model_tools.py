from typing import Dict, Any, List
from pathlib import Path
from ..utils.api_client import APIClient

class ModelTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for current tenant"""
        return self.api_client.get("/model/info")
    
    def add_class(self, class_name: str, image_paths: List[Path]) -> Dict[str, Any]:
        """Add a new class with examples"""
        files = [("files", open(img, "rb")) for img in image_paths]
        try:
            return self.api_client.post(f"/class/add/{class_name}", files=files)
        finally:
            for _, f in files:
                f.close()
    
    def update_class(self, class_name: str, image_paths: List[Path], append: bool = True) -> Dict[str, Any]:
        """Update or append examples to a class"""
        files = [("files", open(img, "rb")) for img in image_paths]
        try:
            return self.api_client.post(
                f"/class/update/{class_name}",
                data={"append": str(append).lower()},
                files=files
            )
        finally:
            for _, f in files:
                f.close()
    
    def remove_class(self, class_name: str) -> Dict[str, Any]:
        """Remove a class"""
        return self.api_client.delete(f"/class/{class_name}")
    
    def get_class_images(self, class_name: str) -> Dict[str, Any]:
        """Get all images for a specific class"""
        return self.api_client.get(f"/class/{class_name}/images") 