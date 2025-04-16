from typing import Dict, Any, List
from pathlib import Path
from ..utils.api_client import APIClient

class PredictionTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def predict_single(self, image_path: Path) -> Dict[str, Any]:
        """Make prediction on a single image"""
        with open(image_path, "rb") as f:
            files = {"file": f}
            return self.api_client.post("/predict", files=files)
    
    def predict_batch(self, image_paths: List[Path]) -> List[Dict[str, Any]]:
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)
        return results 