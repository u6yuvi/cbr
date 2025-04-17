from typing import Dict, Any, List, Union
from pathlib import Path
from ..utils.api_client import APIClient
import mimetypes
import logging
from PIL import Image, UnidentifiedImageError
import io
from torchvision import transforms
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictionTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        # Match server's image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    
    def predict_single(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Make prediction on a single image"""
        logger.info(f"Processing image for prediction: {image_path}")
        try:
            # Convert string path to Path object if needed
            img_path = Path(image_path) if isinstance(image_path, str) else image_path
            
            # Preprocess image
            with Image.open(img_path) as img:
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA', 'P'):
                    # Create white background for transparency
                    if 'transparency' in img.info or img.mode in ('RGBA', 'LA'):
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        if img.mode == 'P':
                            img = img.convert('RGBA')
                        # Alpha composite onto white background
                        background.paste(img, mask=img.split()[-1])
                        img = background
                    else:
                        img = img.convert('RGB')
                elif img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply size transformations
                img = self.transform(img)
                
                # Save to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95, optimize=True)
                buffer.seek(0)
                
                # Send processed image
                files = {'file': (f"{os.path.basename(str(img_path))}", buffer, 'image/jpeg')}
                
                logger.info("Sending prediction request to API")
                result = self.api_client.post("/predict", files=files)
                logger.info("Prediction request successful")
                return result
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {str(e)}")
            raise
    
    def predict_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """Make predictions on multiple images"""
        results = []
        for image_path in image_paths:
            result = self.predict_single(image_path)
            results.append(result)
        return results 