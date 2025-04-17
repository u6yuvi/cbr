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

class ModelTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
        # Match server's image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ])
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get the MIME type for a file based on its extension"""
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'
    
    def _validate_and_convert_image(self, img_path: Path) -> bytes:
        """Validate image and convert to RGB JPEG format with proper sizing"""
        try:
            # Open and validate image
            with Image.open(img_path) as img:
                # Force load the image to catch any corruption issues
                img.load()
                
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
                
                # Save as high-quality JPEG to bytes buffer
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95, optimize=True)
                return buffer.getvalue()
                
        except UnidentifiedImageError:
            logger.error(f"File {img_path} is not a valid image")
            raise ValueError(f"File {img_path} is not a valid image")
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            raise ValueError(f"Error processing image {img_path}: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information for current tenant"""
        return self.api_client.get("/model/info")
    
    def add_class(self, class_name: str, images: List[Union[str, Path]]) -> Dict[str, Any]:
        """Add a new class with example images"""
        logger.info(f"Adding class '{class_name}' with {len(images)} images")
        try:
            # Process all images
            files = []
            for img in images:
                # Convert string path to Path object if needed
                img_path = Path(img) if isinstance(img, str) else img
                
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
                    
                    # Add to files list with proper filename and content type
                    filename = os.path.basename(str(img_path))
                    files.append(("files", (filename, buffer, 'image/jpeg')))
            
            logger.info("Sending add_class request to API")
            result = self.api_client.post(f"/class/add/{class_name}", files=files)
            logger.info(f"Successfully added class '{class_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to add class '{class_name}': {str(e)}")
            raise
    
    def update_class(self, class_name: str, images: List[Union[str, Path]], append: bool = False) -> Dict[str, Any]:
        """Update an existing class with new example images"""
        logger.info(f"Updating class '{class_name}' with {len(images)} images")
        try:
            # Process all images
            files = []
            for img in images:
                # Convert string path to Path object if needed
                img_path = Path(img) if isinstance(img, str) else img
                
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
                    
                    # Add to files list with proper filename and content type
                    filename = os.path.basename(str(img_path))
                    files.append(("files", (filename, buffer, 'image/jpeg')))
            
            # Add append parameter to form data
            data = {'append': str(append).lower()}
            
            logger.info("Sending update_class request to API")
            result = self.api_client.post(f"/class/update/{class_name}", data=data, files=files)
            logger.info(f"Successfully updated class '{class_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Failed to update class '{class_name}': {str(e)}")
            raise
    
    def remove_class(self, class_name: str) -> Dict[str, Any]:
        """Remove a class"""
        return self.api_client.delete(f"/class/{class_name}")
    
    def get_class_images(self, class_name: str) -> Dict[str, Any]:
        """Get all images for a specific class"""
        return self.api_client.get(f"/class/{class_name}/images") 