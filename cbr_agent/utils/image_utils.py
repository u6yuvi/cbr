import requests
import shutil
from pathlib import Path
from typing import List, Union
from urllib.parse import urlparse

class ImageDownloader:
    def __init__(self, download_dir: Union[str, Path] = "downloaded_images"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
    
    def download_image(self, url: str) -> Path:
        """Download an image from URL and save it locally"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Extract filename from URL or generate one
        filename = Path(urlparse(url).path).name
        if not filename:
            filename = f"image_{hash(url)}.jpg"
        
        save_path = self.download_dir / filename
        with open(save_path, 'wb') as f:
            response.raw.decode_content = True
            shutil.copyfileobj(response.raw, f)
        
        return save_path
    
    def download_images(self, urls: List[str]) -> List[Path]:
        """Download multiple images from URLs"""
        paths = []
        for url in urls:
            paths.append(self.download_image(url))
        return paths
    
    def cleanup(self):
        """Remove the download directory and all its contents"""
        shutil.rmtree(self.download_dir) 