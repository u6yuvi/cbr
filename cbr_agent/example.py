import sys
import os
from pathlib import Path
import requests
import shutil
from PIL import Image
import io
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cbr_agent.agents.cbr_agent import CBRAgent

def download_image(url: str, save_path: Path) -> Path:
    """Download an image from URL and save it locally"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    # Load image using PIL to ensure it's valid
    img = Image.open(io.BytesIO(response.content))
    
    # Convert to RGB if needed (handles PNG with transparency)
    if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
        img = img.convert('RGB')
    
    # Save as JPEG with proper extension
    save_path = save_path.with_suffix('.jpg')
    img.save(save_path, 'JPEG', quality=95)
    
    return save_path

def main():
    try:
        # Initialize agent
        agent = CBRAgent()
        
        # Create directories
        image_dir = Path("downloaded_images")
        image_dir.mkdir(exist_ok=True)
        
        # Example workflow
        print("\n1. Creating a new tenant...")
        result = agent.run("Create a new tenant called 'image-classifier'")
        if 'result' in result and 'tenant_id' in result['result']:
            tenant_id = result['result']['tenant_id']
            print(f"Created tenant: {tenant_id}")
            
            # Save tenant ID to file
            with open(image_dir / "tenant_id.txt", "w") as f:
                f.write(tenant_id)
        else:
            print(result['response'])
            return
            
        # Ensure we're using the correct tenant
        agent.run({"tool": "switch_tenant", "args": {"tenant_id": tenant_id}})
        
        print("\n2. Training the model with cat images...")
        # Use local test images
        cat_paths = [str(Path("test_images/cat2.jpg"))]
        
        # Train with cat images
        result = agent.run({
            "tool": "train_model",
            "args": {
                "class_name": "cat",
                "images": cat_paths
            }
        })
        if 'result' in result:
            print("Successfully trained model with cat images:")
            print(f"Status: {result['result'].get('status')}")
            print(f"Message: {result['result'].get('message')}")
        else:
            print(f"Error: {result.get('error', result['response'])}")
        
        print("\n3. Training the model with dog images...")
        # Use local test images
        dog_paths = [str(Path("test_images/dog.jpg"))]
        
        # Train with dog images
        result = agent.run({
            "tool": "train_model",
            "args": {
                "class_name": "dog",
                "images": dog_paths
            }
        })
        if 'result' in result:
            print("Successfully trained model with dog images:")
            print(f"Status: {result['result'].get('status')}")
            print(f"Message: {result['result'].get('message')}")
        else:
            print(f"Error: {result.get('error', result['response'])}")
        
        print("\n4. Making predictions...")
        # Use a local test image
        test_path = str(Path("test_images/dog.jpg"))
        
        # Make prediction
        result = agent.run({
            "tool": "predict_image",
            "args": {
                "image": test_path
            }
        })
        if 'result' in result:
            prediction = result['result']
            print("\nPrediction Results:")
            print(f"Predicted Class: {prediction['predicted_class']}")
            print(f"Confidence: {prediction['confidence']:.2%}")
            print("\nClass Probabilities:")
            for class_name, prob in prediction['class_probabilities'].items():
                print(f"{class_name}: {prob:.2%}")
        else:
            print(f"Error: {result.get('error', result['response'])}")
        
        # Clean up downloaded images
        shutil.rmtree(image_dir)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 