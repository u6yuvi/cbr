import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path
import json
import time
import numpy as np

def get_model_size(model):
    """Calculate model size in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for name, buffer in model.named_buffers():
        size_bytes = buffer.nelement() * buffer.element_size()
        size_mb = size_bytes / 1024 / 1024
        buffer_size += size_bytes
        print(f"Buffer {name}: {size_mb:.4f} MB, shape: {buffer.shape if hasattr(buffer, 'shape') else 'None'}")
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

def load_and_transform_image(image_path, transform):
    """Load and transform a single image"""
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def main():
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the CbR model
    model = ClassificationByRetrieval()
    model.eval()
    
    # Example image paths
    cat_image = 'index_images/cat/cat1.jpg'
    dog_image = 'index_images/dog/dog1.jpg'
    
    print("Loading example images...")
    try:
        # Load and process example images
        cat_tensor = load_and_transform_image(cat_image, transform)
        dog_tensor = load_and_transform_image(dog_image, transform)
        
        # Get embeddings
        with torch.no_grad():
            cat_embedding = model.get_embedding(cat_tensor)
            dog_embedding = model.get_embedding(dog_tensor)
            
            # Print embedding shapes
            print(f"Cat embedding shape: {cat_embedding.shape}")
            print(f"Dog embedding shape: {dog_embedding.shape}")
        
        # Create embeddings with more data to make buffer size visible
        # Use 50 copies of each embedding to simulate a larger dataset
        print("\nCreating larger dataset for better size measurement...")
        cat_embeddings = torch.cat([cat_embedding] * 50, dim=0)
        dog_embeddings = torch.cat([dog_embedding] * 50, dim=0)
        all_embeddings = torch.cat([cat_embeddings, dog_embeddings], dim=0)
        all_labels = ['cat'] * 50 + ['dog'] * 50
        
        print(f"Total embeddings: {len(all_labels)}")
        print(f"Embeddings tensor shape: {all_embeddings.shape}")
        
        # Add index data to the model
        print("Adding example data to model...")
        model.add_index_data(all_embeddings, all_labels)
        
        # Calculate model size with register buffers
        print("\nCalculating model sizes...")
        size_with_buffers = get_model_size(model)
        print(f"Total model size with register buffers: {size_with_buffers:.2f} MB")
        
        # Save model with register buffers
        print("\nSaving model with register buffers...")
        save_path = 'cbr_model_with_buffers.pt'
        
        # Create clean version of the model's state dict
        # that properly handles the buffer issue
        model_save_dict = {
            'state_dict': model.state_dict(),
            'idx_to_classes': model.idx_to_classes,
            'classes_to_idx': model.classes_to_idx,
            'num_classes': model.num_classes,
            'embedding_dim': model.embedding_dim
        }
        torch.save(model_save_dict, save_path)
        
        # Create a new model instance without buffers
        model_no_buffers = ClassificationByRetrieval()
        model_no_buffers.eval()
        size_without_buffers = get_model_size(model_no_buffers)
        print(f"Model size without register buffers: {size_without_buffers:.2f} MB")
        print(f"Buffer size: {size_with_buffers - size_without_buffers:.2f} MB")
        
        # Load the saved model
        print("\nLoading saved model...")
        loaded_model = ClassificationByRetrieval()
        loaded_model.eval()
        
        # Load the saved dictionary
        saved_dict = torch.load(save_path)
        
        # First set the model attributes
        loaded_model.idx_to_classes = saved_dict['idx_to_classes']
        loaded_model.classes_to_idx = saved_dict['classes_to_idx']
        loaded_model.num_classes = saved_dict['num_classes']
        loaded_model.embedding_dim = saved_dict['embedding_dim']
        
        # Now load the state dict, properly handling buffers
        try:
            # Step 1: Try direct loading (might fail due to buffer differences)
            loaded_model.load_state_dict(saved_dict['state_dict'])
        except Exception as e:
            print(f"Standard loading failed: {e}")
            
            # Step 2: Manual loading with buffer initialization
            state_dict = saved_dict['state_dict']
            model_state = loaded_model.state_dict()
            
            # First load parameters (non-buffers)
            for name, param in state_dict.items():
                if name in model_state and 'index_embeddings' not in name and 'class_labels' not in name:
                    model_state[name].copy_(param)
                    
            # Then manually register the buffers with the right values
            if 'index_embeddings' in state_dict:
                embeddings = state_dict['index_embeddings']
                loaded_model.register_buffer('index_embeddings', embeddings)
                
            if 'class_labels' in state_dict:
                labels = state_dict['class_labels']
                loaded_model.register_buffer('class_labels', labels)
            
            print("Manually loaded model state with buffer handling")
        
        # Verify the buffers are properly loaded
        print("\nVerifying loaded model buffers:")
        for name, buffer in loaded_model.named_buffers():
            size_mb = buffer.nelement() * buffer.element_size() / 1024 / 1024
            print(f"Loaded buffer {name}: {size_mb:.4f} MB, shape: {buffer.shape if hasattr(buffer, 'shape') else 'None'}")
            
        # Test predictions on the same images
        print("\nTesting predictions with loaded model...")
        with torch.no_grad():
            # Test cat image
            cat_logits = loaded_model(cat_tensor)
            cat_probs = torch.softmax(cat_logits, dim=1)
            cat_pred = loaded_model.idx_to_classes[torch.argmax(cat_probs, dim=1).item()]
            cat_conf = cat_probs[0, torch.argmax(cat_probs, dim=1).item()].item()
            print(f"Cat image prediction: {cat_pred} (confidence: {cat_conf:.2%})")
            
            # Test dog image
            dog_logits = loaded_model(dog_tensor)
            dog_probs = torch.softmax(dog_logits, dim=1)
            dog_pred = loaded_model.idx_to_classes[torch.argmax(dog_probs, dim=1).item()]
            dog_conf = dog_probs[0, torch.argmax(dog_probs, dim=1).item()].item()
            print(f"Dog image prediction: {dog_pred} (confidence: {dog_conf:.2%})")
        
        # Save model info
        model_info = {
            'classes': loaded_model.idx_to_classes,
            'num_classes': loaded_model.num_classes,
            'embedding_dim': loaded_model.embedding_dim,
            'model_size_with_buffers_mb': size_with_buffers,
            'model_size_without_buffers_mb': size_without_buffers,
            'buffer_size_mb': size_with_buffers - size_without_buffers,
            'export_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open('model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
            
        print("\nModel info saved to model_info.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return

if __name__ == '__main__':
    main() 