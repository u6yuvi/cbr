import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path

def setup_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_category_images(category_path, transform):
    """
    Load all images from a category directory
    Returns: List of transformed image tensors
    """
    images = []
    if not os.path.exists(category_path):
        print(f"Warning: Directory {category_path} does not exist")
        return images
    
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    for img_path in Path(category_path).glob('*'):
        if img_path.suffix.lower() in valid_extensions:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                images.append(image_tensor)
                print(f"Loaded: {img_path}")
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    return images

def analyze_dimensions(model, image_path, transform):
    print(f"\nAnalyzing dimensions for image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    print(f"\nInput image tensor shape: {image_tensor.shape}")  # Should be [1, 3, 224, 224]
    
    with torch.no_grad():
        # 1. Get embedding through the backbone
        embedding = model.get_embedding(image_tensor)
        print(f"Embedding shape: {embedding.shape}")  # Usually [1, embedding_dim]
        
        # 2. Calculate distances to all index embeddings
        print(f"Index embeddings shape: {model.index_embeddings.shape}")  # [num_index_images, embedding_dim]
        distances = torch.cdist(embedding, model.index_embeddings)
        print(f"Distances tensor shape: {distances.shape}")  # [1, num_index_images]
        
        # 3. Get logits (negative distances)
        logits = -distances
        print(f"Logits tensor shape: {logits.shape}")  # [1, num_index_images]
        
        # 4. Apply softmax
        probabilities = torch.softmax(logits, dim=1)
        print(f"Probabilities tensor shape: {probabilities.shape}")  # [1, num_index_images]
        
        # Print actual values for verification
        print("\nDetailed probability distribution:")
        # Use class_labels instead of idx_to_classes for mapping
        unique_labels = list(set(model.class_labels))
        for label in unique_labels:
            # Find all indices for this class
            indices = [i for i, x in enumerate(model.class_labels) if x == label]
            # Sum probabilities for all instances of this class
            class_prob = sum(probabilities[0, idx].item() for idx in indices)
            print(f"{label}: {class_prob:.2%}")

def main():
    transform = setup_transforms()
    model = ClassificationByRetrieval()
    model.eval()
    
    # Load all images from index_images directory
    index_dir = "index_images"
    all_embeddings = []
    all_labels = []
    
    print("Loading images from all categories:")
    print("-" * 50)
    
    # Get all category directories
    categories = [d for d in os.listdir(index_dir) 
                 if os.path.isdir(os.path.join(index_dir, d))]
    
    with torch.no_grad():
        for category in categories:
            category_path = os.path.join(index_dir, category)
            print(f"\nProcessing category: {category}")
            
            # Load all images for this category
            category_images = load_category_images(category_path, transform)
            
            if not category_images:
                continue
                
            # Get embeddings for all images in this category
            for img_tensor in category_images:
                embedding = model.get_embedding(img_tensor)
                all_embeddings.append(embedding)
                all_labels.append(category)
    
    if all_embeddings:
        # Combine all embeddings
        combined_embeddings = torch.cat(all_embeddings)
        print(f"\nTotal number of images loaded: {len(all_labels)}")
        print(f"Combined embeddings shape: {combined_embeddings.shape}")
        
        # Add all data to the model
        model.add_index_data(combined_embeddings, all_labels)
        
        # Analyze dimensions during inference
        print("\n" + "="*50)
        print("Analyzing forward pass dimensions:")
        
        # Use the first image from any category for testing
        test_image_path = os.path.join(index_dir, categories[0], 
                                     os.listdir(os.path.join(index_dir, categories[0]))[0])
        analyze_dimensions(model, test_image_path, transform)
    else:
        print("No images were loaded. Please check the index_images directory.")

if __name__ == "__main__":
    main() 