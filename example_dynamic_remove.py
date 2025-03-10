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

def process_image(model, image_path, transform):
    print(f"\nProcessing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get predictions
        class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = model.idx_to_classes[class_idx]
        confidence = probabilities[0, class_idx].item()
        
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        
        print("\nAll class probabilities:")
        for idx, prob in enumerate(probabilities[0]):
            class_name = model.idx_to_classes[idx]
            print(f"{class_name}: {prob.item():.2%}")

def main():
    transform = setup_transforms()
    model = ClassificationByRetrieval()
    model.eval()
    
    # Step 1: Initial setup with multiple examples
    print("Step 1: Initial Setup (Multiple Examples)")
    print("-" * 50)
    
    # Process initial images
    cat1_path = "index_images/cat/cat1.jpg"
    cat2_path = "index_images/cat/cat2.jpg"
    dog1_path = "index_images/dog/dog1.jpg"
    dog2_path = "index_images/dog/dog2.jpg"
    
    with torch.no_grad():
        # Get embeddings for all initial images
        cat1_img = transform(Image.open(cat1_path).convert('RGB')).unsqueeze(0)
        cat2_img = transform(Image.open(cat2_path).convert('RGB')).unsqueeze(0)
        dog1_img = transform(Image.open(dog1_path).convert('RGB')).unsqueeze(0)
        dog2_img = transform(Image.open(dog2_path).convert('RGB')).unsqueeze(0)
        
        # Get embeddings
        cat1_emb = model.get_embedding(cat1_img)
        cat2_emb = model.get_embedding(cat2_img)
        dog1_emb = model.get_embedding(dog1_img)
        dog2_emb = model.get_embedding(dog2_img)
        
        # Add initial index data
        initial_embeddings = torch.cat([cat1_emb, cat2_emb, dog1_emb, dog2_emb])
        initial_labels = ['cat', 'cat', 'dog', 'dog']
        model.add_index_data(initial_embeddings, initial_labels)
    
    print("\nInitial model state:")
    print(f"Number of classes: {model.num_classes}")
    print(f"Number of examples: {len(model.index_embeddings)}")
    print(f"Available classes: {list(model.classes_to_idx.keys())}")
    
    # Test initial classification
    print("\nTesting initial classification:")
    process_image(model, "test_images/dog.jpg", transform)
    
    # Step 2: Remove specific examples
    print("\nStep 2: Removing Specific Examples")
    print("-" * 50)
    
    # Remove the first example of each class (indices 0 and 2)
    model.remove_examples([0, 2])
    
    print("\nModel state after removing examples:")
    print(f"Number of classes: {model.num_classes}")
    print(f"Number of examples: {len(model.index_embeddings)}")
    print(f"Available classes: {list(model.classes_to_idx.keys())}")
    
    print("\nTesting after removing examples:")
    process_image(model, "test_images/dog.jpg", transform)
    
    # Step 3: Remove entire class
    print("\nStep 3: Removing Entire Class")
    print("-" * 50)
    
    # Remove the 'cat' class
    model.remove_class('cat')
    
    print("\nModel state after removing cat class:")
    print(f"Number of classes: {model.num_classes}")
    print(f"Number of examples: {len(model.index_embeddings)}")
    print(f"Available classes: {list(model.classes_to_idx.keys())}")
    
    print("\nTesting after removing cat class:")
    process_image(model, "test_images/dog.jpg", transform)

if __name__ == "__main__":
    main() 