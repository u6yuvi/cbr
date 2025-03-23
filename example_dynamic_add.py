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
    
    # Step 1: Initial setup with cats and dogs
    print("Step 1: Initial Setup (Cats and Dogs)")
    print("-" * 50)
    
    # Process initial cat and dog images
    cat_path = "index_images/cat/cat1.jpg"
    dog_path = "index_images/dog/dog1.jpg"
    
    with torch.no_grad():
        # Get embeddings for initial images
        cat_img = transform(Image.open(cat_path).convert('RGB')).unsqueeze(0)
        dog_img = transform(Image.open(dog_path).convert('RGB')).unsqueeze(0)
        
        cat_embedding = model.get_embedding(cat_img)
        dog_embedding = model.get_embedding(dog_img)
        
        # Add initial index data
        initial_embeddings = torch.cat([cat_embedding, dog_embedding])
        initial_labels = ['cat', 'dog']
        model.add_index_data(initial_embeddings, initial_labels)
    
    # Test initial classification
    print("\nTesting initial classification:")
    process_image(model, "test_images/dog.jpg", transform)
    
    # Step 2: Add a new cat example
    print("\nStep 2: Adding New Cat Example")
    print("-" * 50)
    
    new_cat_path = "index_images/cat/cat2.jpg"
    with torch.no_grad():
        new_cat_img = transform(Image.open(new_cat_path).convert('RGB')).unsqueeze(0)
        new_cat_embedding = model.get_embedding(new_cat_img)
        model.update_class_embeddings('cat', new_cat_embedding, append=True)
    
    print("\nTesting after adding new cat example:")
    process_image(model, "test_images/cat2.jpg", transform)
    
    # Step 3: Add a completely new class (bird)
    # print("\nStep 3: Adding New Class (Bird)")
    # print("-" * 50)
    
    # # Create bird directory and download a sample bird image
    # os.makedirs("index_images/bird", exist_ok=True)
    # bird_path = "index_images/bird/bird1.jpg"
    
    # # For demonstration, we'll use the dog image as a bird (in real use, you'd have actual bird images)
    # with torch.no_grad():
    #     bird_img = transform(Image.open(dog_path).convert('RGB')).unsqueeze(0)
    #     bird_embedding = model.get_embedding(bird_img)
        
    #     # Combine with existing embeddings
    #     combined_embeddings = torch.cat([model.index_embeddings, bird_embedding])
    #     combined_labels = model.class_labels.tolist() + [len(model.classes_to_idx)]
        
    #     # Update model with new class
    #     model.add_index_data(combined_embeddings, ['cat', 'dog', 'bird'])
    
    # print("\nTesting after adding bird class:")
    # process_image(model, "test_images/dog.jpg", transform)

if __name__ == "__main__":
    main() 