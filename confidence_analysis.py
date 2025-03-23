import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval

def setup_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def analyze_confidence(model, image_path, transform, temperature=1.0):
    """
    Analyze confidence scores using different methods
    Args:
        temperature: Controls the sharpness of the softmax distribution.
                    Higher values make distribution more uniform,
                    Lower values make it more peaked.
    """
    print(f"\nAnalyzing image: {image_path}")
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # 1. Get raw logits and embeddings
        logits = model(image_tensor)
        embedding = model.get_embedding(image_tensor)
        
        # 2. Calculate distances to all index embeddings
        distances = torch.cdist(embedding, model.index_embeddings)
        
        # 3. Calculate probabilities with different temperatures
        standard_probs = torch.softmax(logits, dim=1)
        scaled_probs = torch.softmax(logits / temperature, dim=1)
        
        # Print results
        print("\n1. Raw Logits (before softmax):")
        for idx, logit in enumerate(logits[0]):
            class_name = model.idx_to_classes[idx]
            print(f"{class_name}: {logit.item():.4f}")
            
        print("\n2. Embedding Distances:")
        for idx, dist in enumerate(distances[0]):
            class_name = model.idx_to_classes[idx]
            print(f"Distance to {class_name}: {dist.item():.4f}")
            
        print("\n3. Standard Probabilities (temperature=1.0):")
        for idx, prob in enumerate(standard_probs[0]):
            class_name = model.idx_to_classes[idx]
            print(f"{class_name}: {prob.item():.2%}")
            
        print(f"\n4. Scaled Probabilities (temperature={temperature}):")
        for idx, prob in enumerate(scaled_probs[0]):
            class_name = model.idx_to_classes[idx]
            print(f"{class_name}: {prob.item():.2%}")

def main():
    transform = setup_transforms()
    model = ClassificationByRetrieval()
    model.eval()
    
    # Initial setup with cats and dogs
    cat_path = "index_images/cat/cat1.jpg"
    dog_path = "index_images/dog/dog1.jpg"
    
    with torch.no_grad():
        # Add initial index data
        cat_img = transform(Image.open(cat_path).convert('RGB')).unsqueeze(0)
        dog_img = transform(Image.open(dog_path).convert('RGB')).unsqueeze(0)
        
        cat_embedding = model.get_embedding(cat_img)
        dog_embedding = model.get_embedding(dog_img)
        
        initial_embeddings = torch.cat([cat_embedding, dog_embedding])
        initial_labels = ['cat', 'dog']
        model.add_index_data(initial_embeddings, initial_labels)
    
    # Test with different temperatures
    print("\nAnalyzing with default temperature (1.0)")
    analyze_confidence(model, cat_path, transform)
    
    print("\nAnalyzing with lower temperature (0.5) - Should make predictions more confident")
    analyze_confidence(model, cat_path, transform, temperature=0.5)
    
    print("\nAnalyzing with higher temperature (2.0) - Should make predictions less confident")
    analyze_confidence(model, cat_path, transform, temperature=2.0)

if __name__ == "__main__":
    main() 