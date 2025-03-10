import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path

def load_index_images(index_dir: str, transform):
    """
    Load and process index images from directory.
    Directory structure should be:
    index_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    """
    embeddings_list = []
    labels_list = []
    model = ClassificationByRetrieval()
    model.eval()
    
    # Process each class directory
    for class_dir in Path(index_dir).iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Process each image in the class directory
        for img_path in class_dir.glob("*.jpg"):
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # Get embedding
                with torch.no_grad():
                    embedding = model.get_embedding(image_tensor)
                
                embeddings_list.append(embedding)
                labels_list.append(class_name)
                print(f"  Processed: {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    # Stack all embeddings
    if embeddings_list:
        embeddings = torch.cat(embeddings_list, dim=0)
        return embeddings, labels_list
    else:
        raise ValueError("No valid images found in the index directory")

def main():
    # Define image transforms (same for index and query images)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize the CbR model
    model = ClassificationByRetrieval()
    model.eval()
    
    try:
        # Load and process index images
        print("Loading index images...")
        index_embeddings, index_labels = load_index_images('index_images', transform)
        
        # Add index data to the model
        print("Adding index data to model...")
        model.add_index_data(index_embeddings, index_labels)
        print(f"Added {len(index_labels)} images to index")
        
    except Exception as e:
        print(f"Error setting up index data: {e}")
        return
    
    def process_image(image_path):
        print(f"\nProcessing query image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
            
            # Get predicted class and all class probabilities
            class_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = model.idx_to_classes[class_idx]
            confidence = probabilities[0, class_idx].item()
            
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
            
            # Print all class probabilities
            print("\nClass probabilities:")
            for idx, prob in enumerate(probabilities[0]):
                class_name = model.idx_to_classes[idx]
                print(f"{class_name}: {prob.item():.2%}")

    # Process test images
    test_dir = 'test_images'
    if not os.path.exists(test_dir):
        print(f"\nNo test directory found at {test_dir}")
        return
        
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"\nNo images found in {test_dir}")
        return
        
    print("\nProcessing test images...")
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        process_image(image_path)

if __name__ == '__main__':
    main() 