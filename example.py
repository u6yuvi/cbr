import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os

def main():
    # Initialize the CbR model
    model = ClassificationByRetrieval()
    model.eval()
    
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Example: Create some dummy index data
    # In practice, you would load your real training images and extract embeddings
    dummy_embeddings = torch.randn(10, 512)  # 10 examples, 512-dim embeddings (ResNet18)
    dummy_labels = ['cat'] * 5 + ['dog'] * 5  # 5 cats, 5 dogs
    
    # Add index data to the model
    model.add_index_data(dummy_embeddings, dummy_labels)
    
    def process_image(image_path):
        print(f"Processing image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
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
            for idx, prob in enumerate(probabilities[0]):
                class_name = model.idx_to_classes[idx]
                print(f"{class_name}: {prob.item():.2%}")

    # Create a test_images directory if it doesn't exist
    if not os.path.exists('test_images'):
        os.makedirs('test_images')
        print("Created test_images directory. Please place a dog image in this directory.")
        return

    # Look for images in the test_images directory
    test_dir = 'test_images'
    image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print("No images found in test_images directory. Please add some images and try again.")
        return
        
    # Process each image in the directory
    for image_file in image_files:
        image_path = os.path.join(test_dir, image_file)
        process_image(image_path)

if __name__ == '__main__':
    main() 