import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path
from collections import defaultdict

def load_index_images(index_dir: str, transform, initial_only=False):
    """
    Load and process index images from directory.
    """
    embeddings_list = []
    labels_list = []
    model = ClassificationByRetrieval()
    model.eval()
    
    # First, get all unique classes
    classes = sorted([d.name for d in Path(index_dir).iterdir() if d.is_dir()])
    print(f"Found classes: {classes}")
    
    # Process each class directory
    for class_name in classes:
        class_dir = Path(index_dir) / class_name
        print(f"Processing class: {class_name}")
        
        # Get all image paths for this class
        image_paths = list(class_dir.glob("*.jpg"))
        if initial_only:
            image_paths = image_paths[:1]  # Take only first image if initial loading
        
        # Process each image in the class directory
        for img_path in image_paths:
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
        print(f"Created embeddings tensor of shape: {embeddings.shape}")
        print(f"Labels: {labels_list}")
        return embeddings, labels_list, model, classes
    else:
        raise ValueError("No valid images found in the index directory")

def process_image(model, image_path, transform):
    """Process a single image and get predictions"""
    print(f"\nProcessing image: {image_path}")
    
    print(f"Current model classes: {model.idx_to_classes}")
    print(f"Current index embeddings shape: {model.index_embeddings.shape}")
    print(f"Current class labels: {model.class_labels}")
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        # Get embedding
        embedding = model.get_embedding(image_tensor)
        print(f"Generated embedding of shape: {embedding.shape}")
        
        # Use model's forward pass directly instead of manual computation
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        
        # Get predicted class and confidence
        class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class = model.idx_to_classes[class_idx]
        confidence = probabilities[0, class_idx].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0],
            'embedding': embedding
        }

def main():
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    index_dir = 'index_images'
    
    try:
        # Load initial index (1 image per class)
        print("\nLoading initial index (1 image per class)...")
        initial_embeddings, initial_labels, model, classes = load_index_images(index_dir, transform, initial_only=True)
        
        # Add initial index data to the model
        print("Adding initial index data to model...")
        print(f"Initial embeddings shape: {initial_embeddings.shape}")
        print(f"Initial labels: {initial_labels}")
        
        model.add_index_data(initial_embeddings, initial_labels)
        print(f"Added {len(initial_labels)} images to initial index")
        print(f"Model classes after initialization: {model.idx_to_classes}")
        
        # Process remaining images progressively
        print("\nProcessing remaining images...")
        metrics = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        # Keep track of all embeddings and labels
        all_embeddings = initial_embeddings
        all_labels = initial_labels.copy()
        
        for class_name in classes:
            class_dir = Path(index_dir) / class_name
            # Skip the first image as it's already in the index
            image_paths = list(class_dir.glob("*.jpg"))[1:]
            
            print(f"\nProcessing remaining images for class: {class_name}")
            print(f"Found {len(image_paths)} additional images")
            
            for img_path in image_paths:
                try:
                    # Process image and get predictions
                    results = process_image(model, img_path, transform)
                    
                    # Update metrics
                    metrics[class_name]['total'] += 1
                    if results['predicted_class'] == class_name:
                        metrics[class_name]['correct'] += 1
                    
                    # Print predictions
                    print(f"True class: {class_name}")
                    print(f"Predicted class: {results['predicted_class']}")
                    print(f"Confidence: {results['confidence']:.2%}")
                    print("\nClass probabilities:")
                    for idx, prob in enumerate(results['probabilities']):
                        pred_class = model.idx_to_classes[idx]
                        print(f"{pred_class}: {prob.item():.2%}")
                    
                    # Add image to index after prediction
                    print("Adding new image to index...")
                    # Update all embeddings and labels
                    all_embeddings = torch.cat([all_embeddings, results['embedding']], dim=0)
                    all_labels.append(class_name)
                    # Update model with all data
                    model.add_index_data(all_embeddings, all_labels)
                    
                    print(f"Updated index size: {len(all_labels)} images")
                    
                    # Print current metrics
                    total_correct = sum(m['correct'] for m in metrics.values())
                    total_images = sum(m['total'] for m in metrics.values())
                    if total_images > 0:
                        print(f"\nCurrent overall accuracy: {total_correct/total_images:.2%}")
                        print("Per-class accuracies:")
                        for cn, m in metrics.items():
                            if m['total'] > 0:
                                acc = m['correct'] / m['total']
                                print(f"{cn}: {acc:.2%} ({m['correct']}/{m['total']})")
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    print(f"Current model state - classes: {model.idx_to_classes}")
                    print(f"Current index size: {len(all_labels)}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Print final metrics
        print("\nFinal Results:")
        total_correct = sum(m['correct'] for m in metrics.values())
        total_images = sum(m['total'] for m in metrics.values())
        
        if total_images > 0:
            print(f"\nOverall accuracy: {total_correct/total_images:.2%}")
            print("\nPer-class results:")
            for class_name, m in metrics.items():
                if m['total'] > 0:
                    accuracy = m['correct'] / m['total']
                    print(f"{class_name}: {accuracy:.2%} ({m['correct']}/{m['total']})")
    
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 