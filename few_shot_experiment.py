import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report
import json
from datetime import datetime

def load_index_images(index_dir: str, transform, samples_per_class: int = None):
    """
    Load and process index images from directory with configurable samples per class.
    Directory structure should be:
    index_dir/
        class1/
            image1.jpg
            image2.jpg
        class2/
            image1.jpg
            image2.jpg
    
    Args:
        index_dir: Directory containing class subdirectories
        transform: Image transforms to apply
        samples_per_class: Number of samples to load per class. If None, load all samples.
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
        
        # Get all image paths for this class
        image_paths = list(class_dir.glob("*.jpg"))
        
        # Randomly sample if samples_per_class is specified
        if samples_per_class is not None and samples_per_class < len(image_paths):
            image_paths = random.sample(image_paths, samples_per_class)
        
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
        return embeddings, labels_list
    else:
        raise ValueError("No valid images found in the index directory")

def evaluate_model(model, index_dir: str, transform, samples_per_class: int):
    """
    Evaluate the model using the remaining samples after few-shot learning.
    """
    true_labels = []
    predicted_labels = []
    
    # Process each class directory
    for class_dir in Path(index_dir).iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        print(f"\nEvaluating class: {class_name}")
        
        # Get all image paths for this class
        image_paths = list(class_dir.glob("*.jpg"))
        
        # Remove the samples used for training
        if samples_per_class is not None:
            image_paths = image_paths[samples_per_class:]
        
        # Process remaining images
        for img_path in image_paths:
            try:
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image).unsqueeze(0)
                
                # Get prediction
                with torch.no_grad():
                    logits = model(image_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    class_idx = torch.argmax(probabilities, dim=1).item()
                    predicted_class = model.idx_to_classes[class_idx]
                
                true_labels.append(class_name)
                predicted_labels.append(predicted_class)
                print(f"  Processed: {img_path.name}")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    return true_labels, predicted_labels

def save_metrics(true_labels, predicted_labels, samples_per_class, output_dir: str = "experiment_results"):
    """
    Save evaluation metrics to a JSON file and classification report to a TXT file.
    """
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )
    
    # Generate classification report
    report = classification_report(true_labels, predicted_labels)
    
    # Create metrics dictionary
    metrics = {
        "samples_per_class": samples_per_class,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "classification_report": report,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for consistent filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metrics to JSON file
    json_file = os.path.join(
        output_dir, 
        f"metrics_{samples_per_class}shots_{timestamp}.json"
    )
    
    with open(json_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save classification report to TXT file
    txt_file = os.path.join(
        output_dir,
        f"classification_report_{samples_per_class}shots_{timestamp}.txt"
    )
    
    with open(txt_file, 'w') as f:
        f.write(f"Few-Shot Learning Experiment Results\n")
        f.write(f"====================================\n\n")
        f.write(f"Samples per class: {samples_per_class}\n")
        f.write(f"Timestamp: {metrics['timestamp']}\n\n")
        f.write(f"Overall Metrics:\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write("==============================\n")
        f.write(report)
    
    print(f"\nMetrics saved to: {json_file}")
    print(f"Classification report saved to: {txt_file}")
    return metrics

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
    
    # Configuration
    index_dir = 'index_images'
    samples_per_class = 1  # Configure number of samples per class for few-shot learning
    
    try:
        # Load and process index images for few-shot learning
        print(f"Loading {samples_per_class} samples per class for few-shot learning...")
        index_embeddings, index_labels = load_index_images(index_dir, transform, samples_per_class)
        
        # Add index data to the model
        print("Adding index data to model...")
        model.add_index_data(index_embeddings, index_labels)
        print(f"Added {len(index_labels)} images to index")
        
        # Evaluate on remaining samples
        print("\nEvaluating model on remaining samples...")
        true_labels, predicted_labels = evaluate_model(model, index_dir, transform, samples_per_class)
        
        # Save metrics
        metrics = save_metrics(true_labels, predicted_labels, samples_per_class)
        
        # Print summary
        print("\nExperiment Summary:")
        print(f"Samples per class: {samples_per_class}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        return

if __name__ == '__main__':
    main() 