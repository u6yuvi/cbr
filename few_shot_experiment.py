import torch
from torchvision import transforms
from PIL import Image
from cbr_model import ClassificationByRetrieval
import os
from pathlib import Path
import random
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score
import json
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import argparse

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

def run_single_iteration(model, index_dir: str, transform, samples_per_class: int, seed: int):
    """
    Run a single iteration of the few-shot learning experiment with a specific seed.
    """
    # Set random seed for this iteration
    random.seed(seed)
    
    # Load and process index images for few-shot learning
    print(f"\nIteration with seed {seed}")
    print(f"Loading {samples_per_class} samples per class for few-shot learning...")
    index_embeddings, index_labels = load_index_images(index_dir, transform, samples_per_class)
    
    # Add index data to the model
    print("Adding index data to model...")
    model.add_index_data(index_embeddings, index_labels)
    print(f"Added {len(index_labels)} images to index")
    
    # Evaluate on remaining samples
    print("\nEvaluating model on remaining samples...")
    true_labels, predicted_labels = evaluate_model(model, index_dir, transform, samples_per_class)
    
    # Calculate metrics for this iteration
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_labels': true_labels,
        'predicted_labels': predicted_labels
    }

def run_multiple_iterations(model, index_dir: str, transform, samples_per_class: int, n_iterations: int):
    """
    Run multiple iterations of the few-shot learning experiment and calculate statistics.
    """
    # Store results for each iteration
    results = []
    
    # Run iterations with different seeds
    for i in range(n_iterations):
        # Use current timestamp + iteration number as seed for reproducibility
        seed = int(datetime.now().timestamp()) + i
        iteration_results = run_single_iteration(model, index_dir, transform, samples_per_class, seed)
        results.append(iteration_results)
    
    # Calculate statistics across iterations
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    stats = {}
    
    for metric in metrics:
        values = [r[metric] for r in results]
        stats[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return results, stats

def plot_metrics(results, stats, samples_per_class, n_iterations, output_dir: str = "experiment_results"):
    """
    Create and save plots showing metrics across iterations with error bars.
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Few-Shot Learning Metrics (n={samples_per_class} shots)', fontsize=16)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Iteration numbers for x-axis
    iterations = range(1, n_iterations + 1)
    
    for idx, metric in enumerate(metrics):
        # Get values for this metric
        values = [r[metric] for r in results]
        mean = stats[metric]['mean']
        std = stats[metric]['std']
        
        # Create plot
        ax = axes[idx]
        ax.plot(iterations, values, 'b-', label='Per Iteration')
        ax.errorbar(iterations, [mean] * n_iterations, yerr=std, 
                   fmt='r--', label=f'Mean ± Std\nMean: {mean:.4f}\nStd: {std:.4f}')
        
        # Customize plot
        ax.set_title(metric.capitalize())
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Score')
        ax.grid(True)
        ax.legend()
        
        # Set y-axis limits with some padding
        y_min = min(min(values), mean - std) * 0.95
        y_max = max(max(values), mean + std) * 1.05
        ax.set_ylim(y_min, y_max)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_file = os.path.join(
        output_dir,
        f"metrics_plot_{samples_per_class}shots_{n_iterations}iterations_{timestamp}.png"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {plot_file}")
    return plot_file

def save_iteration_metrics(results, stats, samples_per_class, n_iterations, output_dir: str = "experiment_results"):
    """
    Save metrics from multiple iterations to files.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for consistent filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare data for JSON
    metrics_data = {
        "samples_per_class": samples_per_class,
        "n_iterations": n_iterations,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "statistics": stats,
        "iteration_results": [
            {
                "iteration": i,
                "accuracy": r['accuracy'],
                "precision": r['precision'],
                "recall": r['recall'],
                "f1": r['f1']
            }
            for i, r in enumerate(results)
        ]
    }
    
    # Save metrics to JSON file
    json_file = os.path.join(
        output_dir, 
        f"metrics_{samples_per_class}shots_{n_iterations}iterations_{timestamp}.json"
    )
    
    with open(json_file, 'w') as f:
        json.dump(metrics_data, f, indent=4)
    
    # Save detailed report to TXT file
    txt_file = os.path.join(
        output_dir,
        f"classification_report_{samples_per_class}shots_{n_iterations}iterations_{timestamp}.txt"
    )
    
    with open(txt_file, 'w') as f:
        f.write(f"Few-Shot Learning Experiment Results\n")
        f.write(f"====================================\n\n")
        f.write(f"Samples per class: {samples_per_class}\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Timestamp: {metrics_data['timestamp']}\n\n")
        
        f.write("Statistics Across Iterations:\n")
        f.write("============================\n")
        for metric, values in stats.items():
            f.write(f"\n{metric.capitalize()}:\n")
            f.write(f"  Mean: {values['mean']:.4f}\n")
            f.write(f"  Std:  {values['std']:.4f}\n")
            f.write(f"  Min:  {values['min']:.4f}\n")
            f.write(f"  Max:  {values['max']:.4f}\n")
        
        f.write("\nDetailed Results by Iteration:\n")
        f.write("=============================\n")
        for i, r in enumerate(results):
            f.write(f"\nIteration {i+1}:\n")
            f.write(f"  Accuracy:  {r['accuracy']:.4f}\n")
            f.write(f"  Precision: {r['precision']:.4f}\n")
            f.write(f"  Recall:    {r['recall']:.4f}\n")
            f.write(f"  F1 Score:  {r['f1']:.4f}\n")
    
    # Create and save plots
    plot_file = plot_metrics(results, stats, samples_per_class, n_iterations, output_dir)
    
    print(f"\nMetrics saved to: {json_file}")
    print(f"Detailed report saved to: {txt_file}")
    print(f"Plot saved to: {plot_file}")
    return metrics_data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run few-shot learning experiments with Classification by Retrieval')
    parser.add_argument('--samples', type=int, default=1,
                      help='Number of samples per class for few-shot learning (default: 1)')
    parser.add_argument('--iterations', type=int, default=5,
                      help='Number of iterations to run (default: 5)')
    parser.add_argument('--index-dir', type=str, default='index_images',
                      help='Directory containing class subdirectories (default: index_images)')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                      help='Directory to save results (default: experiment_results)')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility (default: None)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
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
    
    try:
        # Run multiple iterations
        print(f"\nStarting {args.iterations} iterations with {args.samples} samples per class...")
        print(f"Using index directory: {args.index_dir}")
        print(f"Results will be saved to: {args.output_dir}")
        if args.seed is not None:
            print(f"Using random seed: {args.seed}")
        
        results, stats = run_multiple_iterations(
            model, 
            args.index_dir, 
            transform, 
            args.samples, 
            args.iterations
        )
        
        # Save metrics
        metrics = save_iteration_metrics(
            results, 
            stats, 
            args.samples, 
            args.iterations,
            args.output_dir
        )
        
        # Print summary
        print("\nExperiment Summary:")
        print(f"Samples per class: {args.samples}")
        print(f"Number of iterations: {args.iterations}")
        print("\nStatistics across iterations:")
        for metric, values in stats.items():
            print(f"\n{metric.capitalize()}:")
            print(f"  Mean: {values['mean']:.4f}")
            print(f"  Std:  {values['std']:.4f}")
            print(f"  Min:  {values['min']:.4f}")
            print(f"  Max:  {values['max']:.4f}")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        return

if __name__ == '__main__':
    main() 