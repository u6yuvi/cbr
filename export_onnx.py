"""
Export Classification by Retrieval (CbR) model to ONNX format.
"""

import torch
import torch.nn as nn
from cbr_model import ClassificationByRetrieval
import onnx
import json
from pathlib import Path
import numpy as np

def export_model_to_onnx(model: ClassificationByRetrieval, output_dir: str = "web/public/model"):
    """
    Export the CbR model to ONNX format and save necessary metadata.
    
    Args:
        model: Trained CbR model
        output_dir: Directory to save the ONNX model and metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export backbone to ONNX
    dummy_input = torch.randn(1, 3, 224, 224)
    backbone_path = output_dir / "backbone.onnx"
    
    # Set model to eval mode
    model.backbone.eval()
    
    # Export backbone
    torch.onnx.export(
        model.backbone,
        dummy_input,
        backbone_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['embedding'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'embedding': {0: 'batch_size'}
        }
    )
    
    # Save index data
    if model.index_embeddings is not None:
        index_data = {
            'embeddings': model.index_embeddings.cpu().numpy().tolist(),
            'labels': model.class_labels.cpu().numpy().tolist(),
            'classes_to_idx': model.classes_to_idx,
            'idx_to_classes': model.idx_to_classes,
            'num_classes': model.num_classes
        }
    else:
        index_data = {
            'embeddings': [],
            'labels': [],
            'classes_to_idx': {},
            'idx_to_classes': {},
            'num_classes': 0
        }
    
    # Save index data
    with open(output_dir / "index_data.json", "w") as f:
        json.dump(index_data, f)
        
    # Save image preprocessing parameters
    preprocess_params = {
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225],
        'size': 224
    }
    
    with open(output_dir / "preprocess_params.json", "w") as f:
        json.dump(preprocess_params, f)
        
    print(f"Model exported successfully to {output_dir}")
    print("Files created:")
    print(f"- {backbone_path}")
    print(f"- {output_dir}/index_data.json")
    print(f"- {output_dir}/preprocess_params.json")

if __name__ == "__main__":
    # Load your trained model
    model = ClassificationByRetrieval()
    
    # If you have index data, load it here
    # Example:
    # model.add_index_data(embeddings, labels)
    
    # Export the model
    export_model_to_onnx(model) 