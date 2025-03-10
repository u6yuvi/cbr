# Classification by Retrieval (CbR)

This repository implements Classification by Retrieval (CbR) in PyTorch, a neural network model that combines image retrieval and classification capabilities without requiring expensive training.

## Overview

CbR is composed of two main components:
1. A pre-trained embedding network (backbone)
2. Retrieval layers that perform nearest neighbor matching and result aggregation

The key advantages of CbR include:
- No training required - just add your index data
- Scalable to large numbers of classes
- Flexible - easily add or remove classes without retraining
- Few-shot learning capability - works with as little as one example per class

## Installation

### Environment Setup

It's recommended to create a fresh virtual environment to avoid package conflicts:

```bash
# Create and activate a new virtual environment
python -m venv cbr_env
source cbr_env/bin/activate  # On Windows, use: cbr_env\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

Requirements:
```bash
torch
torchvision
Pillow
numpy<2.0.0  # Required for compatibility
```

## Usage

### Basic Usage

```python
from cbr_model import ClassificationByRetrieval

# Initialize model
model = ClassificationByRetrieval()

# Prepare your index data (embeddings and labels)
embeddings = ...  # shape: (num_samples, embedding_dim)
labels = ...      # list of class labels

# Add index data to the model
model.add_index_data(embeddings, labels)

# Make predictions
predictions = model(input_images)
```

### Custom Backbone

You can use your own backbone network:

```python
custom_backbone = ...  # Your custom PyTorch model
model = ClassificationByRetrieval(
    backbone=custom_backbone,
    embedding_dim=your_embedding_dim
)
```

## How It Works

1. **Embedding Extraction**: The backbone network converts input images into fixed-size embeddings
2. **Nearest Neighbor Matching**: Computes similarities between input embeddings and index embeddings
3. **Result Aggregation**: Aggregates similarity scores per class to produce final classification logits

The retrieval layers are implemented as a differentiable neural network layer, making it compatible with standard deep learning workflows.

## Example

See `example.py` for a complete working example of how to use the CbR model.

## Troubleshooting

If you encounter NumPy-related errors, make sure you're using a compatible version of NumPy (<2.0.0) as specified in the requirements. You can fix this by:

```bash
pip uninstall numpy
pip install "numpy<2.0.0"
``` 