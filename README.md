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

### Creating Index Embeddings

The CbR model requires index data (reference images) for classification. Here's how to set it up:

1. **Prepare Your Index Images**:
   ```
   index_images/
   ├── class1/
   │   ├── image1.jpg
   │   └── image2.jpg
   ├── class2/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── ...
   ```
   - Create a directory for each class
   - Place example images for each class in their respective directories
   - Supports common image formats (jpg, png, jpeg)

2. **Load and Process Index Images**:
   ```python
   from cbr_model import ClassificationByRetrieval
   
   # Initialize model and transforms
   model = ClassificationByRetrieval()
   transform = transforms.Compose([
       transforms.Resize(256),
       transforms.CenterCrop(224),
       transforms.ToTensor(),
       transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
       )
   ])
   
   # Load index images and create embeddings
   index_embeddings, index_labels = load_index_images('index_images', transform)
   
   # Add index data to model
   model.add_index_data(index_embeddings, index_labels)
   ```

3. **Using Custom Index Data**:
   - You can use your own embedding function instead of ResNet18
   - Provide pre-computed embeddings if available
   - Mix and match different sources of index data

### Making Predictions

Once index data is added, you can classify new images:

```python
# Load and preprocess your image
image = Image.open('test_image.jpg').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Get predictions
with torch.no_grad():
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)
    
    # Get predicted class
    class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = model.idx_to_classes[class_idx]
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

1. **Embedding Extraction**: 
   - The backbone network (ResNet18 by default) converts images into fixed-size embeddings
   - Index images are processed once to create reference embeddings
   - New images are converted to embeddings during classification

2. **Nearest Neighbor Matching**: 
   - Computes similarities between input embeddings and index embeddings
   - Uses cosine similarity for better matching
   - Efficiently implemented as matrix multiplication

3. **Result Aggregation**: 
   - Aggregates similarity scores per class
   - Uses max pooling to select best match per class
   - Returns probabilities for all classes

The retrieval layers are implemented as a differentiable neural network layer, making it compatible with standard deep learning workflows.

## Example

See `example.py` for a complete working example of how to:
- Set up the directory structure
- Process index images
- Make predictions on new images

## Troubleshooting

If you encounter NumPy-related errors, make sure you're using a compatible version of NumPy (<2.0.0) as specified in the requirements. You can fix this by:

```bash
pip uninstall numpy
pip install "numpy<2.0.0"
```

## Processes index images to create real embeddings
```bash
#Image Directory Structure
   index_images/
       cat/
           cat1.jpg
           cat2.jpg
       dog/
           dog1.jpg
           dog2.jpg
   test_images/
       dog.jpg (your test image)
```