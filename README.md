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

## Key Benefits
- Fast Setup: Just add reference images for your classes
- Flexibility: Easily add or remove classes without retraining
- Few-Shot Learning: Works with as little as one example per class
- Dynamic Updates: Can update the index data on the fly

## Dynamic Updates

One of the powerful features of CbR is the ability to dynamically update the model's index data without retraining. This means you can:

1. **Add New Classes**:
```python
# Existing model with cat/dog classes
model = ClassificationByRetrieval()

# Add a new 'bird' class
new_images = load_images('bird_images/')  # Load new class images
new_embeddings = model.get_embedding(new_images)
new_labels = ['bird'] * len(new_images)

# Combine with existing index data
combined_embeddings = torch.cat([model.index_embeddings, new_embeddings])
combined_labels = model.class_labels + new_labels

# Update model's index data
model.add_index_data(combined_embeddings, combined_labels)
```

2. **Update Existing Classes**:
```python
# Update examples for an existing class
new_cat_images = load_images('new_cat_images/')
new_cat_embeddings = model.get_embedding(new_cat_images)

# Replace or append to existing cat embeddings
model.update_class_embeddings('cat', new_cat_embeddings)
```

3. **Remove Classes or Examples**:
```python
# Remove a class
model.remove_class('bird')

# Or remove specific examples
model.remove_examples(indices=[0, 1, 2])  # Remove first three examples
```

These updates can be performed at any time without:
- Retraining the model
- Disrupting the model's operation
- Affecting other classes
- Requiring a model restart

This makes CbR ideal for:
- Active learning systems
- Continuously evolving datasets
- Interactive applications where classes need to be added/removed frequently
- Production systems that need to be updated without downtime

With CbR, all updates are immediate and don't require retraining, making it much more flexible for real-world applications.


#Check model info
```bash
curl http://localhost:8000/model/info
curl -v -X POST \
  -F "files=@index_images/dog/dog1.jpg" \
  http://localhost:8000/class/add/dog
```

## FastAPI Service

The model is exposed through a REST API built with FastAPI. The service provides endpoints for model management, classification, and dynamic updates.

### Running the Service

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

### API Endpoints

#### 1. Model Information
```bash
# Get model status and information
curl http://localhost:8000/model/info
```
Response:
```json
{
    "num_classes": 2,
    "num_examples": 4,
    "available_classes": ["cat", "dog"],
    "examples_per_class": {
        "cat": 2,
        "dog": 2
    }
}
```

#### 2. Adding Classes
```bash
# Add a new class with example images
curl -X POST \
  -F "files=@path/to/cat1.jpg" \
  -F "files=@path/to/cat2.jpg" \
  http://localhost:8000/class/add/cat
```
Response:
```json
{
    "status": "success",
    "message": "Added class 'cat' with 2 examples",
    "num_classes": 1,
    "available_classes": ["cat"]
}
```

#### 3. Updating Classes
```bash
# Add more examples to existing class
curl -X POST \
  -F "files=@path/to/cat3.jpg" \
  -F "append=true" \
  http://localhost:8000/class/update/cat

# Replace examples of existing class
curl -X POST \
  -F "files=@path/to/new_cat.jpg" \
  -F "append=false" \
  http://localhost:8000/class/update/cat
```
Response:
```json
{
    "status": "success",
    "message": "Added 1 examples for class 'cat'",
    "num_examples": 3
}
```

#### 4. Making Predictions
```bash
# Classify an image
curl -X POST \
  -F "file=@path/to/test_image.jpg" \
  http://localhost:8000/predict
```
Response:
```json
{
    "predicted_class": "cat",
    "confidence": 0.85,
    "class_probabilities": {
        "cat": 0.85,
        "dog": 0.15
    }
}
```

#### 5. Removing Classes
```bash
# Remove an entire class
curl -X DELETE http://localhost:8000/class/cat
```
Response:
```json
{
    "status": "success",
    "message": "Removed class 'cat'",
    "num_classes": 1,
    "available_classes": ["dog"]
}
```

#### 6. Removing Examples
```bash
# Remove specific examples by their indices
curl -X DELETE \
  -H "Content-Type: application/json" \
  -d '[0, 2]' \
  http://localhost:8000/examples
```
Response:
```json
{
    "status": "success",
    "message": "Removed 2 examples",
    "num_examples": 2
}
```

# Multi-tenant Classification by Retrieval API

A FastAPI service that provides dynamic image classification with tenant-specific models. Each tenant gets their own isolated model instance for managing classes and making predictions.

## Features
- Multi-tenant support with isolated model instances
- Dynamic class management (add/update/remove)
- Real-time image classification
- RESTful API with FastAPI
- Docker support for easy deployment
- Automatic tenant ID generation
- Tenant usage tracking

## Deployment Options

### 1. Docker Deployment (Recommended)
```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Create data directory for persistence
mkdir -p data

# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

### 2. Local Development
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn api.main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive documentation is available at `http://localhost:8000/docs`.

## Directory Structure
```
.
├── api/                # API implementation
├── data/              # Persistent data storage (when using Docker)
├── Dockerfile         # Docker image definition
├── docker-compose.yml # Docker compose configuration
└── requirements.txt   # Python dependencies
```

## API Usage

### 1. Tenant Management

First, create a tenant to get a unique tenant ID:

```bash
# Create a new tenant with a name
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"name": "user1"}' \
  http://localhost:8000/tenants

# Create an anonymous tenant
curl -X POST http://localhost:8000/tenants
```

Example response:
```json
{
    "tenant_id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "user1",
    "created_at": "2024-03-14T12:34:56.789Z"
}
```

View existing tenants:
```bash
# List all tenants and their metadata
curl http://localhost:8000/tenants

# Remove a tenant
curl -X DELETE http://localhost:8000/tenants/tenant1
```

### 2. Using the API with Your Tenant ID

All subsequent API calls require the `X-Tenant-ID` header:

```bash
# Get model info
curl -H "X-Tenant-ID: your-tenant-id" \
  http://localhost:8000/model/info

# Add a class
curl -X POST \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "files=@dog1.jpg" \
  -F "files=@dog2.jpg" \
  http://localhost:8000/class/add/dog

# Make predictions
curl -X POST \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "file=@test_image.jpg" \
  http://localhost:8000/predict
```

### 3. Class Management

```bash
# Add more examples to a class
curl -X POST \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "files=@dog3.jpg" \
  -F "append=true" \
  http://localhost:8000/class/update/dog

# Replace class examples
curl -X POST \
  -H "X-Tenant-ID: your-tenant-id" \
  -F "files=@new_dog.jpg" \
  -F "append=false" \
  http://localhost:8000/class/update/dog

# Remove a class
curl -X DELETE \
  -H "X-Tenant-ID: your-tenant-id" \
  http://localhost:8000/class/dog

# Remove specific examples
curl -X DELETE \
  -H "X-Tenant-ID: your-tenant-id" \
  -H "Content-Type: application/json" \
  -d '[0, 2]' \
  http://localhost:8000/examples
```

## Python Client Usage

For programmatic access, use the provided Python client:

```python
from api.client import CbRClient

# Create client for a specific tenant
client = CbRClient(tenant_id="your-tenant-id")

# Add a class
client.add_class("dog", ["dog1.jpg", "dog2.jpg"])

# Make predictions
result = client.predict("test_image.jpg")

# Or specify tenant per-call
client = CbRClient()
client.add_class("cat", ["cat1.jpg"], tenant_id="another-tenant-id")
```

## Docker Notes

1. Data Persistence
   - Model data is stored in the `./data` directory
   - This directory is mounted as a volume in Docker
   - Ensure proper permissions on the data directory

2. Health Monitoring
   - The service includes a health check endpoint
   - Docker will automatically restart the container on failure
   - Health status can be viewed with `docker-compose ps`

3. Environment Configuration
   - Environment variables can be added in docker-compose.yml
   - The service runs on port 8000 by default
   - Port mapping can be modified in docker-compose.yml

4. Development Mode
   - Code changes are reflected immediately due to volume mounting
   - Logs can be viewed in real-time with `docker-compose logs -f`
   - Use `docker-compose down` to clean up resources

## Security Notes

1. Tenant Isolation
   - Each tenant gets their own isolated model instance
   - Tenant IDs are required for all operations
   - Invalid tenant IDs are rejected

2. API Access
   - Use HTTPS in production
   - Implement proper authentication/authorization
   - Consider rate limiting for tenant requests
