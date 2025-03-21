from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Optional
import uvicorn
import logging
import base64
import io
import traceback
from PIL import Image
from torchvision import transforms

from .model_service import TenantModelManager
from .schemas import (
    PredictionResponse,
    ClassAdditionResponse,
    ClassUpdateResponse,
    ClassRemovalResponse,
    ExampleRemovalResponse,
    ModelInfo,
    ErrorResponse,
    TenantsResponse,
    TenantCreationRequest,
    TenantCreationResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multi-tenant Classification by Retrieval API",
    description="API for dynamic image classification using tenant-specific CbR models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize tenant model manager
tenant_manager = TenantModelManager()

@app.get("/", tags=["Info"])
async def root():
    """Get basic API information."""
    return {
        "name": "Multi-tenant Classification by Retrieval API",
        "version": "1.0.0",
        "description": "Dynamic image classification with tenant-specific models"
    }

@app.post("/tenants", response_model=TenantCreationResponse, tags=["Tenant Management"])
async def create_tenant(request: TenantCreationRequest = None):
    """Create a new tenant with a unique ID."""
    request = request or TenantCreationRequest()
    return tenant_manager.create_tenant(request.name)

@app.get("/tenants", response_model=TenantsResponse, tags=["Tenant Management"])
async def list_tenants():
    """Get list of all tenants and their metadata."""
    tenant_ids = tenant_manager.get_tenant_ids()
    tenants = {
        tid: tenant_manager.get_tenant_metadata(tid)
        for tid in tenant_ids
    }
    return {
        "tenant_ids": tenant_ids,
        "tenants": tenants
    }

@app.delete("/tenants/{tenant_id}", tags=["Tenant Management"])
async def remove_tenant(tenant_id: str):
    """Remove a tenant and their model instance."""
    if tenant_manager.remove_tenant(tenant_id):
        return {"status": "success", "message": f"Removed tenant {tenant_id}"}
    raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info(x_tenant_id: str = Header(...)):
    """Get current model state and information for a specific tenant."""
    info = tenant_manager.get_tenant_info(x_tenant_id)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Tenant {x_tenant_id} not found")
    return info

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    file: UploadFile = File(...),
    x_tenant_id: str = Header(...)
):
    """Classify an image using the tenant's model state."""
    try:
        model = tenant_manager.get_or_create_model(x_tenant_id)
        contents = await file.read()
        result = model.process_image(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/class/add/{class_name}", response_model=ClassAdditionResponse, tags=["Class Management"])
async def add_class(
    class_name: str,
    files: List[UploadFile] = File(...),
    x_tenant_id: str = Header(...)
):
    """Add a new class with examples to tenant's model."""
    logger.info(f"Received request to add class '{class_name}' for tenant {x_tenant_id}")
    
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not class_name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")
        
        model = tenant_manager.get_or_create_model(x_tenant_id)
        
        image_bytes_list = []
        for i, file in enumerate(files):
            try:
                content = await file.read()
                if not content:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Empty file content for file {i+1}: {file.filename}"
                    )
                image_bytes_list.append(content)
            except Exception as e:
                logger.error(f"Error reading file {i+1}: {str(e)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Error reading file {i+1}: {file.filename} - {str(e)}"
                )
        
        result = model.add_class(class_name, image_bytes_list)
        logger.info(f"Successfully added class '{class_name}' for tenant {x_tenant_id}")
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/class/update/{class_name}", response_model=ClassUpdateResponse, tags=["Class Management"])
async def update_class(
    class_name: str,
    files: List[UploadFile] = File(...),
    append: bool = Form(False),
    x_tenant_id: str = Header(...)
):
    """Update or append examples for a class in tenant's model."""
    try:
        model = tenant_manager.get_or_create_model(x_tenant_id)
        image_bytes_list = [await file.read() for file in files]
        result = model.update_class(class_name, image_bytes_list, append)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/class/{class_name}", response_model=ClassRemovalResponse, tags=["Class Management"])
async def remove_class(
    class_name: str,
    x_tenant_id: str = Header(...)
):
    """Remove a class from tenant's model."""
    try:
        model = tenant_manager.get_or_create_model(x_tenant_id)
        result = model.remove_class(class_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/examples", response_model=ExampleRemovalResponse, tags=["Example Management"])
async def remove_examples(
    indices: List[int],
    x_tenant_id: str = Header(...)
):
    """Remove examples from tenant's model."""
    try:
        model = tenant_manager.get_or_create_model(x_tenant_id)
        result = model.remove_examples(indices)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/class/{class_name}/images", tags=["Class Management"])
async def get_class_images(
    class_name: str,
    x_tenant_id: str = Header(...)
):
    """Get all images for a specific class."""
    try:
        model_service = tenant_manager.get_or_create_model(x_tenant_id)
        if class_name not in model_service.model.classes_to_idx:
            raise HTTPException(status_code=404, detail=f"Class '{class_name}' not found")
        
        # Get indices for this class
        class_idx = model_service.model.classes_to_idx[class_name]
        class_mask = (model_service.model.class_labels == class_idx)
        
        # Get the original images for this class
        images = []
        for i, is_class in enumerate(class_mask):
            if is_class:
                # Get original image bytes
                img_bytes = model_service.model.original_images[i]
                # Convert to base64 for JSON transport
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                images.append(img_b64)
        
        return JSONResponse(content={"images": images})
        
    except Exception as e:
        logger.error(f"Error getting class images: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) 