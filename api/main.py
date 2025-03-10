from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn
import logging

from .model_service import ModelService
from .schemas import (
    PredictionResponse,
    ClassAdditionResponse,
    ClassUpdateResponse,
    ClassRemovalResponse,
    ExampleRemovalResponse,
    ModelInfo,
    ErrorResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Classification by Retrieval API",
    description="API for dynamic image classification using the CbR model",
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

# Initialize model service
model_service = ModelService()

@app.get("/", tags=["Info"])
async def root():
    """Get basic API information."""
    return {
        "name": "Classification by Retrieval API",
        "version": "1.0.0",
        "description": "Dynamic image classification with real-time updates"
    }

@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get current model state and information."""
    return model_service.get_model_info()

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify an image using the current model state.
    """
    try:
        contents = await file.read()
        result = model_service.process_image(contents)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/class/add/{class_name}", response_model=ClassAdditionResponse, tags=["Class Management"])
async def add_class(
    class_name: str,
    files: List[UploadFile] = File(...)
):
    """
    Add a new class with one or more example images.
    """
    logger.info(f"Received request to add class '{class_name}' with {len(files)} images")
    
    try:
        # Validate input
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        if not class_name:
            raise HTTPException(status_code=400, detail="Class name cannot be empty")
            
        # Read all files
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
        
        # Process the class addition
        result = model_service.add_class(class_name, image_bytes_list)
        logger.info(f"Successfully added class '{class_name}'")
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
    append: bool = Form(False)
):
    """
    Update or append examples for an existing class.
    """
    try:
        image_bytes_list = [await file.read() for file in files]
        result = model_service.update_class(class_name, image_bytes_list, append)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/class/{class_name}", response_model=ClassRemovalResponse, tags=["Class Management"])
async def remove_class(class_name: str):
    """
    Remove a class and all its examples.
    """
    try:
        result = model_service.remove_class(class_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/examples", response_model=ExampleRemovalResponse, tags=["Example Management"])
async def remove_examples(indices: List[int]):
    """
    Remove specific examples by their indices.
    """
    try:
        result = model_service.remove_examples(indices)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True) 