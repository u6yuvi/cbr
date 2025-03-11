from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from fastapi import UploadFile

class PredictionResponse(BaseModel):
    """Response for image classification prediction."""
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]

class ClassAdditionResponse(BaseModel):
    """Response for adding a new class."""
    status: str
    message: str
    num_classes: int
    available_classes: List[str]

class ClassUpdateResponse(BaseModel):
    """Response for updating a class."""
    status: str
    message: str
    num_examples: int

class ClassRemovalResponse(BaseModel):
    """Response for removing a class."""
    status: str
    message: str
    num_classes: int
    available_classes: List[str]

class ExampleRemovalResponse(BaseModel):
    """Response for removing examples."""
    status: str
    message: str
    num_examples: int

class ModelInfo(BaseModel):
    """Model information and current state."""
    num_classes: int
    num_examples: int
    available_classes: List[str]
    examples_per_class: Dict[str, int]

class ErrorResponse(BaseModel):
    """Error response."""
    detail: str 