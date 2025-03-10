from pydantic import BaseModel
from typing import List, Dict, Optional
from fastapi import UploadFile

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]

class ClassAdditionResponse(BaseModel):
    status: str
    message: str
    num_classes: int
    available_classes: List[str]

class ClassUpdateResponse(BaseModel):
    status: str
    message: str
    num_examples: int

class ClassRemovalResponse(BaseModel):
    status: str
    message: str
    num_classes: int
    available_classes: List[str]

class ExampleRemovalResponse(BaseModel):
    status: str
    message: str
    num_examples: int

class ModelInfo(BaseModel):
    num_classes: int
    num_examples: int
    available_classes: List[str]
    examples_per_class: Dict[str, int]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None 