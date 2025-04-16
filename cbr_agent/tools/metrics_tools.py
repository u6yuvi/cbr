from typing import Dict, Any, List
from ..utils.api_client import APIClient
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

class MetricsTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def calculate_metrics(self, true_labels: List[str], predicted_labels: List[str]) -> Dict[str, float]:
        """Calculate various performance metrics"""
        return {
            "accuracy": accuracy_score(true_labels, predicted_labels),
            "f1": f1_score(true_labels, predicted_labels, average='weighted'),
            "recall": recall_score(true_labels, predicted_labels, average='weighted'),
            "precision": precision_score(true_labels, predicted_labels, average='weighted')
        }
    
    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics as a readable string"""
        return "\n".join([
            f"- {key.capitalize()}: {value:.2%}"
            for key, value in metrics.items()
        ]) 