from typing import Dict, Any
from ..utils.api_client import APIClient

class TenantTools:
    def __init__(self, api_client: APIClient):
        self.api_client = api_client
    
    def create_tenant(self, name: str = "") -> Dict[str, Any]:
        """Create a new tenant"""
        data = {"name": name} if name else {}
        return self.api_client.post("/tenants", data=data)
    
    def list_tenants(self) -> Dict[str, Any]:
        """List all tenants"""
        return self.api_client.get("/tenants") 