import requests
from typing import Optional, Dict, Any
from ..config.config import Config

class APIClient:
    def __init__(self, tenant_id: Optional[str] = None):
        self.base_url = Config.API_BASE_URL
        self.tenant_id = tenant_id
        
    def _get_headers(self) -> Dict:
        headers = Config.DEFAULT_HEADERS.copy()
        if self.tenant_id:
            headers["X-Tenant-ID"] = self.tenant_id
        return headers
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
            
        response = requests.request(method, url, headers=headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get(self, endpoint: str, params: Dict = None) -> Dict[str, Any]:
        return self._make_request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Dict = None, files: Dict = None) -> Dict[str, Any]:
        return self._make_request("POST", endpoint, json=data, files=files)
    
    def put(self, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        return self._make_request("PUT", endpoint, json=data)
    
    def delete(self, endpoint: str) -> Dict[str, Any]:
        return self._make_request("DELETE", endpoint) 