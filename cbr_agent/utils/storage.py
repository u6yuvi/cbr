import json
import os
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

class TenantStorage:
    def __init__(self):
        # Create storage directory in user's home directory
        self.storage_dir = Path.home() / '.cbr_agent'
        self.storage_dir.mkdir(exist_ok=True)
        self.tenant_file = self.storage_dir / 'tenants.json'
        
        # Initialize storage file if it doesn't exist
        if not self.tenant_file.exists():
            self._save_tenants({})
    
    def _save_tenants(self, data: Dict[str, Any]) -> None:
        """Save tenants data to file"""
        with open(self.tenant_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _load_tenants(self) -> Dict[str, Any]:
        """Load tenants data from file"""
        try:
            with open(self.tenant_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_tenant(self, tenant_id: str, name: Optional[str] = None) -> None:
        """Save a tenant with metadata"""
        tenants = self._load_tenants()
        tenants[tenant_id] = {
            'name': name,
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat()
        }
        self._save_tenants(tenants)
    
    def get_tenant(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        """Get tenant information"""
        return self._load_tenants().get(tenant_id)
    
    def update_last_used(self, tenant_id: str) -> None:
        """Update the last used timestamp for a tenant"""
        tenants = self._load_tenants()
        if tenant_id in tenants:
            tenants[tenant_id]['last_used'] = datetime.now().isoformat()
            self._save_tenants(tenants)
    
    def list_tenants(self) -> Dict[str, Any]:
        """List all saved tenants"""
        return self._load_tenants()
    
    def get_last_used_tenant(self) -> Optional[str]:
        """Get the most recently used tenant ID"""
        tenants = self._load_tenants()
        if not tenants:
            return None
            
        # Find tenant with most recent last_used timestamp
        return max(tenants.items(), key=lambda x: x[1]['last_used'])[0] 