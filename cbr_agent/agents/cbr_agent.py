from typing import Dict, Any, List
from google import genai
import json
import inspect
from cbr_agent.agents.base_agent import BaseAgent
from cbr_agent.tools.tenant_tools import TenantTools
from cbr_agent.tools.model_tools import ModelTools
from cbr_agent.tools.prediction_tools import PredictionTools
from cbr_agent.tools.metrics_tools import MetricsTools
from cbr_agent.utils.api_client import APIClient
from cbr_agent.utils.storage import TenantStorage
from cbr_agent.config.config import Config
from ..utils.image_utils import ImageDownloader
from urllib.parse import urlparse
from pathlib import Path

class CBRAgent(BaseAgent):
    def __init__(self):
        # Validate config
        Config.validate()
        
        # Initialize storage
        self.storage = TenantStorage()
        
        # Try to load last used tenant
        self.current_tenant_id = self.storage.get_last_used_tenant()
        
        # Initialize API client with last used tenant
        self.api_client = APIClient(tenant_id=self.current_tenant_id)
        
        # Initialize tools
        self.tenant_tools = TenantTools(self.api_client)
        self.model_tools = ModelTools(self.api_client)
        self.prediction_tools = PredictionTools(self.api_client)
        self.metrics_tools = MetricsTools(self.api_client)
        
        # Initialize Google AI client
        self.client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        
        # Initialize image downloader
        self.image_downloader = ImageDownloader()
        
        # Add new tenant management tools
        self.tools = {
            # Tenant tools
            "create_tenant": self._create_tenant_wrapper,
            "list_tenants": self._list_tenants_wrapper,
            "switch_tenant": self._switch_tenant,
            "get_current_tenant": self._get_current_tenant,
            
            # Model tools
            "get_model_info": self.model_tools.get_model_info,
            "add_class": self.model_tools.add_class,
            "update_class": self.model_tools.update_class,
            "remove_class": self.model_tools.remove_class,
            "get_class_images": self.model_tools.get_class_images,
            
            # Prediction tools
            "predict_single": self.prediction_tools.predict_single,
            "predict_batch": self.prediction_tools.predict_batch,
            
            # Metrics tools
            "calculate_metrics": self.metrics_tools.calculate_metrics,
            
            # Image handling tools
            "train_model": self._train_model,
            "predict_image": self._predict_image,
        }
        
        # Create detailed tool descriptions with parameter information
        self.tool_descriptions = []
        for name, func in self.tools.items():
            sig = inspect.signature(func)
            params = []
            for param_name, param in sig.parameters.items():
                if param.default == inspect.Parameter.empty:
                    params.append(f"{param_name} (required)")
                else:
                    default = "None" if param.default is None else param.default
                    params.append(f"{param_name} (optional, default={default})")
            
            params_str = ", ".join(params)
            doc = func.__doc__ or "No description available"
            self.tool_descriptions.append(
                f"Tool: {name}\n"
                f"Parameters: {params_str}\n"
                f"Description: {doc}\n"
            )
        
        self.tool_descriptions = "\n".join(self.tool_descriptions)

    def _create_tenant_wrapper(self, name: str = "") -> Dict[str, Any]:
        """Create a new tenant and store the tenant ID"""
        result = self.tenant_tools.create_tenant(name)
        if "tenant_id" in result:
            self._update_tenant_id(result["tenant_id"])
            # Save tenant to storage
            self.storage.save_tenant(result["tenant_id"], name)
            result["message"] = f"Created and switched to tenant with ID: {result['tenant_id']}"
        return result
    
    def _list_tenants_wrapper(self) -> Dict[str, Any]:
        """List all tenants with their metadata"""
        api_tenants = self.tenant_tools.list_tenants()
        stored_tenants = self.storage.list_tenants()
        
        # Combine API and stored tenant information
        result = {
            "tenants": {},
            "current_tenant_id": self.current_tenant_id
        }
        
        for tenant_id in api_tenants.get("tenant_ids", []):
            result["tenants"][tenant_id] = {
                **stored_tenants.get(tenant_id, {}),
                "active": True
            }
        
        return result
    
    def _switch_tenant(self, tenant_id: str) -> Dict[str, Any]:
        """Switch to a different tenant"""
        # Verify tenant exists
        tenant_info = self.storage.get_tenant(tenant_id)
        if not tenant_info:
            raise ValueError(f"Tenant {tenant_id} not found in storage")
        
        # Update current tenant
        self._update_tenant_id(tenant_id)
        
        # Update last used timestamp
        self.storage.update_last_used(tenant_id)
        
        return {
            "message": f"Switched to tenant: {tenant_id}",
            "tenant_info": tenant_info
        }
    
    def _get_current_tenant(self) -> Dict[str, Any]:
        """Get information about the current tenant"""
        if not self.current_tenant_id:
            return {
                "message": "No tenant currently selected",
                "tenant_id": None
            }
        
        tenant_info = self.storage.get_tenant(self.current_tenant_id)
        return {
            "message": f"Current tenant: {self.current_tenant_id}",
            "tenant_id": self.current_tenant_id,
            "tenant_info": tenant_info
        }

    def _is_url(self, path: str) -> bool:
        """Check if a string is a URL"""
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _get_image_paths(self, images: List[str]) -> List[Path]:
        """Convert image URLs to local paths if needed"""
        paths = []
        for img in images:
            if self._is_url(img):
                paths.append(self.image_downloader.download_image(img))
            else:
                paths.append(Path(img))
        return paths
    
    def _train_model(self, class_name: str, images: List[str]) -> Dict[str, Any]:
        """Train the model with images for a class"""
        if not class_name:
            raise ValueError("Class name is required")
        if not images:
            raise ValueError("At least one image is required")
        
        # Convert URLs to local paths if needed
        image_paths = self._get_image_paths(images)
        
        # Use model_tools to add the class with images
        response = self.model_tools.add_class(
            class_name=class_name,
            images=image_paths
        )
        
        return response
    
    def _predict_image(self, image: str) -> Dict[str, Any]:
        """Make a prediction on an image"""
        if not image:
            raise ValueError("Image path or URL is required")
        
        # Convert URL to local path if needed
        image_paths = self._get_image_paths([image])
        
        # Make prediction
        result = self.prediction_tools.predict_single(image_paths[0])
        
        return result

    def run(self, query: str) -> Dict[str, Any]:
        """Main agent loop"""
        conversation = []
        system_prompt = f"""You are an AI assistant that helps users interact with a Classification by Retrieval (CBR) system.
        You can manage tenants, handle image classifications, and calculate metrics.
        When you need to perform an action, respond with a JSON object containing:
        {{"tool": "tool_name", "args": {{tool_arguments}}}}

        Available tools and their parameters:
        {self.tool_descriptions}

        Important:
        - For create_tenant, use "name" as the parameter name
        - Use switch_tenant to change the active tenant
        - Use get_current_tenant to check which tenant is active
        - All parameters must match exactly as specified
        - Always include tenant information in your responses

        Current tenant ID: {self.current_tenant_id or "None"}
        Current query: {query}
        """
        
        # Get LLM response
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=system_prompt
        )
        
        assistant_message = response.text
        conversation.append({"role": "assistant", "content": assistant_message})
        
        # Try to parse as JSON (tool call) or treat as natural language
        try:
            tool_call = json.loads(assistant_message.strip('`json\n'))
            if "tool" in tool_call and "args" in tool_call:
                # Execute tool
                tool_name = tool_call["tool"]
                tool_args = tool_call["args"]
                
                try:
                    result = self.handle_tool_call(tool_name, tool_args)
                    # Include tenant ID in response if available
                    response_data = {
                        "tool_result": result,
                        "current_tenant_id": self.current_tenant_id
                    }
                    conversation.append({
                        "role": "system",
                        "content": response_data
                    })
                    
                    return {
                        "response": "Success",
                        "result": result,
                        "conversation": conversation,
                        "current_tenant_id": self.current_tenant_id
                    }
                except Exception as e:
                    error_message = f"Error executing tool: {str(e)}"
                    conversation.append({
                        "role": "system",
                        "content": error_message
                    })
                    return {
                        "response": "Error",
                        "error": error_message,
                        "conversation": conversation,
                        "current_tenant_id": self.current_tenant_id
                    }
            else:
                # Not a valid tool call format
                return {
                    "response": assistant_message,
                    "conversation": conversation,
                    "current_tenant_id": self.current_tenant_id
                }
        except json.JSONDecodeError:
            # Not JSON, treat as final response
            return {
                "response": assistant_message,
                "conversation": conversation,
                "current_tenant_id": self.current_tenant_id
            }
    
    def handle_tool_call(self, tool_name: str, tool_args: Dict) -> Any:
        """Handle tool calls from LLM"""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")
        
        return self.tools[tool_name](**tool_args)

    def __del__(self):
        """Cleanup downloaded images when the agent is destroyed"""
        try:
            self.image_downloader.cleanup()
        except:
            pass

    def _update_tenant_id(self, tenant_id: str):
        """Update tenant ID across all components"""
        self.current_tenant_id = tenant_id
        self.api_client.tenant_id = tenant_id
        self.tenant_tools.api_client.tenant_id = tenant_id
        self.model_tools.api_client.tenant_id = tenant_id
        self.prediction_tools.api_client.tenant_id = tenant_id
        self.metrics_tools.api_client.tenant_id = tenant_id 