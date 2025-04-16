from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    @abstractmethod
    def run(self, query: str) -> Dict[str, Any]:
        """Run the agent with a query"""
        pass
    
    @abstractmethod
    def handle_tool_call(self, tool_name: str, tool_args: Dict) -> Any:
        """Handle tool calls from LLM"""
        pass 