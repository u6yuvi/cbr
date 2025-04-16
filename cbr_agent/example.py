import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from cbr_agent.agents.cbr_agent import CBRAgent

def main():
    try:
        # Initialize agent (no need to pass API key anymore)
        agent = CBRAgent()
        
        # Example query
        query = "Create a new tenant called 'test-tenant' and show me the model information"
        
        # Run agent
        result = agent.run(query)
        
        # Print response
        print("Agent Response:", result["response"])
        
        # Create a new tenant
        result = agent.run("Create a new tenant called 'my-project'")
        print("1",result)
        # The tenant ID will be saved and set as current
        
        # List all tenants
        result = agent.run("Show me all available tenants")
        # Will show both API and stored tenant information
        print("2",result)
        
        # Switch to a different tenant
        result = agent.run("Switch to tenant abc-123")
        # Will update the current tenant and last used timestamp
        print("3",result)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 