# Custom MCP Servers

## Overview

While the MarketAgents framework includes built-in MCP servers like the Finance Tools MCP Server, you can also create custom MCP servers to provide specialized capabilities for your agents. This document explains how to design, implement, and deploy custom MCP servers.

## Creating a Custom MCP Server

Creating a custom MCP server involves defining tools that agents can call and implementing the server logic. Here's the basic structure:

```python
from mcp.server.fastmcp import FastMCP

# Create an MCP server with a descriptive name
mcp = FastMCP("CustomToolsServer")

# Define tools as decorated functions
@mcp.tool()
def custom_tool_name(param1: str, param2: int) -> dict:
    """
    Tool description that explains what the tool does.
    
    Args:
        param1: Description of the first parameter
        param2: Description of the second parameter
    
    Returns:
        Description of the return value
    """
    # Tool implementation
    result = {}
    # ... process the parameters and generate a result
    return result

# Run the server when the script is executed directly
if __name__ == "__main__":
    mcp.run()
```

## Tool Definition Best Practices

When defining tools for your MCP server, follow these best practices:

1. **Clear Names**: Use descriptive names that indicate the tool's purpose
2. **Type Annotations**: Include proper type annotations for parameters and return values
3. **Comprehensive Docstrings**: Provide detailed descriptions of the tool, parameters, and return values
4. **Error Handling**: Implement robust error handling to prevent server crashes
5. **Parameter Validation**: Validate input parameters before processing
6. **Reasonable Defaults**: Provide sensible default values for optional parameters

## Example: Custom Data Analysis MCP Server

Here's an example of a custom MCP server for data analysis:

```python
# data_analysis_mcp_server.py
from mcp.server.fastmcp import FastMCP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64

# Create an MCP server for data analysis
mcp = FastMCP("DataAnalysisServer")

@mcp.tool()
def calculate_statistics(data: list) -> dict:
    """
    Calculate basic statistics for a list of numerical values.
    
    Args:
        data: List of numerical values
    
    Returns:
        Dictionary containing statistics (mean, median, std_dev, min, max)
    """
    try:
        # Convert to numpy array for calculations
        arr = np.array(data, dtype=float)
        
        # Calculate statistics
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std_dev": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "count": len(arr)
        }
        
        return stats
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def generate_chart(
    data: list,
    chart_type: str = "line",
    title: str = "Chart",
    x_label: str = "X",
    y_label: str = "Y"
) -> str:
    """
    Generate a chart from data and return as base64-encoded image.
    
    Args:
        data: List of numerical values
        chart_type: Type of chart (line, bar, scatter)
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
    
    Returns:
        Base64-encoded PNG image
    """
    try:
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Generate x values (indices)
        x = list(range(len(data)))
        
        # Create chart based on type
        if chart_type == "line":
            plt.plot(x, data)
        elif chart_type == "bar":
            plt.bar(x, data)
        elif chart_type == "scatter":
            plt.scatter(x, data)
        else:
            plt.plot(x, data)  # Default to line chart
        
        # Add labels and title
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(True)
        
        # Save to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return img_base64
    except Exception as e:
        return {"error": str(e)}

@mcp.tool()
def perform_correlation_analysis(data_x: list, data_y: list) -> dict:
    """
    Calculate correlation between two data series.
    
    Args:
        data_x: First data series
        data_y: Second data series
    
    Returns:
        Dictionary with correlation coefficient and p-value
    """
    try:
        # Check if data lengths match
        if len(data_x) != len(data_y):
            return {"error": "Data series must have the same length"}
        
        # Convert to numpy arrays
        x = np.array(data_x, dtype=float)
        y = np.array(data_y, dtype=float)
        
        # Calculate correlation
        correlation = np.corrcoef(x, y)[0, 1]
        
        return {
            "correlation_coefficient": float(correlation),
            "correlation_strength": interpret_correlation(correlation)
        }
    except Exception as e:
        return {"error": str(e)}

def interpret_correlation(coefficient):
    """Helper function to interpret correlation strength."""
    abs_coef = abs(coefficient)
    if abs_coef < 0.3:
        return "weak"
    elif abs_coef < 0.7:
        return "moderate"
    else:
        return "strong"

# Run the server
if __name__ == "__main__":
    mcp.run()
```

## Deploying Custom MCP Servers

To deploy your custom MCP server:

1. Save your server script to a file (e.g., `custom_mcp_server.py`)
2. Run the script to start the server:

```bash
python custom_mcp_server.py
```

By default, the server will run on `localhost:8000`. You can specify a different port:

```bash
python custom_mcp_server.py --port 8080
```

## Integrating Custom MCP Servers with Agents

To use your custom MCP server with agents, configure an MCP server environment:

```python
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment

# Create MCP server mechanism for your custom server
custom_mcp = MCPServer(
    server_url="http://localhost:8080",  # URL of your custom MCP server
    server_name="CustomToolsServer"      # Name of your custom MCP server
)

# Create MCP server environment
custom_env = MCPServerEnvironment(
    name="custom_tools",
    mechanism=custom_mcp
)

# Add environment to agent
agent.environments["custom_tools"] = custom_env
```

## Example: Using a Custom Data Analysis MCP Server

Here's an example of using the custom data analysis MCP server:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.mcp_server import MCPServer, MCPServerEnvironment
from market_agents.agents.cognitive_steps import CognitiveStep
from minference.lite.models import LLMConfig

class DataAnalysisStep(CognitiveStep):
    step_name: str = Field(default="data_analysis", description="Data analysis step")
    data: list = Field(default=[], description="Data to analyze")
    
    async def execute(self, agent):
        # Call MCP tool to calculate statistics
        stats_action = {
            "tool_name": "calculate_statistics",
            "parameters": {"data": self.data}
        }
        stats_result = agent.environments["data_analysis"].step(stats_action)
        
        # Call MCP tool to generate a chart
        chart_action = {
            "tool_name": "generate_chart",
            "parameters": {
                "data": self.data,
                "chart_type": "line",
                "title": "Data Analysis",
                "x_label": "Index",
                "y_label": "Value"
            }
        }
        chart_result = agent.environments["data_analysis"].step(chart_action)
        
        # Generate analysis based on the data
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Analyze the following data statistics:
        
        {self._format_dict(stats_result.get('result', {}))}
        
        Based on these statistics, provide a comprehensive analysis of the data.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Return analysis and chart
        return {
            "analysis": response.content,
            "statistics": stats_result.get('result', {}),
            "chart_base64": chart_result.get('result', "")
        }
    
    def _format_dict(self, data):
        if not data:
            return "No data available"
        
        formatted = ""
        for key, value in data.items():
            formatted += f"- {key}: {value}\n"
        return formatted

async def run_data_analysis():
    # Create storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Data Analyst",
        persona="I am a data analyst specializing in statistical analysis and data visualization.",
        objectives=["Analyze data patterns", "Provide statistical insights"]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="data_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
    # Configure Data Analysis MCP server
    data_mcp = MCPServer(
        server_url="http://localhost:8080",
        server_name="DataAnalysisServer"
    )
    data_env = MCPServerEnvironment(
        name="data_analysis",
        mechanism=data_mcp
    )
    
    # Add environment to agent
    agent.environments["data_analysis"] = data_env
    
    # Sample data (stock prices over time)
    sample_data = [150.25, 152.75, 151.80, 153.20, 155.45, 154.30, 156.90, 158.25, 157.75, 160.10]
    
    # Run data analysis
    analysis_result = await agent.run_step(
        step=DataAnalysisStep(data=sample_data)
    )
    
    return analysis_result

# Run the analysis
analysis_result = asyncio.run(run_data_analysis())

# Display the analysis
print(analysis_result["analysis"])

# Save the chart if available
if "chart_base64" in analysis_result and analysis_result["chart_base64"]:
    import base64
    
    # Decode base64 image
    img_data = base64.b64decode(analysis_result["chart_base64"])
    
    # Save to file
    with open("data_analysis_chart.png", "wb") as f:
        f.write(img_data)
    
    print("Chart saved to data_analysis_chart.png")
```

## Advanced MCP Server Features

### Authentication

You can add authentication to your MCP server:

```python
from mcp.server.fastmcp import FastMCP, MCPAuth

# Define authentication
auth = MCPAuth(
    api_keys=["your-secret-api-key"],
    header_name="X-API-Key"
)

# Create MCP server with authentication
mcp = FastMCP("SecureServer", auth=auth)

# Define tools
@mcp.tool()
def secure_tool(param: str) -> str:
    # This tool is now protected by authentication
    return f"Processed: {param}"
```

### Tool Dependencies

You can create tools that depend on other tools:

```python
@mcp.tool()
def get_stock_data(symbol: str) -> dict:
    # Get stock data
    # ...
    return data

@mcp.tool()
def analyze_stock(symbol: str) -> dict:
    # This tool depends on get_stock_data
    data = get_stock_data(symbol)
    # Analyze the data
    # ...
    return analysis
```

### Asynchronous Tools

For tools that perform I/O operations, you can use async functions:

```python
@mcp.tool()
async def fetch_market_data(symbol: str) -> dict:
    """
    Fetch market data asynchronously.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(f"https://api.example.com/market/{symbol}") as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"Failed to fetch data: {response.status}"}
```

This concludes our documentation on MCP server integration in the MarketAgents framework. In the next section, we'll create system architecture diagrams to visualize the framework's components and their relationships.
