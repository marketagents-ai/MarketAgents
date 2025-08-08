# Financial Analysis Agent System: Comprehensive Configuration Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Agent Configuration Structure](#agent-configuration-structure)
3. [Detailed Field Explanations](#detailed-field-explanations)
4. [Best Practices](#best-practices)
5. [Advanced Configuration Options](#advanced-configuration-options)
6. [Examples](#examples)

## Introduction

This guide provides comprehensive instructions for creating, modifying, and managing agent configurations in our Financial Analysis Agent System. Our system uses a multi-agent approach to analyze financial data, conduct research, provide investment advice, and summarize findings.

## Agent Configuration Structure

Each agent in our system is defined by a JSON object. Here's the basic structure:

```json
{
    "name": "AgentName",
    "default_tools": ["tool1", "tool2"],
    "tools": ["specialTool1", "specialTool2"],
    "llm_config": {
        "client": "providerName",
        "model": "modelName",
        "temperature": 0.0,
        "response_format": {"type": "format_type"}
    },
    "prompt_type": "promptTypeName",
    "output_format": "OutputFormatName",
    "dependencies": ["DependencyAgent1", "DependencyAgent2"]
}
```

## Detailed Field Explanations

### 1. Name
- **Purpose**: Unique identifier for the agent
- **Format**: String
- **Example**: `"name": "QuantitativeFinancialAnalyst"`
- **Best Practice**: Use CamelCase, be descriptive and specific

### 2. Default Tools
- **Purpose**: Basic tools available to all agents
- **Format**: Array of strings
- **Example**: `"default_tools": ["web_search", "rag_search", "calculator"]`
- **Modification**: Add or remove tools based on global system changes

### 3. Tools
- **Purpose**: Specialized tools for this specific agent
- **Format**: Array of strings
- **Example**:
  ```json
  "tools": [
    "get_stock_data",
    "analyze_financial_statements",
    "calculate_financial_ratios"
  ]
  ```
- **Best Practice**: Only include tools necessary for the agent's specific tasks

### 4. LLM Config
- **Purpose**: Configures the language model for this agent
- **Format**: Object with specific properties
- **Properties**:
  - `client`: LLM provider (e.g., "groq", "openai")
  - `model`: Specific model name
  - `temperature`: Controls randomness (0.0 to 1.0)
  - `response_format`: Specifies output format
- **Example**:
  ```json
  "llm_config": {
    "client": "groq",
    "model": "llama3-70b-8192",
    "temperature": 0.3,
    "response_format": {"type": "json_object"}
  }
  ```
- **Customization**: Adjust based on task requirements and available models

### 5. Prompt Type
- **Purpose**: Defines the type of prompt used by this agent
- **Format**: String
- **Example**: `"prompt_type": "financial_ratio_analyst"`
- **Best Practice**: Create specific prompt types for different analysis tasks

### 6. Output Format
- **Purpose**: Specifies the structure of the agent's output
- **Format**: String (simple) or Object (detailed schema)
- **Example** (detailed):
  ```json
  "output_format": {
    "type": "object",
    "properties": {
      "stockAnalysis": {"type": "string"},
      "keyMetrics": {
        "type": "object",
        "properties": {
          "peRatio": {"type": "number"},
          "debtToEquity": {"type": "number"}
        }
      },
      "recommendation": {"type": "string"}
    },
    "required": ["stockAnalysis", "keyMetrics", "recommendation"]
  }
  ```
- **Best Practice**: Define clear, structured outputs for easy integration

### 7. Dependencies
- **Purpose**: Lists other agents this agent relies on
- **Format**: Array of strings
- **Example**: `"dependencies": ["MarketDataCollector", "EconomicTrendAnalyzer"]`
- **Consideration**: Ensure dependency chain doesn't create circular references


### Error Handling
```json
"error_handling": {
  "retry_attempts": 3,
  "fallback_strategy": "use_cached_data"
}
```

### Rate Limiting
```json
"rate_limit": {
  "max_requests_per_minute": 60,
  "burst_limit": 10
}
```

### Logging Configuration
```json
"logging": {
  "level": "INFO",
  "include_timestamps": true,
  "log_to_file": "agent_name.log"
}
```

### Performance Metrics
```json
"performance_metrics": {
  "accuracy_threshold": 0.95,
  "max_response_time": 5000
}
```

## Examples

### Basic Financial Analyst Agent
```json
{
  "name": "BasicFinancialAnalyst",
  "default_tools": ["web_search", "rag_search"],
  "tools": ["get_stock_data", "calculate_financial_ratios"],
  "llm_config": {
    "client": "groq",
    "model": "llama3-70b-8192",
    "temperature": 0.2,
    "response_format": {"type": "json_object"}
  },
  "prompt_type": "basic_financial_analysis",
  "output_format": "BasicFinancialAnalysisOutput",
}
```
