# Structured Output Methods with VLLM in Parallel AI Inference

This document outlines the structured output methods implemented with VLLM (OpenAI-compatible) in our inference setup for parallel AI inference in multi-agent simulations.

## Implemented Methods

We have implemented three methods to get structured output (JSON object) from LLM API calls:

1. JSON Object
2. JSON Schema
3. Tool Choice (Function Call)

## Implementation Details

### StructuredTool Class

The `StructuredTool` Pydantic class in `message_models.py` is the core implementation for most of our structured output methods. It provides a flexible way to define and manage structured output schemas.

### VLLM Request Creation

In `parallel_inference.py`, we have a method to create VLLM requests with the appropriate structured output response format. This method handles the different structured output methods we've implemented.

### VLLM-specific Configuration

The `VLLMConfig` and `VLLMRequest` classes in `clients_models.py` contain VLLM-specific configurations for structured output. These classes extend the base OpenAI request model with additional VLLM-specific parameters.

## Method Details

### 1. JSON Object

The JSON object method ensures that the response is in JSON format, without guaranteeing a specific schema.

### 2. JSON Schema

This method uses a predefined JSON schema to structure the output, ensuring that the response adheres to a specific format.

### 3. Tool Choice (Function Call)

This method simulates function calls, allowing for more complex structured interactions with the model.

Each of these methods is implemented using the `StructuredTool` class and configured in the VLLM request creation process.
