"""
Entity models with registry integration and serialization support.

This module provides:
1. Base Entity class for registry-integrated, serializable objects
2. CallableTool implementation for managing executable functions
3. Serialization and persistence capabilities
4. Registry integration for both entities and callables
"""
from typing import Dict, Any, Optional, ClassVar, Type, TypeVar, List, Generic, Callable, Literal, Union, Tuple, Self
from enum import Enum

from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator, computed_field
from pathlib import Path
import json
from datetime import datetime
import inspect
from jsonschema import validate
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition
from anthropic.types import ToolParam, CacheControlEphemeralParam
from minference.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

from minference.enregistry import EntityRegistry
from minference.caregistry import (
    CallableRegistry,
    derive_input_schema,
    derive_output_schema,
    validate_schema_compatibility,
)
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema

from openai.types.shared_params import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    FunctionDefinition
)
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletion
)
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolUseBlock,
    ToolParam,
    TextBlockParam,
    ToolResultBlockParam,
    Message as AnthropicMessage
)
T = TypeVar('T', bound='Entity')

class Entity(BaseModel):
    """
    Base class for registry-integrated, serializable entities.
    
    This class provides:
    1. Integration with EntityRegistry for persistence and retrieval
    2. Basic serialization interface for saving/loading
    3. Common entity attributes and operations
    
    Subclasses are responsible for:
    1. Implementing custom serialization if needed (_custom_serialize/_custom_deserialize)
    2. Handling any nested entities or complex relationships
    3. Managing entity-specific validation and business logic
    
    All entities are immutable - modifications require creating new instances.
    """
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for this entity instance"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when this entity was created"
    )
    
    class Config:
        # Add JSON encoders for UUID and datetime
        json_encoders = {
            UUID: str,  # Convert UUID to string
            datetime: lambda v: v.isoformat()  # Convert datetime to ISO format string
        }
    
    @model_validator(mode='after')
    def register_entity(self) -> Self:
        """Register this entity instance in the registry."""
        registry = EntityRegistry
        registry._logger.info(f"{self.__class__.__name__}({self.id}): Registering entity")
        
        try:
            registry.register(self)
            registry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully registered")
        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Registration failed - {str(e)}")
            raise ValueError(f"Entity registration failed: {str(e)}") from e
            
        return self
    
    def _custom_serialize(self) -> Dict[str, Any]:
        """
        Custom serialization hook for subclasses.
        
        Override this method to add custom serialization logic.
        The result will be included in the serialized output under 'custom_data'.
        
        Returns:
            Dict containing any custom serialized data
        """
        return {}
        
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Custom deserialization hook for subclasses.
        
        Override this method to handle custom deserialization logic.
        The input comes from the 'custom_data' field in serialized data.
        
        Args:
            data: Custom data from serialized entity
            
        Returns:
            Dict of deserialized fields to include in entity initialization
        """
        return {}
    
    def save(self, path: Path) -> None:
        """
        Save this entity instance to a file.
        
        Args:
            path: Path where to save the entity
            
        Raises:
            IOError: If saving fails
        """
        registry = EntityRegistry
        registry._logger.info(f"{self.__class__.__name__}({self.id}): Saving to {path}")
        
        try:
            # Get base serialization
            data = self.model_dump()
            
            # Add metadata
            metadata = {
                "entity_type": self.__class__.__name__,
                "schema_version": "1.0",
                "saved_at": datetime.utcnow().isoformat()
            }
            
            # Get custom data
            custom_data = self._custom_serialize()
            
            # Combine all
            serialized = {
                "metadata": metadata,
                "data": data,
                "custom_data": custom_data
            }
            
            # Save
            with open(path, 'w') as f:
                json.dump(serialized, f, indent=2)
                
            registry._logger.info(f"{self.__class__.__name__}({self.id}): Successfully saved")
            
        except Exception as e:
            registry._logger.error(f"{self.__class__.__name__}({self.id}): Save failed - {str(e)}")
            raise IOError(f"Failed to save entity: {str(e)}") from e
    
    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        """
        Load an entity instance from a file.
        
        Args:
            path: Path to the saved entity
            
        Returns:
            Loaded entity instance
            
        Raises:
            IOError: If loading fails
            ValueError: If data validation fails
        """
        registry = EntityRegistry
        registry._logger.info(f"{cls.__name__}: Loading from {path}")
        
        try:
            # Load file
            with open(path) as f:
                serialized = json.load(f)
                
            # Verify entity type
            metadata = serialized["metadata"]
            if metadata["entity_type"] != cls.__name__:
                raise ValueError(
                    f"Entity type mismatch. File contains {metadata['entity_type']}, "
                    f"expected {cls.__name__}"
                )
                
            # Get base and custom data
            base_data = serialized["data"]
            custom_data = cls._custom_deserialize(serialized.get("custom_data", {}))
            
            # Create instance
            instance = cls(**{**base_data, **custom_data})
            registry._logger.info(f"{cls.__name__}({instance.id}): Successfully loaded")
            return instance
            
        except Exception as e:
            registry._logger.error(f"{cls.__name__}: Load failed - {str(e)}")
            raise IOError(f"Failed to load entity: {str(e)}") from e
            
    @classmethod
    def get(cls: Type[T], entity_id: UUID) -> Optional[T]:
        """Get an entity instance from the registry."""
        return EntityRegistry.get(entity_id, expected_type=cls)
        
    @classmethod
    def list_all(cls: Type[T]) -> List[T]:
        """List all entities of this type."""
        return EntityRegistry.list_by_type(cls)
        
    @classmethod
    def get_many(cls: Type[T], entity_ids: List[UUID]) -> List[T]:
        """Get multiple entities by their IDs."""
        return EntityRegistry.get_many(entity_ids, expected_type=cls)


class CallableTool(Entity):
    """
    An immutable callable tool that can be registered and executed with schema validation.
    
    Inherits from Entity for registry integration and serialization support.
    The tool is registered in:
    - CallableRegistry: For the actual function registration and execution
    - EntityRegistry: For the tool metadata and versioning (handled by parent)
    
    Any modifications require creating new instances with new UUIDs.
    """    
    name: str = Field(
        description="Registry name for the callable function",
        min_length=1
    )
    
    docstring: Optional[str] = Field(
        default=None,
        description="Documentation describing the tool's purpose and usage"
    )
    
    input_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining valid input parameters"
    )
    
    output_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining the expected output format"
    )
    
    strict_schema: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation"
    )
    
    callable_text: Optional[str] = Field(
        default=None,
        description="Source code of the callable function"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "multiply",
                    "docstring": "Multiplies two numbers together",
                    "callable_text": "def multiply(x: float, y: float) -> float:\n    return x * y"
                }
            ]
        }
    }
    
    @classmethod
    def _get_or_register_function(
        cls,
        func: Optional[Callable] = None,
        name: Optional[str] = None,
        callable_text: Optional[str] = None
    ) -> tuple[str, Callable]:
        """
        Helper method to get an existing function from registry or register a new one.
        
        Args:
            func: Optional callable to register
            name: Optional name for the function
            callable_text: Optional source code for the function
            
        Returns:
            Tuple of (function_name, callable_function)
            
        Raises:
            ValueError: If insufficient information provided or registration fails
        """
        registry = CallableRegistry
        
        # Determine function name
        if name is None:
            if func is not None:
                name = func.__name__
            else:
                raise ValueError("Must provide either a name or a callable function")
                
        # Check if function already exists in registry
        existing_func = registry.get(name)
        if existing_func is not None:
            return name, existing_func
            
        # Register new function
        if func is not None:
            registry.register(name, func)
            return name, func
        elif callable_text is not None:
            registry.register_from_text(name, callable_text)
            registered_func = registry.get(name)
            if registered_func is None:
                raise ValueError(f"Failed to register function '{name}' from source code")
            return name, registered_func
        else:
            raise ValueError(
                f"No function found in registry for '{name}' and no function or "
                "callable_text provided for registration"
            )

    @classmethod
    def from_callable(cls, func: Callable, name: Optional[str] = None, docstring: Optional[str] = None, strict_schema: bool = True) -> 'CallableTool':
        """Creates a new tool from a callable function."""
        # Get function source if possible
        try:
            callable_text = inspect.getsource(func)
        except (TypeError, OSError):
            callable_text = str(func)
            
        # Get or register the function
        func_name, registered_func = cls._get_or_register_function(
            func=func,
            name=name,
            callable_text=callable_text
        )
        
        # Derive schemas
        input_schema = derive_input_schema(registered_func)
        output_schema = derive_output_schema(registered_func)
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=docstring or func.__doc__,
            input_schema=input_schema,
            output_schema=output_schema,
            strict_schema=strict_schema,
            callable_text=callable_text
        )

    @classmethod
    def from_registry(cls, name: str) -> 'CallableTool':
        """Creates a new tool from an existing registry entry."""
        # Get function from registry
        func_name, registered_func = cls._get_or_register_function(name=name)
        
        # Get function source if possible
        try:
            callable_text = inspect.getsource(registered_func)
        except (TypeError, OSError):
            callable_text = str(registered_func)
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=registered_func.__doc__,
            callable_text=callable_text
        )

    @classmethod
    def from_source(cls, source: str, name: Optional[str] = None, docstring: Optional[str] = None, strict_schema: bool = True) -> 'CallableTool':
        """Creates a new tool from source code string."""
        # Get or register the function
        func_name, registered_func = cls._get_or_register_function(
            callable_text=source,
            name=name
        )
        
        # Create tool instance
        return cls(
            name=func_name,
            docstring=docstring or registered_func.__doc__,
            strict_schema=strict_schema,
            callable_text=source
        )

    @model_validator(mode='after')
    def validate_schemas_and_callable(self) -> Self:
        """Validates the tool's schemas and ensures its callable is registered."""
        # Get or register the function
        func_name, func = self._get_or_register_function(
            name=self.name,
            callable_text=self.callable_text
        )
        
        # Store text representation if not provided
        if not self.callable_text:
            try:
                self.callable_text = inspect.getsource(func)
            except (TypeError, OSError):
                self.callable_text = str(func)
        
        # Derive and validate schemas
        derived_input = derive_input_schema(func)
        derived_output = derive_output_schema(func)
        
        # Validate and set input schema
        if self.input_schema:
            validate_schema_compatibility(derived_input, self.input_schema)
        else:
            self.input_schema = derived_input
            
        # Validate and set output schema
        if self.output_schema:
            validate_schema_compatibility(derived_output, self.output_schema)
        else:
            self.output_schema = derived_output
            
        return self
    
    def _custom_serialize(self) -> Dict[str, Any]:
        """Serialize the callable-specific data."""
        return {
            "schemas": {
                "input": self.input_schema,
                "output": self.output_schema
            }
        }
    
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize the callable-specific data."""
        schemas = data.get("schemas", {})
        return {
            "input_schema": schemas.get("input", {}),
            "output_schema": schemas.get("output", {})
        }
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Executes the callable with the given input data."""
        ca_registry = CallableRegistry
        ca_registry._logger.info(f"CallableTool({self.id}): Executing '{self.name}'")
        
        try:
            result = ca_registry.execute(self.name, input_data)
            ca_registry._logger.info(f"CallableTool({self.id}): Execution successful")
            return result
        except Exception as e:
            ca_registry._logger.error(f"CallableTool({self.id}): Execution failed for '{self.name}'")
            raise ValueError(f"Error executing {self.name}: {str(e)}") from e
    
    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the callable with the given input data asynchronously.
        
        Will execute async functions natively and wrap sync functions in
        asyncio.to_thread automatically.
        
        Args:
            input_data: Input data for the callable
            
        Returns:
            Execution results
            
        Raises:
            ValueError: If execution fails
        """
        ca_registry = CallableRegistry
        ca_registry._logger.info(f"CallableTool({self.id}): Executing '{self.name}' asynchronously")
        
        try:
            result = await ca_registry.aexecute(self.name, input_data)
            ca_registry._logger.info(f"CallableTool({self.id}): Async execution successful")
            return result
        except Exception as e:
            ca_registry._logger.error(f"CallableTool({self.id}): Async execution failed for '{self.name}'")
            raise ValueError(f"Error executing {self.name} asynchronously: {str(e)}") from e
    
    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        """Get OpenAI tool format using the callable's schema."""
        if self.input_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    parameters=self.input_schema
                )
            )
        return None

    def get_anthropic_tool(self, use_cache:bool=True) -> Optional[ToolParam]:
        """Get Anthropic tool format using the callable's schema."""
        if self.input_schema:
            if use_cache:
                return ToolParam(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    input_schema=self.input_schema,
                    cache_control=CacheControlEphemeralParam(type='ephemeral')
                )
            else:
                return ToolParam(
                    name=self.name,
                    description=self.docstring or f"Execute {self.name} function",
                    input_schema=self.input_schema,
            )
        return None

class CallableMCPTool(CallableTool):
    """
    A callable tool intended for MCP server execution.
    delegates the call to the MCP server's call_tool interface.
    """
    mcp_mode: bool = Field(default=True, description="Indicates tool execution via MCP server")
    mcp_mechanism: Optional[Any] = Field(default=None, description="Reference to the MCP server mechanism")

    @classmethod
    def from_callable(
        cls, 
        name: str, 
        description: Optional[str] = None, 
        input_schema: Dict[str, Any] = None
    ) -> 'CallableMCPTool':
        """
        Creates a new MCP tool with only the parameters available from MCP server.
        
        Args:
            name: Unique identifier for the tool
            description: Human-readable description
            input_schema: JSON Schema for the tool's parameters
        """
        # Create a dummy function with type hints based on the input_schema
        def dummy_func(input_data: Dict[str, Any] = None) -> Dict[str, Any]:
            return {}
            
        dummy_func.__name__ = name
        dummy_func.__doc__ = description

        # Call parent class's from_callable with our dummy function
        tool = super().from_callable(
            func=dummy_func,
            name=name,
            docstring=description,
            strict_schema=True
        )

        # Override the schemas with the ones from MCP server
        tool.input_schema = input_schema or {"type": "object", "properties": {}}
        tool.output_schema = {}

        return tool

    async def aexecute(self, chat_thread_id: str, input_data: Dict[str, Any]) -> Any:
            """Execute the tool asynchronously"""
            try:
                # Execute the tool
                result = await self.mcp_mechanism.execute_tool(
                    tool_name=self.name, 
                    arguments=input_data,
                    chat_thread_id=chat_thread_id
                )
                EntityRegistry._logger.info(f"Raw MCP tool result: {result}")
                
                # Early return if result is None
                if result is None:
                    return None
                    
                # Handle MCP-specific response format
                if isinstance(result, dict):
                    # Check if it's an error response
                    if result.get('isError', False):
                        raise Exception(str(result.get('content', 'Unknown error')))
                        
                    # Extract content from MCP response
                    if 'content' in result:
                        for content_item in result['content']:
                            if content_item.get('type') == 'text':
                                text_content = content_item['text']
                                try:
                                    # Try to parse as JSON first
                                    parsed_content = json.loads(text_content)
                                    EntityRegistry._logger.info(f"Successfully parsed tool result: {parsed_content}")
                                    return parsed_content
                                except json.JSONDecodeError:
                                    # Return as plain text if not JSON
                                    EntityRegistry._logger.info(f"Returning text content: {text_content}")
                                    return text_content
                    
                    # If we couldn't extract content, return the raw result
                    EntityRegistry._logger.info(f"Returning raw result: {result}")
                    return result
                
                # If result is not a dict, return as is
                return result
                
            except Exception as e:
                EntityRegistry._logger.error(f"{self.__class__.__name__}({self.name}): Async execution failed: {str(e)}")
                # Don't raise TaskGroup errors if we have a result
                if "unhandled errors in a TaskGroup" not in str(e):
                    raise
                return None
        
class StructuredTool(Entity):
    """
    Entity representing a tool for structured output with schema validation and LLM integration.
    
    Inherits from Entity for registry integration and automatic registration.
    Provides schema validation and LLM format conversion for structured outputs.
    """
    name: str = Field(
        default="generate_structured_output",
        description="Name for the structured output schema"
    )
    
    description: str = Field(
        default="Generate a structured output based on the provided JSON schema.",
        description="Description of what the structured output represents"
    )
    
    instruction_string: str = Field(
        default="Please follow this JSON schema for your response:",
        description="Instruction to prepend to schema for LLM"
    )
    
    json_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description="JSON schema defining the expected structure"
    )
    
    strict_schema: bool = Field(
        default=True,
        description="Whether to enforce strict schema validation"
    )

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute structured output validation."""
        EntityRegistry._logger.info(f"StructuredTool({self.id}): Validating input for '{self.name}'")
        
        try:
            # Validate input against schema
            validate(instance=input_data, schema=self.json_schema)
            EntityRegistry._logger.info(f"StructuredTool({self.id}): Validation successful")
            return input_data
            
        except Exception as e:
            EntityRegistry._logger.error(f"StructuredTool({self.id}): Validation failed - {str(e)}")
            return {"error": str(e)}

    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async version of execute - for consistency with CallableTool interface.
        Since validation is CPU-bound, we don't need true async here.
        """
        return self.execute(input_data)

    def _custom_serialize(self) -> Dict[str, Any]:
        """Serialize tool-specific data."""
        return {
            "json_schema": self.json_schema,
            "description": self.description,
            "instruction": self.instruction_string
        }
    
    @classmethod
    def _custom_deserialize(cls, data: Dict[str, Any]) -> Dict[str, Any]:
        """Deserialize tool-specific data."""
        return {
            "json_schema": data.get("json_schema", {}),
            "description": data.get("description", ""),
            "instruction_string": data.get("instruction", "")
        }

    @property
    def schema_instruction(self) -> str:
        """Get formatted schema instruction for LLM."""
        return f"{self.instruction_string}: {self.json_schema}"

    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        """Get OpenAI tool format."""
        if self.json_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.name,
                    description=self.description,
                    parameters=self.json_schema
                )
            )
        return None

    def get_anthropic_tool(self,use_cache:bool=True) -> Optional[ToolParam]:
        """Get Anthropic tool format."""
        if self.json_schema:
            if use_cache:
                return ToolParam(
                    name=self.name,
                    description=self.description,
                    input_schema=self.json_schema,
                    cache_control=CacheControlEphemeralParam(type='ephemeral')
                )
            else:
                return ToolParam(
                    name=self.name,
                    description=self.description,
                    input_schema=self.json_schema,
            )
        return None

    def get_openai_json_schema_response(self) -> Optional[ResponseFormatJSONSchema]:
        """Get OpenAI JSON schema response format."""
        if self.json_schema:
            schema = JSONSchema(
                name=self.name,
                description=self.description,
                schema=self.json_schema,
                strict=self.strict_schema
            )
            return ResponseFormatJSONSchema(type="json_schema", json_schema=schema)
        return None

    @classmethod
    def from_pydantic(
        cls,
        model: Type[BaseModel],
        name: Optional[str] = None,
        description: Optional[str] = None,
        instruction_string: Optional[str] = None,
        strict_schema: bool = True
    ) -> 'StructuredTool':
        """
        Create a StructuredTool from a Pydantic model.
        
        Args:
            model: Pydantic model class
            name: Optional override for tool name (defaults to model name)
            description: Optional override for description (defaults to model docstring)
            instruction_string: Optional custom instruction
            strict_schema: Whether to enforce strict schema validation
        """
        if not issubclass(model, BaseModel):
            raise ValueError("Model must be a Pydantic model")
            
        # Get model schema
        schema = model.model_json_schema()
        
        # Use model name if not provided
        tool_name = name or model.__name__.lower()
        
        # Use model docstring if no description provided
        tool_description = description or model.__doc__ or f"Generate {tool_name} structured output"
        
        return cls(
            name=tool_name,
            json_schema=schema,
            description=tool_description,
            instruction_string=instruction_string or cls.model_fields["instruction_string"].default,
            strict_schema=strict_schema
        )

class LLMClient(str, Enum):
    openai = "openai"
    azure_openai = "azure_openai"
    anthropic = "anthropic"
    vllm = "vllm"
    litellm = "litellm"

class ResponseFormat(str, Enum):
    json_beg = "json_beg"
    text = "text"
    json_object = "json_object"
    structured_output = "structured_output"
    tool = "tool"
    auto_tools = "auto_tools"
    workflow = "workflow"
    
class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"

class LLMConfig(Entity):
    """
    Configuration entity for LLM interactions.
    
    Specifies the client, model, and response format settings
    for interacting with various LLM providers.
    """
    client: LLMClient = Field(
        description="The LLM client/provider to use"
    )
    
    model: Optional[str] = Field(
        default=None,
        description="Model identifier for the LLM"
    )
    
    max_tokens: int = Field(
        default=400,
        description="Maximum number of tokens in completion",
        ge=1
    )
    
    temperature: float = Field(
        default=0,
        description="Sampling temperature",
        ge=0,
        le=2
    )
    
    response_format: ResponseFormat = Field(
        default=ResponseFormat.text,
        description="Format for LLM responses"
    )
    
    use_cache: bool = Field(
        default=True,
        description="Whether to use response caching"
    )

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        """Validate response format compatibility with selected client."""
        if (self.response_format == ResponseFormat.json_object and 
            self.client in [LLMClient.vllm, LLMClient.litellm, LLMClient.anthropic]):
            raise ValueError(f"{self.client} does not support json_object response format")
            
        if (self.response_format == ResponseFormat.structured_output and 
            self.client == LLMClient.anthropic):
            raise ValueError(
                f"Anthropic does not support structured_output response format. "
                "Use json_beg or tool instead"
            )
            
        return self


ToolType = Literal["Callable", "CallableMCP", "Structured"]

class ChatMessage(Entity):
    """A chat message entity using chatml format."""
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the message was created"
    )
    
    role: MessageRole = Field(
        description="Role of the message sender" 
    )
    
    content: str = Field(
        description="The message content"
    )

    author_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the author of the message"
    )

    chat_thread_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the chat thread this message belongs to"
    )
    
    parent_message_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the parent message in the conversation"
    )
    
    tool_name: Optional[str] = Field(
        default=None, 
        description="Name of the tool if this is a tool-related message"
    )

    tool_uuid: Optional[UUID] = Field(
        default=None,
        description="UUID of the tool in our EntityRegistry"
    )

    tool_type: Optional[ToolType] = Field(
        default=None,
        description="Type of tool - either Callable or Structured"
    )
    
    oai_tool_call_id: Optional[str] = Field(
        default=None,
        description="OAI tool call id"
    )
    
    tool_json_schema: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON schema for tool input/output"
    )
    
    tool_call: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Tool call details and arguments"
    )

    usage: Optional['Usage'] = Field(
        default=None,
        description="Usage statistics for the message"
    )


    @property
    @computed_field
    def is_root(self) -> bool:

        """Check if this is a root message (no parent)."""
        return self.parent_message_uuid is None

    def get_parent(self) -> Optional['ChatMessage']:
        """Get the parent message if it exists."""
        if self.parent_message_uuid:
            return ChatMessage.get(self.parent_message_uuid)
        return None

    def get_tool(self) -> Optional[Union['CallableTool', 'StructuredTool']]:
        """Get the associated tool from the registry if it exists."""
        if not self.tool_uuid:
            return None
        
        if self.tool_type == "Callable":
            return CallableTool.get(self.tool_uuid)
        elif self.tool_type == "Structured":
            return StructuredTool.get(self.tool_uuid)
        return None
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to chatml format dictionary."""
        if self.role == MessageRole.tool:
            return {
                "role": self.role.value,
                "content": self.content,
                "tool_call_id": self.oai_tool_call_id
            }
        elif self.role == MessageRole.assistant and self.oai_tool_call_id is not None:
            return {
                "role": self.role.value,
                "content": self.content,
                "tool_calls": [{
                    "id": self.oai_tool_call_id,
                    "function": {
                        "arguments": json.dumps(self.tool_call),
                        "name": self.tool_name
                    },
                    "type": "function"
                }]
            }
        else:
            return {
                "role": self.role.value,
                "content": self.content
            }

    @classmethod
    def from_dict(cls, message_dict: Dict[str, Any]) -> 'ChatMessage':
        """Create a ChatMessage from a chatml format dictionary."""
        return cls(
            role=MessageRole(message_dict["role"]),
            content=message_dict["content"]
        )

class SystemPrompt(Entity):
    """Entity representing a reusable system prompt."""
    
    name: str = Field(
        description="Identifier name for the system prompt"
    )
    
    content: str = Field(
        description="The system prompt text content"
    )



class Usage(Entity):
    """Tracks token usage for LLM interactions."""
    prompt_tokens: int = Field(
        description="Number of tokens in the prompt"
    )
    completion_tokens: int = Field(
        description="Number of tokens in the completion"
    )
    total_tokens: int = Field(
        description="Total number of tokens used"
    )
    cache_creation_input_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens used in cache creation with Anthropic endpoint"
    )
    cache_read_input_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens read from cache with Anthropic endpoint"
    )
    accepted_prediction_tokens: Optional[int] = Field(
        default=None,
        description="Number of accepted prediction tokens using OpenAI endpoint"
    )   
    audio_tokens: Optional[int] = Field(
        default=None,
        description="Number of audio tokens using OpenAI endpoint"
    )
    reasoning_tokens: Optional[int] = Field(
        default=None,
        description="Number of reasoning tokens using OpenAI endpoint"
    )
    rejected_prediction_tokens: Optional[int] = Field(
        default=None,
        description="Number of rejected prediction tokens using OpenAI endpoint"
    )
    cached_tokens: Optional[int] = Field(
        default=None,
        description="Number of tokens read from cache with OpenAI endpoint"
    )


class GeneratedJsonObject(Entity):
    """Represents a structured JSON object generated by an LLM."""
    name: str = Field(
        description="Name identifier for the generated object"
    )
    object: Dict[str, Any] = Field(
        description="The actual JSON object content"
    )
    tool_call_id: Optional[str] = Field(
        default=None,
        description="Associated tool call ID if generated via tool"
    )

class RawOutput(Entity):
    """Raw output from LLM API calls with metadata."""
    raw_result: Dict[str, Any] = Field(
        description="Raw response from the LLM API"
    )
    completion_kwargs: Dict[str, Any] = Field(
        description="Arguments used in the completion call"
    )
    start_time: float = Field(
        description="Timestamp when the API call started"
    )
    end_time: float = Field(
        description="Timestamp when the API call completed"
    )
    chat_thread_id: Optional[UUID] = Field(
        default=None,
        description="ID of the associated chat thread"
    )
    client: LLMClient = Field(
        description="The LLM client used for this call"
    )
    parsed_result: Optional[Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]] = Field(
        default=None,
        description="Cached parsed results (content, json_object, usage, error)",
        exclude=True
    )

    @property
    def time_taken(self) -> float:
        """Calculate time taken for the LLM call."""
        return self.end_time - self.start_time

    @computed_field
    @property
    def str_content(self) -> Optional[str]:
        """Extract string content from raw result."""
        return self._parse_result()[0]

    @computed_field
    @property
    def json_object(self) -> Optional[GeneratedJsonObject]:
        """Extract JSON object from raw result."""
        return self._parse_result()[1]
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        """Extract usage statistics."""
        return self._parse_result()[2]

    @computed_field
    @property
    def error(self) -> Optional[str]:
        """Extract error message if present."""
        return self._parse_result()[3]

    @computed_field
    @property
    def contains_object(self) -> bool:
        """Check if result contains a JSON object."""
        return self._parse_result()[1] is not None

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], Optional[str]]:
        """Parse raw result into structured components with caching."""
        if self.parsed_result is not None:
            return self.parsed_result
            
        # Check for errors first
        if getattr(self.raw_result, "error", None):
            self.parsed_result = (None, None, None, getattr(self.raw_result, "error", None))
            return self.parsed_result

        provider = self.result_provider
        try:
            if provider == LLMClient.openai:
                self.parsed_result = self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
            elif provider == LLMClient.anthropic:
                self.parsed_result = self._parse_anthropic_message(AnthropicMessage.model_validate(self.raw_result))
            elif provider in [LLMClient.vllm, LLMClient.litellm]:
                self.parsed_result = self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
            else:
                raise ValueError(f"Unsupported result provider: {provider}")
        except Exception as e:
            self.parsed_result = (None, None, None, str(e))
            
        return self.parsed_result

    @computed_field
    @property
    def result_provider(self) -> Optional[LLMClient]:
        """Determine the LLM provider from the result format."""
        return self.search_result_provider() if self.client is None else self.client
    
    def search_result_provider(self) -> Optional[LLMClient]:
        """Identify LLM provider from result structure."""
        try:
            ChatCompletion.model_validate(self.raw_result)
            return LLMClient.openai
        except:
            try:
                AnthropicMessage.model_validate(self.raw_result)
                return LLMClient.anthropic
            except:
                return None

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        """Parse JSON string safely."""
        try:
            # Try direct JSON parsing first
            return json.loads(content)
        except json.JSONDecodeError:
            # Fall back to more lenient parsing if needed
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    return None
            return None

    def _parse_oai_completion(self, chat_completion: ChatCompletion) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], None]:
        """Parse OpenAI completion format."""
        message = chat_completion.choices[0].message
        content = message.content

        json_object = None
        usage = None

        # Handle tool calls
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            tool_call_id = tool_call.id
            try:
                object_dict = json.loads(tool_call.function.arguments)
                json_object = GeneratedJsonObject(name=name, object=object_dict, tool_call_id=tool_call_id)
            except json.JSONDecodeError:
                json_object = GeneratedJsonObject(name=name, object={"raw": tool_call.function.arguments}, tool_call_id=tool_call_id)
        
        # Handle content parsing
        elif content is not None:
            if self.completion_kwargs:
                name = self.completion_kwargs.get("response_format", {}).get("json_schema", {}).get("name", None)
            else:
                name = None
            parsed_json = self._parse_json_string(content)
            if parsed_json:
                json_object = GeneratedJsonObject(
                    name="parsed_content" if name is None else name,
                    object=parsed_json
                )
                content = None

        # Extract usage information
        if chat_completion.usage:

            usage = Usage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens,
                accepted_prediction_tokens=chat_completion.usage.completion_tokens_details.accepted_prediction_tokens if chat_completion.usage.completion_tokens_details else None,
                audio_tokens=chat_completion.usage.completion_tokens_details.audio_tokens if chat_completion.usage.completion_tokens_details else None,
                reasoning_tokens=chat_completion.usage.completion_tokens_details.reasoning_tokens if chat_completion.usage.completion_tokens_details else None,
                rejected_prediction_tokens=chat_completion.usage.completion_tokens_details.rejected_prediction_tokens if chat_completion.usage.completion_tokens_details else None,
                cached_tokens=chat_completion.usage.prompt_tokens_details.cached_tokens if chat_completion.usage.prompt_tokens_details else None
            )
        EntityRegistry._logger.info(f"Parsed OpenAI completion:  {json_object.name if json_object else None}, {usage}")


        return content, json_object, usage, None

    def _parse_anthropic_message(self, message: AnthropicMessage) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], None]:
        """Parse Anthropic message format."""
        content = None
        json_object = None
        usage = None

        if message.content:
            text_block = next((block for block in message.content 
                          if isinstance(block, TextBlock) and block.text.strip()), None)
            

            tool_block = next((block for block in message.content 
                          if isinstance(block, ToolUseBlock)), None)
            # Check if it's a TextBlock
            if isinstance(text_block, TextBlock):
                content = text_block.text

                parsed_json = self._parse_json_string(content)
                if parsed_json:
                    json_object = GeneratedJsonObject(
                        name="parsed_content", 
                        object=parsed_json
                    )
                    content = content
            # Check if it's a ToolUseBlock
            if isinstance(tool_block, ToolUseBlock):
               
                tool_call_id = tool_block.id
                # Cast tool_use.input to Dict[str, Any]
                if isinstance(tool_block.input, dict):
                    json_object = GeneratedJsonObject(
                        name=tool_block.name,
                        object=tool_block.input,
                        tool_call_id=tool_call_id
                    )
                else:
                    # Handle non-dict input by wrapping it

                    json_object = GeneratedJsonObject(
                        name=tool_block.name,
                        object={"value": tool_block.input},
                        tool_call_id=tool_call_id
                    )


        if hasattr(message, 'usage'):
            usage = Usage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                cache_creation_input_tokens=getattr(message.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(message.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage, None


    def create_processed_output(self) -> 'ProcessedOutput':
        """Create a ProcessedOutput from this raw output."""
        content, json_object, usage, error = self._parse_result()
        if (json_object is None and content is None) or self.chat_thread_id is None:
            raise ValueError("No content or JSON object found in raw output")
   
        return ProcessedOutput(
            content=content,
            json_object=json_object,
            usage=usage,
            error=error,
            time_taken=self.time_taken,
            llm_client=self.client,
            raw_output=self,
            chat_thread_id=self.chat_thread_id
        )
    
class ProcessedOutput(Entity):
    """
    Processed and structured output from LLM interactions.
    Contains parsed content, JSON objects, and usage statistics.
    """
    content: Optional[str] = None
    json_object: Optional[GeneratedJsonObject] = None
    usage: Optional[Usage] = None
    error: Optional[str] = None
    time_taken: float
    llm_client: LLMClient
    raw_output: RawOutput
    chat_thread_id: UUID
    
class ChatThread(Entity):
    """A chat thread entity managing conversation flow and message history."""
    
    name: Optional[str] = Field(
        default=None,
        description="Optional name for the thread"
    )
    
    system_prompt: Optional[SystemPrompt] = Field(
        default=None,
        description="Associated system prompt"
    )
    
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Messages in chronological order"
    )
    
    new_message: Optional[str] = Field(
        default=None,
        description="Temporary storage for message being processed"
    )
    
    prefill: str = Field(
        default="Here's the valid JSON object response:```json",
        description="Prefill assistant response with an instruction"
    )
    
    postfill: str = Field(
        default="\n\nPlease provide your response in JSON format.",
        description="Postfill user response with an instruction"
    )
    
    use_schema_instruction: bool = Field(
        default=False,
        description="Whether to use the schema instruction"
    )
    
    use_history: bool = Field(
        default=True,
        description="Whether to use the history"
    )
    
    forced_output: Optional[Union[StructuredTool, CallableTool]] = Field(
        default=None,
        description="Associated forced output tool"
    )
    
    llm_config: LLMConfig = Field(
        description="LLM configuration"
    )
    
    tools: List[Union[CallableTool, StructuredTool]] = Field(
        default_factory=list,
        description="Available tools"
    )
    workflow_step: Optional[int] = Field(
        default=None,
        description="Workflow step number"
    )

    @property
    def oai_response_format(self) -> Optional[Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema]]:
        """Get OpenAI response format based on config."""
        if self.llm_config.response_format == ResponseFormat.text:
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == ResponseFormat.json_object:
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == ResponseFormat.structured_output and isinstance(self.forced_output, StructuredTool):
            assert self.forced_output is not None, "Structured output is not set"
            return self.forced_output.get_openai_json_schema_response()
        elif self.llm_config.response_format == ResponseFormat.tool and isinstance(self.forced_output, CallableTool):
            return 
        return None

    @property
    def use_prefill(self) -> bool:
        """Check if prefill should be used."""
        return (self.llm_config.client in [LLMClient.anthropic, LLMClient.vllm, LLMClient.litellm] and 
                self.llm_config.response_format == ResponseFormat.json_beg)

    @property
    def use_postfill(self) -> bool:
        """Check if postfill should be used."""
        return (self.llm_config.client == LLMClient.openai and 
                self.llm_config.response_format in [ResponseFormat.json_object, ResponseFormat.json_beg] and 
                not self.use_schema_instruction)

    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        """Get system message including schema instruction if needed."""
        content = self.system_prompt.content if self.system_prompt else ""
        if self.use_schema_instruction and self.forced_output and isinstance(self.forced_output, StructuredTool):
            content = "\n".join([content, self.forced_output.schema_instruction])
        return {"role": "system", "content": content} if content else None

    @property
    def message_objects(self) -> List[ChatMessage]:
        """Get all message objects in the conversation."""
        messages = []
        
        # Add system message
        if self.system_message:
            messages.append(ChatMessage(
                role=MessageRole.system,
                content=self.system_message["content"]
            ))
            
        # Add history
        if self.use_history:
            messages.extend(self.history)
            
        # Add new message
        if self.new_message:
            messages.append(ChatMessage(
                role=MessageRole.user,
                content=self.new_message
            ))
            
        # Handle prefill/postfill
        if self.use_prefill and messages:
            messages.append(ChatMessage(
                role=MessageRole.assistant,
                content=self.prefill
            ))
        elif self.use_postfill and messages:
            messages[-1].content += self.postfill
            
        return messages

    @property
    def messages(self) -> List[Dict[str, Any]]:
        """Get messages in chatml dict format."""
        return [msg.to_dict() for msg in self.message_objects]

    @property
    def oai_messages(self) -> List[Dict[str, Any]]:
        """Convert chat history to OpenAI message format."""
        EntityRegistry._logger.info(f"Converting ChatThread({self.id}) history to OpenAI format")
        messages = []
        
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt.content
            })
        
        for msg in self.history:
            EntityRegistry._logger.info(f"Processing message: role={msg.role}, "
                                      f"tool_call_id={msg.oai_tool_call_id}")
            
            if msg.role == MessageRole.user:
                messages.append({
                    "role": "user",
                    "content": msg.content
                })
                
            elif msg.role == MessageRole.assistant:
                if msg.tool_call:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": [{
                            "id": msg.oai_tool_call_id,
                            "type": "function",
                            "function": {
                                "name": msg.tool_name,
                                "arguments": json.dumps(msg.tool_call)
                            }
                        }]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": msg.content
                    })
                
            elif msg.role == MessageRole.tool:
                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.oai_tool_call_id
                })
        
        EntityRegistry._logger.info(f"Final messages for ChatThread({self.id}): {messages}")
        return messages

    @property
    def anthropic_messages(self) -> Tuple[List[TextBlockParam], List[MessageParam]]:
        """Get messages in Anthropic format."""
        
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)

    @property
    def vllm_messages(self) -> List[ChatCompletionMessageParam]:
        """Get messages in vLLM format."""
        return msg_dict_to_oai(self.messages)

    def get_tool_by_name(self, tool_name: str) -> Optional[Union[CallableTool, StructuredTool]]:
        """Get tool by name from available tools."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        if self.forced_output and self.forced_output.name == tool_name:
            return self.forced_output
        return None

    def add_user_message(self) -> Optional[ChatMessage]:
        """Add a user message to history."""
        if not self.new_message and self.llm_config.response_format != ResponseFormat.auto_tools and self.llm_config.response_format != ResponseFormat.workflow:
            raise ValueError("Cannot add user message - no new message content")
        elif not self.new_message and self.llm_config.response_format == ResponseFormat.auto_tools:
            EntityRegistry._logger.info(f"Skipped adding user message to ChatThread({self.id}) since we are in auto_tools mode")
            return None
        elif not self.new_message and self.llm_config.response_format == ResponseFormat.workflow:
            EntityRegistry._logger.info(f"Skipped adding user message to ChatThread({self.id}) since we are in workflow mode")
            return None
        assert self.new_message is not None, "Cannot add user message - no new message content"
        user_message = ChatMessage(
            role=MessageRole.user,
            content=self.new_message,
            parent_message_uuid=self.history[-1].id if self.history else None
        )
        self.history.append(user_message)
        EntityRegistry._logger.info(f"Added user message({user_message.id}) to ChatThread({self.id})")
        self.new_message = None
        return user_message

    async def add_chat_turn_history(self, output: ProcessedOutput) -> Tuple[ChatMessage, ChatMessage]:
        """Add a chat turn to history, including any tool executions or validations."""
        EntityRegistry._logger.info(f"ChatThread({self.id}): Adding chat turn from ProcessedOutput({output.id})")
        
        # Get the parent message - could be user, assistant, or tool message
        if not self.history:
            raise ValueError("Cannot add chat turn to empty history")
        parent_message = self.history[-1]
        
        # Create assistant message with complete metadata
        assistant_message = ChatMessage(
            role=MessageRole.assistant,
            content=output.content or "",
            chat_thread_uuid=self.id,
            parent_message_uuid=parent_message.id,
            tool_call=output.json_object.object if output.json_object else None,
            tool_name=output.json_object.name if output.json_object else None,
            tool_uuid=self.forced_output.id if self.forced_output else None,
            tool_type="Structured" if isinstance(self.forced_output, StructuredTool) else "Callable" if isinstance(self.forced_output, CallableTool) else None,
            oai_tool_call_id=output.json_object.tool_call_id if output.json_object else None,
            tool_json_schema=self.forced_output.json_schema if self.forced_output and isinstance(self.forced_output, StructuredTool) else None,
            usage=output.usage
        )
        self.history.append(assistant_message)
        EntityRegistry._logger.info(f"Added assistant message({assistant_message.id}) with tool_call_id: {assistant_message.oai_tool_call_id}")


        
        # Handle tool execution/validation if present
        if output.json_object and assistant_message:
            tool = self.get_tool_by_name(output.json_object.name)
            if tool:
                try:
                    if isinstance(tool, StructuredTool):
                        # Handle structured tool validation
                        validation_result = tool.execute(input_data=output.json_object.object)
                        tool_message = ChatMessage(
                            role=MessageRole.tool,
                            content=json.dumps({"status": "validated", "message": "Schema validation successful"}),
                            chat_thread_uuid=self.id,
                            tool_name=tool.name,
                            tool_uuid=tool.id,
                            tool_type="Structured",
                            tool_json_schema=tool.json_schema,
                            parent_message_uuid=assistant_message.id,
                            oai_tool_call_id=assistant_message.oai_tool_call_id
                        )
                        self.history.append(tool_message)
                        EntityRegistry._logger.info(f"Added tool validation message({tool_message.id}) with tool_call_id: {tool_message.oai_tool_call_id}")
                    elif isinstance(tool, CallableMCPTool):
                        # Handle MCP tool execution
                        EntityRegistry._logger.info(f"Executing MCP tool: {tool.name} with args: {output.json_object.object}")
                        tool_result = await tool.aexecute(chat_thread_id=self.id, input_data=output.json_object.object)
                        EntityRegistry._logger.info(f"MCP tool result: {tool_result}")
                        
                        tool_message = ChatMessage(
                            role=MessageRole.tool,
                            content=json.dumps(tool_result),
                            chat_thread_uuid=self.id,
                            tool_name=tool.name,
                            tool_uuid=tool.id,
                            tool_type="CallableMCP",
                            parent_message_uuid=assistant_message.id,
                            oai_tool_call_id=assistant_message.oai_tool_call_id
                        )
                        self.history.append(tool_message)
                        EntityRegistry._logger.info(f"Added MCP tool execution message({tool_message.id}) with tool_call_id: {tool_message.oai_tool_call_id}")
                    
                    elif isinstance(tool, CallableTool):
                        # Handle callable tool execution
                        tool_result = await tool.aexecute(input_data=output.json_object.object)
                        tool_message = ChatMessage(
                            role=MessageRole.tool,
                            content=json.dumps(tool_result),
                            chat_thread_uuid=self.id,
                            tool_name=tool.name,
                            tool_uuid=tool.id,
                            tool_type="Callable",
                            parent_message_uuid=assistant_message.id,
                            oai_tool_call_id=assistant_message.oai_tool_call_id
                        )
                        self.history.append(tool_message)
                        EntityRegistry._logger.info(f"Added tool execution message({tool_message.id}) with tool_call_id: {tool_message.oai_tool_call_id}")
                    
                except Exception as e:
                    EntityRegistry._logger.error(f"Tool operation failed: {str(e)}")
                    error_message = ChatMessage(
                        role=MessageRole.tool,
                        content=json.dumps({"error": str(e)}),
                        chat_thread_uuid=self.id,
                        tool_name=tool.name,
                        tool_uuid=tool.id,
                        tool_type="Callable" if isinstance(tool, CallableTool) else "Structured",
                        parent_message_uuid=assistant_message.id,
                        oai_tool_call_id=assistant_message.oai_tool_call_id
                    )
                    self.history.append(error_message)
                    EntityRegistry._logger.info(f"Added error message({error_message.id}) for failed tool operation")
        
        return parent_message, assistant_message
    
    def get_tools_for_llm(self) -> Optional[List[Union[ChatCompletionToolParam, ToolParam]]]:
        """Get tools in format appropriate for current LLM."""
        if not self.tools:
            return None
            
        tools = []
        for idx,tool in enumerate(self.tools):
            if self.llm_config.client in [LLMClient.openai, LLMClient.vllm, LLMClient.litellm]:
                tools.append(tool.get_openai_tool())
            elif self.llm_config.client == LLMClient.anthropic:
                if idx == 0 and self.llm_config.use_cache:
                    use_cache = True
                else:
                    use_cache = False
                tools.append(tool.get_anthropic_tool(use_cache=use_cache))
        return tools if tools else None


    @model_validator(mode='after')
    def validate_workflow(self) -> Self:
        """Validate workflow step configuration."""
        # Initialize workflow step if using workflow response format
        if self.llm_config.response_format == ResponseFormat.workflow:
            if self.workflow_step is None:
                self.workflow_step = 0
            
            # Validate step number against tools list
            if self.workflow_step >= len(self.tools):
                raise ValueError(
                    f"Workflow step {self.workflow_step} is out of range - only {len(self.tools)} tools available"
                )
        
        return self
    
    def get_all_usages(self) -> List[Usage]:
        """Get all usages from history"""
        usages = []
        for message in self.history:
            if message.role == MessageRole.assistant and message.usage:
                usages.append(message.usage)
        return usages



