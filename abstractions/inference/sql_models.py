#market_agents\inference\sql_models.py

from sqlmodel import Field, SQLModel, create_engine, Column, JSON, Session, Relationship, select
from typing import Dict, Any, List, Optional, Literal, Self, Union, Tuple, Callable, get_type_hints
from pydantic import computed_field, ValidationError, model_validator, create_model, BaseModel
from sqlalchemy import Engine
from enum import Enum
from inspect import signature
import json
from ast import literal_eval
import sys
from datetime import datetime
import libcst as cst
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletion
)
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema

from openai.types.shared_params import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    FunctionDefinition
)
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolUseBlock,
    ToolParam,
    Message as AnthropicMessage
)
from anthropic.types.beta.prompt_caching import (
    PromptCachingBetaMessage,
    PromptCachingBetaToolParam,
    PromptCachingBetaMessageParam,
    PromptCachingBetaTextBlockParam,
    message_create_params
)
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam
from anthropic.types.model_param import ModelParam
from abstractions.inference.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
import uuid
from uuid import UUID

class CallableRegistry:
    """Global registry for tool callables"""
    _instance = None
    _registry: Dict[str, Callable] = {}

    def __new__(cls) -> 'CallableRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """Register a new callable. Raises error if name exists."""
        if name in cls._registry:
            raise ValueError(f"Function '{name}' already registered. Use update() to replace existing function.")
        
        # Check for type hints
        type_hints = get_type_hints(func)
        if not type_hints:
            raise ValueError(f"Function '{name}' must have type hints")
        if 'return' not in type_hints:
            raise ValueError(f"Function '{name}' must have a return type hint")
            
        cls._registry[name] = func
    
    @classmethod
    def register_from_text(cls, name: str, func_text: str) -> None:
        """Register a function from its text representation."""
        if name in cls._registry:
            raise ValueError(f"Function '{name}' already registered. Use update() to replace existing function.")
        
        # For lambdas, wrap them in a typed function
        if func_text.strip().startswith('lambda'):
            # Create a wrapper function with type hints
            wrapper_text = f"""
def {name}(x: float) -> float:
    \"\"\"Wrapped lambda function with type hints\"\"\"
    func = {func_text}
    return func(x)
"""
            func_text = wrapper_text
        
        try:
            # Parse with libcst
            module = cst.parse_module(func_text)
            
            # Create a clean namespace with all needed imports
            namespace = {
                # Basic types
                'float': float, 'int': int, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple,
                # Type hints
                'List': List, 'Dict': Dict, 'Tuple': Tuple, 
                'Optional': Optional, 'Union': Union,
                'Any': Any,
                # Models if needed
                'BaseModel': BaseModel
            }
            
            # Execute the code
            exec(module.code, namespace)
            
            # Get the function
            if func_text.strip().startswith('lambda'):
                func = namespace[name]  # Get wrapped function
            else:
                func_name = func_text.split('def ')[1].split('(')[0].strip()
                func = namespace[func_name]
            
            # Verify it has type hints
            type_hints = get_type_hints(func)
            if not type_hints:
                raise ValueError(f"Function must have type hints")
            if 'return' not in type_hints:
                raise ValueError(f"Function must have a return type hint")
            
            cls._registry[name] = func
            
        except Exception as e:
            raise ValueError(f"Failed to parse function: {str(e)}")
    
    @classmethod
    def update(cls, name: str, func: Callable) -> None:
        """Update an existing callable."""
        # Check for type hints
        type_hints = get_type_hints(func)
        if not type_hints:
            raise ValueError(f"Function must have type hints")
        if 'return' not in type_hints:
            raise ValueError(f"Function must have a return type hint")
            
        cls._registry[name] = func
    
    @classmethod
    def delete(cls, name: str) -> None:
        """Delete a callable from registry."""
        if name not in cls._registry:
            raise ValueError(f"Function '{name}' not found in registry.")
        del cls._registry[name]
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        return cls._registry.get(name)
    
    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Get current status of the registry"""
        return {
            "total_functions": len(cls._registry),
            "registered_functions": list(cls._registry.keys()),
            "function_signatures": {
                name: str(signature(func))
                for name, func in cls._registry.items()
            }
        }

class Tool(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    schema_name: str = Field(default="generate_structured_output")
    schema_description: str = Field(default="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default="Please follow this JSON schema for your response:")
    strict_schema: bool = True
    json_schema: Dict = Field(default={}, sa_column=Column(JSON))
    chats: List["ChatThread"] = Relationship(back_populates="structured_output")
    callable: bool = False
    callable_function: Optional[str] = None
    callable_output_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    allow_literal_eval: bool = False

    def _register_callable(self) -> None:
        """Helper method to register callable in registry"""
        if self.callable and self.callable_function:
            try:
                # First try to get the function from DEFAULT_CALLABLE_TOOLS
                from abstractions.hub.callable_tools import DEFAULT_CALLABLE_TOOLS
                if self.schema_name in DEFAULT_CALLABLE_TOOLS:
                    func = DEFAULT_CALLABLE_TOOLS[self.schema_name]["function"]
                    try:
                        CallableRegistry().register(self.schema_name, func)
                    except ValueError:
                        # Function already registered, skip
                        pass
                    return
                
                # If not in defaults and allow_literal_eval is True, try to register from text
                if not CallableRegistry().get(self.schema_name):
                    if not self.allow_literal_eval:
                        raise ValueError(
                            f"Function '{self.callable_function}' not found in registry "
                            f"and allow_literal_eval is False"
                        )
                    try:
                        CallableRegistry().register_from_text(
                            name=self.schema_name, 
                            func_text=self.callable_function
                        )
                    except Exception as e:
                        raise ValueError(f"Could not register callable_function: {str(e)}")
            except Exception as e:
                print(f"Warning: Failed to register callable tool {self.schema_name}: {str(e)}")

    def __init__(self, **data):
        super().__init__(**data)
        self._register_callable()

    @model_validator(mode='after')
    def validate_callable(self) -> Self:
        """Validate and register callable"""
        self._register_callable()
        return self

    def execute(self, input: Dict[str, Any], tool_call_id: Optional[str] = None, tool_call_message_uuid: Optional[UUID] = None) -> 'ChatMessage':
        """Execute the callable function with the given input."""
        print(f"\n=== Tool Execution Debug ===")
        print(f"Tool: {self.schema_name}")
        print(f"Input: {input}")
        print(f"Tool call ID: {tool_call_id}")
        print(f"Parent message UUID: {tool_call_message_uuid}")

        if not self.callable or not self.callable_function:
            error_msg = "Tool is not callable"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)

        callable_func = CallableRegistry().get(self.schema_name)
        if not callable_func:
            error_msg = f"Function '{self.schema_name}' not found in registry. Available functions: {list(CallableRegistry()._registry.keys())}"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg)
        
        try:
            # Get function signature info
            sig = signature(callable_func)
            type_hints = get_type_hints(callable_func)
            first_param = next(iter(sig.parameters.values()))
            param_type = type_hints.get(first_param.name)
            
            print(f"Function signature: {sig}")
            print(f"Type hints: {type_hints}")
            print(f"Parameter type: {param_type}")
            
            # Handle input based on parameter type
            if (isinstance(param_type, type) and 
                issubclass(param_type, BaseModel)):
                print(f"Using Pydantic model validation for input")
                model_input = param_type.model_validate(input)
                response = callable_func(model_input)
            else:
                print(f"Using direct kwargs for input")
                response = callable_func(**input)
            
            print(f"Raw response: {response}")
            
            # Process response
            if isinstance(response, BaseModel):
                content = response.model_dump_json()
                print(f"Serialized Pydantic response: {content}")
            else:
                content = json.dumps({"result": response})
                print(f"Wrapped primitive response: {content}")
                
            message = ChatMessage(
                role=MessageRole.tool,
                content=content,
                tool_name=f"{self.schema_name}_response",
                tool_call_id=tool_call_id,
                parent_message_uuid=tool_call_message_uuid,
                tool_json_schema=self.callable_output_schema,
                tool_executable=True
            )
            print(f"Created ChatMessage: {message}")
            return message
            
        except Exception as e:
            error_msg = f"Error executing tool {self.schema_name}: {str(e)}"
            print(f"Error: {error_msg}")
            print(f"Exception type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            raise ValueError(error_msg) from e

    @computed_field
    @property
    def schema_instruction(self) -> str:
        return f"{self.instruction_string}: {self.json_schema}"

    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        if self.json_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.schema_name,
                    description=self.schema_description,
                    parameters=self.json_schema
                )
            )
        return None

    def get_anthropic_tool(self) -> Optional[PromptCachingBetaToolParam]:
        if self.json_schema:
            return PromptCachingBetaToolParam(
                name=self.schema_name,
                description=self.schema_description,
                input_schema=self.json_schema,
                cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')
            )
        return None

    def get_openai_json_schema_response(self) -> Optional[ResponseFormatJSONSchema]:
        if self.json_schema:
            schema = JSONSchema(
                name=self.schema_name,
                description=self.schema_description,
                schema=self.json_schema,
                strict=self.strict_schema
            )
            return ResponseFormatJSONSchema(type="json_schema", json_schema=schema)
        return None

    @classmethod
    def from_callable(
        cls,
        func: Callable,
        schema_name: Optional[str] = None,
        schema_description: Optional[str] = None,
        instruction_string: Optional[str] = None,
        strict_schema: bool = True,
        json_schema: Optional[Dict[str, Any]] = None
    ) -> Self:
        """Initialize a Tool from a Python callable with type hints."""
        type_hints = get_type_hints(func)
        sig = signature(func)
        
        if 'return' not in type_hints:
            raise ValueError(f"Function {func.__name__} must have a return type hint")
        
        # Use provided name or function name
        final_name = schema_name if schema_name is not None else func.__name__
        
        # Register with the final name
        CallableRegistry().register(final_name, func)
        
        # Handle input schema
        first_param = next(iter(sig.parameters.values()))
        first_param_type = type_hints.get(first_param.name)
        
        if (isinstance(first_param_type, type) and 
            issubclass(first_param_type, BaseModel)):
            derived_input_schema = first_param_type.model_json_schema()
        else:
            input_fields = {}
            for param_name, param in sig.parameters.items():
                if param_name not in type_hints:
                    raise ValueError(f"Parameter {param_name} must have a type hint")
                
                if param.default is param.empty:
                    input_fields[param_name] = (type_hints[param_name], ...)
                else:
                    input_fields[param_name] = (type_hints[param_name], param.default)

            InputModel = create_model(f"{final_name}Input", **input_fields)
            derived_input_schema = InputModel.model_json_schema()
        
        # Validate provided schema against derived
        if json_schema is not None:
            derived_required = set(derived_input_schema.get("required", []))
            provided_required = set(json_schema.get("required", []))
            if derived_required != provided_required:
                raise ValueError(
                    f"Schema mismatch: Required properties don't match.\n"
                    f"Derived: {derived_required}\n"
                    f"Provided: {provided_required}"
                )

            derived_props = derived_input_schema.get("properties", {})
            provided_props = json_schema.get("properties", {})
            
            for prop_name, prop_schema in derived_props.items():
                if prop_name not in provided_props:
                    raise ValueError(
                        f"Schema mismatch: Missing property '{prop_name}' in provided schema"
                    )
                provided_type = provided_props[prop_name].get("type")
                derived_type = prop_schema.get("type")
                if provided_type != derived_type:
                    raise ValueError(
                        f"Schema mismatch: Property '{prop_name}' type mismatch.\n"
                        f"Derived type: {derived_type}\n"
                        f"Provided type: {provided_type}"
                    )

            extra_props = set(provided_props.keys()) - set(derived_props.keys())
            if extra_props:
                raise ValueError(
                    f"Schema mismatch: Extra properties in provided schema: {extra_props}"
                )

            final_schema = json_schema
        else:
            final_schema = derived_input_schema
        
        # Handle output type
        output_type = type_hints['return']
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            OutputModel = output_type
        else:
            OutputModel = create_model(f"{final_name}Output", result=(output_type, ...))
        
        return cls(
            schema_name=final_name,
            schema_description=schema_description if schema_description is not None else (func.__doc__ or f"Execute {final_name} function"),
            instruction_string=instruction_string or "Please follow this JSON schema for your response:",
            strict_schema=strict_schema,
            json_schema=final_schema,
            callable=True,
            callable_function=func.__name__,
            callable_output_schema=OutputModel.model_json_schema()
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

class LLMConfig(SQLModel, table=True):
    id: Optional[int]  = Field(default=None, primary_key=True)
    client: LLMClient
    model: Optional[str] = None
    max_tokens: int = Field(default=400)
    temperature: float = 0
    response_format: ResponseFormat = Field(default=ResponseFormat.text)
    use_cache: bool = True
    chats: List["ChatThread"] = Relationship(back_populates="llm_config")
    
    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        if self.response_format == ResponseFormat.json_object and self.client in [LLMClient.vllm, LLMClient.litellm,LLMClient.anthropic]:
            raise ValueError(f"{self.client} does not support json_object response format")
        elif self.response_format == ResponseFormat.structured_output and self.client == LLMClient.anthropic:
            raise ValueError(f"Anthropic does not support structured_output response format use json_beg or tool instead")
        return self
    
class ChatSnapshot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    chat_thread_id: int = Field(foreign_key="chatthread.id")
    messages: Optional[List[Dict[str, Any]]] = Field(default=None,sa_column=Column(JSON))
    structured_output_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    structured_output_name: Optional[str] = None
    structured_output_id: Optional[int] = Field(default=None, foreign_key="tool.id")
    llm_config_id: int = Field( foreign_key="llmconfig.id")


class ChatThreadProcessedOutputLinkage(SQLModel, table=True):
    chat_thread_id: int = Field(foreign_key="chatthread.id",primary_key=True)
    processed_output_id: int = Field(foreign_key="processedoutput.id",primary_key=True)

class ThreadMessageLinkage(SQLModel, table=True):
    chat_thread_id: int = Field(foreign_key="chatthread.id",primary_key=True)
    chat_message_id: int = Field(foreign_key="chatmessage.id",primary_key=True)

class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"
    system = "system"

class MessageFormat(str, Enum):
    chatml = "chatml"
    json = "json"
    python_dict = "python_dict"

class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    uuid: UUID = Field(default_factory=lambda: uuid.uuid4())
    role: MessageRole
    content: str
    author_name: Optional[str] = None
    parent_message_uuid: Optional[UUID] = None
    chat_thread: 'ChatThread' = Relationship(back_populates="history",link_model=ThreadMessageLinkage,sa_relationship_kwargs={"lazy": "joined"})
    format: MessageFormat = Field(default=MessageFormat.python_dict)
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_json_schema: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tool_call: Dict[str, Any] = Field(default=None, sa_column=Column(JSON))
    tool_executable: bool = Field(default=False)

    def to_chatml_dict(self) -> Dict[str, Any]:
        if self.role == MessageRole.tool:
            return {"role":self.role.value,"content":self.content,"tool_call_id":self.tool_call_id}
        elif self.role == MessageRole.assistant:
            if self.tool_call_id is not None and self.tool_executable:
                print(f"tool_call adding proper dictioanry with tool_call_id: {self.tool_call_id} and content: {self.content}")
                return {"role":self.role.value,"content":self.content,"tool_calls":[{"id":self.tool_call_id,"function":{"arguments":json.dumps(self.tool_call),"name":self.tool_name},"type":"function"}]}
            else:
                return {"role":self.role.value,"content":self.content}
        else:
            return {"role":self.role.value,"content":self.content}
    
    def to_string(self) -> str:
        if self.format == MessageFormat.python_dict:
            return json.dumps(self.to_chatml_dict())
        else:
            raise ValueError(f"Message format {self.format} is not supported")
    def to_share_gpt_dict(self) -> Dict[str, str]:
        return {"from":self.role.value,"value":self.content}
    
    @classmethod
    def from_chatml_dict(cls,message_dict:Dict[str, Any]) -> Self:
        return cls(role=MessageRole(message_dict["role"]),content=message_dict["content"])
    
    @classmethod
    def from_share_gpt_dict(cls,message_dict:Dict[str, Any]) -> Self:
        return cls(role=MessageRole(message_dict["from"]),content=message_dict["value"])
    
    @classmethod
    def from_dict(cls,message_dict:Dict[str, Any]) -> Self:
        if "from" in message_dict and "value" in message_dict:
            return cls.from_share_gpt_dict(message_dict)
        elif "role" in message_dict and "content" in message_dict:
            return cls.from_chatml_dict(message_dict)
        else:
            raise ValueError(f"Message dictionary {message_dict} is not valid, only chatml (role,content) or share_gpt (from,value) are supported")

class ThreadToolLinkage(SQLModel, table=True):
    chat_thread_id: int = Field(foreign_key="chatthread.id",primary_key=True)
    tool_id: int = Field(foreign_key="tool.id",primary_key=True)

    
class ThreadSystemLinkage(SQLModel, table=True):
    chat_thread_id: int = Field(foreign_key="chatthread.id",primary_key=True)
    system_str_id: int = Field(foreign_key="systemstr.id",primary_key=True)

class SystemStr(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: UUID = Field(default_factory=lambda: uuid.uuid4())
    name: str = Field(index=True)
    content: str
    chats: List["ChatThread"] = Relationship(back_populates="system_prompt", link_model=ThreadSystemLinkage, sa_relationship_kwargs={"lazy": "joined"})

class ChatThread (SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: UUID = Field(default_factory=lambda: uuid.uuid4())
    name: Optional[str] = Field(default=None)
    system_prompt: Optional[SystemStr] = Relationship(back_populates="chats", link_model=ThreadSystemLinkage, sa_relationship_kwargs={"lazy": "joined"})
    history: List[ChatMessage] = Relationship(back_populates="chat_thread",link_model=ThreadMessageLinkage,sa_relationship_kwargs={"lazy": "joined","order_by":"ChatMessage.timestamp"})
    new_message: Optional[str] = Field(default=None)
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    
    use_schema_instruction: bool = Field(default=False, description="Whether to use the schema instruction")
    use_history: bool = Field(default=True, description="Whether to use the history")
    structured_output: Optional[Tool] = Relationship(back_populates="chats",sa_relationship_kwargs={"lazy": "joined"})
    structured_output_id: Optional[int] = Field(default=None, foreign_key="tool.id")
    llm_config: LLMConfig = Relationship(back_populates="chats",sa_relationship_kwargs={"lazy": "joined"})
    llm_config_id: Optional[int] = Field(default=None, foreign_key="llmconfig.id")
    tools: List[Tool] = Relationship(link_model=ThreadToolLinkage,sa_relationship_kwargs={"lazy": "joined"})
    processed_outputs: List['ProcessedOutput'] = Relationship(back_populates="chat_thread", link_model=ChatThreadProcessedOutputLinkage,sa_relationship_kwargs={"lazy": "joined"})
    
    def get_last_message_uuid(self) -> Optional[UUID]:
        if len(self.history) == 0:
            return None
        return self.history[-1].uuid

    @computed_field
    @property
    def oai_response_format(self) -> Optional[Union[ResponseFormatText, ResponseFormatJSONObject, ResponseFormatJSONSchema]]:
        if self.llm_config.response_format == "text":
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == "json_object":
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == "structured_output":
            assert self.structured_output is not None, "Structured output is not set"
            return self.structured_output.get_openai_json_schema_response()
        else:
            return None


    @computed_field
    @property
    def use_prefill(self) -> bool:
        if self.llm_config.client in [LLMClient.anthropic,LLMClient.vllm,LLMClient.litellm] and  self.llm_config.response_format in [ResponseFormat.json_beg]:
            return True
        else:
            return False
        
    @computed_field
    @property
    def use_postfill(self) -> bool:
        if self.llm_config.client == LLMClient.openai and  self.llm_config.response_format in [ResponseFormat.json_object,ResponseFormat.json_beg] and not self.use_schema_instruction:
            return True

        else:
            return False
        
    @computed_field
    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        content= self.system_prompt.content if self.system_prompt else ""
        if self.use_schema_instruction and self.structured_output:
            content = "\n".join([content,self.structured_output.schema_instruction])
        return {"role":"system","content":content} if len(content)>0 else None
    
    @computed_field
    @property
    def messages_objects(self) -> List[ChatMessage]:
        system_message = ChatMessage(role=MessageRole.system,content=self.system_message["content"]) if self.system_message else None
        messages = [system_message] if system_message else []
        if self.use_history and self.history:
            messages+= [message for message in self.history]
        elif not self.use_history and not self.new_message:
            raise ValueError("ChatThread has no history and no new message, cannot generate messages")
        if self.new_message:
            messages.append(ChatMessage(role=MessageRole.user,content=self.new_message))
        if self.use_prefill:
            prefill_message = ChatMessage(role=MessageRole.assistant,content=self.prefill)
            messages.append(prefill_message)
        elif self.use_postfill:
            messages[-1].content = messages[-1].content + self.postfill
        
        return messages
    
    @computed_field
    @property
    def messages(self)-> List[Dict[str, Any]]:
        return [message.to_chatml_dict() for message in self.messages_objects]
    
    @computed_field
    @property
    def share_gpt_messages(self) -> List[Dict[str, str]]:
        return [message.to_share_gpt_dict() for message in self.messages_objects]

    
        
        
    
    @computed_field
    @property
    def oai_messages(self)-> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
    
    @computed_field
    @property
    def anthropic_messages(self) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)
    
    @computed_field
    @property
    def vllm_messages(self) -> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
        
    def add_user_message(self) -> ChatMessage:
        """Add the current new_message as a user message to the history"""
        if self.new_message is None:
            raise ValueError("new_message is None, cannot add to history")

        last_message_uuid = self.get_last_message_uuid()
        user_message = ChatMessage(
            role=MessageRole.user, 
            content=self.new_message, 
            parent_message_uuid=last_message_uuid
        )
        self.history.append(user_message)
        self.new_message = None
        return user_message

    def add_assistant_response(self, llm_output: 'ProcessedOutput', user_message_uuid: UUID):
        """Add the assistant's response from the ProcessedOutput to the history"""
        if llm_output.chat_thread_id != self.id:
            raise ValueError(f"ProcessedOutput chat_thread_id {llm_output.chat_thread_id} does not match the chat_thread id {self.id}")

        json_object = llm_output.json_object
        str_content = llm_output.content
        print(f"adding assistant response with content: {str_content} and json_object: {json_object}")
        if not json_object:
            if str_content:
                response = str_content
                tool_name = None
                assistant_message = ChatMessage(
                    role=MessageRole.assistant, 
                    content=response, 
                    parent_message_uuid=user_message_uuid
                )
            else:
                raise ValueError("ProcessedOutput json_object is None and content is None, cannot add to history")
        else:
            print(f"adding assistant response with json_object passed json object check")
            tool_name = json_object.name
            
            tool_json_schema = None
            tool = self.get_tool_by_name(tool_name)
            print(f"extracted toolname: {tool_name} and tool: {tool}")
            if not tool:
                raise ValueError(f"Tool {tool_name} not found, cannot add to history")

            
            if self.llm_config.response_format in [ResponseFormat.auto_tools,ResponseFormat.tool] and tool is not None and tool.callable:
                assert json_object is not None, "json_object is None, cannot add to history"
                assistant_message = ChatMessage(
                    role=MessageRole.assistant, 
                    content=str_content if str_content else "", 
                    parent_message_uuid=user_message_uuid,
                    tool_name=tool_name,
                    tool_call_id=json_object.tool_call_id,
                    tool_json_schema=tool.json_schema,
                    tool_call=json_object.object,
                    tool_executable=tool.callable
                )
            elif self.llm_config.response_format in [ResponseFormat.auto_tools,ResponseFormat.tool] and tool is not None and not tool.callable:
                assert json_object is not None, "json_object is None, cannot add to history"
                structured_response = json.dumps(json_object.object)
                assistant_message = ChatMessage(
                    role=MessageRole.assistant, 
                    content=structured_response, 
                    parent_message_uuid=user_message_uuid,
                    tool_name=tool_name,
                    tool_json_schema=tool.json_schema,
                )
            elif self.llm_config.response_format != ResponseFormat.auto_tools and tool is not None:
                print(f"current response format: {self.llm_config.response_format}")
                assert json_object is not None, "json_object is None, cannot add to history"
                structured_response = json.dumps(json_object.object)
                tool_json_schema = tool.json_schema
                assistant_message = ChatMessage(
                    role=MessageRole.assistant, 
                    content=structured_response, 
                    parent_message_uuid=user_message_uuid,
                    tool_name=tool_name,
                    tool_json_schema=tool_json_schema
                )
            elif self.llm_config.response_format != ResponseFormat.auto_tools and tool is None and json_object is not None:
                print(f"current response format: {self.llm_config.response_format}")
                assert json_object is not None, "json_object is None, cannot add to history"
                structured_response = json.dumps(json_object.object)
                tool_json_schema = tool.json_schema
                assistant_message = ChatMessage(
                    role=MessageRole.assistant, 
                    content=structured_response, 
                    parent_message_uuid=user_message_uuid,
                    tool_name=tool_name,
                    tool_json_schema=tool_json_schema
                )    
        self.history.append(assistant_message)
        self.processed_outputs.append(llm_output)
        self.new_message = None

    def add_assistant_and_tool_execution_response(self, llm_output: 'ProcessedOutput'):
        """Add the assistant's response from the ProcessedOutput to the history"""
        user_message_uuid = self.get_last_message_uuid()
        assert user_message_uuid is not None, "User message uuid is None, cannot add assistant response"
        assert llm_output.json_object is not None, "json_object is None, cannot add to history"
        self.add_assistant_response(llm_output, user_message_uuid)
        #execute the tool

        tool = self.get_tool_by_name(llm_output.json_object.name)
        if tool is None:
            raise ValueError(f"Tool {llm_output.json_object.name} not found, cannot execute tool")
        tool_response = tool.execute(input=llm_output.json_object.object, tool_call_id=llm_output.json_object.tool_call_id)
        self.history.append(tool_response)

    def add_chat_turn_history(self, llm_output: 'ProcessedOutput'):
        """Add a complete chat turn (user message + assistant response) to the history"""
        user_message = self.add_user_message()
        self.add_assistant_response(llm_output, user_message.uuid)

    def get_structured_output_as_tool(self) -> Union[ChatCompletionToolParam, PromptCachingBetaToolParam, None]:
        if not self.structured_output:
            return None
        if self.llm_config.client in [LLMClient.openai,LLMClient.vllm,LLMClient.litellm]:
            return self.structured_output.get_openai_tool()
        elif self.llm_config.client == LLMClient.anthropic:
            return self.structured_output.get_anthropic_tool()
        else:
            return None
    
    def get_tools(self) -> Optional[List[Union[ChatCompletionToolParam, PromptCachingBetaToolParam]]]:
        if len(self.tools) == 0:
            return None
        else:
            tools = []
            for tool in self.tools:
                if self.llm_config.client in [LLMClient.openai,LLMClient.vllm,LLMClient.litellm]:
                    tools.append(tool.get_openai_tool())
                elif self.llm_config.client == LLMClient.anthropic:
                    tools.append(tool.get_anthropic_tool())
            return tools
        
    def get_tool_by_name(self, tool_name: str) -> Optional[Tool]:
        for tool in self.tools:
            if tool.schema_name == tool_name:
                return tool
        if self.structured_output and self.structured_output.schema_name == tool_name:
            return self.structured_output
        return None
        
    def create_snapshot(self) -> 'ChatSnapshot':
        if not self.llm_config.id or not self.id:
            raise ValueError("LLMConfig or ChatThread id is not set, register the chat to the database before creating a snapshot")
        return ChatSnapshot(chat_thread_id=self.id,
                             messages=self.messages,
                            structured_output_schema=self.structured_output.json_schema if self.structured_output else None,
                            structured_output_name=self.structured_output.schema_name if self.structured_output else None,
                            structured_output_id=self.structured_output.id if self.structured_output else None,
                            llm_config_id=self.llm_config.id)
    
    def update_db(self,engine:Engine):
        with Session(engine) as session:
            session.add(self)
            snapshot = self.create_snapshot()
            session.add(snapshot)
            session.commit()

    def update_db_from_session(self,session:Session):
        session.add(self)
        snapshot = self.create_snapshot()
        session.add(snapshot)
        session.commit()

class OutputUsageLinkage(SQLModel, table=True):
    usage_id: int = Field(foreign_key="usage.id",primary_key=True)
    processed_output_id: int = Field(foreign_key="processedoutput.id",primary_key=True)

class OutputJsonObjectLinkage(SQLModel, table=True):
    generated_json_object_id: int = Field(foreign_key="generatedjsonobject.id",primary_key=True)
    processed_output_id: int = Field(foreign_key="processedoutput.id",primary_key=True)

class RawProcessedLinkage(SQLModel, table=True):
    raw_output_id: int = Field(foreign_key="rawoutput.id",primary_key=True)
    processed_output_id: int = Field(foreign_key="processedoutput.id",primary_key=True)

class Usage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None
    processed_output: 'ProcessedOutput' = Relationship(back_populates="usage", link_model=OutputUsageLinkage)

class GeneratedJsonObject(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    object: Dict[str, Any] = Field(sa_column=Column(JSON))
    processed_output: 'ProcessedOutput' = Relationship(back_populates="json_object", link_model=OutputJsonObjectLinkage)
    tool_call_id : Optional[str] = None

class RawOutput(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    raw_result: Union[str, dict, ChatCompletion, AnthropicMessage, PromptCachingBetaMessage] = Field(sa_column=Column(JSON))
    completion_kwargs: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    chat_thread_id: Optional[int] = Field(default=None, foreign_key="chatthread.id")
    start_time: float
    end_time: float
    client:LLMClient 

    @property
    def time_taken(self) -> float:
        return self.end_time - self.start_time

    @computed_field
    @property
    def str_content(self) -> Optional[str]:
        return self._parse_result()[0]

    @computed_field
    @property
    def json_object(self) -> Optional[GeneratedJsonObject]:
        return self._parse_result()[1]
    
    @computed_field
    @property
    def error(self) -> Optional[str]:
        return self._parse_result()[3]

    @computed_field
    @property
    def contains_object(self) -> bool:
        return self._parse_result()[1] is not None
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        return self._parse_result()[2]

    @computed_field
    @property
    def result_provider(self) -> Optional[LLMClient]:
        return self.search_result_provider() if self.client is None else self.client
    
    @model_validator(mode="after")
    def validate_provider_and_client(self) -> Self:
        if self.client is not None and self.result_provider != self.client:
            raise ValueError(f"The inferred result provider '{self.result_provider}' does not match the specified client '{self.client}'")
        return self
    
    
    def search_result_provider(self) -> Optional[LLMClient]:
        try:
            oai_completion = ChatCompletion.model_validate(self.raw_result)
            return LLMClient.openai
        except ValidationError:
            try:
                anthropic_completion = AnthropicMessage.model_validate(self.raw_result)
                return LLMClient.anthropic
            except ValidationError:
                try:
                    antrhopic_beta_completion = PromptCachingBetaMessage.model_validate(self.raw_result)
                    return LLMClient.anthropic
                except ValidationError:
                    return None

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        return parse_json_string(content)
    
    

    def _parse_oai_completion(self,chat_completion:ChatCompletion) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], None]:
        message = chat_completion.choices[0].message
        content = message.content

        json_object = None
        usage = None

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            tool_call_id = tool_call.id
            try:
                object_dict = json.loads(tool_call.function.arguments)
                json_object = GeneratedJsonObject(name=name, object=object_dict, tool_call_id=tool_call_id)
            except json.JSONDecodeError:
                json_object = GeneratedJsonObject(name=name, object={"raw": tool_call.function.arguments}, tool_call_id=tool_call_id)
        elif content is not None:
            if self.completion_kwargs:
                name = self.completion_kwargs.get("response_format",{}).get("json_schema",{}).get("name",None)
            else:
                name = None
            parsed_json = self._parse_json_string(content)
            if parsed_json:
                
                json_object = GeneratedJsonObject(name="parsed_content" if name is None else name,
                                                   object=parsed_json)
                content = None  # Set content to None when we have a parsed JSON object
                #print(f"parsed_json: {parsed_json} with name")
        if chat_completion.usage:
            usage = Usage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens
            )

        return content, json_object, usage, None

    def _parse_anthropic_message(self, message: Union[AnthropicMessage, PromptCachingBetaMessage]) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],None]:
        content = None
        json_object = None
        usage = None

        if message.content:
            first_content = message.content[0]
            if isinstance(first_content, TextBlock):
                content = first_content.text
                parsed_json = self._parse_json_string(content)
                if parsed_json:
                    json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                    content = None  # Set content to None when we have a parsed JSON object
            elif isinstance(first_content, ToolUseBlock):
                name = first_content.name
                input_dict : Dict[str,Any] = first_content.input # type: ignore  # had to ignore due to .input being of object class
                json_object = GeneratedJsonObject(name=name, object=input_dict)

        if hasattr(message, 'usage'):
            usage = Usage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                cache_creation_input_tokens=getattr(message.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(message.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage, None
    

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],Optional[str]]:
        provider = self.result_provider
        if getattr(self.raw_result, "error", None):
            return None, None, None,  getattr(self.raw_result, "error", None)
        if provider == "openai":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "anthropic":
            try: #beta first
                return self._parse_anthropic_message(PromptCachingBetaMessage.model_validate(self.raw_result))
            except ValidationError:
                return self._parse_anthropic_message(AnthropicMessage.model_validate(self.raw_result))
        elif provider == "vllm":
             return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "litellm":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        else:
            raise ValueError(f"Unsupported result provider: {provider}")
    
    def create_processed_output(self) -> 'ProcessedOutput':
        content, json_object, usage, error = self._parse_result()
        if (json_object is None and content is None) or usage is None or self.chat_thread_id is None:
            print(f"content: {content}, json_object: {json_object}, usage: {usage}, error: {error}, chat_thread_id: {self.chat_thread_id}")
            raise ValueError("No JSON object or usage found or chat_thread_id in the raw output, can not create processed output")
   
        processed_output = ProcessedOutput(content=content, json_object=json_object, usage=usage, error=error, time_taken=self.time_taken, llm_client=self.client, raw_output=self, chat_thread_id=self.chat_thread_id)
        return processed_output

    class Config:
        arbitrary_types_allowed = True


class ProcessedOutput(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: Optional[str] = None
    json_object: Optional[GeneratedJsonObject] = Relationship(back_populates="processed_output", link_model=OutputJsonObjectLinkage,sa_relationship_kwargs={"lazy": "joined"})
    usage: Usage = Relationship(back_populates="processed_output", link_model=OutputUsageLinkage,sa_relationship_kwargs={"lazy": "joined"})
    raw_output: 'RawOutput' = Relationship(link_model=RawProcessedLinkage,sa_relationship_kwargs={"lazy": "joined"})
    error: Optional[str] = None
    time_taken: float
    llm_client: LLMClient
    chat_thread_id: int = Field(foreign_key="chatthread.id")
    chat_thread: 'ChatThread' = Relationship(back_populates="processed_outputs", link_model=ChatThreadProcessedOutputLinkage,sa_relationship_kwargs={"lazy": "joined"})
