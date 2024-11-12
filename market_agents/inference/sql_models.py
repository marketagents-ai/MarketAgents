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
from market_agents.inference.utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string
import uuid
from uuid import UUID

class Tool(SQLModel, table=True):
    id: Optional[int]  = Field(default=None, primary_key=True)
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True
    json_schema: Dict = Field(default = {},sa_column=Column(JSON))
    chats: List["ChatThread"] = Relationship(back_populates="structured_output")
    callable: bool = False
    callable_function: Optional[str] = None
    callable_output_schema: Optional[Dict[str, Any]] = Field(default = None,sa_column=Column(JSON))
    allow_literal_eval: bool = False

    def parse_execute_callable_function(self) -> Callable:
        """
        Safely parse the callable function string using ast.literal_eval.
        Only allows access to functions in the current module's globals.
        
        Returns:
            Callable: The parsed callable function
            
        Raises:
            ValueError: If callable_function is not set or function not found
            TypeError: If parsed value is not callable
        """
        if self.callable_function is None:
            raise ValueError("callable_function is not set")
            
        # Get the calling module's globals
        frame = sys._getframe(1)
        while frame:
            if frame.f_globals.get('__name__') != __name__:
                global_dict = frame.f_globals
                break
            frame = frame.f_back
        else:
            raise ValueError("Could not find calling module")

        # Try to get function directly from globals first
        if self.callable_function in global_dict:
            func = global_dict[self.callable_function]
            if callable(func):
                return func
            raise TypeError(f"'{self.callable_function}' is not callable")
        if self.allow_literal_eval:
            # Otherwise try to evaluate as a literal
            try:
                parsed = literal_eval(self.callable_function)
                if not callable(parsed):
                    raise TypeError(f"Parsed value '{self.callable_function}' is not callable")
                return parsed
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"Could not parse callable_function: {e}")
        raise ValueError(f"callable_function '{self.callable_function}' is not present in the global scope and allow_literal_eval is False")

    def execute(self, input: Dict[str, Any], tool_call_id: Optional[str] = None) -> 'ChatMessage':
        """
        Execute the callable function with the given input.
        
        Args:
            input: Dictionary of input parameters. If the function expects a BaseModel,
                this should be the dict representation of that model.
            
        Returns:
            ChatMessage: The response message with JSON-serialized content
        """
        if not self.callable:
            raise ValueError("Tool is not callable")
        
        callable_func = self.parse_execute_callable_function()
        
        # Get the first parameter's type hint
        sig = signature(callable_func)
        type_hints = get_type_hints(callable_func)
        first_param = next(iter(sig.parameters.values()))
        param_type = type_hints.get(first_param.name)
        
        # If the first parameter is a BaseModel, construct it from input
        if (isinstance(param_type, type) and 
            issubclass(param_type, BaseModel)):
            model_input = param_type.model_validate(input)
            response = callable_func(model_input)
        else:
            response = callable_func(**input)
        
        # Convert response to JSON string
        if isinstance(response, BaseModel):
            content = response.model_dump_json()
        else:
            content = json.dumps({"result": response})
            
        return ChatMessage(
            role=MessageRole.tool, 
            content=content, 
            tool_name=f"{self.schema_name}_response",
            tool_call_id=tool_call_id
        )
            

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
            schema = JSONSchema(name=self.schema_name,description=self.schema_description,schema=self.json_schema,strict=self.strict_schema)
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
        
        # Handle input schema
        first_param = next(iter(sig.parameters.values()))
        first_param_type = type_hints.get(first_param.name)
        
        # If first parameter is BaseModel, use its schema
        if (isinstance(first_param_type, type) and 
            issubclass(first_param_type, BaseModel)):
            derived_input_schema = first_param_type.model_json_schema()
        else:
            # Create input model from parameters
            input_fields = {}
            for param_name, param in sig.parameters.items():
                if param_name not in type_hints:
                    raise ValueError(f"Parameter {param_name} must have a type hint")
                
                if param.default is param.empty:
                    input_fields[param_name] = (type_hints[param_name], ...)
                else:
                    input_fields[param_name] = (type_hints[param_name], param.default)

            InputModel = create_model(
                f"{func.__name__}Input",
                **input_fields
            )
            derived_input_schema = InputModel.model_json_schema()
        
        # Validate provided schema if any
        if json_schema is not None:
            # (validation code remains the same)
            pass
        else:
            json_schema = derived_input_schema
        
        # Handle output type
        output_type = type_hints['return']
        if isinstance(output_type, type) and issubclass(output_type, BaseModel):
            OutputModel = output_type
        else:
            OutputModel = create_model(
                f"{func.__name__}Output",
                result=(output_type, ...)
            )
        
        return cls(
            schema_name=schema_name if schema_name is not None else func.__name__,
            schema_description=schema_description if schema_description is not None else (func.__doc__ or f"Execute {func.__name__} function"),
            instruction_string=instruction_string or "Please follow this JSON schema for your response:",
            strict_schema=strict_schema,
            json_schema=json_schema,
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

    def to_chatml_dict(self) -> Dict[str, Any]:
        return {"role":self.role.value,"content":self.content}
    
    def to_string(self) -> str:
        if self.format == MessageFormat.python_dict:
            return json.dumps(self.to_chatml_dict())
        else:
            raise ValueError(f"Message format {self.format} is not supported")
    def to_share_gpt_dict(self) -> Dict[str, Any]:
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

    
class ChatThread (SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    uuid: UUID = Field(default_factory=lambda: uuid.uuid4())
    system_string: Optional[str] = None
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
        if self.llm_config.client == 'openai' and 'json' in self.llm_config.response_format and not self.use_schema_instruction:
            return True

        else:
            return False
        
    @computed_field
    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        content= self.system_string if self.system_string  else ""
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
        
        if not json_object:
            if str_content:
                response = str_content
                tool_name = None
            else:
                raise ValueError("ProcessedOutput json_object is None and content is None, cannot add to history")
        else:
            response = json.dumps(json_object.object)
            tool_name = json_object.name

        assistant_message = ChatMessage(
            role=MessageRole.assistant, 
            content=response, 
            parent_message_uuid=user_message_uuid,
            tool_name=tool_name
        )
        
        self.history.append(assistant_message)
        self.processed_outputs.append(llm_output)
        self.new_message = None

    def add_assistant_and_tool_execution_response(self, llm_output: 'ProcessedOutput'):
        """Add the assistant's response from the ProcessedOutput to the history"""
        user_message_uuid = self.get_last_message_uuid()
        assert user_message_uuid is not None, "User message uuid is None, cannot add assistant response"
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
        if json_object is None or usage is None or self.chat_thread_id is None:
            print(f"content: {content}, json_object: {json_object}, usage: {usage}, error: {error}, chat_thread_id: {self.chat_thread_id}")
            raise ValueError("No JSON object or usage found or chat_thread_id in the raw output, can not create processed output")
        processed_output = ProcessedOutput(content=content, json_object=json_object, usage=usage, error=error, time_taken=self.time_taken, llm_client=self.client, raw_output=self, chat_thread_id=self.chat_thread_id)
        return processed_output

    class Config:
        arbitrary_types_allowed = True


class ProcessedOutput(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    content: Optional[str] = None
    json_object: GeneratedJsonObject = Relationship(back_populates="processed_output", link_model=OutputJsonObjectLinkage,sa_relationship_kwargs={"lazy": "joined"})
    usage: Usage = Relationship(back_populates="processed_output", link_model=OutputUsageLinkage,sa_relationship_kwargs={"lazy": "joined"})
    raw_output: 'RawOutput' = Relationship(link_model=RawProcessedLinkage,sa_relationship_kwargs={"lazy": "joined"})
    error: Optional[str] = None
    time_taken: float
    llm_client: LLMClient
    chat_thread_id: int = Field(foreign_key="chatthread.id")
    chat_thread: 'ChatThread' = Relationship(back_populates="processed_outputs", link_model=ChatThreadProcessedOutputLinkage,sa_relationship_kwargs={"lazy": "joined"})


if __name__ == "__main__":
    def create_processed_output(engine:Engine) -> ProcessedOutput:
        with Session(engine) as session:
            first_chat = session.exec(select(ChatThread)).first()
            if first_chat is None:
                raise ValueError("No chat thread found, can not create processed output")
            elif first_chat.id is None:
                raise ValueError("Chat thread id is not set, can not create processed output")
            dummy_usage = Usage(prompt_tokens=69, completion_tokens=420, total_tokens=69420)
            dummy_json_object = GeneratedJsonObject(name="dummy_json_object", object={"dummy": "object"})
            dummy_raw_output = RawOutput(raw_result="dummy_raw_output", client=LLMClient.openai, start_time=10, end_time=20, chat_thread_id=first_chat.id)
            dummy_processed_output = ProcessedOutput(usage=dummy_usage, json_object=dummy_json_object, raw_output=dummy_raw_output, time_taken=10, llm_client=LLMClient.openai, chat_thread=first_chat, 
                                                     chat_thread_id=first_chat.id    )
            session.add(dummy_processed_output)
            session.commit()
        return dummy_processed_output



    def create_chat(engine: Engine):
        with Session(engine) as session:
            oai_config = LLMConfig(client=LLMClient.openai, model="gpt-4o", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
            edit_tool = Tool(schema_name="edit_tool",
                            schema_description="Edit the provided JSON schema.",
                            instruction_string="Please follow this JSON schema for your response:",
                            strict_schema=True,
                            json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}})
            chat = ChatThread (new_message="Hello, how are you?", structured_output=edit_tool, llm_config=oai_config)      
            chat_italian = ChatThread (new_message="Ciao, come stai?", structured_output=edit_tool, llm_config=oai_config)
            anthropic_config = LLMConfig(client=LLMClient.anthropic, model="claude-3-5-sonnet-20240620", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
            chat_french_anthropic = ChatThread (new_message="Bonjour, comment ça va?", structured_output=Tool(schema_name="outil_edition",
                                schema_description="Éditer le schéma JSON fourni.",
                                instruction_string="Veuillez suivre ce schéma JSON pour votre réponse:",
                                strict_schema=True,
                                json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}}),
                                llm_config=anthropic_config)
            session.add(chat)
            session.add(chat_italian)
            session.add(chat_french_anthropic)
            session.commit()

    def create_all_snapshots(engine: Engine):
        """ we get all the chats and create snapshots for each one """
        with Session(engine) as session:
            statement = select(ChatThread)
            result = session.exec(statement).unique()
            for chat in result:
                snapshot = chat.create_snapshot()
                session.add(snapshot)
            session.commit()

    def add_history_to_all_chats(engine: Engine):
        test_history = [{"role":"user","content":"Hello, how are you?"},{"role":"assistant","content":"I'm fine, thank you!"}]
        with Session(engine) as session:
            statement = select(ChatThread)
            result = session.exec(statement).unique()
            for chat in result:
                chat.history = [ChatMessage.from_dict(message) for message in test_history]
                chat.update_db_from_session(session)

    def select_openai_config(engine: Engine):
        with Session(engine) as session:
            statement = select(ChatThread , LLMConfig).join(LLMConfig).where(LLMConfig.client == LLMClient.openai)
            result = session.exec(statement)
            for chat, config in result:
                print(chat.new_message)

    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)

    SQLModel.metadata.create_all(engine)
    create_chat(engine)
    # select_openai_config(engine)
    create_all_snapshots(engine)
    add_history_to_all_chats(engine)
    create_processed_output(engine)
