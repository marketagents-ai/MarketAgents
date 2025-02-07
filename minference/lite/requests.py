"""
Direct extraction of InferenceOrchestrator methods as standalone functions.
Maintains exact behavior while making functions reusable outside the orchestrator.
"""
import json
import time
from typing import Dict, Any, Optional, List, Union
from pydantic import ValidationError, BaseModel, Field
from uuid import UUID
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool, ToolChoiceToolChoiceAuto
from minference.lite.models import ChatThread, LLMClient, ResponseFormat
from minference.oai_parallel import OAIApiFromFileConfig
from minference.enregistry import EntityRegistry
from pydantic import BaseModel, Field
from typing import  Optional, Union, Dict, List, Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam
)
from openai.types.shared_params import (

    FunctionDefinition
)
from openai.types.chat.completion_create_params import (
    ResponseFormat as OpenAIResponseFormat,
    FunctionCall
)


from anthropic.types import (
    MessageParam,

    ToolParam,
    TextBlockParam,

    ToolChoiceParam,

    
)
from anthropic.types.model_param import ModelParam


class OpenAIRequest(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = Field(default=None)
    function_call: Optional[FunctionCall] = Field(default=None)
    functions: Optional[List[FunctionDefinition]] = Field(default=None)
    logit_bias: Optional[Dict[str, int]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    n: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    response_format: Optional[OpenAIResponseFormat] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = Field(default=None)
    tools: Optional[List[ChatCompletionToolParam]] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    user: Optional[str] = Field(default=None)

    class Config:
        extra = 'forbid'


class AnthropicRequest(BaseModel):
    max_tokens: int
    messages: List[MessageParam]
    model: ModelParam
    metadata: Optional[Dict] = Field(default=None)
    stop_sequences: Optional[List[str]] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    system: Optional[Union[str, List[TextBlockParam]]] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    tool_choice: Optional[ToolChoiceParam] = Field(default=None)
    tools: Optional[List[ToolParam]] = Field(default=None)
    top_k: Optional[int] = Field(default=None)
    top_p: Optional[float] = Field(default=None)

class VLLMConfig(BaseModel):
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
    
class VLLMRequest(OpenAIRequest,VLLMConfig):
    pass
def prepare_requests_file(chat_threads: List[ChatThread], client: str, filename: str):
    """Prepare JSONL file with chat thread requests."""
    requests = []
    EntityRegistry._logger.info(f"Preparing {client} requests for {len(chat_threads)} chat threads")
    for chat_thread in chat_threads:
        request = convert_chat_thread_to_request(chat_thread, client)
        if request:
            metadata = {
                "chat_thread_id": str(chat_thread.id),
                "start_time": time.time(),
                "end_time": None,
                "total_time": None
            }
            requests.append([metadata, request])
    
    with open(filename, 'w') as f:
        for request in requests:
            json.dump(request, f)
            f.write('\n')
    EntityRegistry._logger.info(f"Wrote {len(requests)} requests to {filename}")

def validate_anthropic_request(request: Dict[str, Any]) -> bool:
    """Validate an Anthropic API request."""

    try:
        anthropic_request = AnthropicRequest(**request)
        return True
    except ValidationError as e:
        EntityRegistry._logger.error(f"Error validating Anthropic request: {e}")
        # Re-raise the original ValidationError instead of creating a new one
        raise e
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating Anthropic request: {e}")
        # For other types of errors, raise a ValueError instead
        raise ValueError(f"Error validating Anthropic request: {e} with request: {request}")

def validate_openai_request(request: Dict[str, Any]) -> bool:
    """Validate an OpenAI API request."""
    try:
        openai_request = OpenAIRequest(**request)
        return True
    except ValidationError as e:
        EntityRegistry._logger.error(f"Error validating OpenAI request: {e}")
        raise e
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating OpenAI request: {e}")
        raise ValueError(f"Error validating OpenAI request: {e} with request: {request}")

def validate_vllm_request(request: Dict[str, Any]) -> bool:
    """Validate a vLLM API request."""
    try:
        vllm_request = VLLMRequest(**request)
        return True
    except ValidationError as e:
        EntityRegistry._logger.error(f"Error validating VLLM request: {e}")
        raise e
    except Exception as e:
        EntityRegistry._logger.error(f"Error validating VLLM request: {e}")
        raise ValueError(f"Error validating VLLM request: {e} with request: {request}")

def get_openai_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get OpenAI format request from chat thread."""
    EntityRegistry._logger.info(f"Getting OpenAI request for ChatThread({chat_thread.id}) with response format {chat_thread.llm_config.response_format}")
    messages = chat_thread.oai_messages
    request = {
        "model": chat_thread.llm_config.model,
        "messages": messages,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
    }
    if chat_thread.oai_response_format:
        request["response_format"] = chat_thread.oai_response_format
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_openai_tool()]
            request["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
    elif chat_thread.llm_config.response_format == "auto_tools":
        tools = chat_thread.tools
        if tools:
            request["tools"] = [t.get_openai_tool() for t in tools]
            request["tool_choice"] = "auto"
            EntityRegistry._logger.info(f"Added {len(tools)} tools to OpenAI request as auto_tools")
    elif chat_thread.llm_config.response_format == ResponseFormat.workflow:
        #detected workflow mode 
        if chat_thread.workflow_step is None:
            raise ValueError("Workflow step is None")
        EntityRegistry._logger.info(f"Detected workflow mode for ChatThread({chat_thread.id}) with workflow step {chat_thread.workflow_step}")
        if chat_thread.workflow_step >= len(chat_thread.tools):
            raise ValueError(f"Workflow step {chat_thread.workflow_step} is out of range for tools: {chat_thread.tools}")
        tool = chat_thread.tools[chat_thread.workflow_step]
        if tool:
            request["tools"] = [tool.get_openai_tool()]
            request["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
            EntityRegistry._logger.info(f"Added tool({tool.name}) to OpenAI request as workflow step {chat_thread.workflow_step}")
            chat_thread.workflow_step += 1
        else:
            EntityRegistry._logger.error(f"Tool not found for workflow step {chat_thread.workflow_step}")
    elif chat_thread.llm_config.response_format != ResponseFormat.text:
        raise ValueError(f"Invalid response format: {chat_thread.llm_config.response_format}")
        
    if validate_openai_request(request):
        EntityRegistry._logger.info(f"Validated OpenAI request for ChatThread({chat_thread.id}) with response format {chat_thread.llm_config.response_format}")
        return request
    else:
        EntityRegistry._logger.error(f"Failed to validate OpenAI request for ChatThread({chat_thread.id}) with response format {chat_thread.llm_config.response_format}")
        return None

def get_anthropic_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get Anthropic format request from chat thread."""
    system_content, messages = chat_thread.anthropic_messages    
    request = {
        "model": chat_thread.llm_config.model,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
        "messages": messages,
        "system": system_content if system_content else None
    }
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_anthropic_tool()]
            request["tool_choice"] = ToolChoiceToolChoiceTool(name=tool.name, type="tool")
    elif chat_thread.llm_config.response_format == "auto_tools":
        tools = chat_thread.tools
        if tools:
            request["tools"] =  chat_thread.get_tools_for_llm()
            request["tool_choice"] = ToolChoiceToolChoiceAuto(type="auto")
    elif chat_thread.llm_config.response_format == ResponseFormat.workflow:
        if chat_thread.workflow_step is None:
            raise ValueError("Workflow step is None")
        if chat_thread.workflow_step > len(chat_thread.tools):
            raise ValueError(f"Workflow step {chat_thread.workflow_step} is out of range for tools: {chat_thread.tools}")
        tool = chat_thread.tools[chat_thread.workflow_step]
        if tool:
            request["tools"] = [tool.get_anthropic_tool()]
            request["tool_choice"] = ToolChoiceToolChoiceTool(name=tool.name, type="tool")
            chat_thread.workflow_step += 1
        else:
            EntityRegistry._logger.error(f"Tool not found for workflow step {chat_thread.workflow_step}")



    if validate_anthropic_request(request):
        return request
    else:
        return None

def get_vllm_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get vLLM format request from chat thread."""
    messages = chat_thread.vllm_messages
    request = {
        "model": chat_thread.llm_config.model,
        "messages": messages,
        "max_tokens": chat_thread.llm_config.max_tokens,
        "temperature": chat_thread.llm_config.temperature,
    }
    if chat_thread.llm_config.response_format == "tool" and chat_thread.forced_output:
        tool = chat_thread.forced_output
        if tool:
            request["tools"] = [tool.get_openai_tool()]
            request["tool_choice"] = {"type": "function", "function": {"name": tool.name}}
    if chat_thread.llm_config.response_format == "json_object":
        raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
    if chat_thread.oai_response_format and chat_thread.oai_response_format:
        request["response_format"] = chat_thread.oai_response_format
    
    if validate_vllm_request(request):
        return request
    else:
        return None

def get_litellm_request(chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
    """Get LiteLLM format request from chat thread."""
    if chat_thread.llm_config.response_format == "json_object":
        raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
    return get_openai_request(chat_thread)

def convert_chat_thread_to_request(chat_thread: ChatThread, client: str) -> Optional[Dict[str, Any]]:
    """Convert chat thread to client-specific request format."""
    if client == "openai":
        return get_openai_request(chat_thread)
    elif client == "anthropic":
        return get_anthropic_request(chat_thread)
    elif client == "vllm":
        return get_vllm_request(chat_thread)
    elif client == "litellm":
        return get_litellm_request(chat_thread)
    else:
        raise ValueError(f"Invalid client: {client}")

def create_oai_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    openai_key: str,
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create OpenAI completion configuration."""
    if chat_thread.llm_config.client == "openai" and openai_key:
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url="https://api.openai.com/v1/chat/completions",
            api_key=openai_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_anthropic_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    anthropic_key: str,
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create Anthropic completion configuration."""
    if chat_thread.llm_config.client == "anthropic" and anthropic_key:
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url="https://api.anthropic.com/v1/messages",
            api_key=anthropic_key,
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_vllm_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    vllm_endpoint: str,
    vllm_key: Optional[str],
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create vLLM completion configuration."""
    if chat_thread.llm_config.client == "vllm":
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url=vllm_endpoint,
            api_key=vllm_key if vllm_key else "",
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None

def create_litellm_completion_config(
    chat_thread: ChatThread, 
    requests_file: str, 
    results_file: str,
    litellm_endpoint: str,
    litellm_key: Optional[str],
    max_requests_per_minute: int,
    max_tokens_per_minute: int
) -> Optional[OAIApiFromFileConfig]:
    """Create LiteLLM completion configuration."""
    if chat_thread.llm_config.client == "litellm":
        return OAIApiFromFileConfig(
            requests_filepath=requests_file,
            save_filepath=results_file,
            request_url=litellm_endpoint,
            api_key=litellm_key if litellm_key else "",
            max_requests_per_minute=max_requests_per_minute,
            max_tokens_per_minute=max_tokens_per_minute,
            token_encoding_name="cl100k_base",
            max_attempts=5,
            logging_level=20,
        )
    return None