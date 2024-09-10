import os
import time
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, Dict, Any, List, Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.chat import ChatCompletionToolChoiceOptionParam
from openai.types.chat import completion_create_params
from openai._types import NotGiven
from typing_extensions import TypeAlias
from typing import List as TypeList
from openai.types.shared_params import ResponseFormatText, ResponseFormatJSONObject

#import together
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic
from anthropic.types import MessageParam, TextBlockParam, ModelParam
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam
from anthropic.types.beta.prompt_caching.prompt_caching_beta_text_block_param import PromptCachingBetaTextBlockParam
from anthropic.types import (
    ContentBlock, TextBlockParam, ImageBlockParam, TextBlock,
    ToolUseBlockParam, ToolResultBlockParam,
)
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
)

from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema
from anthropic.types import ToolParam

from pydantic import BaseModel, Field, computed_field
from typing import Union, Optional, List, Tuple, Literal, Dict, Any
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from anthropic.types import Message as AnthropicMessage
from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage, PromptCachingBetaToolParam
from anthropic.types import TextBlock, ToolUseBlock
import json
import re

def parse_json_string(content: str) -> Optional[Dict[str, Any]]:
    # Remove any leading/trailing whitespace and newlines
    cleaned_content = content.strip()
    
    # Remove markdown code block syntax if present
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned_content, flags=re.MULTILINE)
    
    # Attempt to find a JSON object, allowing for newlines and escaped quotes
    json_match = re.search(r'(\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\})', cleaned_content, re.DOTALL)
    if json_match:
        try:
            # Normalize newlines and unescape quotes
            json_str = json_match.group(1).replace('\n', '').replace('\\"', '"')
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # If no valid JSON found, return None
    return None

def get_ai_context_length(ai_vendor: Literal["openai", "azure_openai", "anthropic"]):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")
        else:
            return "Invalid AI vendor"

def msg_dict_to_oai(messages: List[Dict[str, Any]]) -> List[ChatCompletionMessageParam]:
        def convert_message(msg: Dict[str, Any]) -> ChatCompletionMessageParam:
            role = msg["role"]
            if role == "system":
                return ChatCompletionSystemMessageParam(role=role, content=msg["content"])
            elif role == "user":
                return ChatCompletionUserMessageParam(role=role, content=msg["content"])
            elif role == "assistant":
                assistant_msg = ChatCompletionAssistantMessageParam(role=role, content=msg.get("content"))
                if "function_call" in msg:
                    assistant_msg["function_call"] = msg["function_call"]
                if "tool_calls" in msg:
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                return assistant_msg
            elif role == "tool":
                return ChatCompletionToolMessageParam(role=role, content=msg["content"], tool_call_id=msg["tool_call_id"])
            elif role == "function":
                return ChatCompletionFunctionMessageParam(role=role, content=msg["content"], name=msg["name"])
            else:
                raise ValueError(f"Unknown role: {role}")

        return [convert_message(msg) for msg in messages]

def msg_dict_to_anthropic(messages: List[Dict[str, Any]],use_cache:bool=True,use_prefill:bool=False) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        def create_anthropic_system_message(system_message: Optional[Dict[str, Any]],use_cache:bool=True) -> List[PromptCachingBetaTextBlockParam]:
            if system_message and system_message["role"] == "system":
                text = system_message["content"]
                if use_cache:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text, cache_control=PromptCachingBetaCacheControlEphemeralParam(type="ephemeral"))]
                else:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text)]
            return []

        def convert_message(msg: Dict[str, Any],use_cache:bool=False) -> Union[MessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                if not use_cache:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content)]
                else:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content,cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral'))]
            elif isinstance(content, list):
                if not use_cache:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"]) for block in content
                    ]
                else:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block, cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"], cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) for block in content
                    ]
            else:
                raise ValueError("Invalid content type")
            
            return MessageParam(role=role, content=content)
        
        converted_messages = []
        system_message = []
        num_messages = len(messages)
        for i,message in enumerate(messages):
            if message["role"] == "system":
                system_message= create_anthropic_system_message(message,use_cache=use_cache)
            else:
                history_delta = 2 if use_prefill is False else 3
                last_messages = True if i < (num_messages-1-history_delta) and num_messages>=history_delta else False
                converted_messages.append(convert_message(message,use_cache= use_cache if last_messages else False))

        
        return system_message, [msg for msg in converted_messages if msg is not None]



class StructuredTool(BaseModel):
    """ Supported type by OpenAI Structured Output:
    String, Number, Boolean, Integer, Object, Array, Enum, anyOf
    Root must be Object, not anyOf
    Not supported by OpenAI Structured Output: 
    For strings: minLength, maxLength, pattern, format
    For numbers: minimum, maximum, multipleOf
    For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
    oai_reference: https://platform.openai.com/docs/guides/structured-outputs/how-to-use """

    json_schema: Optional[Dict[str, Any]] = None
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True

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

    def get_anthropic_tool(self) -> Optional[ToolParam]:
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
    
class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm"]
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    response_format: Literal["json_beg", "text","json_object","structured_output","tool"] = "text"
    use_cache: bool = True

   


class LLMPromptContext(BaseModel):
    system_string: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    new_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    structured_output : Optional[StructuredTool] = None
    use_schema_instruction: bool = False
    llm_config: LLMConfig


    @computed_field
    @property
    def oai_response_format(self) -> Optional[ResponseFormat]:
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
        if self.llm_config.client == 'anthropic' and  self.llm_config.response_format == "json_beg":

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
    def messages(self)-> List[Dict[str, Any]]:
        messages = [self.system_message] if self.system_message is not None else []
        if self.history:
            messages+=self.history
        messages.append({"role":"user","content":self.new_message})
        if self.use_prefill:
            prefill_message = {"role":"assistant","content":self.prefill}
            messages.append(prefill_message)
        elif self.use_postfill:
            messages[-1]["content"] = messages[-1]["content"] + self.postfill
        return messages
    
    @computed_field
    @property
    def oai_messages(self)-> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
    
    @computed_field
    @property
    def anthropic_messages(self) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)
        
    def update_llm_config(self,llm_config:LLMConfig) -> 'LLMPromptContext':
        
        return self.model_copy(update={"llm_config":llm_config})
       
    
    def update_history(self,history:List[Dict[str, Any]]) -> 'LLMPromptContext':
        return self.model_copy(update={"history":history})
        
    
    def append_to_history(self,new_message:Dict[str, Any]) -> 'LLMPromptContext':
        if  self.history:
            return self.model_copy(update={"history":self.history.append(new_message)})
        else:
            assert not self.history
            return self.update_history(history=[new_message])
       
    
    def get_tool(self) -> Union[ChatCompletionToolParam, ToolParam, None]:
        if not self.structured_output:
            return None
        if self.llm_config.client == "openai":
            return self.structured_output.get_openai_tool()
        elif self.llm_config.client == "anthropic":
            return self.structured_output.get_anthropic_tool()
        else:
            return None

    
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

class GeneratedJsonObject(BaseModel):
    name: str
    object: Dict[str, Any]

class LLMOutput(BaseModel):
    raw_result: Union[str, dict, ChatCompletion, AnthropicMessage, PromptCachingBetaMessage]
    completion_kwargs: Optional[Dict[str, Any]] = None

    def __str__(self):
        if isinstance(self.raw_result, str):
            return self.raw_result
        elif hasattr(self.raw_result, '__dict__'):
            return json.dumps(self.raw_result.__dict__, indent=2, default=str)
        else:
            return str(self.raw_result)

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
    def contains_object(self) -> bool:
        return self._parse_result()[1] is not None
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        return self._parse_result()[2]

    @computed_field
    @property
    def result_type(self) -> Literal["str", "dict", "oaicompletion", "anthropicmessage", "anthropicbetamessage"]:
        if isinstance(self.raw_result, str):
            return "str"
        elif isinstance(self.raw_result, dict):
            return "dict"
        elif isinstance(self.raw_result, ChatCompletion):
            return "oaicompletion"
        elif isinstance(self.raw_result, AnthropicMessage):
            return "anthropicmessage"
        else:
            assert isinstance(self.raw_result, PromptCachingBetaMessage), "Invalid raw result type"
            return "anthropicbetamessage"

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        return parse_json_string(content)
    
    

    def _parse_oai_completion(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage]]:
        assert isinstance(self.raw_result, ChatCompletion), "The result is not an OpenAI ChatCompletion"
        message = self.raw_result.choices[0].message
        content = message.content
        json_object = None
        usage = None

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            try:
                object_dict = json.loads(tool_call.function.arguments)
                json_object = GeneratedJsonObject(name=name, object=object_dict)
            except json.JSONDecodeError:
                json_object = GeneratedJsonObject(name=name, object={"raw": tool_call.function.arguments})
        elif content:
            parsed_json = self._parse_json_string(content)
            if parsed_json:
                json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                content = None  # Set content to None when we have a parsed JSON object

        if self.raw_result.usage:
            usage = Usage(
                prompt_tokens=self.raw_result.usage.prompt_tokens,
                completion_tokens=self.raw_result.usage.completion_tokens,
                total_tokens=self.raw_result.usage.total_tokens
            )

        return content, json_object, usage

    def _parse_anthropic_message(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage]]:
        assert isinstance(self.raw_result, (AnthropicMessage, PromptCachingBetaMessage)), "The message is not an Anthropic message"
        content = None
        json_object = None
        usage = None

        if self.raw_result.content:
            first_content = self.raw_result.content[0]
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

        if hasattr(self.raw_result, 'usage'):
            usage = Usage(
                prompt_tokens=self.raw_result.usage.input_tokens,
                completion_tokens=self.raw_result.usage.output_tokens,
                total_tokens=self.raw_result.usage.input_tokens + self.raw_result.usage.output_tokens,
                cache_creation_input_tokens=getattr(self.raw_result.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(self.raw_result.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage

    def _parse__dict(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage]]:
        assert isinstance(self.raw_result, dict), "The result is not a dictionary"
        content = None
        json_object = None
        usage = None

        if "choices" in self.raw_result and isinstance(self.raw_result["choices"], list):
            first_choice = self.raw_result["choices"][0]
            if "message" in first_choice:
                content = first_choice["message"].get("content")
                if "function_call" in first_choice["message"]:
                    func_call = first_choice["message"]["function_call"]
                    name = func_call["name"]
                    try:
                        object_dict = json.loads(func_call["arguments"])
                        json_object = GeneratedJsonObject(name=name, object=object_dict)
                    except json.JSONDecodeError:
                        json_object = GeneratedJsonObject(name=name, object={"raw": func_call["arguments"]})
        elif "content" in self.raw_result:
            content_list = self.raw_result["content"]
            if isinstance(content_list, list) and len(content_list) > 0:
                first_content = content_list[0]
                if isinstance(first_content, dict):
                    if first_content.get("type") == "text":
                        content = first_content.get("text")
                    elif first_content.get("type") == "tool_use":
                        name = first_content.get("name", "unknown_tool")
                        json_object = GeneratedJsonObject(name=name, object=first_content.get("input", {}))

        if "usage" in self.raw_result:
            usage_data = self.raw_result["usage"]
            usage = Usage(**usage_data)

        return content, json_object, usage

    

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage]]:
        if isinstance(self.raw_result, str):
            return self.raw_result, None, None
        elif self.result_type == "dict":
            return self._parse__dict()
        elif self.result_type == "oaicompletion":
            return self._parse_oai_completion()
        elif self.result_type in ["anthropicmessage", "anthropicbetamessage"]:
            return self._parse_anthropic_message()
        else:
            raise ValueError("Invalid raw result type")

    class Config:
        arbitrary_types_allowed = True


    

class AIUtilities:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        # openai credentials
        self.openai_key = os.getenv("OPENAI_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        # anthropic credentials
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")
    

    def run_ai_completion(self, prompt: LLMPromptContext):
        if prompt.llm_config.client == "openai":
            assert self.openai_key is not None, "OpenAI API key is not set"
            client = OpenAI(api_key=self.openai_key)
            return self.run_openai_completion(client, prompt)
        
        elif prompt.llm_config.client == "anthropic":
            assert self.anthropic_api_key is not None, "Anthropic API key is not set"
            anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.run_anthropic_completion(anthropic, prompt)
        
        else:
            return "Invalid AI vendor"
    
    def run_openai_completion(self, client: OpenAI, prompt: LLMPromptContext):
        try:
            

            completion_kwargs: Dict[str, Any] = {
                "model": prompt.llm_config.model or self.openai_model,
                "messages": prompt.oai_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "response_format": prompt.oai_response_format,
            }
            

            response: ChatCompletion = client.chat.completions.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=f"Error: {str(e)}", completion_kwargs=completion_kwargs)

    def run_anthropic_completion(self, anthropic: Anthropic, prompt: LLMPromptContext):
        
        system_content, anthropic_messages = prompt.anthropic_messages
        model = prompt.llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "system": system_content,
            }
            

            response = anthropic.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)


        
    def run_ai_tool_completion(
        self,
        prompt: LLMPromptContext,
        
    ):
        if prompt.llm_config.client == "openai":
            return self.run_openai_tool_completion(prompt)
        elif prompt.llm_config.client == "anthropic":
            return self.run_anthropic_tool_completion(prompt)
        else:
            raise ValueError("Unsupported client for tool completion")

    def run_openai_tool_completion(
        self,
        prompt: LLMPromptContext,
        
    ):
        client = OpenAI(api_key=self.openai_key)
        
        
        try:
            assert prompt.structured_output is not None, "Tool is not set"
            
            completion_kwargs = {
                "model":  prompt.llm_config.model or self.openai_model,
                "messages": prompt.oai_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
            }

            tool = prompt.get_tool()
            if tool:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
            
            response : ChatCompletion = client.chat.completions.create(**completion_kwargs)

            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)

    def run_anthropic_tool_completion(
        self,
        prompt: LLMPromptContext,
    ):  
        system_content , anthropic_messages = prompt.anthropic_messages
        client = Anthropic(api_key=self.anthropic_api_key)
        model = prompt.llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": prompt.llm_config.max_tokens,
                "temperature": prompt.llm_config.temperature,
                "system": system_content,
            }

            tool = prompt.get_tool()
            if tool and prompt.structured_output is not None:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = ToolChoiceToolChoiceTool(name=prompt.structured_output.schema_name, type="tool")
            response = client.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response, completion_kwargs=completion_kwargs)
        except Exception as e:
            return LLMOutput(raw_result=str(e), completion_kwargs=completion_kwargs)

