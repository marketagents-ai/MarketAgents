
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, Dict, Any, List, Iterable
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from openai.types.shared_params import ResponseFormatText, ResponseFormatJSONObject

#import together
from anthropic.types import MessageParam
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam
from anthropic.types.beta.prompt_caching.prompt_caching_beta_text_block_param import PromptCachingBetaTextBlockParam
from anthropic.types import (
    TextBlock,
    
)

from openai.types.chat import ChatCompletion
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.shared_params import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema
from anthropic.types import ToolParam

from pydantic import BaseModel, Field, computed_field
from typing import Union, Optional, List, Tuple, Literal, Dict, Any
from anthropic.types import Message as AnthropicMessage
from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage, PromptCachingBetaToolParam
from anthropic.types import TextBlock, ToolUseBlock
import json
from .utils import msg_dict_to_oai, msg_dict_to_anthropic, parse_json_string

from dataclasses import dataclass, field
from typing import List, Optional
import time



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
    use_schema_instruction: bool = Field(default=False, description="Whether to use the schema instruction")
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
        if self.llm_config.client == 'anthropic' and  self.llm_config.response_format in ["json_beg", "structured_output","json_object"]:

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
       
    
    def update_history_safely(self,history:List[Dict[str, Any]]) -> 'LLMPromptContext':
        return self.model_copy(update={"history":history})
        
    
    def append_to_history_safely(self,new_message:Dict[str, Any]) -> 'LLMPromptContext':
        if  self.history:
            return self.model_copy(update={"history":self.history.append(new_message)})
        else:
            assert not self.history
            return self.update_history_safely(history=[new_message])
        
    
    def add_chat_turn_history_safely(self, llm_output:'LLMOutput') -> 'LLMPromptContext':
        """
        Safely adds a user-assistant chat turn to the history based on the LLMOutput.
        
        Args:
            llm_output (LLMOutput): The output from the LLM completion.
        
        Returns:
            LLMPromptContext: A new instance with the updated history.
        """
        if llm_output.completion_kwargs is None or "messages" not in llm_output.completion_kwargs:
            raise ValueError("LLMOutput does not contain message history")

        messages = llm_output.completion_kwargs["messages"]
        user_message = next((msg["content"] for msg in reversed(messages) if msg["role"] == "user"), None)
        
        if user_message is None:
            raise ValueError("No user message found in the completion history")

        assistant_response = llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"

        new_turn = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ]
        
        if self.history is None:
            return self.model_copy(update={"history": new_turn})
        else:
            return self.model_copy(update={"history": self.history + new_turn})
        
    def add_chat_turn_history(self, llm_output:'LLMOutput'):
        """ add a chat turn to the history without safely model copy just normal append """
        if self.history is None:
            self.history = []
        self.history.append({"role": "user", "content": self.new_message})
        self.history.append({"role": "assistant", "content": llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"})
    
    def get_tool(self) -> Union[ChatCompletionToolParam, PromptCachingBetaToolParam, None]:
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
    start_time: float = Field(default_factory=time.time)
    end_time: float = Field(default_factory=time.time)

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
        elif content is not None:
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
        provider : Optional[Literal["openai", "anthropic"]] = None

        if "model" in self.raw_result:
            provider = "openai" if "gpt" in self.raw_result["model"] else "anthropic"
        else:
            raise ValueError("No model found in the result")

        if provider == "openai":
            first_choice = self.raw_result["choices"][0]
            if "message" in first_choice:
                if first_choice["message"].get("tool_calls"):
                    tool_call = first_choice["message"]["tool_calls"][0]
                    name = tool_call["function"]["name"]
                    try:
                        object_dict = json.loads(tool_call["function"]["arguments"])
                        json_object = GeneratedJsonObject(name=name, object=object_dict)
                    except json.JSONDecodeError:
                        json_object = GeneratedJsonObject(name=name, object={"raw": tool_call["function"]["arguments"]})
                content = first_choice["message"].get("content")
                if content is not None:
                    parsed_json = self._parse_json_string(content)
                    if parsed_json:
                        json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                        content = None  # Set content to None when we have a parsed JSON object

                if "function_call" in first_choice["message"]:
                    func_call = first_choice["message"]["function_call"]
                    name = func_call["name"]
                    try:
                        object_dict = json.loads(func_call["arguments"])
                        json_object = GeneratedJsonObject(name=name, object=object_dict)
                       
                    except json.JSONDecodeError:
                        json_object = GeneratedJsonObject(name=name, object={"raw": func_call["arguments"]})
 
        elif provider == "anthropic":
            content_list = self.raw_result["content"]
            if isinstance(content_list, list) and len(content_list) > 0:
                first_content = content_list[0]
                if isinstance(first_content, dict):
                    if first_content.get("type") == "text":
                        content = first_content.get("text")
                        if content:
                            parsed_json = self._parse_json_string(content)
                            if parsed_json:
                                    json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                                    content = None  # Set content to None when we have a parsed JSON object
                    elif first_content.get("type") == "tool_use":
                        name = first_content.get("name", "unknown_tool")
                        json_object = GeneratedJsonObject(name=name, object=first_content.get("input", {}))
        if "usage" in self.raw_result and provider is not None:
           
            usage_data = self.raw_result["usage"]
            if provider == "openai":
                usage = Usage(
                    prompt_tokens=usage_data["prompt_tokens"],
                    completion_tokens=usage_data["completion_tokens"],
                    total_tokens=usage_data["total_tokens"]
                )
            elif provider == "anthropic":
                usage = Usage(
                    prompt_tokens=usage_data["input_tokens"],
                    completion_tokens=usage_data["output_tokens"],
                    total_tokens=usage_data["input_tokens"] + usage_data["output_tokens"],
                    cache_creation_input_tokens=usage_data.get("cache_creation_input_tokens", None),
                    cache_read_input_tokens=usage_data.get("cache_read_input_tokens", None)
                )

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

