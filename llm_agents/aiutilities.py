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

from anthropic.types import ToolParam

from pydantic import BaseModel, Field
from typing import Union, Optional, List
from openai.types.chat import ChatCompletionMessage
from anthropic.types import Message as AnthropicMessage
from anthropic.types.beta.prompt_caching import PromptCachingBetaMessage

from anthropic.types import TextBlock, ToolUseBlock

class LLMOutput(BaseModel):
    raw_result: Union[str, dict, ChatCompletionMessage, AnthropicMessage, PromptCachingBetaMessage]
    parsed_content: Optional[str] = None
    parsed_json: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._parse_result()

    def _parse_result(self):
        if isinstance(self.raw_result, str):
            self.parsed_content = self.raw_result
        elif isinstance(self.raw_result, dict):
            self.parsed_json = self.raw_result
        elif isinstance(self.raw_result, ChatCompletionMessage):
            if self.raw_result.content:
                self.parsed_content = self.raw_result.content
            elif self.raw_result.tool_calls:
                self.parsed_json = {
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": tool_call.type,
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        } for tool_call in self.raw_result.tool_calls
                    ]
                }
        elif isinstance(self.raw_result, (AnthropicMessage, PromptCachingBetaMessage)):
            if self.raw_result.content:
                content = self.raw_result.content
                if isinstance(content, list) and len(content) > 0:
                    first_content = content[0]
                    if isinstance(first_content, TextBlock):
                        self.parsed_content = first_content.text
                    elif isinstance(first_content, ToolUseBlock):
                        self.parsed_json = {"tool_use": {"name": first_content.name, "input": first_content.input}}



class StructuredTool(BaseModel):
    json_schema: Optional[Dict[str, Any]] = None
    schema_name: str = "generate_structured_output"
    schema_description: str = "Generate a structured output based on the provided JSON schema."

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
            return ToolParam(
                name=self.schema_name,
                description=self.schema_description,
                input_schema=self.json_schema
            )
        return None

class Prompt(BaseModel):
    system: Optional[str] = None
    history: Optional[List[Dict[str, Any]]] = None
    user_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", alias="prefill_assistant_message")
    use_prefill: bool = False

    def get_messages_no_system(self) -> List[Dict[str, str]]:
        messages = []
        if self.history:
            messages.extend(self.history)
        messages.append({"role": "user", "content": self.user_message})
        if self.use_prefill:
            messages.append({"role": "assistant", "content": self.prefill})
        return messages


class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm"]
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    response_format: Literal["json", "text","json_object","tool"] = "text"
    tool: Optional[StructuredTool] = None

    def get_tool(self) -> Union[ChatCompletionToolParam, ToolParam, None]:
        if not self.tool:
            return None
        if self.client == "openai":
            return self.tool.get_openai_tool()
        elif self.client == "anthropic":
            return self.tool.get_anthropic_tool()
        else:
            return None
    
    

class AIUtilities:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        # openai credentials
        self.openai_key = os.getenv("OPENAI_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")
        # anthropic credentials
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")

    @staticmethod
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

    @staticmethod
    def msg_dict_to_anthropic(prompt: Prompt) -> List[MessageParam]:
        def convert_message(msg: Dict[str, Any]) -> Union[MessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                content = [PromptCachingBetaTextBlockParam(type="text", text=content)]
            elif isinstance(content, list):
                content = [
                    PromptCachingBetaTextBlockParam(type="text", text=block) if isinstance(block, str)
                    else PromptCachingBetaTextBlockParam(type="text", text=block["text"]) for block in content
                ]
            else:
                raise ValueError("Invalid content type")
            
            return MessageParam(role=role, content=content)
        
        messages = prompt.get_messages_no_system() 
        
        converted_messages = [convert_message(msg) for msg in messages]
        return [msg for msg in converted_messages if msg is not None]

    @staticmethod
    def convert_response_format(response_format: str) -> Optional[ResponseFormat]:
        if response_format == "text":
            return ResponseFormatText(type="text")
        elif response_format == "json_object" or response_format == "json":
            return ResponseFormatJSONObject(type="json_object")
        else:
            return None

    def run_ai_completion(self, prompt: Prompt, llm_config: LLMConfig):
        if llm_config.client == "openai":
            assert self.openai_key is not None, "OpenAI API key is not set"
            client = OpenAI(api_key=self.openai_key)
            return self.run_openai_completion(client, prompt, llm_config)
        
        elif llm_config.client == "anthropic":
            assert self.anthropic_api_key is not None, "Anthropic API key is not set"
            anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.run_anthropic_completion(anthropic, prompt, llm_config)
        
        else:
            return "Invalid AI vendor"

    def run_anthropic_completion(self, anthropic: Anthropic, prompt: Prompt, llm_config: LLMConfig):
        system_content = self.create_anthropic_system_message(prompt.system)
        if llm_config.response_format == "json" or llm_config.response_format == "json_object":
                print("json format detected in anthropic")
                prompt = prompt.model_copy(update={"use_prefill": True})
        anthropic_messages = self.msg_dict_to_anthropic(prompt)
        model = llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
                "system": system_content,
            }
            

            
            response = anthropic.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response)
        except Exception as e:
            return LLMOutput(raw_result=str(e))

    def create_anthropic_system_message(self, system_message: Optional[str]) -> List[PromptCachingBetaTextBlockParam]:
        if system_message:
            return [PromptCachingBetaTextBlockParam(type="text", text=system_message, cache_control=PromptCachingBetaCacheControlEphemeralParam(type="ephemeral"))]
        return []

    def get_ai_context_length(self, ai_vendor: Literal["openai", "azure_openai", "anthropic"]):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")
        else:
            return "Invalid AI vendor"
        
    def run_ai_tool_completion(
        self,
        prompt: Prompt,
        llm_config: LLMConfig = LLMConfig(client="openai"),
    ):
        if llm_config.client == "openai":
            return self.run_openai_tool_completion(prompt, llm_config)
        elif llm_config.client == "anthropic":
            return self.run_anthropic_tool_completion(prompt, llm_config)
        else:
            raise ValueError("Unsupported client for tool completion")

    def run_openai_tool_completion(
        self,
        prompt: Prompt,
        llm_config: LLMConfig,
    ):
        client = OpenAI(api_key=self.openai_key)
        model = llm_config.model or self.openai_model
        
        try:
            assert model is not None, "Model is not set"
            assert llm_config.tool is not None, "Tool is not set"
            
            messages = []
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            if prompt.history:
                messages.extend(prompt.history)
            messages.append({"role": "user", "content": prompt.user_message})

            oai_messages = self.msg_dict_to_oai(messages)

            completion_kwargs = {
                "model": model,
                "messages": oai_messages,
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
            }

            tool = llm_config.get_tool()
            if tool:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = {"type": "function", "function": {"name": llm_config.tool.schema_name}}
            
            if llm_config.response_format != "text":
                completion_kwargs["response_format"] = self.convert_response_format(llm_config.response_format)

            response = client.chat.completions.create(**completion_kwargs)
            completion = response.choices[0].message
            print(completion)
            return LLMOutput(raw_result=completion)
        except Exception as e:
            return LLMOutput(raw_result=str(e))

    def run_anthropic_tool_completion(
        self,
        prompt: Prompt,
        llm_config: LLMConfig
    ):  
        system_content = self.create_anthropic_system_message(prompt.system)
        anthropic_messages = self.msg_dict_to_anthropic(prompt)
        client = Anthropic(api_key=self.anthropic_api_key)
        model = llm_config.model or self.anthropic_model

        try:
            assert model is not None, "Model is not set"
            
            completion_kwargs = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
                "system": system_content,
            }

            tool = llm_config.get_tool()
            if tool and llm_config.tool is not None:
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = ToolChoiceToolChoiceTool(name=llm_config.tool.schema_name, type="tool")

            response = client.beta.prompt_caching.messages.create(**completion_kwargs)
            return LLMOutput(raw_result=response)
        except Exception as e:
            return LLMOutput(raw_result=str(e))

    def create_function_definition(self, name: str, json_schema: Dict[str, Any], description: str) -> FunctionDefinition:
        # Ensure additionalProperties is set to false in the schema
        if "additionalProperties" not in json_schema:
            json_schema["additionalProperties"] = False
        
        return FunctionDefinition(
            name=name,
            description=description,
            parameters=json_schema,
            strict=True
        )

    def run_openai_completion(self, client: OpenAI, prompt: Prompt, llm_config: LLMConfig):
        try:
            messages = []
            if prompt.system:
                messages.append({"role": "system", "content": prompt.system})
            if prompt.history:
                messages.extend(prompt.history)
            messages.append({"role": "user", "content": prompt.user_message})

            oai_messages = self.msg_dict_to_oai(messages)

            completion_kwargs: Dict[str, Any] = {
                "model": llm_config.model or self.openai_model,
                "messages": oai_messages,
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
            }

            if llm_config.response_format != "text":
                # Add JSON instruction to the last user message
                last_message = oai_messages[-1]
                if last_message["role"] == "user" and isinstance(last_message["content"], str):
                    last_message["content"] += "\n\nPlease provide your response in JSON format."
                response_format = self.convert_response_format(llm_config.response_format)
                if response_format:
                    completion_kwargs["response_format"] = response_format

                if llm_config.tool is not None and llm_config.tool.json_schema:
                    # Add the JSON schema to the system message or create a new one
                    schema_instruction = f"Please follow this JSON schema for your response: {llm_config.tool.json_schema}"
                    if oai_messages[0]["role"] == "system" and isinstance(oai_messages[0]["content"], str):
                        oai_messages[0]["content"] += f"\n\n{schema_instruction}"
                    else:
                        oai_messages.insert(0, {"role": "system", "content": schema_instruction})

            response: ChatCompletion = client.chat.completions.create(**completion_kwargs)
            return LLMOutput(raw_result=response.choices[0].message.content)
        except Exception as e:
            return LLMOutput(raw_result=f"Error: {str(e)}")


