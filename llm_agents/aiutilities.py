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

class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm"]
    model: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0
    response_format: Literal["json", "text","json_object"] = "text"
    json_schema: Optional[Dict[str, Any]] = None

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
    def msg_dict_to_anthropic(messages: List[Dict[str, Any]]) -> List[MessageParam]:
        def convert_message(msg: Dict[str, Any]) -> Union[MessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                content = [TextBlockParam(type="text", text=content)]
            elif isinstance(content, list):
                content = [
                    TextBlockParam(type="text", text=block) if isinstance(block, str)
                    else block for block in content
                ]
            else:
                raise ValueError("Invalid content type")
            

            
            return MessageParam(role=role, content=content)
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

    def run_ai_completion(self, prompt: Union[str, List[Dict[str, Any]]], llm_config: LLMConfig):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        oai_messages = self.msg_dict_to_oai(prompt)
        
        
        if llm_config.client == "openai":
            assert self.openai_key is not None, "OpenAI API key is not set"
            client = OpenAI(api_key=self.openai_key)
            print(oai_messages)
            return self.run_openai_completion(client, oai_messages, llm_config)
        
        
        elif llm_config.client == "anthropic":
            assert self.anthropic_api_key is not None, "Anthropic API key is not set"
            anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.run_anthropic_completion(anthropic, prompt, llm_config)
        
        
        else:
            return "Invalid AI vendor"
    
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
        prompt: List[Dict[str, Any]],
        tools: Optional[Union[List[Dict[str, Any]], List[ToolParam]]] = None,
        llm_config: LLMConfig = LLMConfig(client="openai"),
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven] = NotGiven()
    ):
        if llm_config.client == "openai":
            if tools is not None:
                openai_tools = []
                for tool in tools:
                    assert isinstance(tool, dict) and "function" in tool, "Invalid tool type for OpenAI"
                    function = tool["function"]
                    openai_tools.append(ChatCompletionToolParam(
                        type="function",
                        function=FunctionDefinition(
                            name=function["name"],
                            description=function.get("description", ""),
                            parameters=function["parameters"]
                        )
                    ))
            else:
                openai_tools = None
            return self.run_openai_tool_completion(prompt, openai_tools, llm_config, tool_choice)
        elif llm_config.client == "anthropic":
            if tools is not None:
                assert isinstance(tools, list) and all(isinstance(tool, dict) and "input_schema" in tool for tool in tools), "Invalid tool type for Anthropic"
                anthropic_tools = []
                for tool in tools:
                    anthropic_tools.append(ToolParam(
                        name=str(tool["name"]),
                        description=str(tool.get("description", "")),
                        input_schema=tool["input_schema"]
                    ))
            else:
                anthropic_tools = None
            return self.run_anthropic_tool_completion(prompt, anthropic_tools, llm_config)
        else:
            raise ValueError("Unsupported client for tool completion")

    def run_openai_tool_completion(
        self,
        prompt: List[Dict[str, Any]],
        tools: Optional[List[ChatCompletionToolParam]],
        llm_config: LLMConfig,
        tool_choice: Union[ChatCompletionToolChoiceOptionParam, NotGiven]
    ):
        oai_messages = self.msg_dict_to_oai(prompt)
        client = OpenAI(api_key=self.openai_key)
        model = llm_config.model or self.openai_model
        
        try:
            assert model is not None, "Model is not set"
            
            completion_kwargs = {
                "model": model,
                "messages": oai_messages,
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
            }

            if tools is not None:
                completion_kwargs["tools"] = tools
                if not isinstance(tool_choice, NotGiven):
                    completion_kwargs["tool_choice"] = tool_choice
            elif llm_config.json_schema is not None:
                function_name = "generate_structured_output"
                function_description = "Generate a structured output based on the provided JSON schema."
                function_def = self.create_function_definition(function_name, llm_config.json_schema, function_description)
                
                tool = ChatCompletionToolParam(type="function", function=function_def)
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] = {"type": "function", "function": {"name": function_name}}

            if llm_config.response_format != "text":
                completion_kwargs["response_format"] = self.convert_response_format(llm_config.response_format)

            response = client.chat.completions.create(**completion_kwargs)
            completion = response.choices[0].message
            print(completion)
            return completion
        except Exception as e:
            return str(e)

    def run_anthropic_tool_completion(
        self,
        prompt: List[Dict[str, Any]],
        tools: Optional[List[ToolParam]],
        llm_config: LLMConfig
    ):  
        #check if hte last message is a assistant and remove it
        if prompt[-1]["role"] == "assistant":
            prompt = prompt[:-1]
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
            }

            if tools is not None:
                completion_kwargs["tools"] = tools
            elif llm_config.json_schema is not None:
                function_name = "generate_structured_output"
                function_description = "Generate a structured output based on the provided JSON schema."
                tool = ToolParam(
                    name=function_name,
                    description=function_description,
                    input_schema=llm_config.json_schema,
                )
                completion_kwargs["tools"] = [tool]
                completion_kwargs["tool_choice"] =  ToolChoiceToolChoiceTool(name= function_name, type="tool")

            response = client.messages.create(**completion_kwargs)
            return response
        except Exception as e:
            return str(e)

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

    def run_openai_completion(self, client: OpenAI, prompt: List[ChatCompletionMessageParam], llm_config: LLMConfig):
        try:
            completion_kwargs: Dict[str, Any] = {
                "model": llm_config.model or self.openai_model,
                "messages": prompt.copy(),  # Create a copy to avoid modifying the original prompt
                "max_tokens": llm_config.max_tokens,
                "temperature": llm_config.temperature,
            }

            if llm_config.response_format != "text":
                # Add JSON instruction to the last user message or create a new one
                json_instruction = "Please provide your response in JSON format."
                if prompt[-1]["role"] == "user":
                    completion_kwargs["messages"][-1]["content"] += f"\n\n{json_instruction}"
                else:
                    completion_kwargs["messages"].append({"role": "user", "content": json_instruction})

                response_format = self.convert_response_format(llm_config.response_format)
                if response_format:
                    completion_kwargs["response_format"] = response_format

                if llm_config.json_schema:
                    # Add the JSON schema to the system message or create a new one
                    schema_instruction = f"Please follow this JSON schema for your response: {llm_config.json_schema}"
                    if completion_kwargs["messages"][0]["role"] == "system":
                        completion_kwargs["messages"][0]["content"] += f"\n\n{schema_instruction}"
                    else:
                        completion_kwargs["messages"].insert(0, {"role": "system", "content": schema_instruction})

            response: ChatCompletion = client.chat.completions.create(**completion_kwargs)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"


    def run_anthropic_completion(
        self,
        anthropic: Anthropic,
        prompt: List[Dict[str, Any]],
        llm_config: LLMConfig
    ):
        # try:
        
        
        system_message = next((msg for msg in prompt if msg["role"] == "system"), None)
        system_content= []
        if system_message:
            content = system_message["content"]
            if isinstance(content, str):
                system_content = content
            elif isinstance(content, list):
                system_content = [
                    TextBlockParam(type="text", text=block) if isinstance(block, str)
                    else TextBlockParam(type="text", text=block["text"]) for block in content
                ]
        
        model_name = llm_config.model or self.anthropic_model
        assert model_name in ["claude-3-sonnet-20240620", "claude-3-haiku-20240307", "claude-3-opus-20240229", "claude-3-5-sonnet-20240620"], "Invalid model name"
        if llm_config.response_format == "json":
            print("json format detected in anthropic")
            prompt.append(
                {"role": "assistant", "content": "Here's the valid JSON object response:```json"}
            )
        anthropic_messages = self.msg_dict_to_anthropic(prompt)
        print(anthropic_messages)
        response = anthropic.messages.create(
            model=model_name,
            max_tokens=llm_config.max_tokens,
            temperature=llm_config.temperature,
            system=system_content if isinstance(system_content, (str, NotGiven)) else system_content,
            messages=anthropic_messages
            
        )
        assert isinstance(response.content[0], TextBlock)
        
        return response.content[0].text
            


def main():
    load_dotenv()  # Load environment variables from .env file

    ai_utilities = AIUtilities()

    # Example prompt
    prompt = [
        {"role": "system", "content": "You are a helpful pirate that tells jokes  pirate language."},
        {"role": "user", "content": "Tell me a programmer joke in italiano"}
    ]
    
    # JSON schema for structured responses
    json_schema = {
        "type": "object",
        "properties": {
            "joke": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["joke", "explanation"],
        "additionalProperties": False  # Add this line
    }

    print("Schema: ", json_schema)
    # OpenAI examples
    print("OpenAI Examples:")
    
    # 1. OpenAI with text response
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="text")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n1. OpenAI Completion Result (Text):")
    print(f"{result}\n")

    # 2. OpenAI with JSON response (no schema)
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="json_object")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n2. OpenAI Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 3. OpenAI with JSON response (with schema)
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="json", json_schema=json_schema)
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n3. OpenAI Completion Result (JSON, with schema):")
    print(f"{result}\n")

    # 7. OpenAI with JSON schema as a tool
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", json_schema=json_schema)
    result = ai_utilities.run_ai_tool_completion(prompt, llm_config=llm_config)
    print("\n7. OpenAI Completion Result (JSON schema as tool):")
    print(f"{result}\n")

    # Anthropic examples
    print("\nAnthropic Examples:")

    # 4. Anthropic with text response
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="text")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n4. Anthropic Completion Result (Text):")
    print(f"{result}\n")

    # 5. Anthropic with JSON response (no schema)
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_object")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n5. Anthropic Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 6. Anthropic with JSON response (with schema)
    # Note: Anthropic doesn't support JSON schema directly, so we'll use the same approach as without schema
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n6. Anthropic Completion Result (JSON, with schema - same as without schema for Anthropic):")
    print(f"{result}\n")

    #8. Anthropic with JSON schema as a tool
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", json_schema=json_schema)
    result = ai_utilities.run_ai_tool_completion(prompt, llm_config=llm_config)
    print("\n8. Anthropic Completion Result (JSON schema as tool):")
    print(f"{result}\n")



if __name__ == "__main__":
    main()