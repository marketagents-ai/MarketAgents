import os
from openai.types.chat import ChatCompletionMessageParam

from anthropic.types import MessageParam
from anthropic.types import CacheControlEphemeralParam
from anthropic.types import TextBlockParam
from anthropic.types import MessageParam
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageParam,
)


from typing import Union, Optional, List, Tuple, Literal, Dict, Any
import json
import re
import tiktoken
import ast

def parse_json_string(content: str) -> Optional[Dict[str, Any]]:
    # Remove any leading/trailing whitespace and newlines
    cleaned_content = content.strip()
    
    # Remove markdown code block syntax if present
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned_content, flags=re.MULTILINE)
    
    try:
        # First, try to parse as JSON
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        try:
            # If JSON parsing fails, try to evaluate as a Python literal
            return ast.literal_eval(cleaned_content)
        except (SyntaxError, ValueError):
            # If both methods fail, try to find and parse a JSON-like structure
            json_match = re.search(r'(\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\})', cleaned_content, re.DOTALL)
            if json_match:
                try:
                    # Normalize newlines, replace single quotes with double quotes, and unescape quotes
                    json_str = json_match.group(1).replace('\n', '').replace("'", '"').replace('\\"', '"')
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # If all parsing attempts fail, return None
    return None

def get_ai_context_length(ai_vendor: Literal["openai", "azure_openai", "anthropic"]):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")


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
                    print(f"validating tool_calls during conversion: {[ChatCompletionMessageToolCallParam(**tool_call) for tool_call in msg["tool_calls"]]}")
                    assistant_msg["tool_calls"] = [ChatCompletionMessageToolCallParam(**tool_call) for tool_call in msg["tool_calls"]]
                return assistant_msg
            elif role == "tool":
                return ChatCompletionToolMessageParam(role=role, content=msg["content"], tool_call_id=msg["tool_call_id"])
            elif role == "function":
                return ChatCompletionFunctionMessageParam(role=role, content=msg["content"], name=msg["name"])
            else:
                raise ValueError(f"Unknown role: {role}")

        return [convert_message(msg) for msg in messages]

def msg_dict_to_anthropic(messages: List[Dict[str, Any]],use_cache:bool=True,use_prefill:bool=False) -> Tuple[List[TextBlockParam],List[MessageParam]]:
        def create_anthropic_system_message(system_message: Optional[Dict[str, Any]],use_cache:bool=True) -> List[TextBlockParam]:
            if system_message and system_message["role"] == "system":
                text = system_message["content"]
                if use_cache:
                    return [TextBlockParam(type="text", text=text, cache_control=CacheControlEphemeralParam(type="ephemeral"))]
                else:
                    return [TextBlockParam(type="text", text=text)]
            return []

        def convert_message(msg: Dict[str, Any],use_cache:bool=False) -> Union[MessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                if not use_cache:
                    content = [TextBlockParam(type="text", text=content)]
                else:
                    content = [TextBlockParam(type="text", text=content,cache_control=CacheControlEphemeralParam(type='ephemeral'))]
            elif isinstance(content, list):
                if not use_cache:
                    content = [
                        TextBlockParam(type="text", text=block) if isinstance(block, str)
                        else TextBlockParam(type="text", text=block["text"]) for block in content
                    ]
                else:
                    content = [
                        TextBlockParam(type="text", text=block, cache_control=CacheControlEphemeralParam(type='ephemeral')) if isinstance(block, str)
                        else TextBlockParam(type="text", text=block["text"], cache_control=CacheControlEphemeralParam(type='ephemeral')) for block in content
                    ]
            else:
                raise ValueError("Invalid content type")
            
            return MessageParam(role=role, content=content)
        converted_messages = []
        system_message = []
        num_messages = len(messages)
        if use_cache:
            use_cache_ids = set([num_messages - 1, max(0, num_messages - 3)])
        else:
            use_cache_ids = set()
        for i,message in enumerate(messages):
            if message["role"] == "system":
                system_message= create_anthropic_system_message(message,use_cache=use_cache)
            else:
                
                use_cache_final = use_cache if  i in use_cache_ids else False
                converted_messages.append(convert_message(message,use_cache= use_cache_final))

        
        return system_message, [msg for msg in converted_messages if msg is not None]
