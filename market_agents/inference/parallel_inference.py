import asyncio
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from .message_models import LLMPromptContext, LLMOutput
from .clients_models import AnthropicRequest, OpenAIRequest, VLLMRequest
from .oai_parallel import process_api_requests_from_file, OAIApiFromFileConfig
import os
from dotenv import load_dotenv
import time
from openai.types.chat import ChatCompletionToolParam
from anthropic.types.beta.prompt_caching import PromptCachingBetaToolParam
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool

import openai
import anthropic


class RequestLimits(BaseModel):
    max_requests_per_minute: int = Field(default=50,description="The maximum number of requests per minute for the API")
    max_tokens_per_minute: int = Field(default=100000,description="The maximum number of tokens per minute for the API")
    provider: Literal["openai", "anthropic", "vllm", "litellm"] = Field(default="openai",description="The provider of the API")

class ParallelAIUtilities:
    def __init__(self, oai_request_limits: Optional[RequestLimits] = None, 
                 anthropic_request_limits: Optional[RequestLimits] = None, 
                 vllm_request_limits: Optional[RequestLimits] = None,
                 litellm_request_limits: Optional[RequestLimits] = None,
                 local_cache: bool = True,
                 cache_folder: Optional[str] = None):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.vllm_key = os.getenv("VLLM_API_KEY")
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_endpoint = os.getenv("LITELLM_ENDPOINT", "http://localhost:8000/v1/chat/completions")
        self.litellm_key = os.getenv("LITELLM_API_KEY")
        self.oai_request_limits = oai_request_limits if oai_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="openai")
        self.anthropic_request_limits = anthropic_request_limits if anthropic_request_limits else RequestLimits(max_requests_per_minute=50,max_tokens_per_minute=40000,provider="anthropic")
        self.vllm_request_limits = vllm_request_limits if vllm_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="vllm")
        self.litellm_request_limits = litellm_request_limits if litellm_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="litellm")
        self.local_cache = local_cache
        self.cache_folder = self._setup_cache_folder(cache_folder)
        self.all_requests = []

    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        if cache_folder:
            full_path = os.path.abspath(cache_folder)
        else:
            # Go up two levels from the current file's directory to reach the project root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            full_path = os.path.join(repo_root, 'outputs', 'inference_cache')
        
        os.makedirs(full_path, exist_ok=True)
        return full_path

    def _create_prompt_hashmap(self, prompts: List[LLMPromptContext]) -> Dict[str, LLMPromptContext]:
        return {p.id: p for p in prompts}
    
    def _update_prompt_history(self, prompts: List[LLMPromptContext], llm_outputs: List[LLMOutput]):
        prompt_hashmap = self._create_prompt_hashmap(prompts)
        for output in llm_outputs:
            prompt_hashmap[output.source_id].add_chat_turn_history(output)
        return list(prompt_hashmap.values())

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext], update_history:bool=True) -> List[LLMOutput]:
        openai_prompts = [p for p in prompts if p.llm_config.client == "openai"]
        anthropic_prompts = [p for p in prompts if p.llm_config.client == "anthropic"]
        vllm_prompts = [p for p in prompts if p.llm_config.client == "vllm"] 
        litellm_prompts = [p for p in prompts if p.llm_config.client == "litellm"]
        tasks = []
        if openai_prompts:
            tasks.append(self._run_openai_completion(openai_prompts))
        if anthropic_prompts:
            tasks.append(self._run_anthropic_completion(anthropic_prompts))
        if vllm_prompts:
            tasks.append(self._run_vllm_completion(vllm_prompts))
        if litellm_prompts:
            tasks.append(self._run_litellm_completion(litellm_prompts))

        results = await asyncio.gather(*tasks)
        flattened_results = [item for sublist in results for item in sublist]
        
        # Track  requests
        self.all_requests.extend(flattened_results)
        
        if update_history:
            prompts = self._update_prompt_history(prompts, flattened_results)
        
        return flattened_results
    
    def get_all_requests(self):
        requests = self.all_requests
        self.all_requests = []  
        return requests

    async def _run_openai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'openai_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openai_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "openai", requests_file)
        config = self._create_oai_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="openai")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_anthropic_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'anthropic_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'anthropic_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "anthropic", requests_file)
        config = self._create_anthropic_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="anthropic")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_vllm_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'vllm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'vllm_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "vllm", requests_file)
        config = self._create_vllm_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="vllm")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_litellm_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'litellm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'litellm_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "litellm", requests_file)
        config = self._create_litellm_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client="litellm")
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    

    def _prepare_requests_file(self, prompts: List[LLMPromptContext], client: str, filename: str):
        requests = []
        for prompt in prompts:
            request = self._convert_prompt_to_request(prompt, client)
            if request:
                metadata = {
                    "prompt_context_id": prompt.id,
                    "start_time": time.time(),
                    "end_time": None,
                    "total_time": None
                }
                requests.append([metadata, request])
        
        with open(filename, 'w') as f:
            for request in requests:
                json.dump(request, f)
                f.write('\n')

    def _validate_anthropic_request(self, request: Dict[str, Any]) -> bool:
        try:
            anthropic_request = AnthropicRequest(**request)
            return True
        except Exception as e:
            raise ValidationError(f"Error validating Anthropic request: {e} with request: {request}")
    
    def _validate_openai_request(self, request: Dict[str, Any]) -> bool:
        try:
            openai_request = OpenAIRequest(**request)
            return True
        except Exception as e:
            raise ValidationError(f"Error validating OpenAI request: {e} with request: {request}")
        
    def _validate_vllm_request(self, request: Dict[str, Any]) -> bool:
        try:
            vllm_request = VLLMRequest(**request)
            return True
        except Exception as e:
            # Instead of raising ValidationError, we'll return False
            raise ValidationError(f"Error validating VLLM request: {e} with request: {request}")
        

    
    def _get_openai_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        messages = prompt.oai_messages
        request = {
            "model": prompt.llm_config.model,
            "messages": messages,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
        }
        if prompt.oai_response_format:
            request["response_format"] = prompt.oai_response_format
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
        if self._validate_openai_request(request):
            return request
        else:
            return None
    
    def _get_anthropic_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        system_content, messages = prompt.anthropic_messages    
        request = {
            "model": prompt.llm_config.model,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
            "messages": messages,
            "system": system_content if system_content else None
        }
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = ToolChoiceToolChoiceTool(name=prompt.structured_output.schema_name, type="tool")

        if self._validate_anthropic_request(request):
            return request
        else:
            return None
        
    def _get_vllm_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        messages = prompt.vllm_messages  # Use vllm_messages instead of oai_messages
        request = {
            "model": prompt.llm_config.model,
            "messages": messages,
            "max_tokens": prompt.llm_config.max_tokens,
            "temperature": prompt.llm_config.temperature,
        }
        if prompt.llm_config.response_format == "tool" and prompt.structured_output:
            tool = prompt.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": prompt.structured_output.schema_name}}
        if prompt.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        if prompt.oai_response_format and prompt.oai_response_format:
            request["response_format"] = prompt.oai_response_format
        
        if self._validate_vllm_request(request):
            return request
        else:
            return None
        
    def _get_litellm_request(self, prompt: LLMPromptContext) -> Optional[Dict[str, Any]]:
        if prompt.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        return self._get_openai_request(prompt)
        
    def _convert_prompt_to_request(self, prompt: LLMPromptContext, client: str) -> Optional[Dict[str, Any]]:
        if client == "openai":
            return self._get_openai_request(prompt)
        elif client == "anthropic":
            return self._get_anthropic_request(prompt)
        elif client == "vllm":
            return self._get_vllm_request(prompt)
        elif client =="litellm":
            return self._get_litellm_request(prompt)
        else:
            raise ValueError(f"Invalid client: {client}")


    def _create_oai_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "openai" and self.openai_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.openai.com/v1/chat/completions",
                api_key=self.openai_key,
                max_requests_per_minute=self.oai_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.oai_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None

    def _create_anthropic_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "anthropic" and self.anthropic_key:
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url="https://api.anthropic.com/v1/messages",
                api_key=self.anthropic_key,
                max_requests_per_minute=self.anthropic_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.anthropic_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    
    def _create_vllm_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "vllm":
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url=self.vllm_endpoint,
                api_key=self.vllm_key if self.vllm_key else "",
                max_requests_per_minute=self.vllm_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.vllm_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    
    def _create_litellm_completion_config(self, prompt: LLMPromptContext, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if prompt.llm_config.client == "litellm":
            return OAIApiFromFileConfig(
                requests_filepath=requests_file,
                save_filepath=results_file,
                request_url=self.litellm_endpoint,
                api_key=self.litellm_key if self.litellm_key else "",
                max_requests_per_minute=self.litellm_request_limits.max_requests_per_minute,
                max_tokens_per_minute=self.litellm_request_limits.max_tokens_per_minute,
                token_encoding_name="cl100k_base",
                max_attempts=5,
                logging_level=20,
            )
        return None
    

    def _parse_results_file(self, filepath: str,client: Literal["openai", "anthropic", "vllm", "litellm"]) -> List[LLMOutput]:
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    llm_output = self._convert_result_to_llm_output(result,client)
                    results.append(llm_output)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    results.append(LLMOutput(raw_result={"error": "JSON decode error"}, completion_kwargs={}, start_time=time.time(), end_time=time.time(), source_id="error"))
                except Exception as e:
                    print(f"Error processing result: {e}")
                    results.append(LLMOutput(raw_result={"error": str(e)}, completion_kwargs={}, start_time=time.time(), end_time=time.time(), source_id="error"))
        return results

    def _convert_result_to_llm_output(self, result: List[Dict[str, Any]],client: Literal["openai", "anthropic", "vllm", "litellm"]) -> LLMOutput:
        metadata, request_data, response_data = result
        
        return LLMOutput(
            raw_result=response_data,
            completion_kwargs=request_data,
            start_time=metadata["start_time"],
            end_time=metadata["end_time"] or time.time(),
            source_id=metadata["prompt_context_id"],
            client=client
        )

    def _delete_files(self, *files):
        for file in files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")