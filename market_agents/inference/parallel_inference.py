import asyncio
import json
from typing import List, Dict, Any, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from .message_models import LLMPromptContext, LLMOutput, LLMConfig
from .oai_parallel import process_api_requests_from_file, OAIApiFromFileConfig
import os
from dotenv import load_dotenv
import time
from openai.types.chat import ChatCompletionToolParam
from anthropic.types.beta.prompt_caching import PromptCachingBetaToolParam
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool

class RequestLimits(BaseModel):
    max_requests_per_minute: int = Field(default=50,description="The maximum number of requests per minute for the API")
    max_tokens_per_minute: int = Field(default=100000,description="The maximum number of tokens per minute for the API")
    provider: Literal["openai", "anthropic"] = Field(default="openai",description="The provider of the API")

class ParallelAIUtilities:
    def __init__(self, oai_request_limits: Optional[RequestLimits] = RequestLimits(), 
                 anthropic_request_limits: RequestLimits = RequestLimits(provider="anthropic"), 
                 local_cache: bool = True,
                 cache_folder: Optional[str] = None):
        load_dotenv()
        self.openai_key = os.getenv("OPENAI_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        self.oai_request_limits = oai_request_limits if oai_request_limits else RequestLimits(max_requests_per_minute=500,max_tokens_per_minute=200000,provider="openai")
        self.anthropic_request_limits = anthropic_request_limits if anthropic_request_limits else RequestLimits(max_requests_per_minute=50,max_tokens_per_minute=40000,provider="anthropic")
        self.local_cache = local_cache
        self.cache_folder = self._setup_cache_folder(cache_folder)

    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        if cache_folder:
            full_path = os.path.abspath(cache_folder)
        else:
            # Corrected path: Go up two levels instead of three
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            full_path = os.path.join(repo_root, 'outputs', 'inference_cache')
        
        os.makedirs(full_path, exist_ok=True)
        return full_path

    async def run_parallel_ai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        openai_prompts = [p for p in prompts if p.llm_config.client == "openai"]
        anthropic_prompts = [p for p in prompts if p.llm_config.client == "anthropic"]

        tasks = []
        if openai_prompts:
            tasks.append(self._run_openai_completion(openai_prompts))
        if anthropic_prompts:
            tasks.append(self._run_anthropic_completion(anthropic_prompts))

        results = await asyncio.gather(*tasks)
        
        # Flatten the results list
        return [item for sublist in results for item in sublist]

    async def _run_openai_completion(self, prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'openai_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openai_results_{timestamp}.jsonl')
        self._prepare_requests_file(prompts, "openai", requests_file)
        config = self._create_oai_completion_config(prompts[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file, prompts)
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
                return self._parse_results_file(results_file, prompts)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    def _prepare_requests_file(self, prompts: List[LLMPromptContext], client: str, filename: str):
        requests = []
        for prompt in prompts:
            request = self._convert_prompt_to_request(prompt, client)
            if request:
                requests.append(request)
        
        with open(filename, 'w') as f:
            for request in requests:
                json.dump(request, f)
                f.write('\n')

    def _convert_prompt_to_request(self, prompt: LLMPromptContext, client: str) -> Optional[Dict[str, Any]]:
        if client == "openai":
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
            return request
        elif client == "anthropic":
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
            return request
        return None

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

    def _parse_results_file(self, filepath: str, original_prompts: List[LLMPromptContext]) -> List[LLMOutput]:
        results = []
        with open(filepath, 'r') as f:
            for line, original_prompt in zip(f, original_prompts):
                try:
                    result = json.loads(line)
                    llm_output = self._convert_result_to_llm_output(result, original_prompt)
                    results.append(llm_output)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                    results.append(LLMOutput(raw_result={"error": "JSON decode error"}, completion_kwargs={}))
                except Exception as e:
                    print(f"Error processing result: {e}")
                    results.append(LLMOutput(raw_result={"error": str(e)}, completion_kwargs={}))
        return results

    def _convert_result_to_llm_output(self, result: List[Dict[str, Any]], original_prompt: LLMPromptContext) -> LLMOutput:
        request_data, response_data = result

        
        if original_prompt.llm_config.client == "openai":
            return LLMOutput(raw_result=response_data, completion_kwargs=request_data)
        elif original_prompt.llm_config.client == "anthropic":
            # Convert Anthropic response format to match LLMOutput expectations
            return LLMOutput(raw_result=response_data, completion_kwargs=request_data)
        else:
            return LLMOutput(raw_result={"error": "Unexpected client type"}, completion_kwargs=request_data)

    def _delete_files(self, *files):
        for file in files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")

