#market_agents\inference\sql_inference.py
import asyncio
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from abstractions.inference.sql_models import RawOutput, ProcessedOutput, ChatThread , LLMClient , ResponseFormat, ChatMessage, MessageRole
from abstractions.inference.clients_models import AnthropicRequest, OpenAIRequest, VLLMRequest
from abstractions.inference.oai_parallel import process_api_requests_from_file, OAIApiFromFileConfig
import os
from dotenv import load_dotenv
import time
from openai.types.chat import ChatCompletionToolParam
from anthropic.types.beta.prompt_caching import PromptCachingBetaToolParam
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool,ToolChoiceToolChoiceAuto
from sqlalchemy import Engine
from sqlmodel import Session

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
                 cache_folder: Optional[str] = None,
                engine: Optional[Engine] = None):
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
        self.engine = engine

    def _setup_cache_folder(self, cache_folder: Optional[str]) -> str:
        if cache_folder:
            full_path = os.path.abspath(cache_folder)
        else:
            # Go up two levels from the current file's directory to reach the project root
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            full_path = os.path.join(repo_root, 'outputs', 'inference_cache')
        
        os.makedirs(full_path, exist_ok=True)
        return full_path
    



    def _check_engine(self) -> None:
        """
        Verifies that the database engine is properly configured.
        
        Raises:
            ValueError: If the engine is not set
        """
        if not self.engine:
            raise ValueError("Engine is not set, cannot update DB")   
                
    def add_user_requests_to_db(self, chat_threads: List[ChatThread]) -> List[ChatThread]:
        """
        Updates chat threads in the database with user messages and returns the filtered/updated list
        
        Args:
            chat_threads: List of ChatThread objects to update
            
        Returns:
            List[ChatThread]: Updated list of chat threads, with failed ones removed unless they use auto_tools
        """
        self._check_engine()
        with Session(self.engine) as session:
            updated_threads = []
            for chat in chat_threads:
                try:
                    chat.add_user_message()
                    session.add(chat)
                    updated_threads.append(chat)
                except Exception as e:
                    if chat.llm_config.response_format not in [ResponseFormat.auto_tools, ResponseFormat.tool, ResponseFormat.text]:
                        print(f"Error adding user message to chat thread {chat.id}: {e}, removed from thread list")
                    else:
                        session.add(chat)
                        updated_threads.append(chat)
            
            session.commit()
            
            # Refresh all remaining chat threads
            for chat in updated_threads:
                session.refresh(chat)
                
        return updated_threads
    
    def add_assistant_responses_to_db(self, flattened_results: List[ProcessedOutput]) -> List[ChatThread]:
        """
        Updates chat threads with LLM outputs and returns the updated chat threads.
        
        Args:
            flattened_results: List[ProcessedOutput] to add to chat threads
            
        Returns:
            List[ChatThread]: Updated chat threads
        """
        self._check_engine()
        
        with Session(self.engine) as session:
            # Step 1: Add and commit each ProcessedOutput result
            for output in flattened_results:
                session.add(output)
            session.commit()
            
            
            updated_threads = []
            # Update the history for each chat thread
            for output in flattened_results:
                chat_thread = session.get(ChatThread, output.chat_thread_id)
                
                if chat_thread:
                    last_message_uuid = chat_thread.get_last_message_uuid()
                    if last_message_uuid:
                        chat_thread.add_assistant_response(output, last_message_uuid)
                        print(f"finished adding assistant response to chat thread {chat_thread.id}")
                    else:
                        raise ValueError(f"Chat thread {chat_thread.id} has no last message uuid")
                    
                    session.refresh(chat_thread)
                    updated_threads.append(chat_thread)
                    
            session.commit()
            
            # Final refresh of all threads
            for thread in updated_threads:
                session.refresh(thread)
                
            return updated_threads
    
    async def run_parallel_ai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        updated_chat_threads = self.add_user_requests_to_db(chat_threads)
        input_threads_ids = [thread.id for thread in updated_chat_threads]

        openai_chat_threads = [p for p in updated_chat_threads if p.llm_config.client == "openai"]
        anthropic_chat_threads = [p for p in updated_chat_threads if p.llm_config.client == "anthropic"]
        vllm_chat_threads = [p for p in updated_chat_threads if p.llm_config.client == "vllm"] 
        litellm_chat_threads = [p for p in updated_chat_threads if p.llm_config.client == "litellm"]
        tasks  = []
        if openai_chat_threads:
            tasks.append(self._run_openai_completion(openai_chat_threads))
        if anthropic_chat_threads:
            tasks.append(self._run_anthropic_completion(anthropic_chat_threads))
        if vllm_chat_threads:
            tasks.append(self._run_vllm_completion(vllm_chat_threads))
        if litellm_chat_threads:
            tasks.append(self._run_litellm_completion(litellm_chat_threads))

        results = await asyncio.gather(*tasks)
        flattened_results : List[ProcessedOutput] = [item for sublist in results for item in sublist]
        
        # Track  requests
        self.all_requests.extend(flattened_results)
        

        print("Updating DB with LLM outputs")
        updated_threads =self.add_assistant_responses_to_db(flattened_results)
        tool_tasks = [thread.execute_tool() for thread in updated_threads if thread.can_execute_tool()]
        print(f"executing {len(tool_tasks)} tools")
        tool_results = await asyncio.gather(*tool_tasks)
        flattened_tool_results = [tool_result for tool_result in tool_results if tool_result is not None]
        print(f"executed {len(flattened_tool_results)} tools")
        with Session(self.engine) as session:
            for tool_result in flattened_tool_results:
                session.add(tool_result)
                
            session.commit()  
            fresh_results = []
            for output in flattened_results:
                fresh_result = session.get(ProcessedOutput, output.id)
                if fresh_result:
                    fresh_results.append(fresh_result)

        return fresh_results
        
    def get_all_requests(self):
        requests = self.all_requests
        self.all_requests = []  
        return requests

    async def _run_openai_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'openai_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'openai_results_{timestamp}.jsonl')
        self._prepare_requests_file(chat_threads, "openai", requests_file)
        config = self._create_oai_completion_config(chat_threads[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client=LLMClient.openai)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    async def _run_anthropic_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'anthropic_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'anthropic_results_{timestamp}.jsonl')
        self._prepare_requests_file(chat_threads, "anthropic", requests_file)
        config = self._create_anthropic_completion_config(chat_threads[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client=LLMClient.anthropic)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_vllm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'vllm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'vllm_results_{timestamp}.jsonl')
        self._prepare_requests_file(chat_threads, "vllm", requests_file)
        config = self._create_vllm_completion_config(chat_threads[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client=LLMClient.vllm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []
    
    async def _run_litellm_completion(self, chat_threads: List[ChatThread]) -> List[ProcessedOutput]:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        requests_file = os.path.join(self.cache_folder, f'litellm_requests_{timestamp}.jsonl')
        results_file = os.path.join(self.cache_folder, f'litellm_results_{timestamp}.jsonl')
        self._prepare_requests_file(chat_threads, "litellm", requests_file)
        config = self._create_litellm_completion_config(chat_threads[0], requests_file, results_file)
        if config:
            try:
                await process_api_requests_from_file(config)
                return self._parse_results_file(results_file,client=LLMClient.litellm)
            finally:
                if not self.local_cache:
                    self._delete_files(requests_file, results_file)
        return []

    

    def _prepare_requests_file(self, chat_threads: List[ChatThread], client: str, filename: str):
        requests = []
        for chat_thread in chat_threads:
            request = self._convert_chat_thread_to_request(chat_thread, client)
            if request:
                metadata = {
                    "chat_thread_id": chat_thread.id,
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
            print(f"Error validating OpenAI request: {e} with request: {request}")
            raise ValidationError(f"Error validating OpenAI request: {e} with request: {request}")
        
    def _validate_vllm_request(self, request: Dict[str, Any]) -> bool:
        try:
            vllm_request = VLLMRequest(**request)
            return True
        except Exception as e:
            # Instead of raising ValidationError, we'll return False
            raise ValidationError(f"Error validating VLLM request: {e} with request: {request}")
        

    
    def _get_openai_request(self, chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
        messages = chat_thread.oai_messages
        request = {
            "model": chat_thread.llm_config.model,
            "messages": messages,
            "max_tokens": chat_thread.llm_config.max_tokens,
            "temperature": chat_thread.llm_config.temperature,
        }
        if chat_thread.oai_response_format:
            request["response_format"] = chat_thread.oai_response_format
        if chat_thread.llm_config.response_format == "tool" and chat_thread.structured_output:
            tool = chat_thread.get_structured_output_as_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": chat_thread.structured_output.schema_name}}
        elif chat_thread.llm_config.response_format == "auto_tools":
            tools = chat_thread.get_tools()
            if tools:
                request["tools"] = tools
                request["tool_choice"] = "auto"
        if self._validate_openai_request(request):
            return request
        else:
            return None
    
    def _get_anthropic_request(self, chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
        system_content, messages = chat_thread.anthropic_messages    
        request = {
            "model": chat_thread.llm_config.model,
            "max_tokens": chat_thread.llm_config.max_tokens,
            "temperature": chat_thread.llm_config.temperature,
            "messages": messages,
            "system": system_content if system_content else None
        }
        if chat_thread.llm_config.response_format == "tool" and chat_thread.structured_output:
            tool = chat_thread.get_structured_output_as_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = ToolChoiceToolChoiceTool(name=chat_thread.structured_output.schema_name, type="tool")
        elif chat_thread.llm_config.response_format == "auto_tools":
            tools = chat_thread.get_tools()
            if tools:
                request["tools"] = tools
                request["tool_choice"] = ToolChoiceToolChoiceAuto(type="auto")

        if self._validate_anthropic_request(request):
            return request
        else:
            return None
        
    def _get_vllm_request(self, chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
        messages = chat_thread.vllm_messages  # Use vllm_messages instead of oai_messages
        request = {
            "model": chat_thread.llm_config.model,
            "messages": messages,
            "max_tokens": chat_thread.llm_config.max_tokens,
            "temperature": chat_thread.llm_config.temperature,
        }
        if chat_thread.llm_config.response_format == "tool" and chat_thread.structured_output:
            tool = chat_thread.get_structured_output_as_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": chat_thread.structured_output.schema_name}}
        if chat_thread.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        if chat_thread.oai_response_format and chat_thread.oai_response_format:
            request["response_format"] = chat_thread.oai_response_format
        
        if self._validate_vllm_request(request):
            return request
        else:
            return None
        
    def _get_litellm_request(self, chat_thread: ChatThread) -> Optional[Dict[str, Any]]:
        if chat_thread.llm_config.response_format == "json_object":
            raise ValueError("VLLM does not support json_object response format otherwise infinite whitespaces are returned")
        return self._get_openai_request(chat_thread)
        
    def _convert_chat_thread_to_request(self, chat_thread: ChatThread, client: str) -> Optional[Dict[str, Any]]:
        if client == "openai":
            return self._get_openai_request(chat_thread)
        elif client == "anthropic":
            return self._get_anthropic_request(chat_thread)
        elif client == "vllm":
            return self._get_vllm_request(chat_thread)
        elif client =="litellm":
            return self._get_litellm_request(chat_thread)
        else:
            raise ValueError(f"Invalid client: {client}")


    def _create_oai_completion_config(self, chat_thread: ChatThread, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if chat_thread.llm_config.client == "openai" and self.openai_key:
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

    def _create_anthropic_completion_config(self, chat_thread: ChatThread, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if chat_thread.llm_config.client == "anthropic" and self.anthropic_key:
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
    
    def _create_vllm_completion_config(self, chat_thread: ChatThread, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if chat_thread.llm_config.client == "vllm":
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
    
    def _create_litellm_completion_config(self, chat_thread: ChatThread, requests_file: str, results_file: str) -> Optional[OAIApiFromFileConfig]:
        if chat_thread.llm_config.client == "litellm":
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
    

    def _parse_results_file(self, filepath: str,client: LLMClient) -> List[ProcessedOutput]:
        results = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    llm_output = self._convert_result_to_llm_output(result,client)
                    results.append(llm_output)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON: {line}")
                except Exception as e:
                    print(f"Error processing result: {e}")
        return results

    def _convert_result_to_llm_output(self, result: List[Dict[str, Any]],client: LLMClient) -> ProcessedOutput:
        metadata, request_data, response_data = result
        

        raw_output = RawOutput(
            raw_result=response_data,
            completion_kwargs=request_data,
            start_time=metadata["start_time"],
            end_time=metadata["end_time"] or time.time(),
            chat_thread_id=metadata["chat_thread_id"],
            client=client
        )

        return raw_output.create_processed_output()

    def _delete_files(self, *files):
        for file in files:
            try:
                os.remove(file)
            except OSError as e:
                print(f"Error deleting file {file}: {e}")