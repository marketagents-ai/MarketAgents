import asyncio
import json
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field, ValidationError
from market_agents.inference.sql_models import RawOutput, ProcessedOutput, ChatThread , LLMClient 
from market_agents.inference.clients_models import AnthropicRequest, OpenAIRequest, VLLMRequest
from market_agents.inference.oai_parallel import process_api_requests_from_file, OAIApiFromFileConfig
import os
from dotenv import load_dotenv
import time
from openai.types.chat import ChatCompletionToolParam
from anthropic.types.beta.prompt_caching import PromptCachingBetaToolParam
from anthropic.types.message_create_params import ToolChoiceToolChoiceTool
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
    
    def _create_chat_thread_hashmap(self, chat_threads: List[ChatThread]) -> Dict[int, ChatThread]:
        return {p.id: p for p in chat_threads if p.id is not None}
    
    def _update_chat_thread_history(self, chat_threads: List[ChatThread], llm_outputs: List[ProcessedOutput]) -> List[ChatThread]:
        chat_thread_hashmap = self._create_chat_thread_hashmap(chat_threads)
        for output in llm_outputs:
            if output.chat_thread_id:
                print(f"updating chat thread history for chat_thread_id: {output.chat_thread_id} with output: {output}")
                chat_thread_hashmap[output.chat_thread_id].add_chat_turn_history(output)
        return list(chat_thread_hashmap.values())
    
    
    def _update_chat_thread_db(self,llm_outputs: List[ChatThread], session: Session):
        for output in llm_outputs:
                output.update_db_from_session(session)

    async def run_parallel_ai_completion(self, chat_threads: List[ChatThread], update_history:bool=True, session: Optional[Session] = None) -> List[ProcessedOutput]:

        for chat in chat_threads:
            print(f"chat message inside run_parallel_ai_completion: {chat.new_message}")

        openai_chat_threads = [p for p in chat_threads if p.llm_config.client == "openai"]
        anthropic_chat_threads = [p for p in chat_threads if p.llm_config.client == "anthropic"]
        vllm_chat_threads = [p for p in chat_threads if p.llm_config.client == "vllm"] 
        litellm_chat_threads = [p for p in chat_threads if p.llm_config.client == "litellm"]
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
        flattened_results = [item for sublist in results for item in sublist]
        
        # Track  requests
        self.all_requests.extend(flattened_results)
        
 
        if self.engine:
            print("Updating DB")
            
            # Open a new session if none is provided
            if session is None:
                with Session(self.engine) as session:
                    # Step 1: Add and commit each ProcessedOutput result to avoid re-adding them
                    for result in flattened_results:
                        session.add(result)
                    session.commit()  # Commit ProcessedOutputs first
                    
                    # Step 2: If history updates are enabled
                    if update_history:
                        print(f"Updating chat thread history for {len(chat_threads)} chat threads")
                        
                        # Update the history for each chat thread directly
                        for output in flattened_results:
                            # Retrieve the corresponding ChatThread instance by ID in this session
                            chat_thread = session.get(ChatThread, output.chat_thread_id)
                            
                            if chat_thread:
                                # Initialize history if it doesn't exist
                                chat_thread.add_chat_turn_history(output)
                                
                                
                                # Use session.merge to ensure the object is attached to the session
                                session.merge(chat_thread)
                                session.commit()  # Commit each history update immediately
                                
                                # Print updated length of the history
                                print(f"The length of the history for chat_thread_id: {chat_thread.id} is {len(chat_thread.history)}")
                    
                    # Step 3: Create snapshots for each updated chat thread
                    for chat in chat_threads:
                        # Retrieve the latest state from the database
                        chat = session.get(ChatThread, chat.id)  # Re-get to avoid detached instance
                        if chat:
                            snapshot = chat.create_snapshot()
                            print(f"snapshot: {snapshot}")
                            session.add(snapshot)
                            print("snapshot added to session:")
                    session.commit()  # Final commit to save all snapshots

            else:
                raise ValueError("can not use external session, call the method independently of any other session")




        
        return flattened_results
        
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
            tool = chat_thread.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = {"type": "function", "function": {"name": chat_thread.structured_output.schema_name}}
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
            tool = chat_thread.get_tool()
            if tool:
                request["tools"] = [tool]
                request["tool_choice"] = ToolChoiceToolChoiceTool(name=chat_thread.structured_output.schema_name, type="tool")

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
            tool = chat_thread.get_tool()
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