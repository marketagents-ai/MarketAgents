import ast
import importlib
import inspect
import json
import os
from typing import Any, Dict, List, Optional, Union
import uuid
from pydantic import BaseModel, Field
from datetime import datetime

from tenacity import retry, stop_after_attempt, wait_random_exponential
import agent_tools
from src.aiutilities import AIUtilities
from src.prompter import PromptManager
from src.utils import agent_logger, create_task_file_path, extract_json_from_response, generate_query, create_folder_path
from src.resources import Resource, ResourceManager
from src.tasks import Task

class Agent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    system: str = None
    task: Task = None
    tools: Union[List[dict], List[str]] = None
    default_tools: List[str] = []
    resources: Resource = None
    output_format: Union[Dict[str, Any], str] = None
    dependencies: Optional[List[str]] = None
    llm_config: Dict = Field(default_factory=dict)
    prompt_type: str
    generation_type: str = None
    few_shot: bool = False
    max_iter: int = 2
    input_messages: List[Dict] = []
    interactions: List[Dict] = []
    verbose: bool = True
    local_embeddings: bool = False

    def __init__(self, task: Task, local_embeddings: bool, generation_type: str, **data: Any):
        super().__init__(**data)
        self.task = task
        self.local_embeddings = local_embeddings
        self.generation_type = generation_type
        if self.output_format and isinstance(self.output_format, str):
            try:
                self.output_format = self.load_output_schema()
            except ImportError:
                pass

        # Check if self.task has a field called Meta_Data
        if hasattr(self.task, "Meta_Data"):
            if self.task.Meta_Data.get("tools"):
                self.tools = self.task.Meta_Data.get("tools")

    def load_output_schema(self):
        schema_module = importlib.import_module("src.schema")
        schema_class = getattr(schema_module, self.output_format)
        json_schema = schema_class.schema_json()
        return json_schema

    def load_agent_resources(self):
        resource_manager = ResourceManager(local_embeddings=self.local_embeddings)

        if self.resources is None:
            self.resources = Resource()

        if self.few_shot:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            example_path = os.path.join(script_dir, '../configs/examples', f"{self.prompt_type}_example.json")
            with open(example_path, 'r') as file:
                examples = json.load(file)

            self.resources.examples = examples

        query = generate_query(self.task, self.prompt_type)

        if "web_search" in self.default_tools:
            results_path = create_folder_path(self.task, folder_type="search_results")
            search_results = ResourceManager(local_embeddings=self.local_embeddings).retrieve_websearch_results(query, 5, results_path)

            self.resources.search_results = search_results

        if "rag_search" in self.default_tools:
            resource_manager.initialize_vector_db()
            documents = resource_manager.retrieve_vectordb_documents(query, num_docs=5)

            self.resources.documents = documents

        return self.resources

    def load_agent_prompts(self):
        ctx_len = AIUtilities().get_ai_context_length(self.llm_config.get("client"))
        #char_limit = (int(ctx_len) - 10000) * 2
        char_limit = 12000
        prompt_manager = PromptManager(self.prompt_type, self.task, self.resources, self.output_format, char_limit) 
        system_prompt = prompt_manager.generate_system_prompt()
        task_prompt = prompt_manager.generate_task_prompt()
        return system_prompt, task_prompt

    def execute(self) -> str:
        self.load_agent_resources()
        messages = []
        system_suffix = ""
        if self.input_messages:
            for input_message in self.input_messages:
                role = input_message["role"]
                content = input_message["content"]
                agent_logger.info(f"Appending input messages from previous agent: {role}")
                system_suffix = f"<agent_messages>\n<{role}>\n{content}\n</{role}>\n</agent_messages>"
                # messages.append({"role": "system", "content": f"<agent_messages>\n<{role}>\n{content}\n</{role}>\n</agent_messages>"})

        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_prefix = f"You are a {self.name} AI agent. Current date and time: {current_datetime}"

        system_prompt, task_prompt = self.load_agent_prompts()
        system_prompt = system_prefix + system_prompt + system_suffix

        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt}
        ]

        messages.extend(prompt_messages)
        agent_logger.info(f"Logging prompt text\n{messages}")

        @retry(wait=wait_random_exponential(multiplier=1, max=30), stop=stop_after_attempt(self.max_iter))
        def run_ai_inference():
            try:
                agent_logger.info(f"Running inference with {self.llm_config.get('client')}")
                ai_utilities = AIUtilities()

                if self.tools:
                    if isinstance(self.tools, list):
                        try:
                            tool_module = importlib.import_module(f"agent_tools.{self.generation_type}")
                            if tool_module:
                                self.tools = tool_module.get_openai_tools(self.tools)
                        except Exception as e:
                            print(f"tool import error: {e}")

                    completion = ai_utilities.run_ai_tool_completion(messages, self.tools, self.llm_config)
                    completion = completion.to_dict()

                    agent_logger.info(f"Assistant Message:\n{completion}")
                    messages.append(completion)

                    # Extract and save results for each task
                    if "tool_calls" in completion and completion["tool_calls"]:
                        try:
                            for tool_call in completion["tool_calls"]:
                                tool_name = tool_call.function.name
                                agent_logger.info(f"Invoking tool: {tool_name}")
                                tool_result = self.execute_function_call(tool_call, tool_module)

                                agent_logger.info(f"Here's the function result:\n{tool_result}")

                                messages.append({
                                    "role":"tool", 
                                    "tool_call_id": tool_call.id, 
                                    "name": tool_name, 
                                    "content":tool_result
                                })
                            completion = ai_utilities.run_ai_completion(messages, self.llm_config)
                            agent_logger.info(f"Assistant Message after exec:\n{completion}")
                        except Exception as e:
                            agent_logger(e)

                        file_path = create_task_file_path(self.task, self.name, folder_type="results", file_type="txt")
                        with open(file_path, 'w') as file:
                            file.write(completion)

                    elif isinstance(json.loads(self.output_format), dict):
                        file_path = create_task_file_path(self.task, self.name, folder_type="results", file_type="json")
                        completion = self.extract_and_save_results(file_path, completion["content"])
                    else:
                        file_path = create_task_file_path(self.task, self.name, folder_type="results", file_type="txt")
                        with open(file_path, 'w') as file:
                            file.write(completion)

                else:   
                    completion = ai_utilities.run_ai_completion(messages, self.llm_config)
                    agent_logger.info(f"Assistant Message:\n{completion}")
                    messages.append({"role": "assistant", "content": completion})

                    # Extract and save results for each task
                    if isinstance(json.loads(self.output_format), dict):
                        file_path = create_task_file_path(self.task, self.name, folder_type="results", file_type="json")
                        agent_logger.info("Extracting JSON object from response")
                        completion = self.extract_and_save_results(file_path, completion)
                        agent_logger.info(f"Extracted JSON Object from Assistant Message:\n{completion}")
                    else:
                        agent_logger.info(f"Assistant Message:\n{completion}")
                        file_path = create_task_file_path(self.task, self.name, folder_type="results", file_type="txt")
                        with open(file_path, 'w') as file:
                            file.write(completion)
            except Exception as e:
                agent_logger.error(e)
                raise e

            return completion

        result = run_ai_inference()
        # Log the final interaction
        self.log_interaction(messages, result)
        return result

    def execute_function_call(self, tool_call, tool_module):
        function_name = tool_call.function.name
        function_to_call = getattr(tool_module, function_name, None)
        function_args = tool_call.function.arguments

        if function_to_call:
            agent_logger.info(f"Invoking function call {function_name} ...")
            function_response = function_to_call(**function_args)
            results_dict = f'{{"name": "{function_name}", "content": {json.dumps(function_response)}}}'
            return results_dict
        else:
            raise ValueError(f"Function '{function_name}' not found.")

    def extract_and_save_results(self, file_path, completion):
        try:
            json_object = None

            # Attempt to parse the completion string as JSON
            try:
                json_object = json.loads(completion)
                agent_logger.info("Parsed with json.loads")
            except json.JSONDecodeError:
                # If JSON decoding fails, attempt to use literal_eval
                try:
                    json_object = ast.literal_eval(completion)
                    agent_logger.info("Parsed with ast.literal_eval")
                except (ValueError, SyntaxError):
                    # If literal_eval also fails, extract JSON manually
                    json_object = extract_json_from_response(completion)
                    agent_logger.info("Extracted JSON manually")

            # Check if json_object is still None after all attempts
            if not json_object:
                raise ValueError("Completion contains an invalid JSON object")

            # Save the JSON object to a file
            with open(file_path, 'w') as json_file:
                json.dump(json_object, json_file, indent=2)

            agent_logger.debug(f"Successfully saved results for {self.task.Task}")
            return json_object

        except Exception as e:
            agent_logger.debug(f"Error extracting and saving results for {self.task.Task}: {str(e)}")

    def log_interaction(self, prompt, response):
        self.interactions.append({
            "name": self.name,
            "messages": prompt,
            "response": response,
            "agent_messages": self.input_messages,
            "tools": self.tools,
            "timestamp": datetime.now().isoformat()
        })
