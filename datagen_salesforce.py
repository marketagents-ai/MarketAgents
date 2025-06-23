import copy
import os
import json
import argparse
from typing import Any, Dict, List
from datetime import datetime

import concurrent.futures
from tenacity import RetryError, retry, stop_after_attempt, wait_random_exponential

from src.resources import Resource
from src.agents import Agent
from src.utils import generate_query, setup_logger, get_source_dir, create_log_directories, create_folder_path_taskid, is_valid_json
from src.tool_formatter import fix_tools_format_oai

from datasets import load_dataset, ReadInstruction
from src.tasks import Task
from src.aiutilities import AIUtilities
from src.vectordb import VectorDB
from src.prompter import PromptManager
from langchain.schema import Document

from openai.types.chat import ChatCompletionMessage

class DataGenOrchestrator:
    def __init__(
            self,
            generation_type: str,
            agent_config: List[Dict],
            log_file: str = "orchestrator_log.log",
            local_embeddings: bool = False
        ):
        self.generation_type = generation_type
        self.agent_logger = setup_logger(log_file)
        self.agent_config = self.load_or_generate_graph(agents_file=agent_config)
        self.log_file = log_file
        self.llama_logs = []
        self.local_embeddings = local_embeddings
        self.ai_utilities = AIUtilities()
        self.agent_outputs = {}
        self.vector_db = None
        self.messages = None

    def run(
            self,
            task: Task,
            results_path: str,
            examples_folder: str,
            results_folder: str,
        ) -> str:

        self.task = task
        self.results_path = results_path

        # initialize vector db
        if not self.vector_db:
            self.initialize_vector_db(examples_folder, results_folder)

        self.load_resources()

        #resources = Resource()
        #resources.examples = self.task.Meta_Data["assistant_message"]
        prompter = PromptManager(
            prompt_type="function_calling_orchestrator",
            task=self.task,
            resources=self.resources,
            output_schema=None,
            char_limit=None
        )
        system_prompt = prompter.format_yaml_prompt()
        self.agent_logger.info(f"Orchestrator Sys Prompt:\n{system_prompt}")

        llm_config = {
            "client": "azure_openai",
            "model": "gpt4o",
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.task.Meta_Data["user_message"]},
        ]
        self.agent_logger.info(self.messages)
        self.agent_logger.info(self.task.Meta_Data['tools'])

        result = self.run_parallel_datagen(llm_config)
        self.save_llama_logs()

        orchestrator_message = f"Task completed: {self.task.Task}"
        self.agent_logger.info(orchestrator_message)

        return result, self.results_path
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))  
    def execute_docstring_agent(self):
        try:
            function_docstring_agent = self.agent_config[0]
            function_docstring_agent["resources"] = self.resources

            self.agent_logger.info(f"Agent resources:\n{function_docstring_agent['resources']}")
            self.agent_logger.info(f"Executing Docstring Generator Agent")

            docstring_task = copy.deepcopy(self.task)
            docstring_task.Meta_Data["tools"] = None
            docstring_task.Meta_Data["proxy_tools"] = self.task.Meta_Data["tools"]

            docstring_agent = Agent(docstring_task, local_embeddings=self.local_embeddings, generation_type=self.generation_type, **function_docstring_agent)
            docstring_results = docstring_agent.execute()

            if docstring_results is None:
                self.agent_logger.info("Docstring results is empty")

            self.agent_logger.info(docstring_results)
            self.agent_logger.info(f"Here are the tool docstrings:\n{docstring_results['doc_strings']}")

            self.agent_outputs[docstring_agent.name] = docstring_results
            self.llama_logs.extend(docstring_agent.interactions)

            for tool in self.task.Meta_Data["tools"]:
                found_tool = False
                for doc_string in docstring_results["doc_strings"]:
                    if tool['function']['name'] == doc_string['name']:
                        tool['function']['description'] = doc_string['doc_string']
                        found_tool = True
                        break
                if not found_tool:
                    raise ValueError("Error: Invalid docstrings")
    
            return docstring_results
        except RetryError as e:
            self.agent_logger.info(e)
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(5))
    def tool_calling_workflow(self, task_item, llm_config):
        try:
            agent_completion = self.ai_utilities.run_ai_tool_completion(self.messages, tools=task_item.Meta_Data['tools'], llm_config=llm_config)
            self.agent_logger.info(agent_completion)
            self.validate_message(agent_completion.to_dict(), type= "tool")
            # self.messages.append(agent_completion.to_dict())

            return agent_completion
        except RetryError as e:
            self.agent_logger.error(e)
            return None
    
    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))  
    def execute_content_schema_agent(self, content_task, completion_message):
        try:
            content_schema_agent = self.agent_config[1]
            content_schema_agent["resources"] = self.resources

            self.agent_logger.info(f"Agent resources:\n{content_schema_agent['resources']}")
            self.agent_logger.info(f"Executing Schema Generator Agent")

            schema_agent = Agent(content_task, local_embeddings=self.local_embeddings, generation_type=self.generation_type, **content_schema_agent)
            schema_results = schema_agent.execute()

            if schema_results is None:
                self.agent_logger.info("Schema results is empty")

            self.agent_logger.info(schema_results)
            self.agent_logger.info(f"Here are the tool content schemas:\n{schema_results['content_schemas']}")

            self.agent_outputs[schema_agent.name] = schema_results
            self.llama_logs.extend(schema_agent.interactions)

            for tool_call in completion_message["tool_calls"]:
                if tool_call['function']['name'] not in [schema['name'] for schema in schema_results['content_schemas']]:
                    raise ValueError("Error: Invalid conent schemas")
    
            return schema_results
        except RetryError as e:
            self.agent_logger.info(e)
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(5))
    def execute_tool_results_agent(self, results_task, completion_message):
        try:
            function_results_agent = self.agent_config[2]
            function_results_agent["resources"] = self.resources

            self.agent_logger.info(f"Agent resources:\n{function_results_agent['resources']}")
            self.agent_logger.info(f"Executing Tool Results Agent")

            tool_results_agent = Agent(results_task, local_embeddings=self.local_embeddings,generation_type=self.generation_type, **function_results_agent)
            tool_results = tool_results_agent.execute()

            if tool_results is None:
                self.agent_logger.info("Tool Results is empty")

            self.agent_logger.info(tool_results)
            self.agent_logger.info(f"Here are the tool results:\n{tool_results['messages']}")

            self.agent_outputs[tool_results_agent.name] = tool_results
            self.llama_logs.extend(tool_results_agent.interactions)

            if len(completion_message["tool_calls"]) == len(tool_results["messages"]):
                for i, tool_call in enumerate(completion_message["tool_calls"]):
                    if tool_call['function']['name'] == tool_results['messages'][i]['name']:
                        if tool_call['function']['arguments']:
                            tool_results['messages'][i]['tool_call_id'] = tool_call['id']
                    else:
                        raise ValueError("Error: Invalid function calls or results")
            else:
                raise ValueError("Error: lengths of tool_calls and results do not match")
                
            self.validate_tool_results(tool_results)

            for message in tool_results['messages']:
                if "content" in message and message["content"]:
                    message['content'] = str(message['content'])

            self.agent_logger.info(tool_results)
            return tool_results
        except RetryError as e:
            self.agent_logger.info(e)
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))
    def recursive_function(self, completion_message, task_item, iteration=0, max_depth=5):
        try:
            if iteration >= max_depth:
                self.agent_logger.info("Max recursion depth reached without success")
                return False

            if isinstance(completion_message, ChatCompletionMessage):
                self.agent_logger.info(f"Converting ChatCompletionMessage to dict")
                completion_message = completion_message.to_dict()

            if "tool_calls" in completion_message and completion_message["tool_calls"] is not None:
                results_task = copy.deepcopy(task_item)
                results_task.Meta_Data["message_history"] = self.messages
                results_task.Meta_Data["assistant_message"] = completion_message
                results_task.Meta_Data["tools"] = None
                results_task.Meta_Data["proxy_tools"] = task_item.Meta_Data["tools"]

                schema_results = self.execute_content_schema_agent(results_task, completion_message)
                if schema_results is not None:
                    results_task.Meta_Data["content_schemas"] = schema_results["content_schemas"]

                tool_results = self.execute_tool_results_agent(results_task, completion_message)
                if tool_results is not None:
                    for tool_result in tool_results['messages']:
                        self.validate_message(tool_result)
                        self.messages.append(tool_result)
                else:
                    raise ValueError("Error: Invalid message role")

                llm_config = {
                    "client": "azure_openai",
                    "model": "gpt4o",
                    "temperature": 0.2
                }

                completion = self.tool_calling_workflow(task_item, llm_config)
                self.messages.append(completion.to_dict())

                # Recursively call the function and return the result
                return self.recursive_function(completion, task_item, iteration + 1)
            else:
                # If there are no tool calls, consider the task successfully completed
                return True

        except RetryError as e:
            self.agent_logger.error(f"RetryError: {e}")
            return False
        except ValueError as e:
            self.agent_logger.error(f"ValueError: {e}")
            return False
        except Exception as e:
            self.agent_logger.error(f"Unhandled exception: {e}")
            return False


    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))  
    def execute_followup_query_agent(self):
        try:
            query_agent = self.agent_config[3]
            followup_task = copy.deepcopy(self.task)
            followup_task.Meta_Data["message_history"] = self.messages
            followup_task.Meta_Data["tools"] = None
            followup_task.Meta_Data["proxy_tools"] = self.task.Meta_Data["tools"]

            query_agent["resources"] = self.resources
            self.agent_logger.info(f"Agent resources:\n{query_agent['resources']}")

            self.agent_logger.info(f"Executing Follow-up Query Agent")
            follow_up_agent = Agent(followup_task, local_embeddings=self.local_embeddings,generation_type=self.generation_type, **query_agent)
            followup_query = follow_up_agent.execute()

            self.agent_outputs[follow_up_agent.name] = followup_query
            self.llama_logs.extend(follow_up_agent.interactions)
            self.agent_logger.info(followup_query)

            return followup_query
        except RetryError as e:
            self.agent_logger(e)
            return None

    @retry(wait=wait_random_exponential(multiplier=2, max=60), stop=stop_after_attempt(2))  
    def attempt_and_validate_tool_completion(self, task_item, llm_config):
        try:
            tool_completion = self.tool_calling_workflow(task_item, llm_config)
            
            if tool_completion is not None:
                #self.agent_logger.info("Validating tool message dict")
                #self.validate_message(tool_completion.to_dict(), type="tool")
                
                self.agent_logger.info(f"Validating tool calls against answers")
                if not self.validate_tool_calls(tool_completion.to_dict(), task_item.Meta_Data["assistant_message"]):
                    raise ValueError("Tool calls validation failed")
            
            return tool_completion
        except RetryError as e:
            self.agent_logger.info(f"Failed to complete after 5 attempts: {e}")
            return None

    def run_parallel_datagen(self, llm_config):

        # run tool docstring agent
        docstrings = self.execute_docstring_agent()

        if docstrings:
            self.agent_logger.info(f"Here are the updated tool descriptions:\n{self.task.Meta_Data['tools']}")
            # run first tool completion and results agent
            tool_completion = self.attempt_and_validate_tool_completion(self.task, llm_config)

            if tool_completion is not None:

                self.messages.append(tool_completion.to_dict())
                success_1 = self.recursive_function(tool_completion, self.task) 

                # run follow-up query agent
                if success_1:
                    followup_query = self.execute_followup_query_agent()

                    if followup_query is not None:
                        self.validate_message(followup_query)
                        self.messages.append(followup_query)

                        # run follow-up tool completion and results agent
                        tool_completion = self.tool_calling_workflow(self.task, llm_config)

                        if tool_completion is not None:
                            self.messages.append(tool_completion.to_dict())
                            success_2 = self.recursive_function(tool_completion, self.task)

                            self.agent_logger.info(f"Final recursion status: {success_2}")

                # index final messages and tools
                self.agent_logger.info("Indexing and saving tool messages json file")
                result = {
                    "tools": self.task.Meta_Data["tools"],
                    "messages": self.messages
                }

                results_file = os.path.join(self.results_path, "tool_messages.json")
                self.save_and_index_results(results_file, result)

                return result
        return None

    def load_or_generate_graph(
            self,
            config_dir: str = "configs",
            agents_dir: str = "agents",
            agents_file: str = "default.json",
        ):
        # Construct the path to the curriculum CSV file
        data_genie_agents_path = get_source_dir()
        agents_config_dir = os.path.join(config_dir, agents_dir)
        agent_metadata_file = os.path.join(agents_config_dir, agents_file)
        total_path = os.path.join(data_genie_agents_path, agent_metadata_file)
        print(f"Total path: {total_path}")

        if os.path.exists(total_path):
            print("Loading agents metadata from file...")

            with open(total_path, "r") as file:
                agents_metadata = json.load(file)

        return agents_metadata
    
    def validate_tool_calls(self, assistant_message, answers):
        for answer in json.loads(answers):
            self.agent_logger.info(f"Validating with answer: {answer}")
            found = False
            for tool_call in assistant_message["tool_calls"]:
                self.agent_logger.info(f"Validating tool call: {tool_call}")
                if "function" in tool_call and "name" in tool_call["function"] and "arguments" in tool_call["function"]:
                    if tool_call["function"]["name"] == answer["name"]:
                        if self.match_arguments(tool_call["function"]["arguments"], answer["arguments"]):
                            found = True
                            self.agent_logger.info(f"tool call-answer pair is VALID:\ntool_call: {tool_call}\nanswer: {answer}")
                            break
            if not found:
                self.agent_logger.info(f"tool_call-answer pair is NOT VALID:\ntool_call: {tool_call}\nanswer: {answer}")
                return False
        return True
    
    def validate_tool_results(self, tool_results):
        # Step 1: Create a dictionary to group messages by function name
        messages_by_name = {}

        for message in tool_results['messages']:
            # Validate the role first
            if "role" in message and message.get("role") != "tool":
                raise ValueError("Error: Invalid message role")

            # Extract the name and content
            name = message.get("name")
            content = message.get("content")

            # Ensure content is present
            if not content:
                raise ValueError(f"Error: Missing or empty content for message with name '{name}'")

            # Step 2: Group messages by their function name
            if name not in messages_by_name:
                messages_by_name[name] = []
            messages_by_name[name].append(content)

            # Additional validation to check for empty keys or invalid values
            for key, value in content.items():
                if key == "" or value in ["", {}]:
                    raise ValueError(f"Error: Invalid content for message '{name}' - key '{key}' has invalid value")

        # Step 3: Check if the keys of their content dict match
        for name, contents in messages_by_name.items():
            if len(contents) > 1:  # Only compare if there are multiple messages with the same name
                # Extract the keys from the first content dictionary
                reference_keys = set(contents[0].keys())

                # Compare the keys of subsequent content dictionaries with the reference keys
                for content in contents[1:]:
                    content_keys = set(content.keys())
                    if reference_keys != content_keys:
                        raise ValueError(f"Error: Content keys do not match for function '{name}'")

        print("Validation passed.")

    def match_arguments(self, tool_arguments, answer_arguments):
        tool_arguments_dict = json.loads(tool_arguments)
        answer_arguments_dict = answer_arguments
        if len(tool_arguments_dict) != len(answer_arguments_dict):
            return False
        for key, value in answer_arguments_dict.items():
            if key not in tool_arguments_dict or tool_arguments_dict[key] != value:
                return False
        return True
    
    def validate_message(self, message, type = None):
        if not isinstance(message, dict):
            raise ValueError("Message must be a dictionary")
        if not is_valid_json(message):
            raise ValueError("Invalid JSON format")
        if type == "tool":
            if message["content"] is None:
                raise ValueError("Error: Invalid content for message - content is null")

    def save_llama_logs(self):
        log_path = "logs"
        qa_interactions_path = os.path.join(log_path, "qa_interactions")
        qa_interaction_path = os.path.join(qa_interactions_path, "qa_interactions" + datetime.now().strftime("%Y%m%d%H%M%S") + ".json")
        with open(qa_interaction_path, "w") as file:
            json.dump(self.llama_logs, file, indent=2)

    def initialize_vector_db(self, examples_folder, results_folder):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            schema_path = os.path.join(script_dir, 'redis_schema.yaml')
            examples_path = os.path.join(script_dir, f'configs/examples/{examples_folder}')

            if results_folder:
                results_path = os.path.join(script_dir, f'/{results_folder}')
            else:
                results_path = None

            self.vector_db = VectorDB()
            try:
                self.vector_db.load_vector_store(schema_path)
                print("Existing VectorDB loaded successfully.")
            except Exception as load_error:
                print(f"Loading existing VectorDB failed: {load_error}. Initializing...")
                try:
                    self.vector_db.initialize_vector_store(examples_path, schema_path)
                    print("VectorDB initialized successfully.")
                    try:
                        if os.path.exists(results_path) and os.listdir(results_path):
                            documents = []
                            for root, dirs, files in os.walk(results_path):
                                for file in files:
                                    if file.endswith("json"):
                                        file_path = os.path.join(root, file)
                                        document = self.vector_db.load_document_from_file(file_path)
                                        documents.extend(document)
                            self.vector_db.rds.add_documents(documents)
                            print("Previous results added to examples index")
                    except Exception as e:
                        print(f"Loading previous examples failed: {e}")
                except Exception as init_error:
                    print("Initialization failed:", init_error)
                    
    def retrieve_and_combine_examples(self, query, num_examples=2):
        try:
            retrieved_docs = self.vector_db.perform_similarity_search(query, num_examples)
            #combined_examples = combine_retrieved_documents(retrieved_docs, type="examples")
        except Exception as e:
            print(f"Error combining examples: {e}")
            retrieved_docs = []
            #combined_examples = 
        return retrieved_docs

    def load_resources(self):

        self.resources = Resource()
        query = generate_query(self.task, generation_type="function_calling")
        examples = self.retrieve_and_combine_examples(query, num_examples=1)
        self.resources.examples = examples
        self.agent_logger.info(f"Here are the retrieved examples\n{examples}")
    
    def save_and_index_results(self, file_path, messages):
        try: 
            document = Document(
                page_content=f"{json.dumps(messages)}",
                metadata={
                    "source": file_path
                }
            )
            self.vector_db.rds.add_documents([document])
            self.agent_logger.info(f"Successfully indexed newly generated messages")
        except Exception as e:
            self.agent_logger.debug(f"Error indexing newly generated example for {file_path}: {str(e)}")

def load_hf_dataset_tasks(dataset_path: str, start_index: int, num_tasks: int) -> list[Task]:
    #dataset = load_dataset(dataset_path)["train[10:20]"]
    if start_index:
        start = start_index
        num_tasks = start + num_tasks
    else:
        start = 0

    ri = ReadInstruction('train', from_=start, to=num_tasks, unit='abs')
    dataset = load_dataset(dataset_path, split=ri)  

    tasks = []
    for row in dataset:
        tools = []
        for tool in json.loads(row['tools']):
            tools.append(fix_tools_format_oai(tool))
        task = Task(
            Category="Function Calling",
            SubCategory="API Calls",
            Task=row['query'],
            Meta_Data={
                "id": row['id'],
                "user_message": row['query'],
                "assistant_message": row['answers'],
                "tools": tools
            }
        )
        tasks.append(task)
    return tasks

def parse_args():
    parser = argparse.ArgumentParser(description="Run the agent orchestrator with dynamic configurations.")
    #parser.add_argument('-q', '--query', type=str, help="user query for agents to assist with", required=True)
    parser.add_argument('--generation_type', type=str, help="type of data generation", required=True)
    parser.add_argument('--start_index', type=int, help="dataset row to start from", required=False)
    parser.add_argument('--num_tasks', type=int, help="number of tasks to generate", required=True)
    parser.add_argument('--agent_config', type=str, help="agent configuration file", required=True)
    
    return parser.parse_args()

def process_task(task, agent_config, orchestrator_log_path, local_embeddings, generation_type):
    """Process a single task."""
    exists, results_path = create_folder_path_taskid(task, "salesforce_results")
    results_file = os.path.join(results_path, "tool_messages.json")
    if exists:
        print(f"Results path for task {task} already exists")
        if os.path.exists(results_file):
            print(f"Results file for task {task} already exists, skipping task")
            return None

    orchestrator = DataGenOrchestrator(
        generation_type=generation_type,
        agent_config=agent_config,
        log_file=os.path.join(orchestrator_log_path, f"orchestrator_log_{datetime.now().strftime('%Y%m%d%H%M%S')}.log"),
        local_embeddings=local_embeddings
    )

    examples_folder = "multi_turn_tools_examples"
    #results_folder = "salesforce_results"
    return orchestrator.run(task, results_path, examples_folder, results_folder=None)

def process_batch_sequentially(batch, agent_config, orchestrator_log_path, local_embeddings, generation_type):
    """Process a batch of tasks sequentially."""
    for task in batch:
        try:
            result = process_task(task, agent_config, orchestrator_log_path, local_embeddings, generation_type)
            if result:
                messages, results_path = result
                results_file = os.path.join(results_path, "tool_messages.json")
                with open(results_file, "w") as file:
                    json.dump(messages, file, indent=2)
                print(f"Completed result for task {task}: {messages}")
        except Exception as e:
            print(f"Task {task} encountered an error: {e}")

def process_batch(batch, agent_config, orchestrator_log_path, local_embeddings, batch_size, generation_type):
    """Process a batch of tasks concurrently."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_task = {executor.submit(process_task, task, agent_config, orchestrator_log_path, local_embeddings, generation_type): task for task in batch}
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                if result:
                    messages, results_path = result
                    if messages:
                        results_file = os.path.join(results_path, "tool_messages.json")
                        with open(results_file, "w") as file:
                            json.dump(messages, file, indent=2)
                        print(f"Completed result for task {task}: {messages}")
            except Exception as e:
                print(f"Task {task} encountered an error: {e}")

def mainflow(generation_type: str, start_index: int, num_tasks: int, agent_config: str, local_embeddings: bool = False):
    """Main flow of the agent orchestrator for data generation."""

    hf_path = "Salesforce/xlam-function-calling-60k"
    
    # Run the orchestrator for each task
    log_path = "logs"
    orchestrator_log_path = os.path.join(log_path, "orchestrator_logs")
    create_log_directories()

    print(f"Generating dataset for {generation_type}")
    tasks = load_hf_dataset_tasks(dataset_path=hf_path, start_index=start_index, num_tasks=num_tasks)

    batch_size = 32  # Number of tasks to process concurrently
    total_tasks = len(tasks)
    total_tasks_completed = 0

    for i in range(0, total_tasks, batch_size):
        batch = tasks[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {total_tasks // batch_size + 1}")
        process_batch(batch, agent_config, orchestrator_log_path, local_embeddings, batch_size, generation_type)
        #process_batch_sequentially(batch, agent_config, orchestrator_log_path, local_embeddings)
        total_tasks_completed += len(batch)

    print(f"Total tasks completed: {total_tasks_completed}")

if __name__ == "__main__":
    args = parse_args()
    mainflow(args.generation_type, args.start_index, args.num_tasks, args.agent_config)
