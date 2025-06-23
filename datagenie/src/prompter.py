from datetime import datetime
from pydantic import BaseModel
from typing import List, Dict
from src.schema import OutputSchema
from src.utils import combine_search_results, combine_retrieved_documents
import yaml
import json
import csv
import os

class PromptSchema(BaseModel):
    Role: str
    Objective: str
    Guidelines: str
    Documents: str
    Examples: str
    Output_instructions: str 
    Output_schema: str
    Assistant: str

class PromptManager:
    def __init__(self, prompt_type, task, resources, output_schema, char_limit):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.char_limit = char_limit
        self.prompt_vars = self.create_vars_dict(task, resources, output_schema)
        self.prompt_path = os.path.join(self.script_dir, '../configs/prompts', f"{prompt_type}_prompt.yaml") 
        self.prompt_schema = self.read_yaml_file(self.prompt_path)
        
    def format_yaml_prompt(self) -> str:
        formatted_prompt = ""
        for field, value in self.prompt_schema.dict().items():
            formatted_value = value.format(**self.prompt_vars)
            formatted_prompt += f"# {field}:\n{formatted_value}"
        return formatted_prompt

    def read_yaml_file(self, file_path: str=None) -> PromptSchema:
        with open(file_path, 'r') as file:
            yaml_content = yaml.safe_load(file)
        
        prompt_schema = PromptSchema(
            Role=yaml_content.get('Role', ''),
            Objective=yaml_content.get('Objective', ''),
            Guidelines=yaml_content.get('Guidelines', ''),
            Documents=yaml_content.get('Documents', ''),
            Examples=yaml_content.get('Examples', ''),
            Output_instructions=yaml_content.get('Output_instructions', ''),
            Output_schema=yaml_content.get('Output_schema', ''),
            Assistant=yaml_content.get('Assistant', '')
        )
        return prompt_schema
    
    def create_vars_dict(self, task, resources, output_schema):
        documents = ""
        examples = ""
        if resources.search_results is not None:
            documents += combine_search_results(resources.search_results, self.char_limit)
        if resources.documents is not None:
            documents += combine_retrieved_documents(resources.documents, type="documents")
        if resources.examples is not None:
            try:
                examples += combine_retrieved_documents(resources.examples, type="examples")
            except Exception as e:
                examples = resources.examples
        input_vars = {
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": task.Category,
            "subcategory": task.SubCategory,
            "task": task.Task, 
            "doc_list": documents,
            "examples": examples,
            "pydantic_schema": output_schema,
        }

        if hasattr(task, "Meta_Data"):
            input_vars.update(task.Meta_Data)
        
        return input_vars

    def generate_system_prompt(self) -> str:
        system_content = f"# Role\n{self.prompt_schema.Role}\n" + \
                         f"# Guidelines\n{self.prompt_schema.Guidelines}\n" + \
                         f"# Output_instructions\n{self.prompt_schema.Output_instructions}\n" + \
                         f"# Output_schema\n{self.prompt_schema.Output_schema.format(**self.prompt_vars)}"
        return system_content

    def generate_task_prompt(self) -> str:
        user_content = f"# Objective\n{self.prompt_schema.Objective.format(**self.prompt_vars)}\n" + \
                       f"# Documents\n{self.prompt_schema.Documents.format(**self.prompt_vars)}\n" + \
                       f"# Examples\n{self.prompt_schema.Examples.format(**self.prompt_vars)}\n" + \
                       f"# Assistant\n{self.prompt_schema.Assistant.format(**self.prompt_vars)}"
        return user_content

    def generate_prompt_messages(self, system_prefix=None):
        if system_prefix:
            system_prompt = system_prefix + self.generate_system_prompt()
        else:
            system_prompt = self.generate_system_prompt()
        user_prompt = self.generate_task_prompt()
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

# Example usage:
if __name__ == "__main__":
    config_path = "./config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    prompt_manager = PromptManager(config)

    generation_type = "function_calling"
    
    examples_json_path = os.path.join(prompt_manager.config["paths"]["examples_path"], generation_type)
    examples = ""
    for file in os.listdir(examples_json_path):
        file_path = os.path.join(examples_json_path, file)
        if os.path.isfile(file_path):
            with open(file_path, "r") as f:
                examples += f"<example = {file}>\n"
                examples += f"{json.load(f)}"
                examples += f"\n</example>\n"
    
    curriculum_csv_path = os.path.join(prompt_manager.config["paths"]["curriculum_csv"], f"{generation_type}.csv")
    with open(curriculum_csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        tasks = [(row['Category'], row['SubCategory'], row['Task']) for row in reader]
        task = tasks[12]
    
    query = f"{task[0]}, {task[1]}, {task[2]}, functions, APIs, documentation"
    variables = {
        "category": task[0],
        "subcategory": task[1],
        "task": task[2],
        "doc_list": "list of documents",
        "examples": examples,
        "pydantic_schema": OutputSchema.schema_json(),
    }

    prompt_messages = prompt_manager.generate_prompt_messages(variables, "./prompt_assets/prompts/function_calling.yaml")
    print(json.dumps(prompt_messages, indent=4))
