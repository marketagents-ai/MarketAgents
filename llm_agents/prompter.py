from typing import Union, List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import yaml
import json
import os

class SystemPromptSchema(BaseModel):
    Role: str
    Objectives: str
    Output_schema: str

class TaskPromptSchema(BaseModel):
    Tasks: Optional[Union[str, List[str]]]
    Assistant: str

class PromptManager:
    def __init__(self, role, task, resources, output_schema, char_limit):
        self.role = role
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.char_limit = char_limit
        self.prompt_vars = self.create_vars_dict(task, resources, output_schema)
        self.prompt_path = os.path.join(self.script_dir,'configs', 'prompts', f"{self.role}_prompt.yaml")
        self.prompt_schema = self.read_yaml_file(self.prompt_path)
        
    def format_yaml_prompt(self) -> str:
        formatted_prompt = ""
        for field, value in self.prompt_schema.dict().items():
            formatted_value = value.format(**self.prompt_vars)
            formatted_prompt += f"# {field}:\n{formatted_value}\n"
        return formatted_prompt

    def read_yaml_file(self, file_path: str=None) -> Dict[str, Any]:
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"The YAML file at {file_path} was not found. Please check the file path and ensure it exists.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        
        return yaml_content
    
    def create_vars_dict(self, task, resources, output_schema):
        documents = ""
        if resources is not None:
            # TODO: @interstellarninja implement document/memory retrieval methods
            raise NotImplementedError("Document retrieval is not implemented yet.")
        input_vars = {
            "role": self.role,
            "objectives": None,
            "task": task,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_list": documents,
            "pydantic_schema": output_schema,
        }
        
        return input_vars

    def generate_system_prompt(self) -> str:
        system_content = f"Role: {self.prompt_schema['Role'].format(**self.prompt_vars)}\n" + \
                         f"Objectives: {self.prompt_schema['Objectives'].format(**self.prompt_vars)}\n" + \
                         f"Output_schema: {self.prompt_schema['Output_schema'].format(**self.prompt_vars)}"
        return system_content
    
    def generate_task_prompt(self) -> str:
        user_content = f"Tasks: {self.prompt_schema['Tasks'].format(**self.prompt_vars)}\n" + \
                       f"Assistant: {self.prompt_schema['Assistant'].format(**self.prompt_vars)}"
        return user_content

    def generate_prompt_messages(self, system_prefix=None):
        system_prompt = self.generate_system_prompt()
        if system_prefix:
            system_prompt = system_prefix + system_prompt
        user_prompt = self.generate_task_prompt()
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        }

# Example usage:
if __name__ == "__main__":

    class Task(BaseModel):
        tasks: List[str]

    class Resources(BaseModel):
        search_results: List[str] = None
        documents: List[str] = None
        examples: List[str] = None

    task = Task(
        tasks=[
            "Analyze Q2 earnings reports",
            "Forecast market trends for next quarter",
            "Evaluate potential investment opportunities"
        ]
    )
    resources = Resources(
        documents=["Q2 earnings report data"],
        examples=["Previous analysis example"]
    )
    
    # Define a sample output schema
    class OutputSchema(BaseModel):
        analysis: str
        key_points: List[str]
        recommendation: str

    prompt_manager = PromptManager(
        role="default",
        task= task.tasks,
        resources=None,
        output_schema=OutputSchema.schema_json(),
        char_limit=1000
    )

    prompt_messages = prompt_manager.generate_prompt_messages()
    print("Generated Prompt Messages:")
    print(json.dumps(prompt_messages, indent=2))

    # Print system and user prompts separately
    system_prompt = prompt_messages["messages"][0]["content"]
    user_prompt = prompt_messages["messages"][1]["content"]

    print("\nSystem Prompt:")
    print(system_prompt)

    print("\nUser Prompt:")
    print(user_prompt)