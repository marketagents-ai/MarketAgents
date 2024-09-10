from typing import Union, List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import yaml
import json
import os


class SystemPromptSchema(BaseModel):
    """Schema for system prompts."""
    Role: str
    Objectives: str
    Output_schema: str


class TaskPromptSchema(BaseModel):
    """Schema for task prompts."""
    Tasks: Optional[Union[str, List[str]]]
    Assistant: str


class PromptManager:
    """
    Manages the creation and formatting of prompts for AI agents.

    This class handles loading prompt templates, formatting prompts with variables,
    and generating system and task prompts for AI agent interactions.
    """

    def __init__(self, role: str, task: Union[str, List[str]], resources: Optional[Any],
                 output_schema: Optional[Union[str, Dict[str, Any]]], char_limit: int):
        """
        Initialize the PromptManager.

        Args:
            role (str): The role of the AI agent.
            task (Union[str, List[str]]): The task or list of tasks for the agent.
            resources (Optional[Any]): Additional resources for prompt generation.
            output_schema (Optional[Union[str, Dict[str, Any]]]): The schema for the expected output or output format.
            char_limit (int): Character limit for the prompts.
        """
        self.role = role
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.char_limit = char_limit
        self.prompt_vars = self._create_prompt_vars_dict(task, resources, output_schema)
        self.prompt_path = os.path.join(self.script_dir, '..', 'configs', 'prompts', f"{self.role}_prompt.yaml")
        self.default_prompt_path = os.path.join(self.script_dir, '..', 'configs', 'prompts', "default_prompt.yaml")
        self.prompt_schema = self._read_yaml_file(self.prompt_path)

    def format_yaml_prompt(self) -> str:
        """
        Format the YAML prompt with variables.

        Returns:
            str: Formatted YAML prompt.
        """
        formatted_prompt = ""
        for field, value in self.prompt_schema.dict().items():
            formatted_value = value.format(**self.prompt_vars)
            formatted_prompt += f"# {field}:\n{formatted_value}\n"
        return formatted_prompt

    def _read_yaml_file(self, file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Read and parse a YAML file.

        Args:
            file_path (Optional[str]): Path to the YAML file.

        Returns:
            Dict[str, Any]: Parsed YAML content.

        Raises:
            FileNotFoundError: If neither the specified file nor the default file is found.
            ValueError: If there's an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r') as file:
                yaml_content = yaml.safe_load(file)
        except FileNotFoundError:
            try:
                with open(self.default_prompt_path, 'r') as file:
                    yaml_content = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Neither the role-specific prompt file at {file_path} "
                                        f"nor the default prompt file at {self.default_prompt_path} were found.")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        return yaml_content

    def _create_prompt_vars_dict(self, task: Union[str, List[str]], resources: Optional[Any],
                          output_schema: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Create a dictionary of variables for prompt formatting.

        Args:
            task (Union[str, List[str]]): The task or list of tasks.
            resources (Optional[Any]): Additional resources.
            output_schema (Optional[Union[str, Dict[str, Any]]]): The output schema or format.

        Returns:
            Dict[str, Any]: Dictionary of variables for prompt formatting.

        Raises:
            NotImplementedError: If document retrieval is not implemented.
        """
        documents = ""
        if resources is not None:
            # TODO: @interstellarninja implement document/memory retrieval methods
            raise NotImplementedError("Document retrieval is not implemented yet.")
        
        pydantic_schema = output_schema if isinstance(output_schema, dict) else None
        output_format = output_schema if isinstance(output_schema, str) else "json_object"
        
        input_vars = {
            "role": self.role,
            "objectives": None,
            "task": task,
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "doc_list": documents,
            "pydantic_schema": json.dumps(pydantic_schema) if pydantic_schema else None,
            "output_format": output_format
        }

        return input_vars

    def generate_system_prompt(self) -> str:
        """
        Generate the system prompt.

        Returns:
            str: Formatted system prompt.
        """
        system_content = (
            f"Role: {self.prompt_schema['Role'].format(**self.prompt_vars)}\n"
            f"Objectives: {self.prompt_schema['Objectives'].format(**self.prompt_vars)}\n"
        )
        if self.prompt_vars['pydantic_schema']:
            system_content += f"Output_schema: {self.prompt_schema['Output_schema'].format(**self.prompt_vars)}"
        else:
            system_content += f"Output_format: {self.prompt_vars['output_format']}"
        return system_content

    def generate_task_prompt(self) -> str:
        """
        Generate the task prompt.

        Returns:
            str: Formatted task prompt.
        """
        user_content = (
            f"Tasks: {self.prompt_schema['Tasks'].format(**self.prompt_vars)}\n"
            f"Assistant: {self.prompt_schema['Assistant'].format(**self.prompt_vars)}"
        )
        return user_content

    def generate_prompt_messages(self, system_prefix: Optional[str] = None) -> Dict[str, List[Dict[str, str]]]:
        """
        Generate prompt messages for AI interaction.

        Args:
            system_prefix (Optional[str]): Prefix for the system prompt.

        Returns:
            Dict[str, List[Dict[str, str]]]: Dictionary containing system and user messages.
        """
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


if __name__ == "__main__":
    class Task(BaseModel):
        """Model for tasks."""
        tasks: List[str]

    class Resources(BaseModel):
        """Model for resources."""
        search_results: Optional[List[str]] = None
        documents: Optional[List[str]] = None
        examples: Optional[List[str]] = None

    class OutputSchema(BaseModel):
        """Model for output schema."""
        analysis: str
        key_points: List[str]
        recommendation: str

    # Example usage
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

    prompt_manager = PromptManager(
        role="default",
        task=task.tasks,
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

    # Example with plain text output
    prompt_manager_plain = PromptManager(
        role="default",
        task=task.tasks,
        resources=None,
        output_schema="plain_text",
        char_limit=1000
    )

    prompt_messages_plain = prompt_manager_plain.generate_prompt_messages()
    print("\nGenerated Prompt Messages (Plain Text):")
    print(json.dumps(prompt_messages_plain, indent=2))