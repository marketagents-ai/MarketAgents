from sqlmodel import Field, SQLModel, create_engine, Column, JSON, Session, Relationship, select
from typing import Dict, Any, List, Optional, Literal, Self
from pydantic import computed_field, ValidationError, model_validator
from sqlalchemy import Engine
from enum import Enum

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletion
)
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema, JSONSchema

from openai.types.shared_params import (
    ResponseFormatText,
    ResponseFormatJSONObject,
    FunctionDefinition
)
from anthropic.types import (
    MessageParam,
    TextBlock,
    ToolUseBlock,
    ToolParam,
    Message as AnthropicMessage
)
from anthropic.types.beta.prompt_caching import (
    PromptCachingBetaMessage,
    PromptCachingBetaToolParam,
    PromptCachingBetaMessageParam,
    PromptCachingBetaTextBlockParam,
    message_create_params
)
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam
from anthropic.types.model_param import ModelParam


class Tool(SQLModel, table=True):
    id: Optional[int]  = Field(default=None, primary_key=True)
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True
    json_schema: Dict = Field(default = {},sa_column=Column(JSON))
    prompts: List["Prompt"] = Relationship(back_populates="structured_output")

    @computed_field
    @property
    def schema_instruction(self) -> str:
        return f"{self.instruction_string}: {self.json_schema}"

    def get_openai_tool(self) -> Optional[ChatCompletionToolParam]:
        if self.json_schema:
            return ChatCompletionToolParam(
                type="function",
                function=FunctionDefinition(
                    name=self.schema_name,
                    description=self.schema_description,
                    parameters=self.json_schema
                )
            )
        return None

    def get_anthropic_tool(self) -> Optional[PromptCachingBetaToolParam]:
        if self.json_schema:
            return PromptCachingBetaToolParam(
                name=self.schema_name,
                description=self.schema_description,
                input_schema=self.json_schema,
                cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')
            )
        return None
    def get_openai_json_schema_response(self) -> Optional[ResponseFormatJSONSchema]:

        if self.json_schema:
            schema = JSONSchema(name=self.schema_name,description=self.schema_description,schema=self.json_schema,strict=self.strict_schema)
            return ResponseFormatJSONSchema(type="json_schema", json_schema=schema)
        return None
    
class LLMClient(str, Enum):
    openai = "openai"
    azure_openai = "azure_openai"
    anthropic = "anthropic"
    vllm = "vllm"
    litellm = "litellm"

class ResponseFormat(str, Enum):
    json_beg = "json_beg"
    text = "text"
    json_object = "json_object"
    structured_output = "structured_output"
    tool = "tool"

class LLMConfig(SQLModel, table=True):
    id: Optional[int]  = Field(default=None, primary_key=True)
    client: LLMClient
    model: Optional[str] = None
    max_tokens: int = Field(default=400)
    temperature: float = 0
    response_format: ResponseFormat = Field(default=ResponseFormat.text)
    use_cache: bool = True
    prompts: List["Prompt"] = Relationship(back_populates="llm_config")
    
    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        if self.response_format == ResponseFormat.json_object and self.client in [LLMClient.vllm, LLMClient.litellm,LLMClient.anthropic]:
            raise ValueError(f"{self.client} does not support json_object response format")
        elif self.response_format == ResponseFormat.structured_output and self.client == LLMClient.anthropic:
            raise ValueError(f"Anthropic does not support structured_output response format use json_beg or tool instead")
        return self
    
class Prompt(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    system_string: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = Field(default=None, sa_column=Column(JSON))
    new_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    structured_output: Optional[Tool] = Relationship(back_populates="prompts")
    use_schema_instruction: bool = Field(default=False, description="Whether to use the schema instruction")
    use_history: bool = Field(default=True, description="Whether to use the history")
    llm_config: LLMConfig = Relationship(back_populates="prompts")
    structured_output_id: Optional[int] = Field(default=None, foreign_key="tool.id")
    llm_config_id: Optional[int] = Field(default=None, foreign_key="llmconfig.id")



def create_prompt(engine: Engine):
    with Session(engine) as session:
        oai_config = LLMConfig(client=LLMClient.openai, model="gpt-4o", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
        edit_tool = Tool(schema_name="edit_tool",
                         schema_description="Edit the provided JSON schema.",
                         instruction_string="Please follow this JSON schema for your response:",
                         strict_schema=True,
                         json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}})
        prompt = Prompt(new_message="Hello, how are you?", structured_output=edit_tool, llm_config=oai_config)      
        prompt_italian = Prompt(new_message="Ciao, come stai?", structured_output=edit_tool, llm_config=oai_config)
        anthropic_config = LLMConfig(client=LLMClient.anthropic, model="claude-3-5-sonnet-20240620", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
        prompt_french_anthropic = Prompt(new_message="Bonjour, comment ça va?", structured_output=Tool(schema_name="outil_edition",
                             schema_description="Éditer le schéma JSON fourni.",
                             instruction_string="Veuillez suivre ce schéma JSON pour votre réponse:",
                             strict_schema=True,
                             json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}}),
                             llm_config=anthropic_config)
        session.add(prompt)
        session.add(prompt_italian)
        session.add(prompt_french_anthropic)
        session.commit()
        

def select_openai_config(engine: Engine):
    with Session(engine) as session:
        statement = select(Prompt, LLMConfig).join(LLMConfig).where(LLMConfig.client == LLMClient.openai)
        result = session.exec(statement)
        for prompt, config in result:
            print(prompt.new_message)

if __name__ == "__main__":
    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)

    SQLModel.metadata.create_all(engine)
    create_prompt(engine)
    select_openai_config(engine)

