from typing import Optional, List, Union
from pydantic import BaseModel, Field
from openai import OpenAI
import json

class VLLMConfig(BaseModel):
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema."
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description="If specified, the output will follow the regex pattern."
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description="If specified, the output will be exactly one of the choices."
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description="If specified, the output will follow the context free grammar."
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"
        )
    )

class ProgrammerJoke(BaseModel):
    setup: str
    punchline: str

config = VLLMConfig(guided_json=ProgrammerJoke.model_json_schema())

client = OpenAI(
    api_key="token-abc123",
    base_url="http://localhost:8000/v1"
)

completion = client.chat.completions.create(
  model="microsoft/Phi-3.5-mini-instruct",
  messages=[
    {"role": "user", "content": "tell me a programmer joke"}
  ],
  extra_body=config.model_dump(exclude_none=True)
)

print(json.dumps(completion.model_dump(), indent=4))