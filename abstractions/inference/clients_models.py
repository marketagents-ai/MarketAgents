from pydantic import BaseModel, Field
from typing import  Optional, Union, Dict, List, Any

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
    ChatCompletionToolChoiceOptionParam
)
from openai.types.shared_params import (

    FunctionDefinition
)
from openai.types.chat.completion_create_params import (
    ResponseFormat,
    FunctionCall
)
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema


from anthropic.types.beta.prompt_caching import (
    PromptCachingBetaToolParam,
    PromptCachingBetaMessageParam,
    PromptCachingBetaTextBlockParam,
    message_create_params
)
from anthropic.types.model_param import ModelParam


class OpenAIRequest(BaseModel):
    messages: List[ChatCompletionMessageParam]
    model: str
    frequency_penalty: Optional[float] = Field(default=None)
    function_call: Optional[FunctionCall] = Field(default=None)
    functions: Optional[List[FunctionDefinition]] = Field(default=None)
    logit_bias: Optional[Dict[str, int]] = Field(default=None)
    max_tokens: Optional[int] = Field(default=None)
    n: Optional[int] = Field(default=None)
    presence_penalty: Optional[float] = Field(default=None)
    response_format: Optional[ResponseFormat] = Field(default=None)
    seed: Optional[int] = Field(default=None)
    stop: Optional[Union[str, List[str]]] = Field(default=None)
    stream: Optional[bool] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = Field(default=None)
    tools: Optional[List[ChatCompletionToolParam]] = Field(default=None)
    top_p: Optional[float] = Field(default=None)
    user: Optional[str] = Field(default=None)

    class Config:
        extra = 'forbid'


class AnthropicRequest(BaseModel):
    max_tokens: int
    messages: List[PromptCachingBetaMessageParam]
    model: ModelParam
    metadata: message_create_params.Metadata | None = Field(default=None)
    stop_sequences: List[str] | None = Field(default=None)
    stream: bool | None = Field(default=None)
    system: Union[str, List[PromptCachingBetaTextBlockParam]] | None = Field(default=None)
    temperature: float | None = Field(default=None)
    tool_choice: message_create_params.ToolChoice | None = Field(default=None)
    tools: List[PromptCachingBetaToolParam] | None = Field(default=None)
    top_k: int | None = Field(default=None)
    top_p: float | None = Field(default=None)

class VLLMConfig(BaseModel):
    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=
        ("If true, the generation prompt will be added to the chat template. "
         "This is a parameter used by chat template in tokenizer config of the "
         "model."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description=
        ("A list of dicts representing documents that will be accessible to "
         "the model if it is performing RAG (retrieval-augmented generation)."
         " If the template does not support RAG, this argument will have no "
         "effect. We recommend that each document should be a dict containing "
         "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    chat_template_kwargs: Optional[Dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the template renderer. "
                     "Will be accessible by the chat template."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"))
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."))
    
class VLLMRequest(OpenAIRequest,VLLMConfig):
    pass