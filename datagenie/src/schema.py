from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field

class UserMessage(BaseModel):
    role: str = "user"
    content: str = Field(
        "The user's query to assist with a task by calling a function. Include any additional context regarding the task. Provide data such as documents, tables, files, code, documentation, etc. that the function call may require."
    )

class FunctionCall(BaseModel):
    name: str
    arguments: Optional[Union[Dict[str, Any], str]] = Field(
        default=None,
        description="The parameters for the function call. It could be a dictionary of key-value pairs or a string, depending on the function."
    )

class ChainedFunctionCall(BaseModel):
    name: str
    kwargs: Optional[Dict[str, Union[Dict[str, Any], str]]] = Field(
        default=None,
        description="The parameters for the function call. It could be a dictionary of key-value pairs or a string, depending on the function."
    )
    returns: Optional[List[str]] = Field(
        default=None,
        description="The expected return variable names from the function call."
    )

class AssistantMessageToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: str
    
class AssistantMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = Field(
        default=None,
        description="If the task is generating function calls and followed by UserMessage this will be step-by-step plan to for calling functions. If the task is generating function call and followed by ToolMessage this will be a summary of the function call results."
    )
    #function_call: Optional[FunctionCall] = Field(
    #    default=None,
    #    description="The 'function_call' property includes 'name' (the name of the function to be called) and 'arguments' (the parameters for the function call)."
    #)
    tool_calls: Optional[List[AssistantMessageToolCall]] = Field(
        default=None,
        description="The 'tool_calls' property includes a list of function calls providing information about the arguments of functions called by the assistant."
    )

class ArgDescription(BaseModel):
    name: str
    description: str
    type: Optional[str] = None

class ReturnDescription(BaseModel):
    description: str
    type: Optional[str] = None

class RaiseDescription(BaseModel):
    exception: str
    description: str

class Docstring(BaseModel):
    summary: str = Field(..., description="A one-line summary of the function")
    args: List[ArgDescription] = Field(default_factory=list, description="List of argument descriptions")
    returns: Optional[ReturnDescription] = Field(None, description="Description of the return value")
    raises: List[RaiseDescription] = Field(default_factory=list, description="List of exceptions that may be raised")

class GoogleStyleDocstring(BaseModel):
    name: str = Field(..., description="The name of the tool that docstring is for")
    doc_string: str = Field(..., description="The google python style docstring with description, args, returns and raises")

class ToolDocStrings(BaseModel):
    doc_strings: List[GoogleStyleDocstring] = Field(
    description="The doc strings array contains the google python style docstring for each tool"    
)

class ContentJSONSchema(BaseModel):
    name: str = Field(..., description="The name of the tool that generated this content.")
    #doc_string: str = Field(..., description="The google python style docstring with description, args, returns and raises")
    content_schema: Dict[str, Any] = Field(..., description="JSON schema for the tool results content to be generated with relevant keys and value types.")
    

class ToolMessage(BaseModel):
    role: str = Field("tool", description="The role of the message, which is 'tool' in this context.")
    tool_call_id: str = Field(..., description="The unique identifier corresponding to the tool call for which this result belongs.")
    name: str = Field(..., description="The name of the tool that generated this message.")
    content: Dict[str, Any] = Field(
        ...,
        description="The content is a function call results dict corresponding'arguments' for the function call that was executed."
    )

class ContentSchemas(BaseModel):
    content_schemas: List[ContentJSONSchema] = Field(
    description="The content schema array contains the JSON schema for each tool call's content to be generated"
)

class ToolResults(BaseModel):
    content_schemas: List[ContentJSONSchema] = Field(
        description="The content schema array contains the JSON schema for each tool call's content to be generated"
    )
    messages: List[ToolMessage] = Field(
        description="The messages array contains the tool results for function calls made to assist with the user query. The contents of the ToolMessage need to correspond to the content JSON schema for function name"
    )

class ChainedToolMessage(BaseModel):
    role: str = "tool"
    tool_call_id: str
    name: str
    content: Optional[ChainedFunctionCall] = Field(
        default=None,
        description="The content of the tool message, which is a function call with 'name' (the name of the function to be called), 'kwargs' (the parameters for the function call), and 'returns' (the expected return variable names from the function call)."
    )

class ParameterProperty(BaseModel):
    type: str
    description: Optional[str]
    enum: Optional[Dict[str, str]]

class Parameter(BaseModel):
    type: str
    properties: Dict[str, ParameterProperty]
    required: List[str]

class Function(BaseModel):
    name: str
    description: str
    parameters: Parameter

# Model to represent the entire JSON object
class FunctionSignature(BaseModel):
    type: str
    function: Function

class OutputSchema(BaseModel):
    messages: List[Union[UserMessage, AssistantMessage, ToolMessage]] = Field(
        description="The messages array contains the chain of messages between the user, assistant, and function to assist with the user query."
    )
    tools: List[FunctionSignature] = Field(
        description="The tools array contains information about available functions or tools that can be called to answer the user query."
    )

class Tools(BaseModel):
    tools: List[FunctionSignature] = Field(
        description="The tools array contains information about available functions or tools that can be called to answer the user query."
    )

class ToolMessages(BaseModel):
    messages: List[ToolMessage] = Field(
        description="The messages array contains the chain of messages between the user, assistant, and function to assist with the user query."
    )

class ChainedToolMessages(BaseModel):
    messages: List[Union[ChainedToolMessage, AssistantMessage]] = Field(
        description="The messages array contains the chain of messages between the user, assistant, and function to assist with the user query."
    )
    
class JsonRequest(BaseModel):
    role: str
    content: str

class JsonResponse(BaseModel):
    role: str
    content: Dict[str, str]

class JsonModeOutput(BaseModel):
    messages: List[Union[JsonRequest, JsonResponse]]
    pydantic_schema: Dict[str, str]

class StockSchema(BaseModel):
    ticker: str = Field(
        default="<ticker>",
        description="The stock ticker symbol."
    )
    stock_rating: str = Field(
        default="<rating>; <analyst_rationale>",
        description="The stock rating and analyst rationale separated by semicolon (;). Stock ratings from analysts such as Buy, Strong Sell, Hold, Outperform, Overweight etc."
    )
    target_price: str = Field(
        default="<price>",
        description="The target stock price mentioned in documents."
    )
    sentiment: str = Field(
        default="<sentiment>",
        description="The NLP based sentiment towards the stock."
    )
    key_catalysts: List[str] = Field(
        default=[
            "<catalyst_1>; <comment>"
        ],
        description="List of top 4 key catalysts with comments separated by semicolon (;). Do not repeat kpis here. Provide qualitative metrics here"
    )
    key_kpis: List[str] = Field(
        default=[
            "<kpi_1>; <comment>",
        ],
        description="List of top 4 key performance indicators with comments separated by semicolon (;). Do not reapeat catalysts here. Provide quantitative metrics here"
    )
    portfolio_action: str = Field(
        default="<long_short_action>; <reason>",
        description="Portfolio action with reason separated by semicolon (;). Portfolio recommendations such as Add Long, Reduce Long, Close Long, Add Short, Reduce Short, Close Short etc."
    )
    broker_name: List[str] = Field(
        default=[
            "<source_1>",
        ],
        description="List of broker names as data sources."
    )

class FinancialMetrics(BaseModel):
    kpi: List[str] = Field(
        default=[
            "<kpi_1>; <comment>",
        ],
        description="List of top 4 key performance indicators with comments separated by semicolon (;)."
    )
    stock_trends: List[str] = Field(
        default=[
            "<trend_1>; <comment>"
        ],
        description="List of significant stock trends with comments separated by semicolon (;)."
    )
    key_events: List[str] = Field(
        default=[
            "<event_1>; <comment>"
        ],
        description="List of critical financial events with comments separated by semicolon (;)."
    )

class FinancialAnalystOutput(BaseModel):
    ticker: str = Field(
        default="<ticker>",
        description="The stock ticker symbol."
    )
    financial_metrics: FinancialMetrics = Field(
        default=None,
        description="Key financial metrics and trends for the stock."
    )
    sources: List[str] = Field(
        default=[
            "<source_1>"
        ],
        description="List of broker names or data sources."
    )

class MarketSentiment(BaseModel):
    sentiment: str = Field(
        default="<sentiment>",
        description="Overall market sentiment towards the stock."
    )
    key_news: List[str] = Field(
        default=[
            "<news_1>; <comment>"
        ],
        description="List of significant news items with comments separated by semicolon (;)."
    )
    catalysts: List[str] = Field(
        default=[
            "<catalyst_1>; <comment>"
        ],
        description="List of key catalysts for the stock with comments separated by semicolon (;)."
    )

class ResearchAnalystOutput(BaseModel):
    ticker: str = Field(
        default="<ticker>",
        description="The stock ticker symbol."
    )
    market_sentiment: MarketSentiment = Field(
        default=None,
        description="Market sentiment and key news for the stock."
    )
    sources: List[str] = Field(
        default=[
            "<source_1>"
        ],
        description="List of broker names or data sources."
    )

class PortfolioRecommendation(BaseModel):
    actionable_insights: List[str] = Field(
        default=[
            "<insight_1>; <comment>"
        ],
        description="List of actionable insights with comments separated by semicolon (;). These should be specific actions that can be taken based on the analysis."
    )
    strategic_recommendations: List[str] = Field(
        default=[
            "<recommendation_1>; <comment>"
        ],
        description="List of strategic recommendations with comments separated by semicolon (;). These are long-term or strategic actions suggested based on the analysis."
    )
    portfolio_recommendation: str = Field(
        default="<long_short_action>; <reason>",
        description="Portfolio action with reason separated by semicolon (;)."
    )
    target_price: str = Field(
        default="<price>",
        description="The target stock price mentioned in documents."
    )

class InvestmentAdvisorOutput(BaseModel):
    ticker: str = Field(
        default="<ticker>",
        description="The stock ticker symbol."
    )
    portfolio_recommendation: PortfolioRecommendation = Field(
        default=None,
        description="Portfolio recommendations for the stock."
    )
    sources: List[str] = Field(
        default=[
            "<source_1>"
        ],
        description="List of broker names or data sources."
    )

class FinancialSummary(BaseModel):
    tldr_summary: str = Field(
        default="<summary>",
        description="A brief overview of the key insights and recommendations."
    )
    key_observations: List[str] = Field(
        default=[
            "<observation_1>; <comment>"
        ],
        description="List of key observations with comments separated by semicolon (;). These observations highlight important insights and trends."
    )
    conclusion: str = Field(
        default="<conclusion>",
        description="A concise conclusion summarizing the overall findings and recommendations."
    )

# Define the aggregate FinancialSummaryOutput class
class FinancialSummaryOutput(BaseModel):
    ticker: str = Field(
        default="<ticker>",
        description="The stock ticker symbol."
    )
    financial_metrics: FinancialMetrics = Field(
        default=None,
        description="Key financial metrics and trends for the stock."
    )
    market_sentiment: MarketSentiment = Field(
        default=None,
        description="Market sentiment and key news for the stock."
    )
    portfolio_recommendation: PortfolioRecommendation = Field(
        default=None,
        description="Portfolio recommendations for the stock."
    )
    financial_summary: FinancialSummary = Field(
        default=None,
        description="Summary of key insights and conclusions."
    )
    sources: List[str] = Field(
        default=[
            "<source_1>"
        ],
        description="List of broker names or data sources."
    )