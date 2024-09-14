from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.econ_models import Basket, Good, Trade, Endowment, Bid, Ask, SellerPreferenceSchedule, BuyerPreferenceSchedule
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, StructuredTool, LLMConfig
from market_agents.environments.mechanisms.auction import AuctionLocalObservation, AuctionGlobalObservation
from typing import Optional, Union, Dict, Any
from pydantic import Field, field_validator, model_validator, computed_field

bid_tool = StructuredTool(
    schema_name="Bid",
    json_schema=Bid.model_json_schema(),
    schema_description="Bid on a good in the market",
    instruction_string="Choose the price to bid on a quantity of 1 of a good in the market. The price must be positive float that must be strictly lower than your current evaluation of the good. You will see your current evalution in the most recent user messae together with the rest of the market state."
)

ask_tool = StructuredTool(
    schema_name="Ask",
    json_schema=Ask.model_json_schema(),
    schema_description="Ask on a good in the market",
    instruction_string="Choose the price to ask for a quantity of 1 of a good in the market. The price must be positive float that must be strictly higher than your current evaluation of the good. You will see your current evalution in the most recent user messae together with the rest of the market state."
)

class BidTool(StructuredTool):
    schema_name: str = Field(default="Bid")
    json_schema: Dict[str, Any] = Field(default=Bid.model_json_schema())
    schema_description: str = Field(default="Bid on a good in the market")
    instruction_string: str = Field(default="Choose the price to bid on a quantity of 1 of a good in the market. The price must be positive float that must be strictly lower than your current evaluation of the good. You will see your current evalution in the most recent user messae together with the rest of the market state.")

class AskTool(StructuredTool):
    schema_name: str = Field(default="Ask")
    json_schema: Dict[str, Any] = Field(default=Ask.model_json_schema())
    schema_description: str = Field(default="Ask on a good in the market")
    instruction_string: str = Field(default="Choose the price to ask for a quantity of 1 of a good in the market. The price must be positive float that must be strictly higher than your current evaluation of the good. You will see your current evalution in the most recent user messae together with the rest of the market state.")

class SimpleAgent(LLMPromptContext, EconomicAgent):
    """ An llm driven agent that can only bid or ask in the market for a single quantity of a single good"""
    system_string: str = Field(default="You are a market agent that can bid and ask for a single good in the market, your objective is to maximize your utility from cash and goods deoending on your evaluation of the goods it might be worth trading in the market for cash or buying more.")
    structured_output: Union[BidTool, AskTool] = Field(default=BidTool(), description="The action to take in the market")
    use_schema_instruction: bool = Field(default=True, description="Whether to use the schema instruction")

    @field_validator("cost_schedules")
    def validate_cost_schedules(cls, v):
        if len(v.keys()) > 1:
            raise ValueError("Simple agent can only have one cost schedule")
        return v
    
    @field_validator("value_schedules")
    def validate_value_schedules(cls, v):
        if len(v.keys()) > 1:
            raise ValueError("Simple agent can only have one value schedule")
        return v
    
    @model_validator(mode='after')
    def either_or(self):
        if len(self.cost_schedules.keys()) == 0 and len(self.value_schedules.keys()) == 0:
            raise ValueError("Simple agent must have either a cost schedule or a value schedule with at least one good")
        if len(self.cost_schedules.keys()) > 0 and len(self.value_schedules.keys()) > 0:
            raise ValueError("Simple agent cannot have both a cost schedule and a value schedule")
        return self
    
    @computed_field
    @property
    def good_name(self) -> str:
        if len(self.cost_schedules.keys()) > 0:
            return list(self.cost_schedules.keys())[0]
        else:
            return list(self.value_schedules.keys())[0]
    


   
    def update_local(self, local_observation: AuctionLocalObservation):
        self.new_message = "\n The market summary of the last round is: " + str(local_observation.observation.market_summary)
        self.new_message += "\n You had the following succesful trades during the last round: " + str(local_observation.observation.trades)
        self.new_message += "\n You currently have the following orders still in the market ledger: " + str(local_observation.observation.waiting_orders)
        


def create_simple_agent(agent_id: str, llm_config: LLMConfig, good: Good, is_buyer: bool, endowment: Endowment, starting_value:float, num_units:int=10):
    if is_buyer:
        value_schedule = BuyerPreferenceSchedule(num_units=num_units, base_value=starting_value)
        cost_schedules = {}
        value_schedules = {good.name: value_schedule}
        new_message = f"You are a buyer of {good.name} and your value schedule is {value_schedule}, this is the first round of the market so the are not bids or asks yet."
        structured_output = BidTool()
    else:
        value_schedules = {}
        cost_schedule = SellerPreferenceSchedule(num_units=num_units, base_value=starting_value)
        cost_schedules = {good.name: cost_schedule}
        new_message = f"You are a seller of {good.name} and your cost schedule is {cost_schedule}, this is the first round of the market so the are not bids or asks yet."
        structured_output = AskTool()
    return SimpleAgent(id=agent_id, llm_config=llm_config,structured_output=structured_output, endowment=endowment, value_schedules=value_schedules, cost_schedules=cost_schedules, new_message=new_message)


