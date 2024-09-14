from market_agents.economics.econ_agent import EconomicAgent, create_economic_agent
from market_agents.economics.econ_models import Basket, Good, Trade, Endowment, Bid, Ask
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, StructuredTool
from market_agents.environments.mechanisms.auction import AuctionLocalObservation, AuctionGlobalObservation
from typing import Optional, Union
from pydantic import Field

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

class SimpleAgent(LLMPromptContext, EconomicAgent):
    """ An llm driven agent that can only bid or ask in the market for a single quantity of a single good"""
    system_string: str = Field(default="You are a market agent that can bid and ask for a single good in the market, your objective is to maximize your utility from cash and goods deoending on your evaluation of the goods it might be worth trading in the market for cash or buying more.")
    structured_output: Union[Bid, Ask] = Field(default=Bid, description="The action to take in the market")
    use_schema_instruction: bool = Field(default=True, description="Whether to use the schema instruction")

   
    def update_local(self, local_observation: AuctionLocalObservation):
        self.new_message = "\n The market summary of the last round is: " + str(local_observation.observation.market_summary)
        self.new_message += "\n You had the following succesful trades during the last round: " + str(local_observation.observation.trades)
        self.new_message += "\n You currently have the following orders still in the market ledger: " + str(local_observation.observation.waiting_orders)
        
