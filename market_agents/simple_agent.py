from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.econ_models import Basket, Good, Trade, Endowment, Bid, Ask, SellerPreferenceSchedule, BuyerPreferenceSchedule
from market_agents.inference.parallel_inference import ParallelAIUtilities
from market_agents.inference.message_models import LLMPromptContext, StructuredTool, LLMConfig, LLMOutput
from market_agents.environments.mechanisms.auction import AuctionLocalObservation, AuctionGlobalObservation
from typing import Optional, Union, Dict, Any, List
from pydantic import Field, field_validator, model_validator, computed_field, BaseModel
import json
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

class SimpleAgentState(BaseModel):
    market_summary: Dict[str, Any]
    trades: List[Trade]
    waiting_orders: List[Union[Bid, Ask]]
    current_basket: Basket
    good_name: str
    evaluation: Optional[float] = None
    is_seller: bool = False

    def __str__(self):
        profit_message = ""
        if self.evaluation is not None:
            if self.is_seller:
                profit_message = f"You can make a profit by selling at {self.evaluation * 1.01} or higher"
            else:
                profit_message = f"You can make a profit by buying at {self.evaluation * 0.99} or lower"

        return (
            f"\n The market summary of the last round is: {self.market_summary}"
            f"\n You had the following successful trades during the last round: {self.trades}"
            f"\n You currently have the following orders still in the market ledger: {self.waiting_orders}"
            f"\n You currently have the following basket, already including the last rounds trades if any: {self.current_basket}"
            f"{f'\n Your evaluation of the good is: {self.evaluation}' if self.evaluation is not None else ''}"
            f"{f'\n {profit_message}' if profit_message else ''}"
        )

    @classmethod
    def from_agent_and_observation(cls, agent: 'SimpleAgent', local_observation: AuctionLocalObservation) -> 'SimpleAgentState':
        good_name = agent.good_name
        is_seller = agent.is_seller(good_name)
        evaluation = agent.get_current_cost(good_name) if is_seller else agent.get_current_value(good_name)

        return cls(
            market_summary=local_observation.observation.market_summary,
            trades=local_observation.observation.trades,
            waiting_orders=local_observation.observation.waiting_orders,
            current_basket=agent.endowment.current_basket,
            good_name=good_name,
            evaluation=evaluation,
            is_seller=is_seller
        )
    
class AuctionInput(BaseModel):
    observation: AuctionLocalObservation
    state: SimpleAgentState
    

class SimpleAgent(LLMPromptContext, EconomicAgent):
    """ An llm driven agent that can only bid or ask in the market for a single quantity of a single good
    It implements the History POMDP with environment observation: AuctionLocalObservation and internal state: SimpleAgentState and actions Union[Bid, Ask]
    The DoubleAuction environment implements the map Untion[Bid,Ask]--> AuctionLocalObservation
    The agent has to process the trades in the endowment and update its state accordingly this is done by combining the local observation with the internal state
    with the map Endowment,AuctionLocalObservation--> SimpleAgentState 
    Finally the agent has to choose an action: Bid or Ask with the map SimpleAgentState--> Union[Bid,Ask]
    The conversation history of this agent is a lit of interleaved messages representing the string counter part of SimpleAgentState and Union[Bid,Ask] """
    system_string: str = Field(
        default="You are a market agent that can bid and ask for a single good in the market, your objective is to maximize your utility from cash and goods deoending on your evaluation of the goods it might be worth trading in the market for cash or buying more.")
    structured_output: Union[BidTool, AskTool] = Field(default=BidTool(), description="The action to take in the market")
    use_schema_instruction: bool = Field(default=True, description="Whether to use the schema instruction")
    input_history: List[AuctionInput] = Field(default=[])
    actions_history: List[Union[Bid, Ask]] = Field(default=[])

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
    
    def update_state(self, local_observation: AuctionLocalObservation,update_message:bool=True):
        """ Update the internal state of the agent with the local observation """
        if local_observation.observation.trades:
            for trade in local_observation.observation.trades:
                if trade not in self.endowment.trades:
                    self.process_trade(trade)
                else:
                    print(f"Trade {trade} already in endowment")
        simple_agent_state = SimpleAgentState.from_agent_and_observation(self, local_observation)
        self.input_history.append(AuctionInput(observation=local_observation, state=simple_agent_state))
        if update_message:
            self.new_message = str(simple_agent_state)

    def add_chat_turn_history(self, llm_output:LLMOutput):
        """ add a chat turn to the history without safely model copy just normal append """
        if llm_output.source_id != self.id:
            raise ValueError(f"LLMOutput source_id {llm_output.source_id} does not match the prompt context id {self.id}")
        #validate the action as a bid or ask
        if llm_output.json_object and llm_output.json_object.name == "Bid":
            try:
                action = Bid(**llm_output.json_object.object)
                self.actions_history.append(action)
            except Exception as e:
                raise ValueError(f"LLMOutput json_object {llm_output.json_object} is not a valid Bid even if name matches {llm_output.json_object.name}")
        elif llm_output.json_object and llm_output.json_object.name == "Ask":
            try:
                action = Ask(**llm_output.json_object.object)
                self.actions_history.append(action)
            except Exception as e:
                raise ValueError(f"LLMOutput json_object {llm_output.json_object} is not a valid Ask even if name matches {llm_output.json_object.name}")
        elif llm_output.json_object:
            raise ValueError(f"LLMOutput json_object name {llm_output.json_object.name} is not a valid action")
        
        if self.history is None:
            self.history = []
        self.history.append({"role": "user", "content": self.new_message})
        self.history.append({"role": "assistant", "content": llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"})

def create_simple_agent(agent_id: str, llm_config: LLMConfig, good: Good, is_buyer: bool, endowment: Endowment, starting_value:float, num_units:int=10):
    if is_buyer:
        value_schedule = BuyerPreferenceSchedule(num_units=num_units, base_value=starting_value)
        cost_schedules = {}
        value_schedules = {good.name: value_schedule}
        new_message = f"You are a buyer of {good.name} and your current value is {value_schedule.get_value(1)}, this is the first round of the market so the are not bids or asks yet. You can make a profit by buying at " + str(value_schedule.get_value(1)*0.99) + " or lower"
        structured_output = BidTool()
    else:
        value_schedules = {}
        cost_schedule = SellerPreferenceSchedule(num_units=num_units, base_value=starting_value)
        cost_schedules = {good.name: cost_schedule}
        new_message = f"You are a seller of {good.name} and your current cost is {cost_schedule.get_value(1)}, this is the first round of the market so the are not bids or asks yet. You can make a profit by selling at " + str(cost_schedule.get_value(1)*1.01) + " or higher"
        structured_output = AskTool()
    return SimpleAgent(id=agent_id, llm_config=llm_config,structured_output=structured_output, endowment=endowment, value_schedules=value_schedules, cost_schedules=cost_schedules, new_message=new_message)


