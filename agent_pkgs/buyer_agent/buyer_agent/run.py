import random 
import json 
import re 
import time 
import ast 
from pydantic.main import BaseModel 
from pydantic.fields import Field 
from pydantic.fields import computed_field 
from typing import List 
from copy import deepcopy 
from pydantic.functional_validators import model_validator 
from datetime import datetime 
from functools import cached_property 
from typing import Dict 
from typing import Literal 
from typing import Optional 
from typing_extensions import Self 
from typing import Any 
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam 
from openai.types.chat.chat_completion import ChatCompletion 
from openai.types.shared_params.function_definition import FunctionDefinition 
# from typing import ResponseFormat 
from openai.types.shared_params.response_format_json_schema import ResponseFormatJSONSchema 
from openai.types.shared_params.response_format_json_schema import JSONSchema 
from anthropic.types.tool_param import ToolParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_tool_param import PromptCachingBetaToolParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_cache_control_ephemeral_param import PromptCachingBetaCacheControlEphemeralParam 
from typing import Union 
from typing import Tuple 
# from typing import ChatCompletionMessageParam 
from openai.types.shared_params.response_format_text import ResponseFormatText 
from openai.types.shared_params.response_format_json_object import ResponseFormatJSONObject 
from anthropic.types.message_param import MessageParam 
from anthropic.types.text_block import TextBlock 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_text_block_param import PromptCachingBetaTextBlockParam 
from openai.types.chat.chat_completion_system_message_param import ChatCompletionSystemMessageParam 
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam 
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam 
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam 
from openai.types.chat.chat_completion_function_message_param import ChatCompletionFunctionMessageParam 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message_param import PromptCachingBetaMessageParam 
from pydantic_core._pydantic_core import ValidationError 
from anthropic.types.tool_use_block import ToolUseBlock 
from anthropic.types.message import AnthropicMessage 
from anthropic.types.beta.prompt_caching.prompt_caching_beta_message import PromptCachingBetaMessage 
from abc import ABC 
from pydantic.functional_validators import field_validator 
from dotenv import load_dotenv
from buyer_agent.schemas import InputSchema
from naptha_sdk.utils import get_logger

# manually added (having some scraping issues for TypeAlias objects):
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat

logger = get_logger(__name__)

load_dotenv()

class Good(BaseModel):
    name: str
    quantity: float

class MarketAction(BaseModel):
    price: float = Field(..., description="Price of the order")
    quantity: int = Field(default=1, ge=1,le=1, description="Quantity of the order")

class Ask(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return False

class Trade(BaseModel):
    trade_id: int = Field(..., description="Unique identifier for the trade")
    buyer_id: str = Field(..., description="ID of the buyer")
    seller_id: str = Field(..., description="ID of the seller")
    price: float = Field(..., description="The price at which the trade was executed")
    ask_price: float = Field(ge=0, description="The price at which the ask was executed")
    bid_price: float = Field(ge=0, description="The price at which the bid was executed")
    quantity: int = Field(default=1, description="The quantity traded")
    good_name: str = Field(default="consumption_good", description="The name of the good traded")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of the trade")

    @model_validator(mode='after')
    def rational_trade(self):
        if self.ask_price > self.bid_price:
            raise ValueError(f"Ask price {self.ask_price} is more than bid price {self.bid_price}")
        return self

class Basket(BaseModel):
    cash: float
    goods: List[Good]

    @computed_field
    @cached_property
    def goods_dict(self) -> Dict[str, int]:
        return {good.name: int(good.quantity) for good in self.goods}

    def update_good(self, name: str, quantity: float):
        for good in self.goods:
            if good.name == name:
                good.quantity = quantity
                return
        self.goods.append(Good(name=name, quantity=quantity))

    def get_good_quantity(self, name: str) -> int:
        return int(next((good.quantity for good in self.goods if good.name == name), 0))

class Endowment(BaseModel):
    initial_basket: Basket
    trades: List[Trade] = Field(default_factory=list)
    agent_id: str

    @computed_field
    @property
    def current_basket(self) -> Basket:
        temp_basket = deepcopy(self.initial_basket)

        for trade in self.trades:
            if trade.buyer_id == self.agent_id:
                temp_basket.cash -= trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
            elif trade.seller_id == self.agent_id:
                temp_basket.cash += trade.price * trade.quantity
                temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        # Create a new Basket instance with the calculated values
        return Basket(
            cash=temp_basket.cash,
            goods=[Good(name=good.name, quantity=good.quantity) for good in temp_basket.goods]
        )

    def add_trade(self, trade: Trade):
        self.trades.append(trade)
        # Clear the cached property to ensure it's recalculated
        if 'current_basket' in self.__dict__:
            del self.__dict__['current_basket']

    def simulate_trade(self, trade: Trade) -> Basket:
        temp_basket = deepcopy(self.current_basket)

        if trade.buyer_id == self.agent_id:
            temp_basket.cash -= trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) + trade.quantity)
        elif trade.seller_id == self.agent_id:
            temp_basket.cash += trade.price * trade.quantity
            temp_basket.update_good(trade.good_name, temp_basket.get_good_quantity(trade.good_name) - trade.quantity)

        return temp_basket

class Bid(MarketAction):
    @computed_field
    @property
    def is_buyer(self) -> bool:
        return True

class PreferenceSchedule(BaseModel):
    num_units: int = Field(..., description="Number of units")
    base_value: float = Field(..., description="Base value for the first unit")
    noise_factor: float = Field(default=0.1, description="Noise factor for value generation")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        raise NotImplementedError("Subclasses must implement this method")

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        raise NotImplementedError("Subclasses must implement this method")

    def get_value(self, quantity: int) -> float:
        return self.values.get(quantity, 0.0)

    def plot_schedule(self, block=False):
        quantities = list(self.values.keys())
        values = list(self.values.values())
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(quantities, values, marker='o')
        plt.title(f"{'Demand' if self.is_buyer else 'Supply'} Schedule")
        plt.xlabel("Quantity")
        plt.ylabel("Value/Cost")
        plt.grid(True)
        plt.show(block=block)

class SellerPreferenceSchedule(PreferenceSchedule):
    is_buyer: bool = Field(default=False, description="Whether the agent is a buyer")
    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Increase current_value by 2% to 5% plus noise
            increment = current_value * random.uniform(0.02, self.noise_factor)
            new_value = current_value+increment
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values())

class BuyerPreferenceSchedule(PreferenceSchedule):
    endowment_factor: float = Field(default=1.2, description="Factor to calculate initial endowment")
    is_buyer: bool = Field(default=True, description="Whether the agent is a buyer")

    @computed_field
    @cached_property
    def values(self) -> Dict[int, float]:
        values = {}
        current_value = self.base_value
        for i in range(1, self.num_units + 1):
            # Decrease current_value by 2% to 5% plus noise
            decrement = current_value * random.uniform(0.02, self.noise_factor)
            
            new_value = current_value-decrement
            values[i] = new_value
            current_value = new_value
        return values

    @computed_field
    @cached_property
    def initial_endowment(self) -> float:
        return sum(self.values.values()) * self.endowment_factor

class LLMConfig(BaseModel):
    client: Literal["openai", "azure_openai", "anthropic", "vllm", "litellm"]
    model: Optional[str] = None
    max_tokens: int = Field(default=400)
    temperature: float = 0
    response_format: Literal["json_beg", "text","json_object","structured_output","tool"] = "text"
    use_cache: bool = True

    @model_validator(mode="after")
    def validate_response_format(self) -> Self:
        if self.response_format == "json_object" and self.client in ["vllm", "litellm","anthropic"]:
            raise ValueError(f"{self.client} does not support json_object response format")
        elif self.response_format == "structured_output" and self.client == "anthropic":
            raise ValueError(f"Anthropic does not support structured_output response format use json_beg or tool instead")
        return self

class StructuredTool(BaseModel):
    """ Supported type by OpenAI Structured Output:
    String, Number, Boolean, Integer, Object, Array, Enum, anyOf
    Root must be Object, not anyOf
    Not supported by OpenAI Structured Output: 
    For strings: minLength, maxLength, pattern, format
    For numbers: minimum, maximum, multipleOf
    For objects: patternProperties, unevaluatedProperties, propertyNames, minProperties, maxProperties
    For arrays: unevaluatedItems, contains, minContains, maxContains, minItems, maxItems, uniqueItems
    oai_reference: https://platform.openai.com/docs/guides/structured-outputs/how-to-use """

    json_schema: Optional[Dict[str, Any]] = None
    schema_name: str = Field(default = "generate_structured_output")
    schema_description: str = Field(default ="Generate a structured output based on the provided JSON schema.")
    instruction_string: str = Field(default = "Please follow this JSON schema for your response:")
    strict_schema: bool = True

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

class EconomicAgent(BaseModel):
    id: str
    endowment: Endowment
    value_schedules: Dict[str, BuyerPreferenceSchedule] = Field(default_factory=dict)
    cost_schedules: Dict[str, SellerPreferenceSchedule] = Field(default_factory=dict)
    max_relative_spread: float = Field(default=0.2)
    pending_orders: Dict[str, List[MarketAction]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_schedules(self):
        overlapping_goods = set(self.value_schedules.keys()) & set(self.cost_schedules.keys())
        if overlapping_goods:
            raise ValueError(f"Agent cannot be both buyer and seller of the same good(s): {overlapping_goods}")
        return self
    
    @computed_field
    @property
    def initial_utility(self) -> float:
        utility = self.endowment.initial_basket.cash
        for good, quantity in self.endowment.initial_basket.goods_dict.items():
            if self.is_buyer(good):
                # Buyers start with cash and no goods
                pass
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                initial_quantity = int(quantity)
                initial_cost = sum(schedule.get_value(q) for q in range(1, initial_quantity + 1))
                utility += initial_cost  # Add total cost of initial inventory
        return utility

    @computed_field
    @property
    def current_utility(self) -> float:
        return self.calculate_utility(self.endowment.current_basket)

    @computed_field
    @property
    def current_cash(self) -> float:
        return self.endowment.current_basket.cash
    
    @computed_field
    @property
    def pending_cash(self) -> float:
        return sum(order.price * order.quantity for orders in self.pending_orders.values() for order in orders if isinstance(order, Bid))
    
    @computed_field
    @property
    def available_cash(self) -> float:
        return self.current_cash - self.pending_cash
    
    
    def get_pending_bid_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
    
    def get_pending_ask_quantity(self, good_name: str) -> int:
        """ returns the quantity of the good that the agent has pending orders for"""
        return sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
    
    def get_quantity_for_bid(self, good_name: str) -> Optional[int]:
        if not self.is_buyer(good_name):
            return None
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_bid_quantity(good_name)
        total_quantity = int(current_quantity + pending_quantity)
        if total_quantity > self.value_schedules[good_name].num_units:
            return None
        print(f"total_quantity: {total_quantity}")
        return total_quantity+1

    
    def get_current_value(self, good_name: str) -> Optional[float]:
        """ returns the current value of the next unit of the good  for buyers only
        it takes in consideration both the current inventory and the pending orders"""
        total_quantity = self.get_quantity_for_bid(good_name)
        if total_quantity is None:
            return None
        current_value = self.value_schedules[good_name].get_value(total_quantity)
        return current_value if current_value  > 0 else None
    
    def get_previous_value(self, good_name: str) -> Optional[float]:
        """ returns the previous value of the good for buyers only
        it takes in consideration both the current inventory and the pending orders"""
        total_quantity = self.get_quantity_for_bid(good_name)
        if total_quantity is None:
            return None
        value = self.value_schedules[good_name].get_value(total_quantity+1)
        return value if value > 0 else None
    
    
    def get_current_cost(self, good_name: str) -> Optional[float]:
        """ returns the current cost of the good for sellers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_seller(good_name):
            return None
        starting_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_ask_quantity(good_name)
        total_quantity = int(starting_quantity - current_quantity + pending_quantity)+1
        cost = self.cost_schedules[good_name].get_value(total_quantity)

        return cost if cost > 0 else None
    def get_previous_cost(self, good_name: str) -> Optional[float]:
        """ returns the previous cost of the good for sellers only
        it takes in consideration both the current inventory and the pending orders"""
        if not self.is_seller(good_name):
            return None
        starting_quantity = self.endowment.initial_basket.get_good_quantity(good_name)
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = self.get_pending_ask_quantity(good_name)
        total_quantity = int(starting_quantity - current_quantity + pending_quantity)
        return self.cost_schedules[good_name].get_value(total_quantity)

    def is_buyer(self, good_name: str) -> bool:
        return good_name in self.value_schedules

    def is_seller(self, good_name: str) -> bool:
        return good_name in self.cost_schedules

    def generate_bid(self, good_name: str) -> Optional[Bid]:
        if not self._can_generate_bid(good_name):
            return None
        price = self._calculate_bid_price(good_name)
        if price is not None:
            bid = Bid(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(bid)
            return bid
        else:
            return None

    def generate_ask(self, good_name: str) -> Optional[Ask]:
        if not self._can_generate_ask(good_name):
            return None
        price = self._calculate_ask_price(good_name)
        if price is not None:
            ask = Ask(price=price, quantity=1)
            # Update pending orders
            self.pending_orders.setdefault(good_name, []).append(ask)
            return ask
        else:
            return None
    
    def would_accept_trade(self, trade: Trade) -> bool:
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            marginal_value = self.get_previous_value(trade.good_name)
            if marginal_value is None:
                print("trade rejected because marginal_value is None")
                return False
            print(f"Buyer {self.id} would accept trade with marginal_value: {marginal_value}, trade.price: {trade.price}")
            return marginal_value >= trade.price
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            marginal_cost = self.get_previous_cost(trade.good_name)
            if marginal_cost is None:
                print("trade rejected because marginal_cost is None")
                return False
            print(f"Seller {self.id} would accept trade with marginal_cost: {marginal_cost}, trade.price: {trade.price}")
            return trade.price >= marginal_cost
        else:
            self_type = "buyer" if self.is_buyer(trade.good_name) else "seller"
            print(f"trade rejected because it's not for the agent {self.id} vs trade.buyer_id: {trade.buyer_id}, trade.seller_id: {trade.seller_id} with type {self_type}")
            return False

    def process_trade(self, trade: Trade):
        if self.is_buyer(trade.good_name) and trade.buyer_id == self.id:
            # Find the exactly matching bid from pending_orders
            bids = self.pending_orders.get(trade.good_name, [])
            matching_bid = next((bid for bid in bids if bid.quantity == trade.quantity and bid.price == trade.bid_price), None)
            if matching_bid:
                bids.remove(matching_bid)
                if not bids:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching bid not found for agent {self.id}")
        elif self.is_seller(trade.good_name) and trade.seller_id == self.id:
            # Find the exactly matching ask from pending_orders
            asks = self.pending_orders.get(trade.good_name, [])
            matching_ask = next((ask for ask in asks if ask.quantity == trade.quantity and ask.price == trade.ask_price), None)
            if matching_ask:
                asks.remove(matching_ask)
                if not asks:
                    del self.pending_orders[trade.good_name]
            else:
                raise ValueError(f"Trade {trade.trade_id} processed but matching ask not found for agent {self.id}")
        
        # Only update the endowment after passing the value error checks
        self.endowment.add_trade(trade)
        new_utility = self.calculate_utility(self.endowment.current_basket)
        logger.info(f"Agent {self.id} processed trade. New utility: {new_utility:.2f}")

    def reset_pending_orders(self,good_name:str):
        self.pending_orders[good_name] = []

    def reset_all_pending_orders(self):
        self.pending_orders = {}
        

    def calculate_utility(self, basket: Basket) -> float:
        utility = basket.cash
        
        for good, quantity in basket.goods_dict.items():
            if self.is_buyer(good):
                schedule = self.value_schedules[good]
                value_sum = sum(schedule.get_value(q) for q in range(1, int(quantity) + 1))
                utility += value_sum
            elif self.is_seller(good):
                schedule = self.cost_schedules[good]
                starting_quantity = self.endowment.initial_basket.get_good_quantity(good)
                unsold_units = int(basket.get_good_quantity(good))
                sold_units = starting_quantity - unsold_units
                # Unsold inventory should be valued at its cost, not higher
                unsold_cost = sum(schedule.get_value(q) for q in range(sold_units+1, starting_quantity + 1))

                utility += unsold_cost  # Add the cost of unsold units
        return utility



    def calculate_individual_surplus(self) -> float:
        current_utility = self.calculate_utility(self.endowment.current_basket)
        surplus = current_utility - self.initial_utility
        print(f"current_utility: {current_utility}, initial_utility: {self.initial_utility}, surplus: {surplus}")
        return surplus



    def _can_generate_bid(self, good_name: str) -> bool:
        if not self.is_buyer(good_name):
            return False

        available_cash = self.available_cash 
        if available_cash <= 0:
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Bid))
        total_quantity = current_quantity + pending_quantity
        return total_quantity < self.value_schedules[good_name].num_units

    def _can_generate_ask(self, good_name: str) -> bool:
        if not self.is_seller(good_name):
            return False
        current_quantity = self.endowment.current_basket.get_good_quantity(good_name)
        pending_quantity = sum(order.quantity for order in self.pending_orders.get(good_name, []) if isinstance(order, Ask))
        total_quantity = current_quantity - pending_quantity
        return total_quantity > 0

    def _calculate_bid_price(self, good_name: str) -> Optional[float]:

        current_value = self.get_current_value(good_name)
        if current_value is None:
            return None
        max_bid = min(self.endowment.current_basket.cash, current_value*0.99)
        price = random.uniform(max_bid * (1 - self.max_relative_spread), max_bid)
        return price

    def _calculate_ask_price(self, good_name: str) -> Optional[float]:
        current_cost = self.get_current_cost(good_name)
        if current_cost is None:
            return None
        min_ask = current_cost * 1.01
        price = random.uniform(min_ask, min_ask * (1 + self.max_relative_spread))
        return price

    def print_status(self):
        print(f"\nAgent ID: {self.id}")
        print(f"Current Endowment:")
        print(f"  Cash: {self.endowment.current_basket.cash:.2f}")
        print(f"  Goods: {self.endowment.current_basket.goods_dict}")
        current_utility = self.calculate_utility(self.endowment.current_basket)
        print(f"Current Utility: {current_utility:.2f}")
        self.calculate_individual_surplus()

def msg_dict_to_oai(messages: List[Dict[str, Any]]) -> List[ChatCompletionMessageParam]:
        def convert_message(msg: Dict[str, Any]) -> ChatCompletionMessageParam:
            role = msg["role"]
            if role == "system":
                return ChatCompletionSystemMessageParam(role=role, content=msg["content"])
            elif role == "user":
                return ChatCompletionUserMessageParam(role=role, content=msg["content"])
            elif role == "assistant":
                assistant_msg = ChatCompletionAssistantMessageParam(role=role, content=msg.get("content"))
                if "function_call" in msg:
                    assistant_msg["function_call"] = msg["function_call"]
                if "tool_calls" in msg:
                    assistant_msg["tool_calls"] = msg["tool_calls"]
                return assistant_msg
            elif role == "tool":
                return ChatCompletionToolMessageParam(role=role, content=msg["content"], tool_call_id=msg["tool_call_id"])
            elif role == "function":
                return ChatCompletionFunctionMessageParam(role=role, content=msg["content"], name=msg["name"])
            else:
                raise ValueError(f"Unknown role: {role}")

        return [convert_message(msg) for msg in messages]

def msg_dict_to_anthropic(messages: List[Dict[str, Any]],use_cache:bool=True,use_prefill:bool=False) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        def create_anthropic_system_message(system_message: Optional[Dict[str, Any]],use_cache:bool=True) -> List[PromptCachingBetaTextBlockParam]:
            if system_message and system_message["role"] == "system":
                text = system_message["content"]
                if use_cache:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text, cache_control=PromptCachingBetaCacheControlEphemeralParam(type="ephemeral"))]
                else:
                    return [PromptCachingBetaTextBlockParam(type="text", text=text)]
            return []

        def convert_message(msg: Dict[str, Any],use_cache:bool=False) -> Union[PromptCachingBetaMessageParam, None]:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                return None
            
            if isinstance(content, str):
                if not use_cache:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content)]
                else:
                    content = [PromptCachingBetaTextBlockParam(type="text", text=content,cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral'))]
            elif isinstance(content, list):
                if not use_cache:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"]) for block in content
                    ]
                else:
                    content = [
                        PromptCachingBetaTextBlockParam(type="text", text=block, cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) if isinstance(block, str)
                        else PromptCachingBetaTextBlockParam(type="text", text=block["text"], cache_control=PromptCachingBetaCacheControlEphemeralParam(type='ephemeral')) for block in content
                    ]
            else:
                raise ValueError("Invalid content type")
            
            return PromptCachingBetaMessageParam(role=role, content=content)
        converted_messages = []
        system_message = []
        num_messages = len(messages)
        if use_cache:
            use_cache_ids = set([num_messages - 1, max(0, num_messages - 3)])
        else:
            use_cache_ids = set()
        for i,message in enumerate(messages):
            if message["role"] == "system":
                system_message= create_anthropic_system_message(message,use_cache=use_cache)
            else:
                
                use_cache_final = use_cache if  i in use_cache_ids else False
                converted_messages.append(convert_message(message,use_cache= use_cache_final))

        
        return system_message, [msg for msg in converted_messages if msg is not None]

def parse_json_string(content: str) -> Optional[Dict[str, Any]]:
    # Remove any leading/trailing whitespace and newlines
    cleaned_content = content.strip()
    
    # Remove markdown code block syntax if present
    cleaned_content = re.sub(r'^```(?:json)?\s*|\s*```$', '', cleaned_content, flags=re.MULTILINE)
    
    try:
        # First, try to parse as JSON
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        try:
            # If JSON parsing fails, try to evaluate as a Python literal
            return ast.literal_eval(cleaned_content)
        except (SyntaxError, ValueError):
            # If both methods fail, try to find and parse a JSON-like structure
            json_match = re.search(r'(\{[^{}]*\{.*?\}[^{}]*\}|\{.*?\})', cleaned_content, re.DOTALL)
            if json_match:
                try:
                    # Normalize newlines, replace single quotes with double quotes, and unescape quotes
                    json_str = json_match.group(1).replace('\n', '').replace("'", '"').replace('\\"', '"')
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
    
    # If all parsing attempts fail, return None
    return None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: Optional[int] = None
    cache_read_input_tokens: Optional[int] = None

class GeneratedJsonObject(BaseModel):
    name: str
    object: Dict[str, Any]

class LLMOutput(BaseModel):
    raw_result: Union[str, dict, ChatCompletion, AnthropicMessage, PromptCachingBetaMessage]
    completion_kwargs: Optional[Dict[str, Any]] = None
    start_time: float
    end_time: float
    source_id: str
    client: Optional[Literal["openai", "anthropic","vllm","litellm"]] = Field(default=None)

    @property
    def time_taken(self) -> float:
        return self.end_time - self.start_time

    @computed_field
    @property
    def str_content(self) -> Optional[str]:
        return self._parse_result()[0]

    @computed_field
    @property
    def json_object(self) -> Optional[GeneratedJsonObject]:
        return self._parse_result()[1]
    
    @computed_field
    @property
    def error(self) -> Optional[str]:
        return self._parse_result()[3]

    @computed_field
    @property
    def contains_object(self) -> bool:
        return self._parse_result()[1] is not None
    
    @computed_field
    @property
    def usage(self) -> Optional[Usage]:
        return self._parse_result()[2]

    @computed_field
    @property
    def result_provider(self) -> Optional[Literal["openai", "anthropic","vllm","litellm"]]:
        return self.search_result_provider() if self.client is None else self.client
    
    @model_validator(mode="after")
    def validate_provider_and_client(self) -> Self:
        if self.client is not None and self.result_provider != self.client:
            raise ValueError(f"The inferred result provider '{self.result_provider}' does not match the specified client '{self.client}'")
        return self
    
    
    def search_result_provider(self) -> Optional[Literal["openai", "anthropic"]]:
        try:
            oai_completion = ChatCompletion.model_validate(self.raw_result)
            return "openai"
        except ValidationError:
            try:
                anthropic_completion = AnthropicMessage.model_validate(self.raw_result)
                return "anthropic"
            except ValidationError:
                try:
                    antrhopic_beta_completion = PromptCachingBetaMessage.model_validate(self.raw_result)
                    return "anthropic"
                except ValidationError:
                    return None

    def _parse_json_string(self, content: str) -> Optional[Dict[str, Any]]:
        return parse_json_string(content)
    
    

    def _parse_oai_completion(self,chat_completion:ChatCompletion) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage], None]:
        message = chat_completion.choices[0].message
        content = message.content

        json_object = None
        usage = None

        if message.tool_calls:
            tool_call = message.tool_calls[0]
            name = tool_call.function.name
            try:
                object_dict = json.loads(tool_call.function.arguments)
                json_object = GeneratedJsonObject(name=name, object=object_dict)
            except json.JSONDecodeError:
                json_object = GeneratedJsonObject(name=name, object={"raw": tool_call.function.arguments})
        elif content is not None:
            if self.completion_kwargs:
                name = self.completion_kwargs.get("response_format",{}).get("json_schema",{}).get("name",None)
            else:
                name = None
            parsed_json = self._parse_json_string(content)
            if parsed_json:
                
                json_object = GeneratedJsonObject(name="parsed_content" if name is None else name,
                                                   object=parsed_json)
                content = None  # Set content to None when we have a parsed JSON object
                #print(f"parsed_json: {parsed_json} with name")
        if chat_completion.usage:
            usage = Usage(
                prompt_tokens=chat_completion.usage.prompt_tokens,
                completion_tokens=chat_completion.usage.completion_tokens,
                total_tokens=chat_completion.usage.total_tokens
            )

        return content, json_object, usage, None

    def _parse_anthropic_message(self, message: Union[AnthropicMessage, PromptCachingBetaMessage]) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],None]:
        content = None
        json_object = None
        usage = None

        if message.content:
            first_content = message.content[0]
            if isinstance(first_content, TextBlock):
                content = first_content.text
                parsed_json = self._parse_json_string(content)
                if parsed_json:
                    json_object = GeneratedJsonObject(name="parsed_content", object=parsed_json)
                    content = None  # Set content to None when we have a parsed JSON object
            elif isinstance(first_content, ToolUseBlock):
                name = first_content.name
                input_dict : Dict[str,Any] = first_content.input # type: ignore  # had to ignore due to .input being of object class
                json_object = GeneratedJsonObject(name=name, object=input_dict)

        if hasattr(message, 'usage'):
            usage = Usage(
                prompt_tokens=message.usage.input_tokens,
                completion_tokens=message.usage.output_tokens,
                total_tokens=message.usage.input_tokens + message.usage.output_tokens,
                cache_creation_input_tokens=getattr(message.usage, 'cache_creation_input_tokens', None),
                cache_read_input_tokens=getattr(message.usage, 'cache_read_input_tokens', None)
            )

        return content, json_object, usage, None
    

    def _parse_result(self) -> Tuple[Optional[str], Optional[GeneratedJsonObject], Optional[Usage],Optional[str]]:
        provider = self.result_provider
        if getattr(self.raw_result, "error", None):
            return None, None, None,  getattr(self.raw_result, "error", None)
        if provider == "openai":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "anthropic":
            try: #beta first
                return self._parse_anthropic_message(PromptCachingBetaMessage.model_validate(self.raw_result))
            except ValidationError:
                return self._parse_anthropic_message(AnthropicMessage.model_validate(self.raw_result))
        elif provider == "vllm":
             return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        elif provider == "litellm":
            return self._parse_oai_completion(ChatCompletion.model_validate(self.raw_result))
        else:
            raise ValueError(f"Unsupported result provider: {provider}")

    class Config:
        arbitrary_types_allowed = True

class LLMPromptContext(BaseModel):
    id: str
    system_string: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None
    new_message: str
    prefill: str = Field(default="Here's the valid JSON object response:```json", description="prefill assistant response with an instruction")
    postfill: str = Field(default="\n\nPlease provide your response in JSON format.", description="postfill user response with an instruction")
    structured_output : Optional[StructuredTool] = None
    use_schema_instruction: bool = Field(default=False, description="Whether to use the schema instruction")
    llm_config: LLMConfig
    use_history: bool = Field(default=True, description="Whether to use the history")
    
    @computed_field
    @property
    def oai_response_format(self) -> Optional[ResponseFormat]:
        if self.llm_config.response_format == "text":
            return ResponseFormatText(type="text")
        elif self.llm_config.response_format == "json_object":
            return ResponseFormatJSONObject(type="json_object")
        elif self.llm_config.response_format == "structured_output":
            assert self.structured_output is not None, "Structured output is not set"
            return self.structured_output.get_openai_json_schema_response()
        else:
            return None


    @computed_field
    @property
    def use_prefill(self) -> bool:
        if self.llm_config.client in ['anthropic','vllm','litellm'] and  self.llm_config.response_format in ["json_beg"]:

            return True
        else:
            return False
        
    @computed_field
    @property
    def use_postfill(self) -> bool:
        if self.llm_config.client == 'openai' and 'json' in self.llm_config.response_format and not self.use_schema_instruction:
            return True

        else:
            return False
        
    @computed_field
    @property
    def system_message(self) -> Optional[Dict[str, str]]:
        content= self.system_string if self.system_string  else ""
        if self.use_schema_instruction and self.structured_output:
            content = "\n".join([content,self.structured_output.schema_instruction])
        return {"role":"system","content":content} if len(content)>0 else None
    
    @computed_field
    @property
    def messages(self)-> List[Dict[str, Any]]:
        messages = [self.system_message] if self.system_message is not None else []
        if  self.use_history and self.history:
            messages+=self.history
        messages.append({"role":"user","content":self.new_message})
        if self.use_prefill:
            prefill_message = {"role":"assistant","content":self.prefill}
            messages.append(prefill_message)
        elif self.use_postfill:
            messages[-1]["content"] = messages[-1]["content"] + self.postfill
        return messages
    
    @computed_field
    @property
    def oai_messages(self)-> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
    
    @computed_field
    @property
    def anthropic_messages(self) -> Tuple[List[PromptCachingBetaTextBlockParam],List[MessageParam]]:
        return msg_dict_to_anthropic(self.messages, use_cache=self.llm_config.use_cache)
    
    @computed_field
    @property
    def vllm_messages(self) -> List[ChatCompletionMessageParam]:
        return msg_dict_to_oai(self.messages)
        
    def update_llm_config(self,llm_config:LLMConfig) -> 'LLMPromptContext':
        
        return self.model_copy(update={"llm_config":llm_config})
       
    

        
    def add_chat_turn_history(self, llm_output:'LLMOutput'):
        """ add a chat turn to the history without safely model copy just normal append """
        if llm_output.source_id != self.id:
            raise ValueError(f"LLMOutput source_id {llm_output.source_id} does not match the prompt context id {self.id}")
        if self.history is None:
            self.history = []
        self.history.append({"role": "user", "content": self.new_message})
        self.history.append({"role": "assistant", "content": llm_output.str_content or json.dumps(llm_output.json_object.object) if llm_output.json_object else "{}"})
    
    def get_tool(self) -> Union[ChatCompletionToolParam, PromptCachingBetaToolParam, None]:
        if not self.structured_output:
            return None
        if self.llm_config.client in ["openai","vllm","litellm"]:
            return self.structured_output.get_openai_tool()
        elif self.llm_config.client == "anthropic":
            return self.structured_output.get_anthropic_tool()
        else:
            return None

class LocalObservation(BaseModel, ABC):
    """Represents an observation for a single agent."""
    agent_id: str
    observation: BaseModel

class AuctionObservation(BaseModel):
    trades: List[Trade] = Field(default_factory=list, description="List of trades the agent participated in")
    market_summary: Dict[str, Any] = Field(default_factory=dict, description="Summary of market activity")
    waiting_orders: List[Union[Bid, Ask]] = Field(default_factory=list, description="List of orders waiting to be executed")

class AuctionLocalObservation(LocalObservation):
    observation: AuctionObservation

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
            f"{f'Your evaluation of the good is: {self.evaluation}' if self.evaluation is not None else ''}"
            f"{f' {profit_message}' if profit_message else ''}"
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

apple = Good(name='apple', quantity=0, )

buyer_llm_config = LLMConfig(client='anthropic', model='claude-3-5-sonnet-20240620', response_format='tool', )

def buyer_agent():
    # Create simple agents
    buyer = create_simple_agent(
        agent_id="buyer_1",
        llm_config=buyer_llm_config,
        good=apple,
        is_buyer=True,
        endowment=Endowment(agent_id="buyer_1", initial_basket=Basket(cash=1000, goods=[Good(name="apple", quantity=0)])),
        starting_value=20.0,
        num_units=10
    )
    return buyer

def run(inputs: InputSchema, *args, **kwargs):
    buyer_agent_0 = buyer_agent()

    tool_input_class = globals().get(inputs.tool_input_type)
    tool_input = tool_input_class(**inputs.tool_input_value)
    method = getattr(buyer_agent_0, inputs.tool_name, None)

    return method(tool_input)

if __name__ == "__main__":
    from naptha_sdk.utils import load_yaml
    from buyer_agent.schemas import InputSchema

    cfg_path = "buyer_agent/component.yaml"
    cfg = load_yaml(cfg_path)

    # You will likely need to change the inputs dict
    # original: inputs = {"tool_name": "execute_task", "tool_input_type": "Task", "tool_input_value": {"description": "What is the market cap of AMZN?", "expected_output": "The market cap of AMZN"}}
    # manually changed: 
    inputs = {"tool_name": "generate_bid", "tool_input_type": "Good", "tool_input_value": {"name": "apple", "quantity":1}}
    inputs = InputSchema(**inputs)

    response = run(inputs)
    print(response)
