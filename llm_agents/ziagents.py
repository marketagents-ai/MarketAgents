import random
import logging
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, model_validator
from typing import Dict, Optional, List
from typing_extensions import Self

# Set up logger
logger = logging.getLogger(__name__)

# Global counter for trade IDs
trade_counter = 0

class PreferenceSchedule(BaseModel):
    values: Dict[int, float] = Field(..., description="Dictionary mapping quantity to value/cost")
    is_buyer: bool = Field(..., description="True if this is a buyer's schedule, False for seller")
    initial_endowment: float = Field(..., description="Initial cash endowment for buyers or value of goods for sellers")

    @classmethod
    def generate(cls, is_buyer: bool, num_units: int, base_value: float, noise_factor: float = 0.1, endowment_factor: float = 1.2):
        values = {}
        current_value = base_value
        for i in range(1, num_units + 1):
            noise = random.uniform(-noise_factor, noise_factor) * current_value
            new_value = max(1, current_value + noise)  # Ensure no zero values
            if is_buyer:
                if i > 1:
                    new_value = min(new_value, values[i-1])
                current_value *= random.uniform(0.95, 1.0)
            else:
                if i > 1:
                    new_value = max(new_value, values[i-1])
                current_value *= random.uniform(1.0, 1.05)
            values[i] = new_value
        
        initial_endowment = sum(values.values()) * endowment_factor if is_buyer else sum(values.values())
        return cls(values=values, is_buyer=is_buyer, initial_endowment=initial_endowment)

    def get_value(self, quantity: int) -> float:
        return self.values.get(quantity, 0)

    def get_cost(self, quantity: int) -> float:
        if self.is_buyer:
            return 0  # Buyers don't have a cost
        else:
            return sum(self.get_value(i) for i in range(1, quantity + 1))

    def plot_schedule(self):
        quantities = list(self.values.keys())
        values = list(self.values.values())
        
        plt.figure(figsize=(10, 6))
        plt.plot(quantities, values, marker='o')
        plt.title(f"{'Demand' if self.is_buyer else 'Supply'} Schedule")
        plt.xlabel("Quantity")
        plt.ylabel("Value/Cost")
        plt.grid(True)
        plt.show()

class Allocation(BaseModel):
    goods: int = Field(default=0, description="Quantity of goods")
    cash: float = Field(default=0.0, description="Amount of cash")
    locked_goods: int = Field(default=0, description="Quantity of locked goods")
    locked_cash: float = Field(default=0.0, description="Amount of locked cash")
    initial_goods: int = Field(default=0, description="Initial quantity of goods")
    initial_cash: float = Field(default=0.0, description="Initial amount of cash")

class Order(BaseModel):
    agent_id: int
    is_buy: bool
    quantity: int
    price: float
    base_value: Optional[float] = None  # Value for buyers
    base_cost: Optional[float] = None   # Cost for sellers

    @model_validator(mode='after')
    def validate_order(self) -> Self:
        if self.is_buy:
            if self.base_value is None:
                raise ValueError("base_value must be provided for buy orders")
            if self.price > self.base_value:
                raise ValueError("buy price cannot exceed base value")
        else:
            if self.base_cost is None:
                raise ValueError("base_cost must be provided for sell orders")
            if self.price < self.base_cost:
                raise ValueError("sell price cannot be less than base cost")

        @classmethod
        def validate_order(cls, order: 'Order') -> bool:
            try:
                order.validate_order()
                return True
            except ValueError:
                return False

class Trade(BaseModel):
    trade_id: int  # Unique identifier for the trade
    buyer_id: int
    seller_id: int
    quantity: int
    price: float
    buyer_value: float
    seller_cost: float
    round: int  # Round number

class ZIAgent(BaseModel):
    id: int = Field(..., description="Unique identifier for the agent")
    preference_schedule: PreferenceSchedule
    allocation: Allocation
    max_relative_spread: float = Field(default=0.2, description="Maximum relative price spread")
    posted_orders: List[Order] = Field(default_factory=list, description="History of successfully posted orders")
    rejected_orders: List[Order] = Field(default_factory=list, description="History of rejected orders")

    @classmethod
    def generate(cls, agent_id: int, is_buyer: bool, num_units: int, base_value: float, max_relative_spread: float = 0.2):
        preference_schedule = PreferenceSchedule.generate(is_buyer, num_units, base_value)
        allocation = Allocation(
            cash=preference_schedule.initial_endowment if is_buyer else 0.0,
            goods=0 if is_buyer else num_units,
            initial_cash=preference_schedule.initial_endowment if is_buyer else 0.0,
            initial_goods=0 if is_buyer else num_units
        )
        return cls(id=agent_id, preference_schedule=preference_schedule, allocation=allocation, max_relative_spread=max_relative_spread)

    def calculate_trade_surplus(self, trade: Trade) -> float:
        if trade.buyer_id == self.id:
            surplus = trade.buyer_value - trade.price
            logger.info(f"Calculating Buyer Surplus:")
            logger.info(f"  Buyer ID: {self.id}")
            logger.info(f"  Buyer's Value: {trade.buyer_value:.2f}")
            logger.info(f"  Trade Price: {trade.price:.2f}")
            logger.info(f"  Surplus: {surplus:.2f}")
        elif trade.seller_id == self.id:
            surplus = trade.price - trade.seller_cost
            logger.info(f"Calculating Seller Surplus:")
            logger.info(f"  Seller ID: {self.id}")
            logger.info(f"  Seller's Cost: {trade.seller_cost:.2f}")
            logger.info(f"  Trade Price: {trade.price:.2f}")
            logger.info(f"  Surplus: {surplus:.2f}")
        else:
            surplus = 0.0

        assert surplus >= 0, (
            f"\nNegative surplus detected:\n"
            f"  Agent ID: {self.id}\n"
            f"  Surplus: {surplus:.2f}\n"
            f"  Buyer ID: {trade.buyer_id}\n"
            f"  Seller ID: {trade.seller_id}\n"
            f"  Trade Price: {trade.price:.2f}\n"
            f"  Buyer's Value: {trade.buyer_value if trade.buyer_id == self.id else 'N/A'}\n"
            f"  Seller's Cost: {trade.seller_cost if trade.seller_id == self.id else 'N/A'}\n"
        )

        return surplus

    def generate_bid(self) -> Optional[Order]:
        is_buy = self.preference_schedule.is_buyer

        if is_buy:
            current_quantity = self.allocation.goods
            available_cash = self.allocation.cash
            base_value = self.preference_schedule.get_value(current_quantity + 1)

            if base_value <= 0 or available_cash < base_value:
                logger.debug(f"Agent {self.id} cannot generate buy order: base_value={base_value}, available_cash={available_cash}")
                return None

            price = random.uniform(base_value * (1 - self.max_relative_spread), base_value)
            price = min(price, available_cash, base_value)

            logger.debug(f"Agent {self.id} generated buy order: price={price}, base_value={base_value}")
            return Order(
                agent_id=self.id,
                is_buy=True,
                quantity=1,
                price=price,
                base_value=base_value
            )

        else:
            current_quantity = self.allocation.initial_goods - self.allocation.goods
            base_cost = self.preference_schedule.get_value(current_quantity + 1)

            if base_cost <= 0:
                logger.debug(f"Agent {self.id} cannot generate sell order: base_cost={base_cost}")
                return None

            price = random.uniform(base_cost, base_cost * (1.0 + self.max_relative_spread))
            price = max(price, base_cost)

            logger.debug(f"Agent {self.id} generated sell order: price={price}, base_cost={base_cost}")
            return Order(
                agent_id=self.id,
                is_buy=False,
                quantity=1,
                price=price,
                base_cost=base_cost
            )

    def finalize_trade(self, trade: Trade):
        if trade.buyer_id == self.id:
            cash_change = trade.price * trade.quantity
            self.allocation.cash = round(self.allocation.cash - cash_change, 10)
            self.allocation.goods += trade.quantity
            logger.info(f"Agent {self.id} finalized buy trade: cash_change={cash_change}, new_cash={self.allocation.cash}, new_goods={self.allocation.goods}")
        elif trade.seller_id == self.id:
            cash_change = trade.price * trade.quantity
            self.allocation.cash = round(self.allocation.cash + cash_change, 10)
            self.allocation.goods -= trade.quantity
            logger.info(f"Agent {self.id} finalized sell trade: cash_change={cash_change}, new_cash={self.allocation.cash}, new_goods={self.allocation.goods}")

    def respond_to_order(self, order: Order, accepted: bool):
        if accepted:
            logger.info(f"Order Accepted: {'Buyer' if order.is_buy else 'Seller'} {self.id} {'spends' if order.is_buy else 'earns'} ${order.price:.2f} per unit.")
            self.posted_orders.append(order)
        else:
            logger.info(f"Order Rejected: {'Buyer' if order.is_buy else 'Seller'} {self.id}")
            self.rejected_orders.append(order)

    def plot_order_history(self):
        rounds = list(range(1, len(self.posted_orders) + 1))
        prices = [order.price for order in self.posted_orders]
        ir_bounds = [self.preference_schedule.get_value(round_num) for round_num in rounds]
        
        plt.figure(figsize=(12, 6))
        plt.plot(rounds, prices, marker='o', linestyle='-', color='green', label='Proposed Bid/Ask Price')
        plt.plot(rounds, ir_bounds, color='red', linestyle='--', label='Individually Rational Value/Cost')
        
        plt.xlabel('Round')
        plt.ylabel('Price')
        plt.title(f'Agent {self.id} {"Bid" if self.preference_schedule.is_buyer else "Ask"} Price Series')
        plt.legend()
        plt.grid(True)
        plt.show()

    def calculate_individual_surplus(self) -> float:
        if self.preference_schedule.is_buyer:
            goods_utility = sum(self.preference_schedule.get_value(q) for q in range(1, self.allocation.goods + 1))
            surplus = goods_utility - (self.allocation.initial_cash - self.allocation.cash)
        else:
            goods_cost = sum(self.preference_schedule.get_value(q) for q in range(1, self.allocation.initial_goods - self.allocation.goods + 1))
            surplus = self.allocation.cash - self.allocation.initial_cash - goods_cost
        
        logger.info(f"Agent {self.id} individual surplus: {surplus:.2f}")
        return surplus


def run_simulation(agent: ZIAgent, num_rounds: int):
    global trade_counter  # Use the global counter
    for round_num in range(num_rounds):
        logger.info(f"Starting round {round_num + 1} for Agent {agent.id}")
        order = agent.generate_bid()
        if order:
            global trade_counter
            trade = Trade(
                trade_id=trade_counter,  # Assign the current trade counter
                buyer_id=order.agent_id if order.is_buy else -1,
                seller_id=-1 if order.is_buy else order.agent_id,
                quantity=order.quantity,
                price=order.price,
                buyer_value=order.base_value if order.is_buy else 0.0,
                seller_cost=order.base_cost if not order.is_buy else 0.0,
                round=round_num + 1
            )
            trade_counter += 1  # Increment the counter after creating a trade
            
            agent.finalize_trade(trade)
            agent.respond_to_order(order, accepted=True)
        else:
            agent.respond_to_order(order, accepted=False)
        logger.info(f"Finished round {round_num + 1} for Agent {agent.id}")

if __name__ == "__main__":
    # Test PreferenceSchedule
    logger.info("Testing PreferenceSchedule:")
    buyer_schedule = PreferenceSchedule.generate(is_buyer=True, num_units=10, base_value=100)
    logger.info("Buyer Schedule:")
    logger.info(f"Initial Endowment: {buyer_schedule.initial_endowment}")
    for q, v in buyer_schedule.values.items():
        logger.info(f"Quantity: {q}, Value: {v:.2f}")
    buyer_schedule.plot_schedule()

    seller_schedule = PreferenceSchedule.generate(is_buyer=False, num_units=10, base_value=50)
    logger.info("\nSeller Schedule:")
    logger.info(f"Initial Endowment: {seller_schedule.initial_endowment}")
    for q, v in seller_schedule.values.items():
        logger.info(f"Quantity: {q}, Cost: {v:.2f}")
    seller_schedule.plot_schedule()

    # Test ZIAgent and trades
    logger.info("\nTesting ZIAgent and trades:")
    buyer_agent = ZIAgent.generate(agent_id=1, is_buyer=True, num_units=100, base_value=100, max_relative_spread=0.5)
    seller_agent = ZIAgent.generate(agent_id=2, is_buyer=False, num_units=100, base_value=50, max_relative_spread=0.5)
    
    num_rounds = 100

    # Run simulations
    run_simulation(buyer_agent, num_rounds)
    run_simulation(seller_agent, num_rounds)
    
    # Plot results
    buyer_agent.plot_order_history()
    seller_agent.plot_order_history()

    # Print final allocations and surpluses
    logger.info("\nFinal Allocations and Surpluses:")
    logger.info(f"Buyer Agent:")
    logger.info(f"  Final Cash: {buyer_agent.allocation.cash:.2f}")
    logger.info(f"  Final Goods: {buyer_agent.allocation.goods}")
    logger.info(f"  Surplus: {buyer_agent.calculate_individual_surplus():.2f}")

    logger.info(f"\nSeller Agent:")
    logger.info(f"  Final Cash: {seller_agent.allocation.cash:.2f}")
    logger.info(f"  Final Goods: {seller_agent.allocation.goods}")