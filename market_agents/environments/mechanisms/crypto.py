# crypto_market.py

from datetime import datetime
import logging
import random
from typing import Any, List, Dict, Type, Optional, Tuple
import uuid
from pydantic import BaseModel, Field, field_validator, ConfigDict
from market_agents.environments.environment import (
    EnvironmentHistory, Mechanism, LocalAction, GlobalAction, LocalObservation, GlobalObservation,
    EnvironmentStep, ActionSpace, ObservationSpace, MultiAgentEnvironment
)
from market_agents.memecoin_orchestrators.crypto_models import OrderType, MarketAction, Trade
from market_agents.memecoin_orchestrators.crypto_agent import CryptoEconomicAgent
from agent_evm_interface.agent_evm_interface import EthereumInterface
logger = logging.getLogger(__name__)


class MarketSummary(BaseModel):
    """Summary of market activity for multiple tokens"""
    trades_count: int = Field(default=0, description="Total number of trades across all tokens")
    token_summaries: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Summary statistics for each token"
    )

class CryptoMarketAction(LocalAction):
    action: MarketAction

    @field_validator('action')
    def validate_action(cls, v):
        if v.order_type in [OrderType.BUY, OrderType.SELL]:
            if v.quantity <= 0:
                raise ValueError("Quantity must be positive for buy and sell orders")
            if v.price <= 0:
                raise ValueError("Price must be positive for buy and sell orders")
        return v

    @classmethod
    def sample(cls, agent_id: str) -> 'CryptoMarketAction':
        order_type = random.choice(list(OrderType))
        if order_type == OrderType.HOLD:
            action = MarketAction(order_type=order_type)
        else:
            random_price = random.uniform(0.01, 1.0)
            random_quantity = random.randint(1, 1000)
            action = MarketAction(
                order_type=order_type,
                price=random_price,
                quantity=random_quantity,
                token="USDC")
        return cls(agent_id=agent_id, action=action)

    @classmethod
    def action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class GlobalCryptoMarketAction(GlobalAction):
    actions: Dict[str, CryptoMarketAction]


class CryptoMarketObservation(BaseModel):
    """Observation of the crypto market state for an agent"""
    trades: List[Trade] = Field(
        default_factory=list, 
        description="List of trades the agent participated in"
    )
    market_summary: MarketSummary = Field(
        default_factory=MarketSummary, 
        description="Summary of market activity"
    )
    current_prices: Dict[str, float] = Field(
        default_factory=dict,
        description="Current prices for each supported token"
    )
    portfolio_value: float = Field(
        default=0.0, 
        description="Total value of the agent's portfolio in USDC"
    )
    eth_balance: int = Field(
        default=0, 
        description="Agent's ETH balance in wei"
    )
    token_balances: Dict[str, float] = Field(
        default_factory=dict,
        description="Map of token symbol to balance"
    )
    price_histories: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Historical prices for each supported token"
    )


class CryptoMarketLocalObservation(BaseModel):
    """Local observation for an individual agent in the crypto market"""
    agent_id: str
    observation: CryptoMarketObservation


class CryptoMarketGlobalObservation(GlobalObservation):
    observations: Dict[str, CryptoMarketLocalObservation]
    all_trades: List[Trade] = Field(default_factory=list, description="All trades executed in this round")
    market_summary: MarketSummary = Field(default_factory=MarketSummary, description="Summary of market activity")
    current_prices: Dict[str, float] = Field(default_factory=dict, description="Current market prices")
    price_histories: Dict[str, List[float]] = Field(default_factory=dict, description="Historical prices")

class CryptoMarketActionSpace(ActionSpace):
    allowed_actions: List[Type[LocalAction]] = [CryptoMarketAction]

    @classmethod
    def get_action_schema(cls) -> Dict[str, Any]:
        return MarketAction.model_json_schema()


class CryptoMarketObservationSpace(ObservationSpace):
    allowed_observations: List[Type[LocalObservation]] = [CryptoMarketLocalObservation]


class CryptoMarketMechanism(Mechanism):
    max_rounds: int = Field(default=100, description="Maximum number of trading rounds")
    current_round: int = Field(default=0, description="Current round number")
    trades: List[Trade] = Field(default_factory=list, description="List of executed trades")
    tokens: List[str] = Field(default=["DOGE"], description="List of supported tokens")
    current_prices: Dict[str, float] = Field(
        default_factory=lambda: {"DOGE": 0.1},
        description="Current market prices for each token"
    )
    price_histories: Dict[str, List[float]] = Field(
        default_factory=lambda: {"DOGE": [0.1]},
        description="Price history for each token"
    )
    
    sequential: bool = Field(default=False, description="Whether the mechanism is sequential")
    agent_registry: Dict[str, Any] = Field(default_factory=dict, description="Registry of agents")
    ethereum_interface: EthereumInterface = Field(
        default_factory=EthereumInterface,
        description="Ethereum Interface"
    )
    token_addresses: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of token symbols to addresses"
    )
    orderbook_address: str = Field(default="", description="Orderbook contract address")
    minter_private_key: str = Field(default="", description="Private key of the minter account")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def register_agent(self, agent_id: str, agent: CryptoEconomicAgent):
        """Register an agent with the mechanism."""
        if not isinstance(agent, CryptoEconomicAgent):
            raise ValueError(f"Agent must be a CryptoEconomicAgent, got {type(agent)}")
        
        self.agent_registry[str(agent_id)] = agent
        logger.info(f"Registered agent {agent_id} with address {agent.ethereum_address}")

    def setup(self):
        """Initialize token addresses, blockchain parameters, and current prices."""
        self.token_addresses = self.ethereum_interface.testnet_data['token_addresses']
        self.orderbook_address = self.ethereum_interface.testnet_data['orderbook_address']
        self.minter_private_key = self.ethereum_interface.accounts[0]['private_key']
        
        # Initialize prices for all supported tokens
        quote_address = self.ethereum_interface.get_token_address('USDC')
        
        for token in self.tokens:
            try:
                token_address = self.ethereum_interface.get_token_address(token)
                if not token_address:
                    logger.warning(f"Address not found for token {token}")
                    continue
                    
                pair_info = self.ethereum_interface.get_pair_info(
                    token_address,
                    quote_address
                )
                
                base_unit_price = pair_info['token0_price_in_token1']
                self.current_prices[token] = base_unit_price / 1e18
                if self.current_prices[token] == 0:
                    self.current_prices[token] = 0.1
                    
                # Initialize price history
                if token not in self.price_histories:
                    self.price_histories[token] = [self.current_prices[token]]
                    
            except Exception as e:
                logger.warning(f"Failed to get initial price for {token}: {str(e)}. Using default price of 0.1")
                self.current_prices[token] = 0.1
                self.price_histories[token] = [0.1]

    def step(self, action: GlobalCryptoMarketAction) -> EnvironmentStep:
        """Execute one step in the mechanism"""
        self.current_round += 1

        # Process actions and collect new trades
        new_trades = self._process_actions(action.actions)
        
        # Update prices based on new trades
        self._update_price(new_trades)
        
        # Create market summary and observations
        market_summary = self._create_market_summary(new_trades)
        observations = self._create_observations(market_summary)
        
        # Store trades in mechanism history
        self.trades.extend(new_trades)

        # Check if simulation is done
        done = self.current_round >= self.max_rounds

        return EnvironmentStep(
            global_observation=CryptoMarketGlobalObservation(
                observations=observations,
                all_trades=new_trades,
                market_summary=market_summary,
                current_prices=self.current_prices.copy(),
                price_histories=self.price_histories.copy()
            ),
            done=done,
            current_round=self.current_round,  # Pass the current round
            info={
                "market_prices": self.current_prices
            }
        )
    
    def _process_actions(self, actions: Dict[str, MarketAction]) -> List[Trade]:
        """Process all market actions and return list of executed trades."""
        trades = []
        
        for agent_id, market_action in actions.items():
            try:
                agent = self.agent_registry.get(agent_id)
                if not agent:
                    logger.error(f"Agent {agent_id} not found in registry")
                    continue

                if market_action.action.order_type == OrderType.BUY:
                    trade = self._execute_buy(agent, market_action.action)
                    if trade:
                        trades.append(trade)
                        self.trades.append(trade)
                        
                elif market_action.action.order_type == OrderType.SELL:
                    trade = self._execute_sell(agent, market_action.action)
                    if trade:
                        trades.append(trade)
                        self.trades.append(trade)
                        
                # HOLD orders require no execution
                
            except Exception as e:
                logger.error(f"Error processing action for agent {agent_id}: {str(e)}")
                logger.error("Exception details:", exc_info=True)
                continue

        return trades
    
    def _create_market_summary(self, trades: List[Trade]) -> MarketSummary:
        """Create market summary from trades, supporting multiple tokens"""
        if not trades:
            # Create empty summary with current prices for all tokens
            token_summaries = {}
            for token in self.tokens:
                current_price = self.current_prices.get(token, 0.1)
                token_summaries[token] = {
                    'trades_count': 0,
                    'average_price': current_price,
                    'total_volume': 0,
                    'price_range': (current_price, current_price)
                }
            return MarketSummary(
                trades_count=0,
                token_summaries=token_summaries
            )

        # Group trades by token
        trades_by_token = {}
        for trade in trades:
            if trade.coin not in trades_by_token:
                trades_by_token[trade.coin] = []
            trades_by_token[trade.coin].append(trade)

        # Calculate summary for each token
        token_summaries = {}
        for token in self.tokens:
            token_trades = trades_by_token.get(token, [])
            if token_trades:
                prices = [t.price for t in token_trades]
                volumes = [t.quantity for t in token_trades]
                token_summaries[token] = {
                    'trades_count': len(token_trades),
                    'average_price': sum(prices) / len(prices),
                    'total_volume': sum(volumes),
                    'price_range': (min(prices), max(prices))
                }
            else:
                current_price = self.current_prices.get(token, 0.1)
                token_summaries[token] = {
                    'trades_count': 0,
                    'average_price': current_price,
                    'total_volume': 0,
                    'price_range': (current_price, current_price)
                }

        # Create overall summary
        return MarketSummary(
            trades_count=len(trades),
            token_summaries=token_summaries
        )

    def _create_observations(self, market_summary: MarketSummary) -> Dict[str, CryptoMarketLocalObservation]:
        """Create observations for all agents, including multi-token balances"""
        observations = {}
        
        for agent_id, agent in self.agent_registry.items():
            # Get balances for all supported tokens
            token_balances = {}
            portfolio_value = 0.0
            
            # Get USDC balance first
            usdc_address = self.ethereum_interface.get_token_address('USDC')
            usdc_info = self.ethereum_interface.get_erc20_info(usdc_address)
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                usdc_address
            ) / (10 ** usdc_info['decimals'])
            token_balances['USDC'] = usdc_balance
            portfolio_value += usdc_balance

            # Get balances for all trading tokens
            for token in self.tokens:
                token_address = self.ethereum_interface.get_token_address(token)
                if not token_address:
                    continue
                    
                token_info = self.ethereum_interface.get_erc20_info(token_address)
                balance = self.ethereum_interface.get_erc20_balance(
                    agent.ethereum_address,
                    token_address
                ) / (10 ** token_info['decimals'])
                
                current_price = self.current_prices.get(token, 0)
                token_balances[token] = balance
                portfolio_value += balance * current_price

            base_observation = CryptoMarketObservation(
                trades=self.trades,
                market_summary=market_summary,
                current_prices=self.current_prices.copy(),
                portfolio_value=portfolio_value,
                eth_balance=self.ethereum_interface.get_eth_balance(agent.ethereum_address),
                token_balances=token_balances,
                price_histories=self.price_histories.copy()
            )

            observations[agent_id] = CryptoMarketLocalObservation(
                agent_id=agent_id,
                observation=base_observation
            )

        return observations

    def _update_price(self, trades: List[Trade]) -> None:
        """Update prices for all tokens based on recent trades"""
        if not trades:
            return

        # Group trades by token
        trades_by_token = {}
        for trade in trades:
            if trade.coin not in trades_by_token:
                trades_by_token[trade.coin] = []
            trades_by_token[trade.coin].append(trade)

        # Update prices for each token
        for token, token_trades in trades_by_token.items():
            if token not in self.current_prices:
                self.current_prices[token] = 0.1
                self.price_histories[token] = [0.1]

            # Calculate volume-weighted average price
            total_volume = sum(t.quantity for t in token_trades)
            if total_volume > 0:
                vwap = sum(t.price * t.quantity for t in token_trades) / total_volume
                self.current_prices[token] = vwap
                self.price_histories[token].append(vwap)

    def _execute_p2p_trade(self, buyer: CryptoEconomicAgent, seller: CryptoEconomicAgent, 
                          price: float, quantity: int) -> None:
        """Execute a peer-to-peer trade between two agents."""
        # Get token addresses
        token_address = self.ethereum_interface.get_token_address(self.coin_name)
        usdc_address = self.ethereum_interface.get_token_address('USDC')
        
        # Get decimals
        token_decimals = self.ethereum_interface.get_erc20_info(token_address)['decimals']
        usdc_decimals = self.ethereum_interface.get_erc20_info(usdc_address)['decimals']
        
        # Convert amounts to proper decimals
        usdc_amount = int(price * quantity * (10 ** usdc_decimals))
        token_amount = int(quantity * (10 ** token_decimals))

        # Verify balances
        if not self._verify_balances(buyer, seller, usdc_amount, token_amount, usdc_address, token_address):
            raise ValueError("Insufficient balance for trade")

        # Execute the transfers
        self._transfer_tokens(buyer, seller, usdc_amount, token_amount, usdc_address, token_address)

    def _verify_balances(self, buyer: CryptoEconomicAgent, seller: CryptoEconomicAgent,
                    usdc_amount: int, token_amount: int,
                    usdc_address: str, token_address: str, token: str) -> bool:
        """Verify that both parties have sufficient balances for the trade."""
        try:
            # Check buyer's USDC balance
            buyer_usdc_balance = self.ethereum_interface.get_erc20_balance(
                buyer.ethereum_address,
                usdc_address
            )
            if buyer_usdc_balance < usdc_amount:
                logger.error(f"Buyer {buyer.id} has insufficient USDC balance. " +
                           f"Has: {buyer_usdc_balance}, Needs: {usdc_amount}")
                return False

            # Check seller's token balance
            seller_token_balance = self.ethereum_interface.get_erc20_balance(
                seller.ethereum_address,
                token_address
            )
            if seller_token_balance < token_amount:
                logger.error(f"Seller {seller.id} has insufficient {self.coin_name} balance. " +
                           f"Has: {seller_token_balance}, Needs: {token_amount}")
                return False

            # Check allowances
            buyer_usdc_allowance = self.ethereum_interface.get_erc20_allowance(
                owner=buyer.ethereum_address,
                spender=self.orderbook_address,
                contract_address=usdc_address
            )
            if buyer_usdc_allowance < usdc_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=usdc_amount,
                    contract_address=usdc_address,
                    private_key=buyer.private_key
                )
                logger.info(f"Buyer {buyer.id} approved {usdc_amount} USDC. TxHash: {tx_hash}")

            seller_token_allowance = self.ethereum_interface.get_erc20_allowance(
                owner=seller.ethereum_address,
                spender=self.orderbook_address,
                contract_address=token_address
            )
            if seller_token_allowance < token_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=token_amount,
                    contract_address=token_address,
                    private_key=seller.private_key
                )
                logger.info(f"Seller {seller.id} approved {token_amount} {token}. TxHash: {tx_hash}")

            return True

        except Exception as e:
            logger.error(f"Error verifying balances: {str(e)}")
            return False

    def _transfer_tokens(self, buyer, seller, usdc_amount, token_amount, usdc_address, token_address):
        try:
            # Transfer USDC from buyer to seller
            tx_hash = self.ethereum_interface.send_erc20(
                to=seller.ethereum_address,
                amount=usdc_amount,
                contract_address=usdc_address,
                private_key=buyer.private_key
            )
            logger.info(f"USDC transfer tx hash: {tx_hash}")

            # Transfer tokens from seller to buyer
            tx_hash = self.ethereum_interface.send_erc20(
                to=buyer.ethereum_address,
                amount=token_amount,
                contract_address=token_address,
                private_key=seller.private_key
            )
            logger.info(f"Token transfer tx hash: {tx_hash}")
        except Exception as e:
            logger.error(f"Error executing transfers: {str(e)}")
            raise

    def _execute_buy(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent buys tokens using USDC."""
        source_token_address = self.ethereum_interface.get_token_address('USDC')
        target_token_address = self.ethereum_interface.get_token_address(market_action.token)
        
        if not target_token_address:
            logger.error(f"Token address not found for {market_action.token}")
            return None

        try:
            # Get token decimals
            usdc_decimals = self.ethereum_interface.get_erc20_info(source_token_address)['decimals']
            token_decimals = self.ethereum_interface.get_erc20_info(target_token_address)['decimals']

            # Convert amounts to proper decimals
            usdc_amount = int(market_action.price * market_action.quantity * (10 ** usdc_decimals))
            token_amount = int(market_action.quantity * (10 ** token_decimals))

            # Check USDC balance
            usdc_balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                source_token_address
            )
            if usdc_balance < usdc_amount:
                logger.error(f"Agent {agent.id} has insufficient USDC balance. " + 
                            f"Has: {usdc_balance / 10**usdc_decimals}, " +
                            f"Needs: {market_action.price * market_action.quantity}")
                return None

            # Check and update allowance if needed
            allowance = self.ethereum_interface.get_erc20_allowance(
                owner=agent.ethereum_address,
                spender=self.orderbook_address,
                contract_address=source_token_address
            )
            
            if allowance < usdc_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=usdc_amount,
                    contract_address=source_token_address,
                    private_key=agent.private_key
                )
                logger.info(f"Agent {agent.id} approved {usdc_amount/(10**usdc_decimals)} USDC. TxHash: {tx_hash}")

            # Execute the swap
            tx_hash = self.ethereum_interface.swap(
                source_token_address=source_token_address,
                source_token_amount=usdc_amount,
                target_token_address=target_token_address,
                private_key=agent.private_key
            )
            logger.info(f"Agent {agent.id} executed buy {market_action.quantity} {market_action.token} " +
                    f"for {usdc_amount/(10**usdc_decimals)} USDC. TxHash: {tx_hash}")

            return Trade(
                trade_id=len(self.trades),
                buyer_id=agent.id,
                seller_id="MARKET_MAKER",
                price=market_action.price,
                bid_price=market_action.price,
                ask_price=market_action.price,
                quantity=market_action.quantity,
                coin=market_action.token,
                tx_hash=tx_hash,
                timestamp=datetime.now(),
                action_type="BUY"
            )

        except Exception as e:
            logger.error(f"Error executing buy for agent {agent.id}: {str(e)}")
            return None

    def _execute_sell(self, agent: CryptoEconomicAgent, market_action: MarketAction) -> Optional[Trade]:
        """Agent sells tokens for USDC."""
        source_token_address = self.ethereum_interface.get_token_address(market_action.token)
        target_token_address = self.ethereum_interface.get_token_address('USDC')
        
        if not source_token_address:
            logger.error(f"Token address not found for {market_action.token}")
            return None

        try:
            # Get token decimals
            token_decimals = self.ethereum_interface.get_erc20_info(source_token_address)['decimals']
            usdc_decimals = self.ethereum_interface.get_erc20_info(target_token_address)['decimals']

            # Convert quantity to proper decimals
            token_amount = int(market_action.quantity * (10 ** token_decimals))

            # Check token balance
            token_balance = self.ethereum_interface.get_erc20_balance(
                agent.ethereum_address,
                source_token_address
            )
            if token_balance < token_amount:
                logger.error(f"Agent {agent.id} has insufficient {market_action.token} balance. " +
                            f"Has: {token_balance/(10**token_decimals)}, " +
                            f"Needs: {market_action.quantity}")
                return None

            # Check and update allowance if needed
            allowance = self.ethereum_interface.get_erc20_allowance(
                owner=agent.ethereum_address,
                spender=self.orderbook_address,
                contract_address=source_token_address
            )
            
            if allowance < token_amount:
                tx_hash = self.ethereum_interface.approve_erc20(
                    spender=self.orderbook_address,
                    amount=token_amount,
                    contract_address=source_token_address,
                    private_key=agent.private_key
                )
                logger.info(f"Agent {agent.id} approved {market_action.quantity} {market_action.token}. TxHash: {tx_hash}")

            # Execute the swap
            tx_hash = self.ethereum_interface.swap(
                source_token_address=source_token_address,
                source_token_amount=token_amount,
                target_token_address=target_token_address,
                private_key=agent.private_key
            )
            logger.info(f"Agent {agent.id} executed sell {market_action.quantity} {market_action.token} " +
                    f"for {market_action.price * market_action.quantity} USDC. TxHash: {tx_hash}")

            return Trade(
                trade_id=len(self.trades),
                buyer_id="MARKET_MAKER",
                seller_id=agent.id,
                price=market_action.price,
                bid_price=market_action.price,
                ask_price=market_action.price,
                quantity=market_action.quantity,
                coin=market_action.token,
                tx_hash=tx_hash,
                timestamp=datetime.now(),
                action_type="SELL"
            )

        except Exception as e:
            logger.error(f"Error executing sell for agent {agent.id}: {str(e)}")
            return None
    
    def _convert_to_decimal_price(self, base_unit_price: int, decimals: int = 18) -> float:
        return base_unit_price / (10 ** decimals)

    def _convert_to_base_units(self, decimal_price: float, decimals: int = 18) -> int:
        return int(decimal_price * (10 ** decimals))

    def get_global_state(self) -> Dict[str, Any]:
        """Get the current global state of the mechanism"""
        return {
            "trades": self.trades,
            "current_prices": self.current_prices,
            "price_histories": self.price_histories,
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "tokens": self.tokens,
            "market_summary": self._create_market_summary(self.trades)
        }

    def reset(self) -> None:
        """Reset the mechanism state"""
        self.current_round = 0
        self.trades = []
        
        # Reset prices for all tokens
        for token in self.tokens:
            self.current_prices[token] = 0.1
            self.price_histories[token] = [0.1]

class CryptoMarket(MultiAgentEnvironment):
    name: str = Field(default="Crypto Market", description="Name of the crypto market")
    action_space: CryptoMarketActionSpace = Field(default_factory=CryptoMarketActionSpace, description="Action space of the crypto market")
    observation_space: CryptoMarketObservationSpace = Field(default_factory=CryptoMarketObservationSpace, description="Observation space of the crypto market")
    mechanism: CryptoMarketMechanism = Field(default_factory=CryptoMarketMechanism, description="Mechanism of the crypto market")
    agents: Dict[str, CryptoEconomicAgent] = Field(default_factory=dict, description="Dictionary of agents in the market")

    def __init__(self, agents: Dict[str, CryptoEconomicAgent], **kwargs):
        super().__init__(**kwargs)
        self.agents = agents
        self.mechanism.agent_registry = {}
        
        # Fix: Properly register agents with string IDs
        for agent_id, agent in agents.items():
            str_id = str(agent_id)
            if hasattr(agent, 'economic_agent'):
                # If agent is wrapped, register the economic agent
                self.mechanism.agent_registry[str_id] = agent.economic_agent
            else:
                # If agent is direct CryptoEconomicAgent instance
                self.mechanism.agent_registry[str_id] = agent
            
            # Ensure agent has its ID set
            if hasattr(agent, 'economic_agent'):
                agent.economic_agent.id = str_id
            else:
                agent.id = str_id

        # Setup mechanism after registry is populated
        self.mechanism.setup()

    def reset(self) -> GlobalObservation:
        self.current_step = 0
        self.history = EnvironmentHistory()
        self.mechanism.reset()
        observations = self.mechanism._create_observations(MarketSummary())

        return CryptoMarketGlobalObservation(
            observations=observations,
            all_trades=[],
            market_summary=MarketSummary(),
            current_price=self.mechanism.current_price,
            price_history=self.mechanism.price_history.copy()
        )

    def step(self, actions: GlobalAction) -> EnvironmentStep:
        step_result = self.mechanism.step(actions)
        self.current_step += 1
        self.update_history(actions, step_result)
        return step_result

    def render(self):
        pass
