import logging
from typing import List, Dict, Any, Optional
from market_agents.economics.econ_models import AgentWallet, BaseHoldings, BaseWallet, Portfolio
from pydantic import BaseModel, Field
from uuid import UUID

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EconomicAgent(BaseModel):
    id: Optional[UUID] = None
    rewards: List[float] = Field(default_factory=list)
    total_reward: float = 0.0
    wallet: Optional[BaseWallet] = None
    holdings: Optional[BaseHoldings] = None
    generate_wallet: bool = Field(default=False, exclude=True)

    def __init__(self, initial_holdings: Optional[Dict[str, float]] = None, **data):
        super().__init__(**data)
        if self.wallet is not None or self.generate_wallet:
            if self.wallet is None:
                self.wallet = AgentWallet(chain="ethereum")
            self.wallet.ensure_valid_wallet()
            if self.holdings is None:
                self.holdings = Portfolio(token_balances=initial_holdings or {})
            else:
                if initial_holdings and isinstance(self.holdings, Portfolio):
                    self.holdings.token_balances.update(initial_holdings)
        else:
            self.wallet = None
            self.holdings = None

    def add_reward(self, reward: float) -> None:
        self.rewards.append(reward)
        self.total_reward += reward
        logger.info(f"[EconomicAgent] {self.id} +reward={reward}, total={self.total_reward}")

    def add_transaction(self, tx: Dict[str, Any]) -> None:
        if not self.holdings:
            logger.warning(f"[EconomicAgent] No holdings => cannot record transaction: {tx}")
            return
        self.holdings.record_transaction(tx)
    
    def get_token_balance(self, symbol: str) -> float:
        """Get the balance of a specific token. Returns 0.0 if no holdings or token not found."""
        if not self.holdings:
            logger.warning(f"[EconomicAgent] No holdings => cannot get balance: symbol={symbol}")
            return 0.0
            
        if hasattr(self.holdings, "get_token_balance"):
            return self.holdings.get_token_balance(symbol)
        else:
            logger.warning(
                f"[EconomicAgent] Holdings type {type(self.holdings).__name__} does not support get_token_balance"
            )
            return 0.0

    def adjust_token_balance(self, symbol: str, delta: float) -> None:
        if not self.holdings:
            logger.warning(
                f"[EconomicAgent] No holdings => cannot adjust balance: symbol={symbol}, delta={delta}"
            )
            return
        
        if hasattr(self.holdings, "adjust_token_balance"):
            self.holdings.adjust_token_balance(symbol, delta)
        else:
            logger.warning(
                f"[EconomicAgent] Holdings type {type(self.holdings).__name__} does not support adjust_token_balance"
            )

    def get_portfolio_value(
        self, 
        price_feeds: Optional[Dict[str, float]] = None, 
        default_price: float = 0.0
    ) -> float:
        if not self.holdings:
            return 0.0
        return self.holdings.get_total_value(price_feeds, default_price)

    # --- Wallet / Chain usage stubs ---
    def sign_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.wallet:
            logger.warning("[EconomicAgent] No wallet => cannot sign transaction.")
            return {}
        return self.wallet.sign_transaction(tx_data)
    
    def serialize(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'rewards': self.rewards,
            'total_reward': self.total_reward,
            'wallet': self.wallet.dict() if self.wallet else None,
            'holdings': self.holdings.dict() if self.holdings else None
        }

    def __str__(self):
        w_str = "No wallet"
        if self.wallet:
            pk_str = (
                self.wallet.private_key[:10] + "..."
            ) if self.wallet.private_key else "None"
            w_str = f"chain={self.wallet.chain}, address={self.wallet.address}, pk={pk_str}"

        h_str = "No holdings"
        if self.holdings:
            if isinstance(self.holdings, Portfolio):
                h_str = f"tokens={self.holdings.token_balances}, tx_count={len(self.holdings.transaction_history)}"
            elif isinstance(self.holdings, MyCustomHoldings):
                h_str = f"points={self.holdings.points}"
            else:
                h_str = f"custom holdings type: {type(self.holdings).__name__}"

        return (
            f"EconomicAgent(id={self.id}, total_reward={self.total_reward})\n"
            f"Wallet: {w_str}\n"
            f"Holdings: {h_str}\n"
            f"Rewards: {self.rewards}"
        )

if __name__ == "__main__":
    # Agent with NO wallet -> no holdings
    agent_none = EconomicAgent()
    agent_none.add_reward(10.0)
    agent_none.adjust_token_balance("ETH", 2.0)
    agent_none.add_transaction({"type": "deposit", "symbol": "ETH", "amount": 2.0})
    print("\n-- Agent with NO WALLET --")
    print(agent_none)

    # Agent with auto-generated wallet -> auto-creates a Portfolio
    agent_auto = EconomicAgent(generate_wallet=True)
    agent_auto.add_reward(20.0)
    agent_auto.adjust_token_balance("ETH", 2.0)
    agent_auto.add_transaction({"type": "faucet", "symbol": "ETH", "amount": 2.0})
    # sign a transaction
    signed_tx = agent_auto.sign_transaction({"action": "transfer", "amount": 50, "symbol": "ETH"})
    print("\n-- Agent with AUTO wallet --")
    print(agent_auto)
    print("Signed TX =>", signed_tx)

    # Agent with user-supplied wallet -> needs holdings (auto if not given)
    custom_wallet = AgentWallet(chain="ethereum", address="0xMyAddr", private_key="0xMyPrivKey")
    agent_with_wallet = EconomicAgent(wallet=custom_wallet)
    agent_with_wallet.adjust_token_balance("USDC", 500.0)
    agent_with_wallet.add_transaction({"type": "deposit", "symbol": "USDC", "amount": 500.0})
    print("\n-- Agent with user-supplied WALLET --")
    print(agent_with_wallet)

    # Provide your own custom holdings
    class MyCustomHoldings(BaseHoldings):
        """Example user-defined holdings with a single 'points' field."""
        points: float = 0.0

        def record_transaction(self, tx: Dict[str, Any]) -> None:
            logger.info(f"[MyCustomHoldings] transaction={tx} (just logging, no effect)")

        def get_total_value(
            self, price_feeds: Optional[Dict[str, float]] = None, default_price: float = 0.0
        ) -> float:
            return self.points

        def add_points(self, amt: float) -> None:
            self.points += amt

    # we can pass both a wallet and a custom holdings
    custom_holdings = MyCustomHoldings(points=100)
    agent_custom = EconomicAgent(wallet=custom_wallet, holdings=custom_holdings)
    agent_custom.add_transaction({"type": "test_tx"})

    if isinstance(agent_custom.holdings, MyCustomHoldings):
        agent_custom.holdings.add_points(50)

    print("\n-- Agent with custom holdings + user-supplied wallet --")
    print(agent_custom)
    val_custom = agent_custom.get_portfolio_value()
    print(f"Custom holdings value => {val_custom}")