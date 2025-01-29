from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from eth_account import Account

class BaseWallet(BaseModel, ABC):
    """
    Abstract wallet class for a chain identity:
      - chain: e.g. 'ethereum', 'solana'
      - address, private_key: user-supplied or auto-generated
    """
    chain: Optional[str] = None
    address: Optional[str] = None
    private_key: Optional[str] = None

    @abstractmethod
    def ensure_valid_wallet(self) -> None:
        """
        Subclasses handle auto-generation or validation.
        """
        pass

    @abstractmethod
    def sign_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Subclasses handle real chain transaction signing.
        """
        pass


class AgentWallet(BaseWallet):
    """
    Agent wallet that auto-generates an Ethereum address/private_key if
    chain='ethereum' and none is provided.
    """
    def ensure_valid_wallet(self) -> None:
        if self.chain and self.chain.lower() == "ethereum":
            if not self.address or not self.private_key:
                logger.info("[AgentWallet] Auto-generating Ethereum wallet.")
                self._generate_ethereum_wallet()

    def _generate_ethereum_wallet(self):
        Account.enable_unaudited_hdwallet_features()
        acct = Account.create()
        self.address = acct.address
        self.private_key = acct.key.hex()
        logger.info(f"[AgentWallet] Generated Ethereum wallet: {self.address}")

    def sign_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"[AgentWallet] sign_transaction with tx_data={tx_data}")
        return {"signed_tx": "dummy_signature", "tx_data": tx_data}


class BaseHoldings(BaseModel, ABC):
    """
    Abstract holdings class
    """
    @abstractmethod
    def record_transaction(self, tx: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def get_total_value(
        self, price_feeds: Optional[Dict[str, float]] = None, default_price: float = 0.0
    ) -> float:
        pass


class Portfolio(BaseHoldings):
    """
    A multi-token dictionary-based portfolio:
      token_balances = {'ETH': 2.0, 'USDC': 500.0, ...}
      transaction_history logs local deposit/trade events, etc.
    """
    token_balances: Dict[str, float] = Field(default_factory=dict)
    transaction_history: List[Dict[str, Any]] = Field(default_factory=list)

    def record_transaction(self, tx: Dict[str, Any]) -> None:
        self.transaction_history.append(tx)
        logger.info(f"[Portfolio] Recorded transaction: {tx}")

    def get_token_balance(self, symbol: str) -> float:
        """Get the balance of a specific token. Returns 0.0 if token not found."""
        return self.token_balances.get(symbol, 0.0)

    def get_total_value(
        self, 
        price_feeds: Optional[Dict[str, float]] = None, 
        default_price: float = 0.0
    ) -> float:
        if not price_feeds:
            return 0.0
        total = 0.0
        for symbol, qty in self.token_balances.items():
            price = price_feeds.get(symbol, default_price)
            total += qty * price
        return total

    def adjust_token_balance(self, symbol: str, delta: float) -> None:
        old_balance = self.token_balances.get(symbol, 0.0)
        new_balance = old_balance + delta
        self.token_balances[symbol] = new_balance
        logger.info(f"[Portfolio] {symbol} balance changed: {old_balance} -> {new_balance}")
