from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid
from eth_account import Account
import logging

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class EconomicAgent(BaseModel):
    """Base economic agent class with crypto wallet capabilities"""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    ethereum_address: str = Field(default="")
    private_key: str = Field(default="")
    balance: float = Field(default=0.0)
    rewards: List[float] = Field(default_factory=list)
    total_reward: float = Field(default=0.0)
    transaction_history: List[Dict[str, Any]] = Field(default_factory=list)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.ethereum_address:
            self._generate_ethereum_account()

    def _generate_ethereum_account(self):
        """Generate new Ethereum account with address and private key"""
        try:
            Account.enable_unaudited_hdwallet_features()
            acct = Account.create()
            self.ethereum_address = acct.address
            self.private_key = acct.key.hex()
            logger.info(f"Generated new Ethereum account with address: {self.ethereum_address}")
        except Exception as e:
            logger.error(f"Error generating Ethereum account: {str(e)}")
            raise

    def add_reward(self, reward: float):
        """Add a reward and update total rewards"""
        self.rewards.append(reward)
        self.total_reward += reward
        logger.info(f"Added reward {reward} to agent {self.id}. Total reward: {self.total_reward}")

    def update_balance(self, new_balance: float):
        """Update agent's balance"""
        self.balance = new_balance
        logger.info(f"Updated balance to {new_balance} for agent {self.id}")

    def add_transaction(self, transaction: Dict[str, Any]):
        """Record a transaction in history"""
        self.transaction_history.append(transaction)
        logger.info(f"Added transaction to history for agent {self.id}")

if __name__ == "__main__":
    test_agents = [EconomicAgent() for _ in range(3)]
    
    for i, agent in enumerate(test_agents, 1):
        print(f"\nAgent {i}:")
        print(f"Address: {agent.ethereum_address}")
        print(f"Private key (first 10 chars): {agent.private_key[:10]}...")
        print(f"Agent ID: {agent.id}")