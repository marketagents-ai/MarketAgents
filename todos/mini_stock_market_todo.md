# Mini Stock Market TODO

## Classes

### StockMarketMechanism (inherits from Mechanism)
- Attributes:
  - `stock_symbol: str`
  - `current_price: float`
  - `order_book: Dict[str, List[Order]]`  # 'buy' and 'sell' keys
  - `trade_history: List[Trade]`
  - `price_history: List[float]`
  - `trading_fee: float`
  - `max_price_change: float`  # Limit on price changes per round

- Methods:
  - `step(action: GlobalStockMarketAction) -> EnvironmentStep`
  - `match_orders() -> List[Trade]`
  - `update_price()`
  - `calculate_rewards() -> Dict[str, float]`
  - `get_global_state() -> Dict[str, Any]`
  - `reset()`

### Order (BaseModel)
- Attributes:
  - `agent_id: str`
  - `order_type: str`  # 'buy' or 'sell'
  - `quantity: int`
  - `price: float`
  - `timestamp: datetime`

### StockMarketAction (inherits from LocalAction)
- Attributes:
  - `action: Order`

- Methods:
  - `sample(agent_id: str) -> 'StockMarketAction'`

### GlobalStockMarketAction (inherits from GlobalAction)
- Attributes:
  - `actions: Dict[str, StockMarketAction]`

### StockMarketObservation (BaseModel)
- Attributes:
  - `current_price: float`
  - `agent_portfolio: Portfolio`
  - `recent_trades: List[Trade]`
  - `order_book_summary: Dict[str, List[Tuple[float, int]]]`  # Price levels and quantities
  - `market_sentiment: float`  # -1 to 1, bearish to bullish

### StockMarketLocalObservation (inherits from LocalObservation)
- Attributes:
  - `observation: StockMarketObservation`

### StockMarketGlobalObservation (inherits from GlobalObservation)
- Attributes:
  - `observations: Dict[str, StockMarketLocalObservation]`
  - `global_market_state: Dict[str, Any]`

### Portfolio (BaseModel)
- Attributes:
  - `cash: float`
  - `stocks: Dict[str, int]`  # symbol: quantity

### StockMarketActionSpace (inherits from ActionSpace)
- Attributes:
  - `allowed_actions: List[Type[LocalAction]] = [StockMarketAction]`

- Methods:
  - `get_action_schema() -> Dict[str, Any]`

### StockMarketObservationSpace (inherits from ObservationSpace)
- Attributes:
  - `allowed_observations: List[Type[LocalObservation]] = [StockMarketLocalObservation]`

### StockMarketEnvironment (inherits from MultiAgentEnvironment)
- Attributes:
  - `name: str = "Stock Market"`
  - `action_space: StockMarketActionSpace`
  - `observation_space: StockMarketObservationSpace`
  - `mechanism: StockMarketMechanism`

## TODO List

1. Implement `StockMarketMechanism` class:
   - [ ] Define all attributes
   - [ ] Implement `step` method
   - [ ] Implement `match_orders` method
   - [ ] Implement `update_price` method
   - [ ] Implement `calculate_rewards` method
   - [ ] Implement `get_global_state` method
   - [ ] Implement `reset` method

2. Implement `Order` class

3. Implement `StockMarketAction` class:
   - [ ] Define attributes
   - [ ] Implement `sample` method

4. Implement `GlobalStockMarketAction` class

5. Implement `StockMarketObservation` class

6. Implement `StockMarketLocalObservation` class

7. Implement `StockMarketGlobalObservation` class

8. Implement `Portfolio` class

9. Implement `StockMarketActionSpace` class:
   - [ ] Define attributes
   - [ ] Implement `get_action_schema` method

10. Implement `StockMarketObservationSpace` class

11. Implement `StockMarketEnvironment` class

12. Update `EconomicAgent` class:
    - [ ] Modify `endowment` to use `Portfolio` instead of `Endowment`
    - [ ] Add methods for portfolio management (buy, sell, calculate value)
    - [ ] Update strategy formation and reflection methods for stock trading

13. Update `MarketAgent` class:
    - [ ] Modify `perceive` method to handle stock market observations
    - [ ] Update `generate_action` method for stock trading actions
    - [ ] Adjust `reflect` method to consider stock market performance

14. Implement utility functions:
    - [ ] Calculate portfolio value
    - [ ] Compute trading fees
    - [ ] Generate market sentiment

15. Create test scenarios:
    - [ ] Basic order matching and price updates
    - [ ] Multiple agents with different strategies
    - [ ] Edge cases (e.g., market crashes, bubbles)

16. Implement reward calculation:
    - [ ] Consider changes in portfolio value
    - [ ] Account for realized and unrealized gains/losses
    - [ ] Factor in trading fees

17. Add market events:
    - [ ] Implement random news events that affect stock price
    - [ ] Create scheduled events (e.g., earnings reports)

18. Enhance realism:
    - [ ] Implement order types (market, limit, stop)
    - [ ] Add trading volume effects on price
    - [ ] Implement basic technical indicators

19. Documentation:
    - [ ] Write detailed docstrings for all classes and methods
    - [ ] Create usage examples and tutorials

20. Testing:
    - [ ] Write unit tests for all components
    - [ ] Perform integration testing of the entire system
    - [ ] Conduct performance testing for large-scale simulations
