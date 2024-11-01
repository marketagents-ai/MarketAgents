# Stock Market Trading Simulation Project

A sophisticated multi-agent stock market simulation system implementing realistic trading mechanics, agent-based learning, and group discussions. The system uses advanced parallel processing and AI-driven decision making to simulate complex market dynamics.

## Detailed Architecture

### Core Components

#### Stock Models (`stock_models.py`)
- **Base Models**
  - `SavableBaseModel`: Extended Pydantic model with JSON persistence
  - `OrderType`: Enum for Buy/Sell/Hold operations
  - `MarketAction`: Validates and processes trading actions
  - `StockOrder`: Extends MarketAction with agent-specific logic
  - `Trade`: Represents completed transactions with validation
  
- **Portfolio Management**
  - `Position`: Tracks quantity and purchase price for holdings
  - `Stock`: Manages positions with FIFO (First-In-First-Out) logic
  - `Portfolio`: Handles cash and stock management
    - Computed properties for quick access
    - Real-time portfolio valuation
    - Position adjustment methods
  
- **State Management**
  - `Endowment`: Tracks initial and current portfolio states
  - Transaction history maintenance
  - Portfolio simulation capabilities
  - Automatic state validation

#### Stock Agent (`stock_agent.py`)
- **Trading Logic**
  - Dynamic order generation based on market conditions
  - Risk-adjusted position sizing
  - Profit/loss tracking
  - Portfolio rebalancing logic

- **Risk Management**
  - Customizable risk aversion parameters
  - Position limits enforcement
  - Cash management rules
  - Stop-loss implementation

- **Performance Analytics**
  - Real-time P&L calculation
  - Portfolio value tracking
  - Performance metrics computation
  - Trading efficiency analysis

### System Architecture

#### Database Layer (`setup_stock_database.py`)
- **Tables Structure**
  ```sql
  agents (
      id UUID PRIMARY KEY,
      role VARCHAR(10),
      is_llm BOOLEAN,
      max_iter INTEGER,
      llm_config JSONB
  )

  trades (
      id SERIAL PRIMARY KEY,
      buyer_id UUID,
      seller_id UUID,
      quantity INTEGER,
      price DECIMAL(15,2),
      round INTEGER
  )

  orders (
      id SERIAL PRIMARY KEY,
      agent_id UUID,
      order_type VARCHAR(10),
      quantity INTEGER,
      price DECIMAL(15,2)
  )
  ```

- **Extensions**
  - pgvector for embedding storage
  - JSONB for flexible data storage
  - UUID handling for unique identifiers

#### Orchestration System

##### Stock Market Orchestrator (`orchestrator_stock_market.py`)
- **Parallel Processing**
  - Asynchronous action execution
  - Batch order processing
  - Concurrent agent updates
  
- **Market Mechanics**
  - Order matching algorithm
  - Price discovery mechanism
  - Volume tracking
  - Market depth maintenance

- **State Management**
  - Transaction logging
  - State persistence
  - Rollback capabilities
  - Checkpoint creation

##### Group Chat Orchestrator (`orchestrator_group_chat.py`)
- **Discussion Management**
  - Topic generation and tracking
  - Message routing system
  - Conversation flow control
  - Sentiment analysis

- **Integration Features**
  - Market data broadcasting
  - Trading signal generation
  - Strategy sharing
  - Performance discussion

### Advanced Features

#### AI Integration
- **LLM Integration**
  ```yaml
  llm_configs:
    - name: "gpt-4o-mini-low-temp"
      client: "openai"
      model: "gpt-4o-mini"
      temperature: 0.2
      max_tokens: 2048
      use_cache: true
  ```

- **Trading Strategies**
  - Momentum-based trading
  - Mean reversion
  - Trend following
  - Sentiment-driven decisions

#### Data Processing Pipeline
- **Real-time Processing**
  - Order flow analysis
  - Market impact calculation
  - Volume profile tracking
  - Price trend detection

- **Historical Analysis**
  - Performance backtesting
  - Strategy evaluation
  - Risk assessment
  - Pattern recognition

## Configuration Details

### Agent Configuration
```yaml
agent_config:
  initial_cash_min: 500.0
  initial_cash_max: 2000.0
  initial_stocks_min: 5
  initial_stocks_max: 20
  risk_aversion: 0.5
  expected_return: 0.05
  use_llm: False
  stock_symbol: 'AAPL'
  max_relative_spread: 0.05
```

### Environment Settings
```yaml
environment_configs:
  stock_market:
    name: "AAPL Stock Market"
    address: "aapl_stock_market"
    max_rounds: 100
    stock_symbol: "AAPL"
```

## Implementation Guide

### Setting Up the Environment
1. Database Initialization
   ```bash
   # Create database and tables
   python setup_stock_database.py
   
   # Verify installation
   psql -d market_simulation -c "\dt"
   ```

2. Configuration Setup
   ```bash
   # Copy and modify config template
   cp orchestrator_config_stock.yaml.template orchestrator_config_stock.yaml
   
   # Set environment variables
   export DB_USER=your_user
   export DB_PASSWORD=your_password
   ```

3. Run Simulation
   ```bash
   # Start simulation with monitoring
   ./run_stock_simulation.sh --monitor
   
   # Run in background
   nohup ./run_stock_simulation.sh &
   ```

### Data Management
- **Backup Strategy**
  ```bash
  # Automated backup script
  pg_dump market_simulation > backup_$(date +%Y%m%d).sql
  ```

- **Data Analysis**
  ```sql
  -- Example queries
  SELECT 
    DATE(created_at) as trade_date,
    COUNT(*) as trade_count,
    AVG(price) as avg_price
  FROM trades
  GROUP BY DATE(created_at)
  ORDER BY trade_date;
  ```



## Notes

### Known Limitations
- Single stock trading only
- Limited order types
- Simplified price discovery
- Basic risk management

### Future Enhancements
- Multi-asset trading
- Advanced order types
- Machine learning integration
- Real-time market data feeds