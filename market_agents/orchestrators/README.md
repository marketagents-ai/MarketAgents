# Orchestrators

A sophisticated multi-agent simulation framework for modeling economic markets with AI agents that can participate in both auctions and group discussions.

## Overview

This framework simulates a market environment where AI agents can engage in two primary activities:
1. Double auctions for trading goods
2. Group discussions about market conditions and strategies

The simulation uses LLMs (Large Language Models) to power agent decision-making, allowing for complex, human-like interactions and strategic behavior.

## Key Components

### Core Configuration (`orchestrator_config.yaml`)
- Defines simulation parameters including number of agents, rounds, and environment settings
- Configures agent economics (initial cash, goods, base values)
- Specifies LLM configurations for agent intelligence
- Sets up environment-specific parameters for auctions and group chats

### Meta Orchestrator (`meta_orchestrator.py`)
- Main simulation controller that manages the overall flow
- Coordinates between different environments (auction and group chat)
- Handles agent generation and initialization
- Manages simulation rounds and data collection

### Environment Orchestrators
1. **Auction Orchestrator** (`auction_orchestrator.py`)
   - Manages double auction markets
   - Handles bid/ask matching
   - Tracks trades and calculates economic surplus
   - Processes agent actions and market outcomes

2. **Group Chat Orchestrator** (`groupchat_orchestrator.py`)
   - Facilitates discussions between agents
   - Manages cohort formation and topic selection
   - Coordinates multi-round conversations
   - Tracks message history and discussion outcomes

### Data Management
- **Database Integration** (`insert_simulation_data.py`)
  - Stores comprehensive simulation data
  - Tracks agent actions, trades, and conversations
  - Maintains economic metrics and performance data
  - Supports post-simulation analysis

### Logging System (`logger_utils.py`)
- Provides detailed logging of simulation events
- Uses color-coded output for better readability
- Tracks agent actions, market events, and system states
- Supports debugging and analysis

## Key Features

### Economic Modeling
- Double auction market mechanism
- Dynamic price discovery
- Supply and demand simulation
- Surplus calculation and efficiency metrics

### Agent Intelligence
- LLM-powered decision making
- Adaptive strategies
- Market perception and analysis
- Learning from experience

### Social Interaction
- Group discussions
- Topic-based conversations
- Dynamic cohort formation
- Multi-round dialogues

### Data Collection
- Comprehensive transaction logging
- Agent state tracking
- Market efficiency metrics
- Conversation analysis

## Setup and Configuration

1. **Environment Setup**
   ```bash
   # Set up database configuration in .env file
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5433
   ```

2. **Basic Configuration**
   ```yaml
   # Key settings in orchestrator_config.yaml
   num_agents: 10
   max_rounds: 2
   environment_order:
    - group_chat
    - auction
   ```

3. **Agent Configuration**
   ```yaml
   agent_config:
     buyer_initial_cash: 1000.0
     seller_initial_goods: 10
     good_name: "strawberry"
     noise_factor: 0.05
   ```

## Running the Simulation

```python
# Basic usage
python meta_orchestrator.py

# With specific environments
python meta_orchestrator.py --environments group_chat auction
```

## Project Structure

```
market_agents/
├── orchestrator_config.yaml
├── meta_orchestrator.py
├── logger_utils.py
├── insert_simulation_data.py
├── config.py
├── base_orchestrator.py
├── auction_orchestrator.py
└── groupchat_orchestrator.py
```

## Dependencies

- Python 3.8+
- PostgreSQL
- Required Python packages:
  - pydantic
  - asyncio
  - psycopg2
  - colorama
  - pyfiglet
  - yaml

## Best Practices

1. **Configuration Management**
   - Keep all configuration in `orchestrator_config.yaml`
   - Use environment variables for sensitive data
   - Maintain separate configs for development/production

2. **Data Handling**
   - Regular database backups
   - Proper error handling for data insertion
   - Comprehensive logging of all operations

3. **Performance Optimization**
   - Use parallel processing where possible
   - Implement caching for LLM responses
   - Monitor resource usage
