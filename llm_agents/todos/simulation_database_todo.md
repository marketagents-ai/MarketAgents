# Simulation Database Integration TODO

## 1. PostgreSQL with pgvector Extension Setup

- [ ] Install PostgreSQL and pgvector extension
- [ ] Create a new database for the simulation
- [ ] Enable pgvector extension in the database

## 2. Relational Database Design

### Agent State Table
- [ ] Design schema for agent state
  - Agent ID (primary key)
  - Type (buyer/seller)
  - Current cash
  - Current goods
  - Preference schedule (JSONB)
  - Last updated timestamp

### Market State Table
- [ ] Design schema for market state
  - Round number (primary key)
  - Current average price
  - Total trades
  - Total surplus extracted
  - Timestamp

### Order Book Table
- [ ] Design schema for order book
  - Order ID (primary key)
  - Agent ID (foreign key to Agent State)
  - Type (bid/ask)
  - Price
  - Quantity
  - Timestamp
  - Round number

### Trade Table
- [ ] Design schema for trades
  - Trade ID (primary key)
  - Buyer ID (foreign key to Agent State)
  - Seller ID (foreign key to Agent State)
  - Price
  - Quantity
  - Buyer surplus
  - Seller surplus
  - Round number
  - Timestamp

### Memory Embedding Table
- [ ] Design schema for memory embeddings
  - Memory ID (primary key)
  - Agent ID (foreign key to Agent State)
  - Embedding vector (using pgvector)
  - Original text
  - Timestamp
  - Round number

## 3. Database Connection and ORM Setup

- [ ] Set up SQLAlchemy as ORM
- [ ] Create database connection string
- [ ] Implement database connection pool

## 4. Data Insertion Methods

- [ ] Implement `insert_agent_state(agent: ZIAgent) -> None`
  - Serialize PreferenceSchedule to JSONB
  - Insert or update agent state in database
- [ ] Implement `insert_market_state(market_state: Dict) -> None`
- [ ] Implement `insert_order(order: Order) -> None`
- [ ] Implement `insert_trade(trade: Trade) -> None`
- [ ] Implement `insert_memory_embedding(agent_id: int, embedding: List[float], original_text: str, round_number: int) -> None`

## 5. Data Retrieval Methods

- [ ] Implement `get_agent_state(agent_id: int) -> Dict`
- [ ] Implement `get_market_state(round_number: int) -> Dict`
- [ ] Implement `get_order_book(round_number: int) -> List[Dict]`
- [ ] Implement `get_trades(round_number: int) -> List[Dict]`
- [ ] Implement `get_similar_memories(agent_id: int, query_embedding: List[float], limit: int) -> List[Dict]`

## 6. Integration with Existing Modules

- [ ] Modify ZIAgent class to use database methods for state persistence
- [ ] Update DoubleAuction class to log order book and trades in the database
- [ ] Integrate vector database queries into LLM agent decision-making process
- [ ] Modify Environment class to store and retrieve market state from the database

## 7. Batch Operations and Performance Optimization

- [ ] Implement batch insert methods for high-frequency data (e.g., orders, trades)
- [ ] Set up appropriate indexing for frequently queried columns
- [ ] Implement caching mechanism for frequently accessed data
- [ ] Profile database performance and optimize slow queries

## 8. Data Management and Maintenance

- [ ] Implement data retention policies (e.g., archiving old simulations)
- [ ] Set up regular backups for the database
- [ ] Create data export functionality for further analysis in external tools

## 9. Testing and Validation

- [ ] Develop unit tests for database interactions
- [ ] Create integration tests for the entire simulation with database components
- [ ] Validate data integrity and consistency across different modules

## 10. Documentation

- [ ] Document database schema and relationships
- [ ] Create user guide for running simulations with the new database integration
- [ ] Write API documentation for database interaction functions

## 11. Future Enhancements

- [ ] Investigate using PostgreSQL's time-series capabilities for high-resolution price and volume data
- [ ] Explore graph query capabilities in PostgreSQL for modeling complex agent relationships
- [ ] Investigate machine learning opportunities using the collected data for predictive modeling
