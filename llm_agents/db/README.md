# Market Simulation Database

## Overview

This repository contains the database design and setup for a market simulation system. The database is designed to support a complex market environment with agents, auctions, trades, and various market activities.

## Table of Contents

1. [Database Schema](#database-schema)
2. [Table Descriptions](#table-descriptions)
3. [Relationships](#relationships)
4. [Setup Instructions](#setup-instructions)
5. [Development Guidelines](#development-guidelines)
6. [Query Examples](#query-examples)
7. [Performance Considerations](#performance-considerations)
8. [Backup and Recovery](#backup-and-recovery)

## Database Schema

The database consists of the following tables:

1. agents
2. preference_schedules
3. allocations
4. orders
5. trades
6. auctions
7. environments
8. environment_agents
9. interactions

## Table Descriptions

### agents
Stores information about each agent in the simulation, including their role and configuration.

### preference_schedules
Holds the preference data for each agent, including their valuation or cost schedules.

### allocations
Tracks the current allocation of goods and cash for each agent.

### orders
Records all orders (bids and asks) placed by agents during auctions.

### trades
Stores information about completed trades between agents.

### auctions
Represents individual auction sessions, including their current state and results.

### environments
Stores information about each simulation environment.

### environment_agents
Links environments to the agents participating in them.

### interactions
Logs interactions between agents and the language model, if applicable.

## Relationships

- An **agent** has one **preference_schedule** and one **allocation**.
- An **agent** can have many **orders** and be involved in many **trades**.
- An **environment** has many **agents** through **environment_agents**.
- An **auction** has many **trades**.
- An **agent** can have many **interactions**.

## Setup Instructions

1. Install PostgreSQL on your system if not already installed.
2. Create a new database for the market simulation project:
   ```
   createdb market_simulation
   ```
3. Run the SQL script to create the tables and set up the schema:
   ```
   psql -d market_simulation -f setup.sql
   ```
4. Configure your application to connect to this database.

## Development Guidelines

1. Always use prepared statements or ORM queries to prevent SQL injection.
2. Create indexes for frequently queried columns.
3. Use transactions for operations that modify multiple tables.
4. Implement proper error handling and logging for database operations.
5. Write unit tests for database interactions.
6. Document any schema changes in the project's change log.

## Query Examples

Here are some example queries for common operations:

1. Get all active buyers:
   ```sql
   SELECT * FROM agents WHERE role = 'buyer' AND id IN (SELECT agent_id FROM environment_agents);
   ```

2. Get the latest trade for a specific auction:
   ```sql
   SELECT * FROM trades WHERE auction_id = :auction_id ORDER BY created_at DESC LIMIT 1;
   ```

3. Calculate the total volume traded in an auction:
   ```sql
   SELECT SUM(quantity) FROM trades WHERE auction_id = :auction_id;
   ```

## Performance Considerations

- Use EXPLAIN ANALYZE to optimize complex queries.
- Consider partitioning large tables (e.g., trades, orders) if they grow significantly.
- Regularly update table statistics using ANALYZE.
- Monitor query performance and add indexes as needed.

## Backup and Recovery

1. Implement regular backups using pg_dump:
   ```
   pg_dump market_simulation > backup.sql
   ```

2. For point-in-time recovery, configure WAL archiving and use pg_basebackup for full backups.

3. Test your recovery process regularly to ensure data can be restored successfully.

## Docker Setup and Database Initialization

To set up and run the PostgreSQL database using Docker:

1. Ensure Docker and Docker Compose are installed on your system.

2. Navigate to the directory containing the `docker-compose.yaml` file.

3. Build and start the Docker containers:

   ```bash
   docker-compose up --build
   ```

   This command will:
   - Build the Docker images if they don't exist
   - Start the PostgreSQL database container
   - Start the application container, which will run the setup script

4. Wait for the setup to complete. You should see output indicating that the database and tables have been created successfully.

5. The database is now ready for use with the following connection details:
   - Host: localhost
   - Port: 5433
   - Database: market_simulation
   - Username: db_user
   - Password: db_pwd@123

6. To stop the containers, use:

   ```bash
   docker-compose down
   ```

7. If you need to reset the database completely, remove the volume:

   ```bash
   docker-compose down -v
   ```

   Then, rebuild and start the containers again using step 3.

These steps will set up your PostgreSQL database in a Docker container and initialize it with the necessary tables and data.

For any questions or issues, please open an issue in the repository or contact the database administrator.