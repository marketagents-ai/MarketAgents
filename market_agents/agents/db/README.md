# Market Simulation Database

## Overview

This directory contains the database setup and testing scripts for the Market Agents project. It uses PostgreSQL with the pgvector extension for vector similarity search capabilities.

## Table of Contents

1. [Database Schema](#database-schema)
2. [Table Descriptions](#table-descriptions)
3. [Relationships](#relationships)
4. [Setup Instructions](#setup-instructions)
5. [Development Guidelines](#development-guidelines)
6. [Query Examples](#query-examples)
7. [Performance Considerations](#performance-considerations)
8. [Backup and Recovery](#backup-and-recovery)
9. [Docker Setup and Database Initialization](#docker-setup-and-database-initialization)
10. [Setting up pgvector](#setting-up-pgvector)
11. [Testing pgvector](#testing-pgvector)

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
10. memory_embeddings

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

### memory_embeddings
Stores vector embeddings of agent memories for similarity search.

## Relationships

- An **agent** has one **preference_schedule** and one **allocation**.
- An **agent** can have many **orders** and be involved in many **trades**.
- An **environment** has many **agents** through **environment_agents**.
- An **auction** has many **trades**.
- An **agent** can have many **interactions**.
- An **agent** can have many **memory_embeddings**.

## Setup Instructions

### Prerequisites

- PostgreSQL 14 or higher
- pgvector extension
- Python 3.8 or higher
- psycopg2
- numpy
## Docker Setup and Database Initialization

Follow these steps to set up and run the PostgreSQL database with pgvector using Docker:

1. **Prerequisites**
   - Docker
   - Docker Compose
   - Python 3.9 or higher

2. **Setup Steps**

   a. Navigate to the project root directory containing the `docker-compose.yaml` file.

   b. Ensure the following files are present in the `market_agents/agents/db/` directory:
      - `Dockerfile`
      - `run_setup.sh`
      - `setup_pgvector.sh`
      - `setup_database.py`

   c. Make sure the shell scripts are executable:
      ```
      chmod +x market_agents/agents/db/run_setup.sh
      chmod +x market_agents/agents/db/setup_pgvector.sh
      ```

   d. Build and start the Docker containers:
      ```
      docker-compose up --build -d
      ```
      This command will:
      - Build the custom PostgreSQL image with pgvector (version 0.5.1)
      - Start the database container
      - Start the application container
      - Run the `run_setup.sh` script, which waits for the database to be ready and then runs `setup_database.py`

   e. Wait for the containers to fully start and the setup to complete. You can check the logs to monitor the progress:
      ```
      docker-compose logs -f app
      ```

   f. Once the setup is complete, run the pgvector setup script:
      ```
      docker-compose exec app ./setup_pgvector.sh
      ```
      This script will ensure pgvector is properly installed and configured.

3. **Verification**
   To verify that the database is set up correctly:
   ```
   docker-compose exec db psql -U db_user -d market_simulation -c "\dt"
   ```
   This should list all the tables in your database.

4. **Usage**
   The database is now ready for use with the following connection details:
   - Host: localhost
   - Port: 5433
   - Database: market_simulation
   - Username: db_user
   - Password: db_pwd@123

5. **Management Commands**
   - To stop the containers:
     ```
     docker-compose down
     ```
   - To reset the database completely:
     ```
     docker-compose down -v
     docker-compose up --build -d
     ```
     The setup scripts will run automatically on container startup.

6. **Troubleshooting**
   - If you encounter connection issues, verify that the Docker containers are running:
     ```
     docker-compose ps
     ```
   - To view the logs of the Docker containers:
     ```
     docker-compose logs
     ```
   - If you need to access the database container directly:
     ```
     docker-compose exec db bash
     ```

This setup ensures that:
- The PostgreSQL database (version 14) with pgvector (version 0.5.1) is running in a Docker container.
- The database schema is initialized using the `setup_database.py` script.
- pgvector is properly set up using the `setup_pgvector.sh` script.

Note: The `run_setup.sh` script automatically runs on container startup, handling the database initialization. The `setup_pgvector.sh` script needs to be run manually after the containers are up to ensure pgvector is properly configured.

Make sure to keep your `setup_database.py` and `setup_pgvector.sh` scripts up to date with your latest schema and configuration requirements.

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

## Setting up pgvector

This project uses pgvector for vector similarity search. To set it up:

1. Ensure you have Homebrew and PostgreSQL 14 installed.
2. Run the setup script: `./setup_pgvector.sh`
3. After the script completes, connect to your database and run:
   ```sql
   CREATE EXTENSION vector;
   ```

If you encounter any issues, please refer to the [pgvector documentation](https://github.com/pgvector/pgvector).

## Testing pgvector

To test if pgvector is working correctly:

1. Ensure your database is set up and running.

2. Set the following environment variables (or update the values in `test_pgvector.py`):
   - `DB_NAME` (default: 'market_simulation')
   - `DB_USER` (default: 'db_user')
   - `DB_PASSWORD` (default: 'db_pwd@123')
   - `DB_HOST` (default: 'localhost')
   - `DB_PORT` (default: '5433')

3. Run the test script:

   ```sh
   python test_pgvector.py
   ```

   This script will:
   - Connect to the database
   - Create the vector extension if it doesn't exist
   - Perform a vector similarity search on the `memory_embeddings` table
   - Print the top 3 similar vectors

4. If successful, you should see output similar to:

   ```
   Top 3 similar vectors:
   ID: 1, Memory: {'text': 'Test memory 0', 'timestamp': '2023-04-01T12:00:00Z'}, Distance: 0.123456
   ID: 2, Memory: {'text': 'Test memory 1', 'timestamp': '2023-04-01T12:00:00Z'}, Distance: 0.234567
   ID: 3, Memory: {'text': 'Test memory 2', 'timestamp': '2023-04-01T12:00:00Z'}, Distance: 0.345678
   ```

## Troubleshooting

- If you encounter errors related to the vector extension, ensure it's properly installed and created in your database:
  ```sql
  CREATE EXTENSION vector;
  ```

- For connection issues, verify your environment variables match your PostgreSQL setup.

- If the `test_pgvector.py` script fails, ensure you have test data in the `memory_embeddings` table. You can run `setup_database.py` again to insert test data.

For any other issues, please refer to the main project README or open an issue on the project repository.