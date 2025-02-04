#!/bin/bash
set -e

# Print ASCII art from logger_utils
echo "MarketAgents Swarm Initializing..."
python -c "from market_agents.orchestrators.logger_utils import print_ascii_art; print_ascii_art()"

# Start the Dashboard API
echo "Starting MarketAgents Dashboard API..."
python market_agents/agents/db/dashboard/dashboard.py &

# Build and start the Docker containers for PostgreSQL
docker-compose -f market_agents/agents/db/docker-compose.yaml up -d

echo "PostgreSQL is starting up. Waiting a few seconds before launching the APIs..."
sleep 5

# Start the Agent Storage API via python, which internally uses uvicorn
echo "Starting Agent Storage API..."
python market_agents/memory/agent_storage/agent_storage_api.py &

# Start the Group Chat API via python, which internally uses uvicorn
echo "Starting Group Chat API..."
python market_agents/orchestrators/group_chat/groupchat_api.py &

# Keep the script running so all APIs continue to serve requests
wait