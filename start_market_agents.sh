#!/bin/bash
set -e

DASHBOARD_PORT=8000
POSTGRES_PORT=5433
STORAGE_PORT=8001
GROUPCHAT_PORT=8002

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

check_api_health() {
    local port=$1
    local retries=3
    local wait_time=2
    
    for i in $(seq 1 $retries); do
        if curl -s "http://localhost:$port/health" > /dev/null; then
            return 0
        fi
        echo "Attempt $i: Waiting for API on port $port to become healthy..."
        sleep $wait_time
    done
    return 1
}

open_dashboard() {
    local port=$1
    local retries=5
    local wait_time=2
    
    echo "Waiting for dashboard to be ready..."
    for i in $(seq 1 $retries); do
        if curl -s "http://localhost:$port" > /dev/null; then
            echo "Opening dashboard in your default browser..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                open "http://localhost:$port" || python -m webbrowser "http://localhost:$port"
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                # Linux
                xdg-open "http://localhost:$port" || sensible-browser "http://localhost:$port" || \
                python -m webbrowser "http://localhost:$port"
            else
                # Windows
                start "http://localhost:$port" || python -m webbrowser "http://localhost:$port"
            fi
            return 0
        fi
        echo "Attempt $i: Waiting for dashboard to become available..."
        sleep $wait_time
    done
    echo "Warning: Could not open dashboard automatically. Please open http://localhost:$port in your browser manually."
    return 1
}

# Print MarketAgents ASCII art
echo "MarketAgents Swarm Initializing..."
python -c "from market_agents.orchestrators.logger_utils import print_ascii_art; print_ascii_art()"

declare -a SERVICE_PIDS

# Start Dashboard API if not already running
if check_port $DASHBOARD_PORT; then
    echo "Dashboard API is already running on port $DASHBOARD_PORT"
else
    echo "Starting MarketAgents Dashboard API..."
    python market_agents/agents/db/dashboard/dashboard.py &
    SERVICE_PIDS+=($!)
    sleep 2
fi

# Check and start PostgreSQL
if check_port $POSTGRES_PORT; then
    echo "PostgreSQL is already running on port $POSTGRES_PORT"
else
    echo "Starting PostgreSQL..."
    docker-compose -f market_agents/agents/db/docker-compose.yaml up -d
    echo "Waiting for PostgreSQL to start..."
    sleep 5
fi

# Start Agent Storage API if not already running
if check_port $STORAGE_PORT; then
    echo "Agent Storage API is already running on port $STORAGE_PORT"
else
    echo "Starting Agent Storage API..."
    python market_agents/memory/agent_storage/agent_storage_api.py &
    SERVICE_PIDS+=($!)
    
    if ! check_api_health $STORAGE_PORT; then
        echo "Failed to start Agent Storage API"
        exit 1
    fi
fi

# Start Group Chat API if not already running
if check_port $GROUPCHAT_PORT; then
    echo "Group Chat API is already running on port $GROUPCHAT_PORT"
else
    echo "Starting Group Chat API..."
    python market_agents/orchestrators/group_chat/groupchat_api.py &
    SERVICE_PIDS+=($!)
    
    if ! check_api_health $GROUPCHAT_PORT; then
        echo "Failed to start Group Chat API"
        exit 1
    fi
fi

# Trap SIGINT and SIGTERM signals
cleanup() {
    echo "Shutting down services..."
    for pid in "${SERVICE_PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid
            echo "Stopped service with PID $pid"
        fi
    done
    
    # Stop Docker containers if we started them
    if [ -f "market_agents/agents/db/docker-compose.yaml" ]; then
        echo "Stopping Docker containers..."
        docker-compose -f market_agents/agents/db/docker-compose.yaml down
    fi
    
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "All services are running. Opening dashboard..."
open_dashboard $DASHBOARD_PORT &

echo "Press Ctrl+C to stop all services."
wait