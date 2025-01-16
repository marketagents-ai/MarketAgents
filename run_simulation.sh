#!/bin/bash

# Script to run market simulation with parallel orchestration, dashboard, and time tracking

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# File names for the Python scripts
ORCHESTRATOR_SCRIPT="market_agents/orchestrators/meta_orchestrator.py"
DASHBOARD_SCRIPT="market_agents/agents/db/dashboard/dashboard.py"
GROUPCHAT_API_SCRIPT="market_agents/orchestrators/group_chat/groupchat_api.py"
DASHBOARD_PORT=8000
GROUPCHAT_API_PORT=8001

# Check if the Python scripts exist
for script in "$ORCHESTRATOR_SCRIPT" "$DASHBOARD_SCRIPT" "$GROUPCHAT_API_SCRIPT"; do
    if [ ! -f "$script" ]; then
        echo "Error: $script not found!"
        exit 1
    fi
done

check_api_health() {
    local port=$1
    local retries=3
    local wait_time=2
    
    for i in $(seq 1 $retries); do
        if curl -s "http://localhost:$port/health" > /dev/null; then
            return 0  # API is healthy
        fi
        echo "Attempt $i: Waiting for API to become healthy..."
        sleep $wait_time
    done
    return 1  # API failed health check
}

# Check and start GroupChat API if needed
if check_port $GROUPCHAT_API_PORT; then
    echo "GroupChat API is already running at http://localhost:$GROUPCHAT_API_PORT"
    GROUPCHAT_STARTED=false
else
    echo "Starting GroupChat API..."
    python3 "$GROUPCHAT_API_SCRIPT" &
    GROUPCHAT_PID=$!
    GROUPCHAT_STARTED=true

    # Check if API becomes healthy
    if ! check_api_health $GROUPCHAT_API_PORT; then
        echo "Failed to start GroupChat API. Killing process..."
        kill $GROUPCHAT_PID
        exit 1
    fi
    echo "GroupChat API is running at http://localhost:$GROUPCHAT_API_PORT"
fi

# Check and start dashboard if needed
if check_port $DASHBOARD_PORT; then
    echo "Dashboard is already running at http://localhost:$DASHBOARD_PORT"
    DASHBOARD_STARTED=false
else
    echo "Starting dashboard..."
    python3 "$DASHBOARD_SCRIPT" &
    DASHBOARD_PID=$!
    DASHBOARD_STARTED=true

    sleep 2
    echo "Dashboard is running at http://localhost:$DASHBOARD_PORT"
fi

# Get the start time
start_time=$(date +%s)

# Run the orchestrator script
echo "Starting market simulation with parallel orchestration..."
python3 "$ORCHESTRATOR_SCRIPT" 2>&1 | tee simulation_output.log

orchestrator_exit_code=${PIPESTATUS[0]}

# Get the end time and calculate duration
end_time=$(date +%s)
duration=$((end_time - start_time))

# Print results
echo "----------------------------------------"
echo "Simulation completed with exit code: $orchestrator_exit_code"
echo "Total execution time: $duration seconds"
echo "----------------------------------------"
echo "Full output has been saved to simulation_output.log"
echo "----------------------------------------"

if [ $orchestrator_exit_code -ne 0 ]; then
    echo "Error: Orchestrator script failed with exit code $orchestrator_exit_code"
else
    echo "Simulation completed successfully."
fi

# Cleanup services that we started
if [ "$DASHBOARD_STARTED" = true ] || [ "$GROUPCHAT_STARTED" = true ]; then
    echo "Press Enter to stop services and exit."
    read

    if [ "$DASHBOARD_STARTED" = true ]; then
        kill $DASHBOARD_PID
        echo "Dashboard stopped."
    fi

    if [ "$GROUPCHAT_STARTED" = true ]; then
        kill $GROUPCHAT_PID
        echo "GroupChat API stopped."
    fi
else
    echo "Services were already running and will continue running."
fi

exit $orchestrator_exit_code