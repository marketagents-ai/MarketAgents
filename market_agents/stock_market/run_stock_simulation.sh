#!/bin/bash

# Script to run stock market simulation with parallel orchestration, dashboard, and time tracking

# File names for the Python scripts
ORCHESTRATOR_SCRIPT="market_agents/stock_market/orchestrator_stock_market.py"
#ORCHESTRATOR_SCRIPT="market_agents/stock_market/orchestrator_group_chat.py"
DASHBOARD_SCRIPT="market_agents/agents/db/dashboard/dashboard.py"

# Check if the Python scripts exist
if [ ! -f "$ORCHESTRATOR_SCRIPT" ]; then
    echo "Error: $ORCHESTRATOR_SCRIPT not found!"
    exit 1
fi

if [ ! -f "$DASHBOARD_SCRIPT" ]; then
    echo "Error: $DASHBOARD_SCRIPT not found!"
    exit 1
fi

# Start the dashboard in the background
echo "Starting dashboard..."
python3 "$DASHBOARD_SCRIPT" &
DASHBOARD_PID=$!

# Give the dashboard a moment to start up
sleep 2

# Print dashboard access information
echo "Dashboard is running. Access it at http://localhost:8000"

# Get the start time
start_time=$(date +%s)

# Run the orchestrator script and print its output in real-time
echo "Starting stock market simulation with parallel orchestration..."
python3 "$ORCHESTRATOR_SCRIPT" 2>&1 | tee simulation_output.log
orchestrator_exit_code=${PIPESTATUS[0]}

# Get the end time for the orchestrator
end_time=$(date +%s)

# Calculate the duration
duration=$((end_time - start_time))

# Print the results
echo "----------------------------------------"
echo "Simulation completed with exit code: $orchestrator_exit_code"
echo "Total execution time: $duration seconds"
echo "----------------------------------------"
echo "Full output has been saved to simulation_output.log"
echo "----------------------------------------"

# Check if the orchestrator script ran successfully
if [ $orchestrator_exit_code -ne 0 ]; then
    echo "Error: Orchestrator script failed with exit code $orchestrator_exit_code"
else
    echo "Simulation completed successfully."
fi

echo "Dashboard is still running. Press Enter to stop the dashboard and exit."
read

# Stop the dashboard
kill $DASHBOARD_PID

exit $orchestrator_exit_code
