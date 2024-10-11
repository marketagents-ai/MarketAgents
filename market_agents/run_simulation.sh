#!/bin/bash

# Script to run market simulation with parallel orchestration and time tracking

# File name for the Python script
PYTHON_SCRIPT="market_agents/orchestrator_parallel_with_db.py"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: $PYTHON_SCRIPT not found!"
    exit 1
fi

# Get the start time
start_time=$(date +%s)

# Run the Python script and print its output in real-time
echo "Starting market simulation with parallel orchestration..."
python3 "$PYTHON_SCRIPT" 2>&1 | tee simulation_output.log
exit_code=${PIPESTATUS[0]}

# Get the end time
end_time=$(date +%s)

# Calculate the duration
duration=$((end_time - start_time))

# Print the results
echo "----------------------------------------"
echo "Simulation completed with exit code: $exit_code"
echo "Total execution time: $duration seconds"
echo "----------------------------------------"
echo "Full output has been saved to simulation_output.log"
echo "----------------------------------------"

# You can add more post-processing or logging here if needed

exit $exit_code