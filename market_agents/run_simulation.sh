#!/bin/bash

# Default values
MAX_ROUNDS=5
LOG_LEVEL="INFO"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --max_rounds)
        MAX_ROUNDS="$2"
        shift # past argument
        shift # past value
        ;;
        --log_level)
        LOG_LEVEL="$2"
        shift # past argument
        shift # past value
        ;;
        *)
        echo "Unknown option: $1"
        exit 1
        ;;
    esac
done

# Run the simulation app
python simulation_app.py --max_rounds $MAX_ROUNDS --log_level $LOG_LEVEL