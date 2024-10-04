#!/bin/bash

# Set variables
#MODEL_NAME="NousResearch/Hermes-3-Llama-3.1-8B"
MODEL_NAME="microsoft/Phi-3.5-mini-instruct"
QUANTIZATION="fp8"
PORT=8000
VLLM_IMAGE="vllm/vllm-openai:latest"

# Function to read environment variables from .env file
read_env() {
    if [[ -f .env ]]; then
        export $(grep -v '^#' .env | xargs)
    fi
}

# Check if Docker is installed and accessible
if ! docker version &> /dev/null; then
    echo "Docker is not accessible. Please ensure Docker Desktop is running and properly configured with WSL2."
    exit 1
fi

# Read .env file if it exists
read_env

# Check if vLLM image exists, if not pull it
if ! docker image inspect $VLLM_IMAGE &> /dev/null; then
    echo "vLLM image not found. Pulling from Docker Hub..."
    docker pull $VLLM_IMAGE
    
    if [ $? -ne 0 ]; then
        echo "Failed to pull vLLM image. Please check your internet connection and Docker setup."
        exit 1
    fi
else
    echo "vLLM image found. Skipping pull step."
fi

# Prepare the Docker run command
DOCKER_CMD="docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p $PORT:8000 \
    --ipc=host \
    $VLLM_IMAGE \
    --model $MODEL_NAME \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --gpu_memory_utilization 0.8 \
    --api_key token-abc123 \
    --quantization $QUANTIZATION"


# Add Hugging Face token if available
if [[ -n "$HUGGING_FACE_HUB_TOKEN" ]]; then
    DOCKER_CMD="$DOCKER_CMD --env HUGGING_FACE_HUB_TOKEN=$HUGGING_FACE_HUB_TOKEN"
    echo "Using Hugging Face token from environment."
else
    echo "No Hugging Face token found. Proceeding without authentication."
fi

# Run the vLLM server with the specified model
echo "Starting vLLM server with $MODEL_NAME..."
eval $DOCKER_CMD

echo "vLLM server is running on http://localhost:$PORT"