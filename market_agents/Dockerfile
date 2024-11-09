FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/app

# Copy the entire project
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set Python path
ENV PYTHONPATH=/usr/src/app

# Make sure we can find the market_agents package
RUN pip install -e .