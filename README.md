# Market Agents

## Overview

This project provides agent framework for creating market agents with economic incentive. The agents have utilities for parallel AI inference and prompt caching using large language models (LLMs).

<p align="center">
  <img src="assets/marketagents.jpeg" alt="Image Alt Text" width="80%" height="80%">
</p>

## Installation

To install the `market_agents` package in editable mode, follow these steps:

1. Clone the repository:

    ```sh
    git clone https://github.com/marketagents-ai/MarketAgents.git
    cd MarketAgents
    ```

2. Install the package in editable mode:

    ```sh
    pip install -e .
    ```

3. Install the required dependencies:

    ```sh
    pip install -r requirements.txt
    ```

4. Follow the README.md (just navigate to market_agents/agents/db)
    ```sh
    cat ./market_agents/agents/db/README.md
    ```

5. Make a copy of .env.example
    ```sh
    cp .env.example .env
    ```

    *Note: Setup API keys and more...*

7. Edit the ```market_agents/orchestrator_config.yaml``` accoding to your configuration

## Running Examples

You can run the `run_simulation.sh` as follows:

```sh
python3 market_agents/run_simulation.sh
```

