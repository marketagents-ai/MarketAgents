# Agent Memory System Usage Instructions

This document provides instructions for using the updated Agent Memory System script, which integrates index search and CSV input for market history.

## Prerequisites

1. Python 3.7 or higher installed on your system.
2. Required Python packages: `asyncio`, `argparse`, `json`, `csv`, `enum`.
3. The `reptarINDEX.py` script and its dependencies.

## Setup

1. Ensure you have built an index using the `reptarINDEX.py` script. This will create an index file (e.g., `index.reptar`) that contains the document graph and embeddings.

2. Prepare a CSV file with market history data according to the appropriate template for your desired abstraction level (Generalist, High-Level, or Detailed).

## Command Line Arguments

The script accepts the following command-line arguments:

- `--level`: Abstraction level (choices: generalist, high_level, detailed; default: high_level)
- `--action`: Action to perform (choices: add, search, format_input, format_output; required)
- `--data`: JSON string containing data for the action
- `--index_file`: Path to the index file (required)
- `--csv_file`: Path to the CSV file containing market history (optional)

## Usage Examples

### 1. Adding a Memory

To add a new memory to the system:

```bash
python agent_memory_system.py --level high_level --action add --index_file index.reptar --data '{"event": "Market crash", "impact": "High"}'
```

### 2. Searching Memories

To search for memories based on a query:

```bash
python agent_memory_system.py --level high_level --action search --index_file index.reptar --data "market volatility"
```

### 3. Formatting Input Prompt

To generate an input prompt based on the current context and market history:

```bash
python agent_memory_system.py --level high_level --action format_input --index_file index.reptar --csv_file market_history.csv --data '{"additional_context": "Unexpected economic report released"}'
```

### 4. Formatting Output Prompt

To generate an output prompt based on an action and its result:

```bash
python agent_memory_system.py --level high_level --action format_output --index_file index.reptar --data '{"action": "Sell tech stocks", "result": "Profit realized"}'
```

## Abstraction Levels

The script supports three abstraction levels, each with different CSV templates and prompt structures:

1. Generalist: Focuses on overall system state and equilibrium.
2. High-Level: Considers market state, recent events, and key insights.
3. Detailed: Includes comprehensive market dynamics, economic indicators, trade analysis, agent interactions, and strategy performance.

Choose the appropriate level based on the complexity and detail required for your use case.

## CSV File Format

Ensure your CSV file follows the correct format for the chosen abstraction level:

### Generalist Level
```csv
timestamp,system_state
2024-08-31 00:00:00,Stable equilibrium with minor fluctuations
```

### High-Level
```csv
timestamp,market_state,recent_events,key_insights
2024-08-31 00:00:00,Bullish,Interest rate cut,Increased market liquidity expected
```

### Detailed Level
```csv
timestamp,market_dynamics,economic_indicators,trade_analysis,agent_interactions,strategy_performance
2024-08-31 00:00:00,High volatility,Inflation: 2.1%,EUR/USD correlation with oil prices,Increased algo trading,Momentum strategies outperforming
```

## Tips for Effective Use

1. Regularly update your index file to ensure the most recent information is available for searches.
2. Keep your market history CSV file up-to-date with the latest data for accurate context in input prompts.
3. When using the `format_input` action, provide additional context in the `--data` argument to supplement the information from the CSV file.
4. Experiment with different abstraction levels to find the right balance of detail for your specific use case.
5. Use the search functionality to retrieve relevant past experiences before making decisions or generating prompts.

## Troubleshooting

- If you encounter errors related to missing modules, ensure all required packages are installed.
- Verify that the paths to your index file and CSV file are correct.
- Check that your JSON data is properly formatted when using the `--data` argument.
- If searches are not returning expected results, review your index and consider rebuilding it with updated content.

For further assistance or to report issues, please contact the development team.

# Explanation of CSV Keys for Different Abstraction Levels

The Agent Memory System uses three levels of abstraction, each with its own set of keys in the corresponding CSV file. These levels allow for varying degrees of detail and complexity in representing the market state and agent context. Below is a detailed explanation of each level and its associated keys.

## 1. Generalist Level

CSV Keys: `timestamp, system_state`

This is the most abstract level, focusing on the overall state of the system.

- `timestamp`: The date and time of the recorded state.
- `system_state`: A high-level description of the entire system's condition.

Example:
```csv
timestamp,system_state
2024-08-31 00:00:00,Stable equilibrium with minor fluctuations in resource distribution
```

Use Case: This level is suitable for agents that need to make decisions based on broad, system-wide patterns without getting into specific market details. It's useful for high-level strategy planning or when dealing with complex systems where detailed information might be overwhelming or unnecessary.

## 2. High-Level

CSV Keys: `timestamp, market_state, recent_events, key_insights`

This level provides more specific information about the market, recent occurrences, and important observations.

- `timestamp`: The date and time of the recorded state.
- `market_state`: A general description of the current market condition (e.g., bullish, bearish, volatile).
- `recent_events`: Significant occurrences that may impact the market.
- `key_insights`: Important observations or conclusions drawn from market analysis.

Example:
```csv
timestamp,market_state,recent_events,key_insights
2024-08-31 00:00:00,Bullish,Interest rate cut announced,Increased market liquidity expected
```

Use Case: This level is appropriate for agents that need to make informed decisions based on overall market trends and recent developments. It's suitable for portfolio managers, economic advisors, or AI systems that need to understand the broader context of the market without delving into highly specific details.

## 3. Detailed Level

CSV Keys: `timestamp, market_dynamics, economic_indicators, trade_analysis, agent_interactions, strategy_performance`

This level provides the most granular and comprehensive view of the market and agent activities.

- `timestamp`: The date and time of the recorded state.
- `market_dynamics`: Specific patterns or behaviors observed in the market.
- `economic_indicators`: Quantitative measures of economic performance.
- `trade_analysis`: Insights derived from recent trading activities.
- `agent_interactions`: Observations about how different agents (human or AI) are behaving in the market.
- `strategy_performance`: Evaluation of how different trading or investment strategies are performing.

Example:
```csv
timestamp,market_dynamics,economic_indicators,trade_analysis,agent_interactions,strategy_performance
2024-08-31 00:00:00,High volatility in currency markets,Inflation rate: 2.1% Unemployment: 3.8%,Currency pair EUR/USD showing strong correlation with oil prices,Increased algorithmic trading activity detected,Momentum strategies outperforming value strategies
```

Use Case: This level is designed for sophisticated agents that require in-depth market analysis and a comprehensive understanding of various factors affecting the market. It's suitable for advanced trading systems, detailed economic modeling, or AI agents that need to make highly informed decisions based on a wide range of factors.

## Choosing the Right Level

The choice of abstraction level depends on several factors:

1. Complexity of the decision-making process: More complex decisions may require more detailed information.
2. Type of agent: Different agents (e.g., high-frequency traders vs. long-term investors) may need different levels of detail.
3. Computational resources: More detailed levels require more processing power and memory.
4. Time sensitivity: In fast-moving markets, the generalist or high-level views might be more appropriate for quick decision-making.
5. Scope of operation: Agents focusing on specific market segments might benefit from the detailed level, while those with a broader focus might prefer the high-level or generalist views.

By choosing the appropriate abstraction level, you can ensure that your agent has access to the most relevant information for its specific needs and operational context.

# CSV and Index Usage in Agent Memory System

The Python script uses both the CSV file and the created index to populate relevant fields. Here's a detailed explanation of how this works:

1. CSV Usage:

The `load_market_history` function reads the CSV file and extracts the most recent data:

```python
def load_market_history(csv_file: str, abstraction_level: AbstractionLevel) -> Dict[str, Any]:
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    
    if abstraction_level == AbstractionLevel.GENERALIST:
        return {
            "system_state": data[-1]["system_state"] if data else ""
        }
    elif abstraction_level == AbstractionLevel.HIGH_LEVEL:
        return {
            "market_state": data[-1]["market_state"] if data else "",
            "recent_events": data[-1]["recent_events"] if data else "",
            "key_insights": data[-1]["key_insights"] if data else ""
        }
    else:  # DETAILED
        return {
            "market_dynamics": data[-1]["market_dynamics"] if data else "",
            "economic_indicators": data[-1]["economic_indicators"] if data else "",
            "trade_analysis": data[-1]["trade_analysis"] if data else "",
            "agent_interactions": data[-1]["agent_interactions"] if data else "",
            "strategy_performance": data[-1]["strategy_performance"] if data else ""
        }
```

This function is called in the main script when formatting input:

```python
if args.csv_file:
    market_history = load_market_history(args.csv_file, AbstractionLevel(args.level))
    context.update(market_history)
```

2. Index Usage:

The index is loaded when initializing the MemorySystem:

```python
class MemorySystem:
    def __init__(self, abstraction_level: AbstractionLevel, index_file: str):
        self.abstraction_level = abstraction_level
        self.memories: List[Dict[str, Any]] = []
        self.graph, self.embeddings = load_graph(type('Args', (), {'input_file': index_file})())
```

The `search_memories` method uses the loaded graph to perform searches:

```python
async def search_memories(self, query: str) -> List[Dict[str, Any]]:
    results = await self.graph.search(query, self.embeddings)
    return [
        {
            "text": node.text,
            "score": score,
            "depth": depth,
            "is_cluster": node.is_cluster
        }
        for node, score, depth, _ in results
    ]
```

3. Combining CSV and Index Data:

In the `format_input_prompt` method, both CSV data and index search results are used:

```python
async def format_input_prompt(self, context: Dict[str, Any]) -> str:
    if self.abstraction_level == AbstractionLevel.GENERALIST:
        system_state = context.get('system_state', '')
        relevant_info = await self.search_memories(system_state)
        return f"""
        Role: Universal Agent
        Context: {system_state}
        Relevant Information: {relevant_info[:3]}
        Objective: Maintain equilibrium through continuous interaction
        Action: Analyze the current state and decide on the next action
        """
    elif self.abstraction_level == AbstractionLevel.HIGH_LEVEL:
        market_state = context.get('market_state', '')
        recent_events = context.get('recent_events', '')
        relevant_info = await self.search_memories(f"{market_state} {recent_events}")
        return f"""
        Role: Intelligent Economic Agent
        Market State: {market_state}
        Recent Events: {recent_events}
        Relevant Information: {relevant_info[:5]}
        Key Insights: {context.get('key_insights', '')}
        Objective: Maximize outcomes while managing risk
        Action: Analyze market conditions, consider potential outcomes, and make an informed decision
        """
    else:  # DETAILED
        market_dynamics = context.get('market_dynamics', '')
        economic_indicators = context.get('economic_indicators', '')
        relevant_info = await self.search_memories(f"{market_dynamics} {economic_indicators}")
        return f"""
        Role: Highly Sophisticated Economic Agent
        Market Dynamics: {market_dynamics}
        Economic Indicators: {economic_indicators}
        Relevant Information: {relevant_info[:7]}
        Trade Analysis: {context.get('trade_analysis', '')}
        Agent Interactions: {context.get('agent_interactions', '')}
        Strategy Performance: {context.get('strategy_performance', '')}
        Objective: Achieve superior risk-adjusted returns while contributing to market efficiency
        Action: Conduct comprehensive market analysis, develop and test hypotheses, implement sophisticated trading strategies
        """
```

In this method:
1. CSV data is accessed via the `context` dictionary (e.g., `context.get('market_state', '')`).
2. Index searches are performed using `self.search_memories()` with queries built from CSV data.
3. Both CSV data and search results are incorporated into the final prompt.

This approach allows the system to combine recent market history (from CSV) with relevant historical information and analysis (from the index) to create comprehensive and context-aware prompts for the agent.