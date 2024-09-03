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


## EXAMPLES AT THE DIFFERENT LEVELS OF RESOLUTION

```
# Universal Agent Input Prompt

## System State
Corrective measures applied, system returning to equilibrium

## Relevant Information
1. Equilibrium restoration often involves balancing resource allocation across system components.
   - Relevance Score: 0.95
   - Source: System Dynamics Principles

2. Minor fluctuations in stable systems can lead to self-correction without external intervention.
   - Relevance Score: 0.92
   - Source: Complexity Theory in Economics

3. System entropy increases may indicate temporary inefficiencies in resource distribution.
   - Relevance Score: 0.88
   - Source: Thermodynamics of Economic Systems

4. Corrective measures in economic systems often involve policy adjustments or market interventions.
   - Relevance Score: 0.85
   - Source: Macroeconomic Policy Analysis

5. Monitoring key system indicators is crucial for maintaining long-term equilibrium.
   - Relevance Score: 0.82
   - Source: Economic Stability Metrics

## Objective
Maintain equilibrium through continuous interaction

## Action
Analyze the current state and decide on the next action

# Universal Agent Analysis and Action Plan

Based on the current system state and relevant information, here's my analysis and proposed action plan:

1. System Status Assessment:
   - The system is currently in a recovery phase, moving back towards equilibrium.
   - Recent corrective measures appear to be effective in addressing the previous resource imbalance.

2. Key Considerations:
   - While the system is returning to equilibrium, we must be cautious of overcorrection.
   - Minor fluctuations are normal and can contribute to system resilience.
   - Monitoring entropy levels will be crucial to ensure the system doesn't slip back into imbalance.

3. Proposed Actions:
   a) Implement a comprehensive monitoring system to track key stability indicators in real-time.
   b) Conduct a thorough analysis of the effectiveness of the applied corrective measures.
   c) Develop contingency plans for potential future imbalances, focusing on rapid response mechanisms.
   d) Initiate a gradual reduction of any emergency interventions to allow the system to stabilize naturally.
   e) Engage with all system components to ensure equitable resource distribution is maintained.

4. Long-term Strategy:
   - Develop predictive models to anticipate potential future imbalances before they occur.
   - Establish a feedback loop system to continuously refine and improve our equilibrium maintenance strategies.

By taking these actions, we aim to solidify the current return to equilibrium and enhance the system's overall resilience against future disturbances.

Next Review: Schedule a comprehensive system status review in 24 hours to assess the impact of these actions and make any necessary adjustments.
```
```
# Intelligent Economic Agent Input Prompt

## Market State
Bearish

## Recent Events
Geopolitical tensions rising

## Key Insights
Risk-off sentiment growing among investors

## Relevant Information
1. Geopolitical tensions often lead to increased volatility in financial markets and a flight to safe-haven assets.
   - Relevance Score: 0.94
   - Source: Geopolitical Risk Analysis

2. Bearish markets during geopolitical crises may present value investment opportunities in oversold sectors.
   - Relevance Score: 0.91
   - Source: Contrarian Investment Strategies

3. Risk-off sentiment typically results in increased demand for government bonds, gold, and defensive stocks.
   - Relevance Score: 0.89
   - Source: Safe-Haven Asset Performance

4. Historical data shows that markets often overreact to geopolitical events in the short term.
   - Relevance Score: 0.86
   - Source: Market Behavior During Crises

5. Diversification across uncorrelated assets becomes crucial during periods of heightened geopolitical risk.
   - Relevance Score: 0.83
   - Source: Portfolio Management in Volatile Markets

## Objective
Maximize outcomes while managing risk

## Action
Analyze market conditions, consider potential outcomes, and make an informed decision
```
```
# Intelligent Economic Agent: Market Analysis and Strategy

Given the current bearish market state, rising geopolitical tensions, and growing risk-off sentiment, here's my analysis and recommended strategy:

1. Market Assessment:
   - The bearish trend is likely to persist in the short term due to geopolitical uncertainties.
   - Expect increased volatility across most asset classes.
   - Safe-haven assets are likely to outperform riskier assets in the near term.

2. Risk Management:
   a) Increase allocation to safe-haven assets:
      - Government bonds: +5% (focus on US Treasuries and German Bunds)
      - Gold: +3%
      - Defensive stocks: +4% (utilities, consumer staples)
   b) Reduce exposure to cyclical and growth stocks by 10%
   c) Implement stop-loss orders on volatile positions to limit downside risk

3. Opportunity Identification:
   a) Watch for oversold quality companies in stable sectors for potential value plays
   b) Prepare a watchlist of fundamentally strong companies in geopolitically sensitive sectors (e.g., energy, defense) for potential entry points
   c) Consider slowly building positions in emerging market bonds, which may offer attractive yields

4. Hedging Strategies:
   a) Implement a put option strategy on major indices to protect against further downside
   b) Consider long volatility positions through VIX futures or options

5. Monitoring and Analysis:
   a) Set up real-time alerts for significant geopolitical developments
   b) Closely track safe-haven asset flows and correlations
   c) Monitor credit spreads as an indicator of market stress

6. Client Communication:
   Prepare a client memo explaining the current market situation, our risk management approach, and potential opportunities we're monitoring.

7. Review and Adjustment:
   Schedule daily strategy reviews to assess the effectiveness of our approach and make necessary adjustments based on new developments.

By implementing this strategy, we aim to protect capital in the short term while positioning ourselves to capitalize on opportunities that may arise from market overreactions. Our focus is on balancing risk management with the potential for long-term growth.

Next Steps: Begin implementing the proposed asset allocation changes and hedging strategies. Prepare a detailed report on potential value opportunities for review in 48 hours.
```
```
# Highly Sophisticated Economic Agent Input Prompt

## Market Dynamics
Emerging markets outperforming developed markets

## Economic Indicators
Consumer confidence index: 110.2

## Relevant Information
1. Emerging market outperformance often correlates with periods of global economic expansion and increased risk appetite.
   - Relevance Score: 0.96
   - Source: Emerging Markets Economic Analysis

2. Consumer confidence above 100 indicates optimism, potentially leading to increased consumer spending and economic growth.
   - Relevance Score: 0.93
   - Source: Consumer Behavior and Economic Indicators

3. Divergence between emerging and developed markets may signal a shift in global economic dynamics and investment flows.
   - Relevance Score: 0.90
   - Source: Global Market Trends Analysis

4. High consumer confidence in developed markets can drive demand for emerging market exports, supporting their outperformance.
   - Relevance Score: 0.87
   - Source: International Trade and Market Correlations

5. Currency fluctuations play a crucial role in the relative performance of emerging vs. developed markets.
   - Relevance Score: 0.85
   - Source: Forex Impact on Global Equities

## Trade Analysis
Commodities futures showing signs of backwardation

## Agent Interactions
Collaborative decision-making between human and AI agents increasing

## Strategy Performance
Long-short equity strategies showing positive alpha

## Objective
Achieve superior risk-adjusted returns while contributing to market efficiency

## Action
Conduct comprehensive market analysis, develop and test hypotheses, implement sophisticated trading strategies
```
```
# Highly Sophisticated Economic Agent: Comprehensive Analysis and Strategic Deployment

Based on the provided market dynamics, economic indicators, and additional contextual information, here's a detailed analysis and strategic plan:

1. Market Dynamics Analysis:
   a) Emerging Markets (EM) Outperformance:
      - Likely drivers: Improving global risk appetite, potential undervaluation, and structural reforms in key EM countries.
      - Risks: Currency volatility, geopolitical tensions, and potential overheating in certain EM economies.
   b) Developed Markets (DM) Underperformance:
      - Potential causes: Valuation concerns, slower growth prospects, and policy uncertainty in major DM economies.
      - Opportunity: Look for specific DM sectors or companies that might buck the trend.

2. Economic Indicator Interpretation:
   - Consumer Confidence Index at 110.2 suggests strong economic optimism.
   - Potential impacts: Increased consumer spending, potential inflationary pressures, and possible monetary policy tightening in the medium term.

3. Commodities and Currency Analysis:
   - Backwardation in commodities futures indicates potential near-term supply constraints or strong current demand.
   - Strategy: Consider long positions in spot commodities and short-dated futures contracts.
   - Monitor currency pairs, especially EM currencies against major DM currencies, for carry trade opportunities.

4. Agent Interaction Optimization:
   - Leverage increased collaborative decision-making between human and AI agents.
   - Implement a hybrid decision-making model that combines AI-driven quantitative analysis with human qualitative insights.

5. Strategy Refinement:
   - Given the positive alpha in long-short equity strategies:
     a) Increase capital allocation to these strategies by 10%.
     b) Focus on EM vs. DM pairs trades to capitalize on the performance divergence.
     c) Implement machine learning algorithms to identify optimal entry and exit points for these strategies.

6. Risk Management:
   a) Implement dynamic hedging strategies to manage EM-specific risks (e.g., currency, liquidity, geopolitical).
   b) Develop stress testing scenarios that account for potential reversals in EM outperformance.
   c) Set up real-time monitoring of key risk indicators with automated alert systems.

7. Alpha Generation Hypotheses:
   a) Test the hypothesis that EM outperformance is sector-specific rather than broad-based.
   b) Investigate the correlation between consumer confidence and subsequent market performance in both EM and DM.
   c) Analyze the impact of AI-human collaborative decisions on strategy performance and market efficiency.

8. Tactical Asset Allocation:
   a) Overweight EM equities: +7% allocation increase.
   b) Underweight DM equities: -5% allocation reduction.
   c) Increase allocation to EM local currency bonds: +3%.
   d) Implement a commodities overlay strategy focused on backwardated markets.

9. Quantitative Model Enhancement:
   a) Incorporate consumer confidence data into our predictive models for both EM and DM performance.
   b) Develop a new ML model to predict regime changes in EM-DM relative performance.
   c) Enhance our NLP algorithms to better interpret and act on geopolitical news affecting EM countries.

10. Contribution to Market Efficiency:
    a) Publish a research paper on the impact of AI-human collaboration on market price discovery.
    b) Engage with EM regulators to share insights on improving market structure and liquidity.
    c) Develop and share (with a time delay) anonymized datasets on EM-DM correlations to promote academic research.

11. Performance Monitoring and Adjustment:
    - Implement a daily review of strategy performance with automated adjustments for strategies falling below predefined Sharpe ratio thresholds.
    - Conduct weekly human-AI collaborative sessions to review and refine our overall market approach.

By executing this comprehensive plan, we aim to capitalize on the current market dynamics while maintaining a robust risk management framework. Our focus on combining quantitative strategies with human oversight, continuous hypothesis testing, and contribution to overall market efficiency positions us to achieve superior risk-adjusted returns in this complex market environment.

Next Steps: Begin immediate implementation of the tactical asset allocation changes and initiate the proposed hypothesis tests. Schedule a full review of preliminary results in 72 hours.
```