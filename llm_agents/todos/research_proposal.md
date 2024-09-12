# Economic Multi-Agent Simulation: Executive Summary

**Authors**: Interstellarninja, Iriden [Tomasso Furlanello], Everyone Is Gross, AtakanTekparmark, mudler, Bexboy

## Project Overview

Develop a minimally viable yet rigorous economic multi-agent simulation, inspired by Stanford's generative agents research. Initial scope: a spatial-less market simulation with double auction (DA) as a base environment with LLMs as consumer, producer and/or institutional agents.

## Framework Architecture

### 1. Core Economic Components

#### Basic Economic Units
- Goods
- Households (consumer agents)
- Firms (producer/entrepreneurial agents)

#### Goods Variety
Hierarchical structure (categories, subcategories, specific goods) representing the diversity of goods available in the economy.

#### Consumer Agent Characteristics
- **Utility Function**: Cobb-Douglas function (U = x1^a1 * x2^a2 * ... * xn^an)
- **Initial Endowment**: Initial amount of assets or goods that each consumer agent starts with
- **Income Distribution (Gini Coefficient)**: A measure of income inequality among consumer agents, ranging from 0 (perfect equality) to 1 (maximum inequality)
- **Preference Baskets**: Distinguish necessities (high initial utility, rapid diminishing returns) from luxuries
- **Bounded Rationality** [as a function of LLM's internal limitations]:
  - Rationality Parameter (r): 0 (random) to 1 (optimal decisions)
  - Limited observability of the world
  - Limited capacity to observe the world
  - Limited memory
  - Limited test time tokens
- **Exploration-Exploitation Parameter (ε)**: Balances known strategies with new opportunities
- **Zero Intelligence Agents**: Random bids/offers within budget constraint, gradual introduction of heuristics
- **Demographic Characteristics**: Age, Gender, Education Level, Household Size and Composition, Income Levels
- **Behavioral Factors**: Incorporate biases such as overconfidence, loss aversion, herd behavior etc.

#### Producer/Entrepreneurial Agent Characteristics
- **Production Function**: Cobb-Douglas function (Y = A * L^α * K^β)
- **Specialization**: Type of good or service the firm produces
- **Capital**: Numeric value representing the firm's financial resources
- **Pricing Strategies**: Cost-plus pricing, competitive pricing, and dynamic pricing based on market conditions
- **Revenue Maximization**: Strategies to balance pricing, sales volume, and profit margins
- **Cost Efficiency**: 0-1 scale, simplified representation of fixed and variable cost management
- **Entrepreneurial Adaptability**: 0-1 scale, speed at which the firm can adjust production based on market signals
- **Nominal Rigidity**: 0-1 scale, speed at which the firm adjusts prices in response to market changes

#### Institutional Characteristics
- **Monetary Authority**:
  - Policy Rate: The interest rate set by the central bank
- **Fiscal Authority**:
  - Taxation Policy: 0-1 scale, level of taxes imposed on individuals and businesses
- **Regulatory Bodies**:
  - Business Regulation Index: 0-1 scale, measures the ease of doing business
  - Social Security Index: 0-1 scale, degree of social support provided to individuals

### 2. Agent Interaction Mechanisms

- **Economic Exchanges**: Double Auctions (DA) as primary mechanism, potential for limit order books
- **Blockchain Integration**: Potentially use Double Auctions (DA) as a consensus mechanism on the blockchain
  - Nodes, Order Book, Smart Contracts, Tokens, Liquidity Pool
- **Debate Protocol**: Allow bargaining through natural language bids/offers bypassing DA matching/consensus
- **LLM as auctioneer**: Explore LLM as a market institution to receive bids and offers
- **Information Sharing**: Public board [4chan], signaling ("whisper words"), news cycle simulation
- **Agent Social Network**: A directed graph to capture the influence of peer interactions on economic decisions
- **Dynamic Adaptation**: Agents learn and adapt their strategies over time

### 3. Simulation Dynamics

- Use general equilibrium theory to measure the efficiency of the bounded/rationality + market + communication protocol
- Introduce experimental modules for different economic scenarios
- Simulate economic shocks, seasonality, policy implementation and transmission channels

### 4. Agent Decision-Making and Learning Framework

- **Goal-Oriented Action Planning (GOAP)**
- **Memory and Retrieval**
- **Reflection**
- **Planning and Reacting**

## First PoC Milestone: Achieving Competitive Equilibrium (C.E.)

1. Define Agents
2. Market Interaction
3. Run Simulation
4. Check Convergence
5. Analyze Results

## Technical Implementation

- Develop framework for distributed agent orchestration, scheduling and execution [LocalAI API]
- Custom inference stack for serving AI models in distributed multi-GPU clusters [Nous Hermes]

## Research Objectives

- Explore event forecasting and "wisdom of the crowd" phenomena
- Develop arbitrage opportunity identification algorithms
- Create framework for learning policies from simulated data
- Study policy transmission mechanisms to economic indicators
- Investigate investment strategies in simulated economy
- Generate agentic dataset for economic modeling
- Conduct randomized controlled trials (RCTs) and difference-in-differences (DiD) studies
- Evaluate causal impacts of economic policies and interventions
- Study emergent social structures from aggregate patterns
- Evaluate the LLMs in-context learning through utility functions and production possibility

## Expected Outcomes

- Flexible, scalable framework for economic multi-agent simulations
- Insights into micro-economic foundations and macro-economic implications
- Novel approaches to economic forecasting and policy analysis
- Applications in financial strategy development and risk assessment

## Conclusion

This PoC project aims to create a framework for economic multi-agent simulations, orchestrating LLM agents in a framework with established economic principles. By incorporating double auctions environment and agentic behavior such as GOAP, memory, reflection, and planning mechanisms, we anticipate generating valuable insights for research in both LLM's in-context learning and economic theory. The scalable nature allows for future expansions with potential for large scale economic modeling and policy experiments. Besides the application of blockchain technology and distributed inference will have potential contributions on the infrastructure for distributed agent orchestration.

## References

1. Epstein, J. M., & Axtell, R. (1996). Growing artificial societies: social science from the bottom up. Brookings Institution Press.
2. Park, J. S., O'Brien, J., Cai, C. J., Morris, M. R., Liang, P., & Bernstein, M. S. (2023, October). Generative agents: Interactive simulacra of human behavior. In Proceedings of the 36th annual acm symposium on user interface software and technology (pp. 1-22).
3. Smith, V. L. (1982). Microeconomic systems as an experimental science. The American economic review, 72(5), 923-955.