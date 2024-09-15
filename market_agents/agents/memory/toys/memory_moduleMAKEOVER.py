import argparse
import json
from typing import List, Dict, Any
from enum import Enum
import csv
import os
import asyncio
from reptarINDEX import load_graph, Graph

class AbstractionLevel(Enum):
    GENERALIST = "generalist"
    HIGH_LEVEL = "high_level"
    DETAILED = "detailed"

class MemorySystem:
    def __init__(self, abstraction_level: AbstractionLevel, index_file: str):
        self.abstraction_level = abstraction_level
        self.memories: List[Dict[str, Any]] = []
        self.graph, self.embeddings = load_graph(type('Args', (), {'input_file': index_file})())

    def add_memory(self, memory: Dict[str, Any]):
        self.memories.append(memory)

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

    def format_output_prompt(self, action: str, result: Any) -> str:
        if self.abstraction_level == AbstractionLevel.GENERALIST:
            return f"Action Taken: {action}\nResult: {result}\nNext Step: Evaluate the outcome and adjust approach if necessary"
        elif self.abstraction_level == AbstractionLevel.HIGH_LEVEL:
            return f"Action Executed: {action}\nOutcome: {result}\nNext Steps: Analyze the results, update market knowledge, and refine strategy"
        else:  # DETAILED
            return f"""
            Action Implemented: {action}
            Outcome Analysis:
            - Result: {result}
            - Performance Metrics: [Include relevant metrics]
            - Market Impact: [Assess the action's effect on the market]
            Next Steps:
            1. Conduct post-trade analysis
            2. Update economic models and predictions
            3. Refine strategies based on new data
            4. Prepare for next market engagement
            """

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

async def main():
    parser = argparse.ArgumentParser(description="Agent Memory System with Index Search and CSV Input")
    parser.add_argument("--level", choices=[e.value for e in AbstractionLevel], default=AbstractionLevel.HIGH_LEVEL.value, help="Abstraction level")
    parser.add_argument("--action", choices=["add", "search", "format_input", "format_output"], required=True, help="Action to perform")
    parser.add_argument("--data", type=str, help="JSON string containing data for the action")
    parser.add_argument("--index_file", type=str, required=True, help="Path to the index file")
    parser.add_argument("--csv_file", type=str, help="Path to the CSV file containing market history")
    
    args = parser.parse_args()
    
    memory_system = MemorySystem(AbstractionLevel(args.level), args.index_file)
    
    if args.action == "add":
        memory_data = json.loads(args.data)
        memory_system.add_memory(memory_data)
        print("Memory added successfully")
    elif args.action == "search":
        query = args.data
        results = await memory_system.search_memories(query)
        print(json.dumps(results, indent=2))
    elif args.action == "format_input":
        context = json.loads(args.data)
        if args.csv_file:
            market_history = load_market_history(args.csv_file, AbstractionLevel(args.level))
            context.update(market_history)
        prompt = await memory_system.format_input_prompt(context)
        print(prompt)
    elif args.action == "format_output":
        output_data = json.loads(args.data)
        prompt = memory_system.format_output_prompt(output_data["action"], output_data["result"])
        print(prompt)

if __name__ == "__main__":
    asyncio.run(main())