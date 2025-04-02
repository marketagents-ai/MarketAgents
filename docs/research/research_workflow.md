# Research Workflow

## Overview

The research workflow in MarketAgents defines how agents collaborate to conduct comprehensive research on a topic. This document explains the research process, including information gathering, analysis, and synthesis phases.

## Research Process

The multi-agent research process typically follows these phases:

1. **Task Assignment**: Agents are assigned specific research subtopics
2. **Information Gathering**: Agents collect relevant information from various sources
3. **Analysis**: Agents analyze the collected information
4. **Synthesis**: Findings are combined into a cohesive research output
5. **Review**: The final output is reviewed and refined

## Task Assignment

In the research environment, tasks can be assigned to agents based on their expertise:

```python
# Automatic subtopic assignment based on agent expertise
research_mechanism = Research(
    topic="Impact of AI on financial markets",
    subtopics=[
        "AI adoption in trading",
        "Algorithmic trading impact",
        "Regulatory considerations"
    ],
    assign_subtopics=True  # Automatically assign subtopics to agents
)

# Manual subtopic assignment
research_mechanism.assign_subtopic(
    agent_id="researcher_0",
    subtopic="AI adoption in trading"
)
```

## Information Gathering

Agents gather information through various methods:

```python
# Agent submitting a research finding
finding_action = {
    "action_type": "research_finding",
    "content": "AI-driven trading algorithms now account for over 70% of daily trading volume in major exchanges.",
    "subtopic": "AI adoption in trading",
    "sources": ["Financial Times report", "Market analysis data"],
    "confidence": 0.85
}

# Execute the action in the environment
observation = agent.environments["research"].step(finding_action)
```

### Research Finding Structure

Research findings typically include:

- **Content**: The actual information or insight
- **Subtopic**: The relevant subtopic
- **Sources**: References to information sources
- **Confidence**: Confidence level in the finding (0.0-1.0)
- **Metadata**: Additional contextual information

## Analysis

Agents analyze gathered information to extract insights:

```python
# Agent submitting an analysis
analysis_action = {
    "action_type": "research_analysis",
    "content": "The high adoption rate of AI in trading has led to increased market efficiency but also raises concerns about systemic risks.",
    "subtopic": "AI adoption in trading",
    "related_findings": ["finding_1", "finding_3"],
    "confidence": 0.8
}

# Execute the action in the environment
observation = agent.environments["research"].step(analysis_action)
```

## Synthesis

After gathering and analyzing information, agents synthesize their findings:

```python
# Agent submitting a synthesis
synthesis_action = {
    "action_type": "research_synthesis",
    "content": "AI has fundamentally transformed financial markets through increased automation, improved efficiency, and new risk patterns...",
    "covers_subtopics": ["AI adoption in trading", "Algorithmic trading impact"],
    "confidence": 0.9
}

# Execute the action in the environment
observation = agent.environments["research"].step(synthesis_action)
```

### Collaborative Synthesis

For complex research topics, agents can collaborate on synthesis:

```python
# Enable collaborative synthesis
research_mechanism = Research(
    topic="Impact of AI on financial markets",
    collaborative_synthesis=True
)

# In collaborative mode, each agent contributes to the synthesis
# The mechanism then combines these contributions
```

## Research Cognitive Steps

The research process is typically implemented through specialized cognitive steps:

```python
from market_agents.agents.cognitive_steps import CognitiveStep
from pydantic import Field

class ResearchGatheringStep(CognitiveStep):
    step_name: str = Field(default="research_gathering", description="Research gathering step")
    
    async def execute(self, agent):
        # Get current research state
        research_state = agent.environments[self.environment_name].get_global_state(agent_id=agent.id)
        
        # Generate research findings
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Research topic: {research_state.get('topic')}
        Your assigned subtopic: {research_state.get('assigned_subtopic')}
        
        Based on your knowledge and expertise, provide 3-5 key findings about your assigned subtopic.
        For each finding, include:
        1. The finding itself
        2. Sources or references
        3. Your confidence level (0.0-1.0)
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Process and submit findings
        # ...
        
        return response.content

class ResearchSynthesisStep(CognitiveStep):
    step_name: str = Field(default="research_synthesis", description="Research synthesis step")
    
    async def execute(self, agent):
        # Get current research state with all findings
        research_state = agent.environments[self.environment_name].get_global_state()
        
        # Generate synthesis
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Research topic: {research_state.get('topic')}
        
        All research findings:
        {self._format_findings(research_state.get('findings', []))}
        
        Based on all the research findings, synthesize a comprehensive analysis of the topic.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Submit synthesis
        # ...
        
        return response.content
    
    def _format_findings(self, findings):
        formatted = ""
        for i, finding in enumerate(findings):
            formatted += f"Finding {i+1}: {finding.get('content')}\n"
            formatted += f"Subtopic: {finding.get('subtopic')}\n"
            formatted += f"Sources: {', '.join(finding.get('sources', []))}\n"
            formatted += f"Confidence: {finding.get('confidence')}\n\n"
        return formatted
```

## Research Episode

A complete research episode typically includes multiple cognitive steps:

```python
from market_agents.agents.cognitive_steps import CognitiveEpisode

# Define a research episode
research_episode = CognitiveEpisode(
    steps=[
        ResearchGatheringStep,
        ResearchAnalysisStep,
        ResearchSynthesisStep
    ],
    environment_name="research"
)

# Run the research episode
results = await agent.run_episode(research_episode)
```

## Practical Example: Research Workflow

Here's a complete example demonstrating the research workflow:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.cognitive_steps import CognitiveStep, CognitiveEpisode
from market_agents.environments.mechanisms.research import Research, ResearchEnvironment

class ResearchGatheringStep(CognitiveStep):
    step_name: str = Field(default="research_gathering", description="Research gathering step")
    
    async def execute(self, agent):
        # Get current research state
        research_state = agent.environments[self.environment_name].get_global_state(agent_id=agent.id)
        
        # Generate research findings
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Research topic: {research_state.get('topic')}
        Your assigned subtopic: {research_state.get('assigned_subtopic', 'any relevant aspect of the main topic')}
        
        Based on your knowledge and expertise, provide 3-5 key findings about your assigned subtopic.
        For each finding, include:
        1. The finding itself
        2. Sources or references
        3. Your confidence level (0.0-1.0)
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Process response to extract findings
        # This is a simplified example - in practice, you would parse the response
        findings = [
            {
                "content": "AI-driven trading algorithms now account for over 70% of daily trading volume in major exchanges.",
                "subtopic": research_state.get('assigned_subtopic', 'AI adoption in trading'),
                "sources": ["Financial Times report", "Market analysis data"],
                "confidence": 0.85
            }
        ]
        
        # Submit each finding
        for finding in findings:
            action = {
                "action_type": "research_finding",
                **finding
            }
            agent.environments[self.environment_name].step(action)
        
        return response.content

class ResearchSynthesisStep(CognitiveStep):
    step_name: str = Field(default="research_synthesis", description="Research synthesis step")
    
    async def execute(self, agent):
        # Get current research state with all findings
        research_state = agent.environments[self.environment_name].get_global_state()
        
        # Generate synthesis
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Research topic: {research_state.get('topic')}
        
        All research findings:
        {self._format_findings(research_state.get('findings', []))}
        
        Based on all the research findings, synthesize a comprehensive analysis of the topic.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Submit synthesis
        action = {
            "action_type": "research_synthesis",
            "content": response.content,
            "covers_subtopics": [f.get('subtopic') for f in research_state.get('findings', [])],
            "confidence": 0.9
        }
        agent.environments[self.environment_name].step(action)
        
        return response.content
    
    def _format_findings(self, findings):
        formatted = ""
        for i, finding in enumerate(findings):
            formatted += f"Finding {i+1}: {finding.get('content')}\n"
            formatted += f"Subtopic: {finding.get('subtopic')}\n"
            formatted += f"Sources: {', '.join(finding.get('sources', []))}\n"
            formatted += f"Confidence: {finding.get('confidence')}\n\n"
        return formatted

async def run_research_workflow(agents, research_env):
    # Define research episode
    research_episode = CognitiveEpisode(
        steps=[ResearchGatheringStep, ResearchSynthesisStep],
        environment_name="research"
    )
    
    # Run the research episode for each agent
    results = []
    for agent in agents:
        agent_results = await agent.run_episode(research_episode)
        results.append({
            "agent_id": agent.id,
            "role": agent.role,
            "results": agent_results
        })
    
    # Get final research output
    final_output = research_env.mechanism.get_research_output()
    
    return {
        "agent_results": results,
        "final_output": final_output
    }

# Run the research workflow
research_results = asyncio.run(run_research_workflow(agents, research_env))
```

In the next section, we'll explore how to integrate knowledge bases with the research process.
