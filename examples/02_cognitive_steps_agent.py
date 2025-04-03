import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.cognitive_steps import CognitiveStep, CognitiveEpisode
from minference.lite.models import LLMConfig, ResponseFormat
from pydantic import Field, BaseModel

# Define a custom cognitive step for research analysis
class ResearchAnalysisStep(CognitiveStep):
    """A cognitive step for conducting research analysis on a specific topic."""
    
    step_name: str = Field(default="research_analysis", description="Research analysis step")
    topic: str = Field(..., description="Topic to research and analyze")
    
    async def execute(self, agent):
        """Execute the research analysis step."""
        
        # Create a prompt for the agent to analyze the topic
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Conduct a thorough analysis on the following topic: {self.topic}
        
        Your analysis should include:
        1. Overview of the current state
        2. Key trends and developments
        3. Future outlook and predictions
        4. Potential implications for investors
        
        Provide a comprehensive and well-structured analysis based on your knowledge.
        """
        
        # Generate response using the agent's LLM
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Store the analysis in the agent's short-term memory
        agent.stm.add(f"analysis_{self.topic}", response.content)
        
        return response.content

# Define a custom cognitive step for generating recommendations
class RecommendationStep(CognitiveStep):
    """A cognitive step for generating recommendations based on previous analysis."""
    
    step_name: str = Field(default="recommendation", description="Recommendation generation step")
    topic: str = Field(..., description="Topic for which to generate recommendations")
    
    async def execute(self, agent):
        """Execute the recommendation generation step."""
        
        # Retrieve the analysis from short-term memory
        analysis = agent.stm.get(f"analysis_{self.topic}")
        
        if not analysis:
            return "No analysis found for this topic. Please run the analysis step first."
        
        # Create a prompt for generating recommendations
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Based on the following analysis of {self.topic}:
        
        {analysis}
        
        Generate 3-5 specific, actionable recommendations for investors or businesses 
        interested in this area. Each recommendation should include:
        
        1. A clear, concise recommendation title
        2. Detailed explanation and rationale
        3. Potential risks or considerations
        4. Timeline for implementation or expected results
        
        Format your recommendations in a clear, structured manner.
        """
        
        # Generate response using the agent's LLM
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Store the recommendations in the agent's short-term memory
        agent.stm.add(f"recommendations_{self.topic}", response.content)
        
        return response.content

# Define a structured output model for the final report
class ResearchReport(BaseModel):
    """Structured output model for a research report."""
    
    topic: str = Field(..., description="Topic of the research report")
    summary: str = Field(..., description="Executive summary of the research")
    key_findings: List[str] = Field(..., description="List of key findings")
    recommendations: List[str] = Field(..., description="List of recommendations")
    risk_factors: List[str] = Field(..., description="List of risk factors to consider")
    conclusion: str = Field(..., description="Concluding thoughts")

# Define a custom cognitive step for generating a structured report
class StructuredReportStep(CognitiveStep):
    """A cognitive step for generating a structured research report."""
    
    step_name: str = Field(default="structured_report", description="Structured report generation step")
    topic: str = Field(..., description="Topic for the structured report")
    
    async def execute(self, agent):
        """Execute the structured report generation step."""
        
        # Retrieve the analysis and recommendations from short-term memory
        analysis = agent.stm.get(f"analysis_{self.topic}")
        recommendations = agent.stm.get(f"recommendations_{self.topic}")
        
        if not analysis or not recommendations:
            return "Missing analysis or recommendations. Please run the previous steps first."
        
        # Create a prompt for generating a structured report
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Based on the following analysis and recommendations for {self.topic}:
        
        ANALYSIS:
        {analysis}
        
        RECOMMENDATIONS:
        {recommendations}
        
        Generate a structured research report with the following components:
        
        1. A concise executive summary (2-3 sentences)
        2. 4-6 key findings as bullet points
        3. 3-5 specific recommendations as bullet points
        4. 2-4 risk factors to consider as bullet points
        5. A brief conclusion (2-3 sentences)
        
        Format your response as a JSON object with the following structure:
        {{
            "topic": "{self.topic}",
            "summary": "Executive summary here",
            "key_findings": ["Finding 1", "Finding 2", ...],
            "recommendations": ["Recommendation 1", "Recommendation 2", ...],
            "risk_factors": ["Risk 1", "Risk 2", ...],
            "conclusion": "Conclusion here"
        }}
        
        Ensure your response is valid JSON.
        """
        
        # Generate response using the agent's LLM with JSON response format
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        # Parse the JSON response into the structured model
        try:
            report_data = json.loads(response.content)
            report = ResearchReport(**report_data)
            return report
        except Exception as e:
            return f"Error parsing structured report: {str(e)}\n\nRaw response: {response.content}"

async def create_research_agent():
    """Create a research agent with cognitive capabilities."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Technology Research Analyst",
        persona="I am a technology research analyst with expertise in emerging technologies, market trends, and investment opportunities. I provide in-depth analysis and actionable recommendations based on thorough research.",
        objectives=[
            "Analyze emerging technology trends and their market impact",
            "Identify investment opportunities in the technology sector",
            "Provide actionable recommendations based on research findings",
            "Assess risks and challenges in technology investments"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="tech_research_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
    print(f"Created research agent with ID: {agent.id}")
    print(f"Role: {agent.role}")
    
    return agent

async def run_research_episode(agent, topic):
    """Run a complete research episode with multiple cognitive steps."""
    
    print(f"\nRunning research episode on topic: {topic}")
    print("-" * 80)
    
    # Define the cognitive episode with multiple steps
    research_episode = CognitiveEpisode(
        steps=[
            ResearchAnalysisStep(topic=topic),
            RecommendationStep(topic=topic),
            StructuredReportStep(topic=topic)
        ]
    )
    
    # Run the episode
    results = await agent.run_episode(research_episode)
    
    # Print the results of each step
    print("\nAnalysis:")
    print("-" * 40)
    print(results.get("research_analysis", "No analysis generated"))
    
    print("\nRecommendations:")
    print("-" * 40)
    print(results.get("recommendation", "No recommendations generated"))
    
    print("\nStructured Report:")
    print("-" * 40)
    structured_report = results.get("structured_report")
    if isinstance(structured_report, ResearchReport):
        print(f"Topic: {structured_report.topic}")
        print(f"Summary: {structured_report.summary}")
        print("\nKey Findings:")
        for i, finding in enumerate(structured_report.key_findings, 1):
            print(f"{i}. {finding}")
        print("\nRecommendations:")
        for i, rec in enumerate(structured_report.recommendations, 1):
            print(f"{i}. {rec}")
        print("\nRisk Factors:")
        for i, risk in enumerate(structured_report.risk_factors, 1):
            print(f"{i}. {risk}")
        print(f"\nConclusion: {structured_report.conclusion}")
    else:
        print(structured_report)
    
    return results

async def main():
    # Create a research agent
    agent = await create_research_agent()
    
    # Run research episodes on different topics
    topics = [
        "Artificial General Intelligence (AGI) Development",
        "Quantum Computing Commercial Applications",
        "Edge Computing and 5G Integration"
    ]
    
    for topic in topics:
        await run_research_episode(agent, topic)

if __name__ == "__main__":
    import json
    asyncio.run(main())
