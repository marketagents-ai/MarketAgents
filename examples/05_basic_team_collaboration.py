import asyncio
import os
from pathlib import Path

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.environments.mechanisms.chat import ChatEnvironment
from minference.lite.models import LLMConfig, ResponseFormat

async def create_product_manager():
    """Create a product manager agent."""
    
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
        role="Product Manager",
        persona="I am a product manager responsible for defining product vision, roadmap, and features. I prioritize user needs and business goals to create successful products.",
        objectives=[
            "Define clear product vision and strategy",
            "Prioritize features based on user needs and business impact",
            "Coordinate with design and engineering teams",
            "Ensure product meets market requirements"
        ],
        communication_style="Clear and decisive",
        skills=[
            "Strategic planning",
            "User research",
            "Feature prioritization",
            "Cross-functional leadership"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="product_manager",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Product Manager with ID: {agent.id}")
    
    return agent

async def create_ux_designer():
    """Create a UX designer agent."""
    
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
        role="UX Designer",
        persona="I am a UX designer focused on creating intuitive, accessible, and delightful user experiences. I advocate for user needs and translate them into effective design solutions.",
        objectives=[
            "Create user-centered designs that solve real problems",
            "Ensure accessibility and usability for all users",
            "Develop consistent design systems and patterns",
            "Collaborate with product and engineering teams"
        ],
        communication_style="Creative and empathetic",
        skills=[
            "User research",
            "Interaction design",
            "Wireframing and prototyping",
            "Usability testing"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="ux_designer",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created UX Designer with ID: {agent.id}")
    
    return agent

async def create_software_engineer():
    """Create a software engineer agent."""
    
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
        role="Software Engineer",
        persona="I am a software engineer specializing in full-stack development. I build robust, scalable, and maintainable software solutions that meet product requirements and technical standards.",
        objectives=[
            "Develop high-quality, maintainable code",
            "Implement features according to specifications",
            "Ensure performance, security, and reliability",
            "Collaborate with design and product teams"
        ],
        communication_style="Logical and precise",
        skills=[
            "Full-stack development",
            "System architecture",
            "Problem-solving",
            "Technical documentation"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="software_engineer",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    print(f"Created Software Engineer with ID: {agent.id}")
    
    return agent

async def create_product_team():
    """Create a product development team with multiple agents."""
    
    # Create individual agents
    product_manager = await create_product_manager()
    ux_designer = await create_ux_designer()
    software_engineer = await create_software_engineer()
    
    # Define chat environment
    chat_env = {
        "name": "product_team_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 3,
        "task_prompt": "Collaborate to design and develop a new product feature."
    }
    
    # Create the product team
    product_team = MarketAgentTeam(
        name="Product Development Team",
        manager=product_manager,
        agents=[
            ux_designer,
            software_engineer
        ],
        mode="collaborative",
        use_group_chat=True,
        shared_context={
            "product_vision": "Create intuitive, accessible, and powerful tools that help users achieve their goals efficiently.",
            "target_audience": "Knowledge workers and professionals who need to manage complex information and workflows.",
            "company_values": ["User-centered design", "Technical excellence", "Continuous improvement", "Inclusive collaboration"]
        },
        environments=[
            chat_env
        ]
    )
    
    print(f"Created Product Team: {product_team.name}")
    print(f"Team members: {[agent.id for agent in product_team.agents]}")
    print(f"Team manager: {product_team.manager.id}")
    
    return product_team

async def run_product_team_discussion(team, feature_request):
    """Run a product team discussion about a feature request."""
    
    task = f"""
    Collaborate as a product development team to design and plan the implementation of the following feature request:
    
    FEATURE REQUEST:
    {feature_request}
    
    Your team should:
    
    1. The Product Manager should:
       - Clarify the feature requirements and user needs
       - Define success metrics for the feature
       - Prioritize aspects of the feature
       - Create a high-level roadmap
    
    2. The UX Designer should:
       - Propose a user-centered design approach
       - Identify potential usability challenges
       - Suggest design solutions and user flows
       - Consider accessibility requirements
    
    3. The Software Engineer should:
       - Evaluate technical feasibility
       - Identify potential technical challenges
       - Propose implementation approach and architecture
       - Estimate development effort and timeline
    
    Through discussion, develop a comprehensive plan that addresses user needs, design considerations, and technical implementation.
    
    The final output should be a structured feature specification that includes:
    - Feature overview and goals
    - User experience design approach
    - Technical implementation plan
    - Timeline and milestones
    - Success metrics
    """
    
    print(f"\nStarting team discussion about: {feature_request}")
    print("-" * 80)
    
    # Run the team discussion
    result = await team.execute(task)
    
    print("\nTeam Discussion Complete")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create a product development team
    team = await create_product_team()
    
    # Run team discussions for different feature requests
    feature_requests = [
        """
        Implement a "Smart Document Summarization" feature that automatically generates concise summaries of long documents.
        Users should be able to customize the summary length and focus areas. The feature should work with multiple document
        formats (PDF, DOCX, TXT) and preserve the key information while reducing reading time by 70%.
        """,
        
        """
        Create a "Collaborative Workflow Builder" that allows teams to design, customize, and automate their work processes.
        The feature should include a visual drag-and-drop interface, pre-built workflow templates, integration with popular
        productivity tools, and real-time collaboration capabilities.
        """
    ]
    
    for request in feature_requests:
        await run_product_team_discussion(team, request)

if __name__ == "__main__":
    asyncio.run(main())
