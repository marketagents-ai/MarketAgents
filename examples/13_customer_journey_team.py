import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.agents.market_agent_team import MarketAgentTeam
from market_agents.environments.mechanisms.chat import ChatEnvironment
from market_agents.orchestrators.mcp_server.finance_mcp_server import FinanceMCPServer
from minference.lite.models import LLMConfig, ResponseFormat

# Define a simple knowledge base for the agent
KNOWLEDGE_BASE = {
    "company_info": {
        "name": "TechInnovate Solutions",
        "founded": 2015,
        "headquarters": "San Francisco, CA",
        "employees": 250,
        "industry": "Enterprise Software",
        "products": [
            "CloudManage Pro - Cloud infrastructure management platform",
            "DataInsight Suite - Business intelligence and analytics solution",
            "SecureConnect - Enterprise security and compliance platform"
        ],
        "competitors": ["CloudTech Systems", "DataWorks Inc.", "SecureSoft Technologies"]
    },
    "product_faqs": {
        "CloudManage Pro": [
            {
                "question": "What cloud providers does CloudManage Pro support?",
                "answer": "CloudManage Pro supports all major cloud providers including AWS, Microsoft Azure, Google Cloud Platform, and IBM Cloud. It also supports hybrid cloud environments."
            },
            {
                "question": "Does CloudManage Pro offer auto-scaling capabilities?",
                "answer": "Yes, CloudManage Pro includes intelligent auto-scaling that optimizes resource allocation based on workload patterns and custom rules you define."
            },
            {
                "question": "What security certifications does CloudManage Pro have?",
                "answer": "CloudManage Pro is SOC 2 Type II certified, GDPR compliant, and meets ISO 27001 security standards."
            }
        ],
        "DataInsight Suite": [
            {
                "question": "Can DataInsight Suite connect to our existing databases?",
                "answer": "Yes, DataInsight Suite supports integration with all major database systems including SQL Server, Oracle, MySQL, PostgreSQL, and MongoDB, as well as data warehouses like Snowflake and Redshift."
            },
            {
                "question": "Does DataInsight Suite include predictive analytics capabilities?",
                "answer": "Yes, DataInsight Suite includes advanced predictive analytics powered by machine learning algorithms that can forecast trends and identify patterns in your data."
            },
            {
                "question": "Is there a limit to how much data DataInsight Suite can process?",
                "answer": "DataInsight Suite is built on a scalable architecture that can handle petabytes of data. The Enterprise tier includes unlimited data processing capabilities."
            }
        ],
        "SecureConnect": [
            {
                "question": "Does SecureConnect offer single sign-on (SSO) capabilities?",
                "answer": "Yes, SecureConnect includes robust SSO capabilities and integrates with all major identity providers including Okta, Auth0, Microsoft Azure AD, and Google Workspace."
            },
            {
                "question": "How does SecureConnect help with compliance requirements?",
                "answer": "SecureConnect includes pre-built compliance templates for GDPR, HIPAA, PCI DSS, and other regulatory frameworks. It also provides automated compliance reporting and real-time monitoring."
            },
            {
                "question": "Can SecureConnect detect and prevent security breaches?",
                "answer": "Yes, SecureConnect includes advanced threat detection powered by AI that can identify suspicious activities and potential security breaches in real-time. It also provides automated incident response capabilities."
            }
        ]
    },
    "pricing": {
        "CloudManage Pro": {
            "Starter": {
                "price": "$499/month",
                "features": ["Up to 50 cloud resources", "Basic monitoring", "Email support"]
            },
            "Professional": {
                "price": "$999/month",
                "features": ["Up to 200 cloud resources", "Advanced monitoring", "Priority support", "Custom alerts"]
            },
            "Enterprise": {
                "price": "Custom pricing",
                "features": ["Unlimited cloud resources", "Full feature set", "Dedicated support manager", "Custom integrations"]
            }
        },
        "DataInsight Suite": {
            "Basic": {
                "price": "$799/month",
                "features": ["Up to 5 data sources", "Standard visualizations", "Basic reporting"]
            },
            "Advanced": {
                "price": "$1,499/month",
                "features": ["Up to 20 data sources", "Advanced visualizations", "Custom dashboards", "API access"]
            },
            "Enterprise": {
                "price": "Custom pricing",
                "features": ["Unlimited data sources", "Full feature set", "Dedicated support", "Custom ML models"]
            }
        },
        "SecureConnect": {
            "Standard": {
                "price": "$599/month",
                "features": ["Up to 100 users", "Basic security features", "Standard compliance reports"]
            },
            "Professional": {
                "price": "$1,299/month",
                "features": ["Up to 500 users", "Advanced security features", "Custom compliance reports", "Threat detection"]
            },
            "Enterprise": {
                "price": "Custom pricing",
                "features": ["Unlimited users", "Full feature set", "Dedicated security team", "Custom security policies"]
            }
        }
    }
}

async def create_customer_support_agent():
    """Create a customer support agent with knowledge base integration."""
    
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
        role="Customer Support Specialist",
        persona="I am a customer support specialist for TechInnovate Solutions. I provide helpful, accurate information about our products, answer customer questions, and resolve issues with a friendly and professional approach.",
        objectives=[
            "Provide accurate information about our products and services",
            "Answer customer questions clearly and comprehensively",
            "Resolve customer issues efficiently and effectively",
            "Maintain a positive and helpful tone in all interactions"
        ],
        communication_style="Friendly, professional, and solution-oriented",
        skills=[
            "Product knowledge",
            "Problem-solving",
            "Clear communication",
            "Customer empathy"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="customer_support",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    # Add knowledge base to agent memory
    await agent.add_to_memory("knowledge_base", json.dumps(KNOWLEDGE_BASE))
    
    print(f"Created Customer Support Agent with ID: {agent.id}")
    print("Added knowledge base to agent memory")
    
    return agent

async def create_sales_agent():
    """Create a sales agent with knowledge base integration."""
    
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
        role="Sales Representative",
        persona="I am a sales representative for TechInnovate Solutions. I help potential customers understand our products, identify the right solutions for their needs, and guide them through the purchasing process with a consultative approach.",
        objectives=[
            "Understand customer needs and requirements",
            "Match customers with the right products and pricing tiers",
            "Articulate product value propositions effectively",
            "Guide customers through the sales process"
        ],
        communication_style="Consultative, persuasive, and solution-focused",
        skills=[
            "Needs assessment",
            "Product knowledge",
            "Value articulation",
            "Objection handling"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="sales_rep",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    # Add knowledge base to agent memory
    await agent.add_to_memory("knowledge_base", json.dumps(KNOWLEDGE_BASE))
    
    print(f"Created Sales Agent with ID: {agent.id}")
    print("Added knowledge base to agent memory")
    
    return agent

async def create_customer_success_agent():
    """Create a customer success agent with knowledge base integration."""
    
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
        role="Customer Success Manager",
        persona="I am a customer success manager for TechInnovate Solutions. I help customers maximize value from our products, ensure successful implementation, and drive adoption and retention through proactive engagement.",
        objectives=[
            "Ensure successful product implementation and adoption",
            "Identify opportunities to expand product usage",
            "Proactively address potential issues before they impact customers",
            "Build strong relationships with key customer stakeholders"
        ],
        communication_style="Proactive, strategic, and relationship-focused",
        skills=[
            "Customer relationship management",
            "Strategic advisory",
            "Product expertise",
            "Value realization"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="customer_success",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7,
            response_format=ResponseFormat.text
        ),
        persona=persona
    )
    
    # Add knowledge base to agent memory
    await agent.add_to_memory("knowledge_base", json.dumps(KNOWLEDGE_BASE))
    
    print(f"Created Customer Success Agent with ID: {agent.id}")
    print("Added knowledge base to agent memory")
    
    return agent

async def create_customer_journey_team():
    """Create a team of agents to handle the complete customer journey."""
    
    # Create individual agents
    sales_agent = await create_sales_agent()
    support_agent = await create_customer_support_agent()
    success_agent = await create_customer_success_agent()
    
    # Define chat environment
    chat_env = {
        "name": "customer_journey_chat",
        "mechanism": "chat",
        "form_cohorts": False,
        "sub_rounds": 3,
        "group_size": 3,
        "task_prompt": "Collaborate to provide a seamless customer experience across the entire customer journey."
    }
    
    # Create the customer journey team
    customer_journey_team = MarketAgentTeam(
        name="Customer Journey Team",
        manager=sales_agent,  # Sales leads the initial customer engagement
        agents=[
            support_agent,
            success_agent
        ],
        mode="collaborative",
        use_group_chat=True,
        shared_context={
            "company_name": "TechInnovate Solutions",
            "customer_journey_stages": ["Awareness", "Consideration", "Purchase", "Onboarding", "Adoption", "Renewal"],
            "team_objective": "Provide a seamless, positive experience across the entire customer journey"
        },
        environments=[
            chat_env
        ]
    )
    
    print(f"Created Customer Journey Team: {customer_journey_team.name}")
    print(f"Team members: {[agent.id for agent in customer_journey_team.agents]}")
    print(f"Team manager: {customer_journey_team.manager.id}")
    
    return customer_journey_team

async def handle_customer_inquiry(team, customer_inquiry):
    """Handle a customer inquiry with the customer journey team."""
    
    task = f"""
    As a customer journey team, collaborate to address the following customer inquiry:
    
    CUSTOMER INQUIRY:
    {customer_inquiry}
    
    Your team should:
    
    1. The Sales Representative should:
       - Identify if this is a new prospect or existing customer
       - Understand the customer's needs and requirements
       - Provide relevant product and pricing information if needed
       - Guide the customer through the sales process if appropriate
    
    2. The Customer Support Specialist should:
       - Address any technical questions or issues
       - Provide accurate product information
       - Offer troubleshooting assistance if needed
       - Ensure the customer's immediate needs are met
    
    3. The Customer Success Manager should:
       - Identify opportunities to enhance the customer's experience
       - Suggest ways to maximize value from our products
       - Provide strategic advice on implementation and adoption
       - Ensure long-term customer success and satisfaction
    
    Collaborate to provide a comprehensive, seamless response that addresses all aspects of the customer's inquiry and ensures a positive experience.
    
    The final response should be a unified message from the team that:
    - Directly addresses the customer's inquiry
    - Provides clear, accurate information
    - Offers next steps or solutions
    - Maintains a friendly, helpful tone
    """
    
    print(f"\nHandling customer inquiry: {customer_inquiry}")
    print("-" * 80)
    
    # Run the team discussion to handle the inquiry
    result = await team.execute(task)
    
    print("\nCustomer Inquiry Handled")
    print("-" * 80)
    print(result)
    
    return result

async def main():
    # Create a customer journey team
    team = await create_customer_journey_team()
    
    # Handle various customer inquiries
    customer_inquiries = [
        """
        Hello, I'm the CTO of a mid-sized financial services company. We're currently evaluating cloud management solutions and I'd like to learn more about CloudManage Pro. Specifically, I'm interested in how it handles multi-cloud environments and what security certifications it has. We're also concerned about scalability as we expect significant growth over the next 18 months. Could you provide information on these aspects and your pricing tiers?
        """,
        
        """
        Hi there, I'm a current customer using DataInsight Suite on the Advanced plan. We're having some issues connecting to our new MongoDB database. The connection keeps timing out after about 30 seconds. We've verified that the database is accessible from other tools. Is this a known issue? Do you have any recommendations for resolving this? We need to get this working by the end of the week for an important project.
        """,
        
        """
        Good morning, we implemented SecureConnect about three months ago and while it's working well overall, we're not seeing the level of adoption we expected among our team members. Some are still using workarounds that don't comply with our security policies. We're on the Professional plan with about 300 users. Do you have any recommendations for improving adoption and ensuring compliance across our organization?
        """
    ]
    
    for inquiry in customer_inquiries:
        await handle_customer_inquiry(team, inquiry)

if __name__ == "__main__":
    asyncio.run(main())
