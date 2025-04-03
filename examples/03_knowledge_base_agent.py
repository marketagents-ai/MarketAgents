import asyncio
import os
from pathlib import Path

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.memory.knowledge_base import MarketKnowledgeBase
from market_agents.memory.knowledge_base_agent import KnowledgeBaseAgent
from minference.lite.models import LLMConfig, ResponseFormat

async def create_knowledge_base():
    """
    Create and initialize a knowledge base for the agent.
    """
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Create knowledge base
    kb = MarketKnowledgeBase(
        config=storage_config,
        table_prefix="ai_ethics_kb"
    )
    
    # Initialize the knowledge base
    await kb.initialize()
    
    # Check if tables exist, create if not
    tables_exist = await kb.check_table_exists()
    if not tables_exist:
        await kb.create_tables()
        print("Created knowledge base tables")
    else:
        print("Knowledge base tables already exist")
    
    return kb

async def populate_knowledge_base(kb):
    """
    Populate the knowledge base with AI ethics information.
    """
    # Sample AI ethics principles and guidelines
    ai_ethics_documents = [
        {
            "title": "Transparency in AI Systems",
            "content": """
            Transparency in AI systems refers to the ability to understand how AI systems make decisions.
            This includes:
            1. Explainability: The ability to explain in understandable terms how the AI reached its conclusion.
            2. Interpretability: The ability to understand the internal mechanics of an AI system.
            3. Disclosure: Clear communication about when AI systems are being used.
            
            Transparency is essential for building trust in AI systems and allowing users to understand when and how
            decisions are being made that affect them. It enables users to contest decisions and provides a basis
            for accountability when systems cause harm.
            
            Best practices for transparency include:
            - Providing clear documentation of AI system capabilities and limitations
            - Explaining key factors that influenced a particular decision
            - Making information about data sources and model training accessible
            - Using visualization tools to help users understand complex AI processes
            """,
            "metadata": {
                "category": "AI Ethics",
                "principle": "Transparency",
                "importance": "High"
            }
        },
        {
            "title": "Fairness and Bias Mitigation",
            "content": """
            AI systems should be designed to avoid creating or reinforcing unfair bias against certain individuals or groups.
            Key considerations include:
            
            1. Representational fairness: Ensuring training data represents diverse populations
            2. Allocational fairness: Ensuring resources or opportunities are distributed fairly
            3. Quality of service: Ensuring the system works equally well for all users
            
            Bias can enter AI systems through:
            - Biased training data reflecting historical or societal inequalities
            - Problem formulation that inadvertently encodes biased assumptions
            - Feature selection that proxies for protected attributes
            - Algorithmic design choices that amplify existing patterns of inequality
            
            Mitigation strategies include:
            - Diverse and representative training data
            - Regular bias audits throughout the development lifecycle
            - Fairness constraints in model optimization
            - Diverse development teams to identify potential bias issues
            - Post-deployment monitoring for disparate impact
            """,
            "metadata": {
                "category": "AI Ethics",
                "principle": "Fairness",
                "importance": "High"
            }
        },
        {
            "title": "Privacy and Data Protection",
            "content": """
            AI systems often rely on large amounts of data, including personal data, raising important privacy concerns.
            Key privacy principles for AI include:
            
            1. Data minimization: Only collecting data necessary for the intended purpose
            2. Purpose limitation: Using data only for specified, explicit, and legitimate purposes
            3. Storage limitation: Keeping data only as long as necessary
            4. Security: Protecting data against unauthorized access or breaches
            5. Individual rights: Respecting rights to access, correct, and delete personal data
            
            Privacy-enhancing technologies for AI include:
            - Federated learning: Training models across multiple devices without centralizing data
            - Differential privacy: Adding noise to data to protect individual privacy while preserving utility
            - Homomorphic encryption: Computing on encrypted data without decryption
            - Synthetic data: Using artificially generated data that preserves statistical properties
            
            Regulatory frameworks like GDPR, CCPA, and others establish legal requirements for data protection
            that AI systems must comply with in various jurisdictions.
            """,
            "metadata": {
                "category": "AI Ethics",
                "principle": "Privacy",
                "importance": "High"
            }
        },
        {
            "title": "Accountability in AI Systems",
            "content": """
            Accountability refers to the obligation of AI developers, deployers, and users to take responsibility
            for the functioning and impacts of AI systems. Key aspects include:
            
            1. Clear allocation of responsibility: Determining who is responsible when AI systems cause harm
            2. Auditability: Enabling third-party verification of system behavior
            3. Redress mechanisms: Providing ways for affected individuals to seek remedies
            4. Liability frameworks: Establishing legal responsibility for AI-related harms
            
            Implementing accountability requires:
            - Documentation of design decisions and system limitations
            - Impact assessments before deployment in high-risk contexts
            - Monitoring systems for unexpected behaviors or outcomes
            - Establishing clear chains of responsibility within organizations
            - Regular audits by internal teams or external entities
            
            Governance structures like AI ethics boards, algorithmic impact assessments, and
            third-party auditing frameworks help operationalize accountability principles.
            """,
            "metadata": {
                "category": "AI Ethics",
                "principle": "Accountability",
                "importance": "High"
            }
        },
        {
            "title": "Human-Centered AI Design",
            "content": """
            Human-centered AI design puts human needs, capabilities, and values at the center of AI development.
            Core principles include:
            
            1. Human autonomy: AI systems should respect human agency and decision-making authority
            2. Human oversight: Humans should maintain appropriate control over AI systems
            3. Augmentation not replacement: AI should enhance human capabilities rather than replace humans
            4. Accessibility: AI benefits should be accessible to diverse users with different abilities
            
            Design approaches include:
            - Participatory design involving potential users and affected stakeholders
            - Value-sensitive design that explicitly accounts for human values
            - Universal design principles to ensure accessibility
            - Appropriate levels of human control based on context and risk
            
            Human-centered AI requires interdisciplinary collaboration between technical experts,
            domain specialists, ethicists, designers, and representatives of affected communities.
            """,
            "metadata": {
                "category": "AI Ethics",
                "principle": "Human-Centered Design",
                "importance": "Medium"
            }
        }
    ]
    
    # Add documents to knowledge base
    for doc in ai_ethics_documents:
        doc_id = await kb.add_document(
            content=doc["content"],
            metadata=doc["metadata"]
        )
        print(f"Added document: {doc['title']} with ID: {doc_id}")
    
    return len(ai_ethics_documents)

async def create_ethics_agent():
    """
    Create an AI ethics agent with knowledge base integration.
    """
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create and populate knowledge base
    kb = await create_knowledge_base()
    doc_count = await populate_knowledge_base(kb)
    
    # Create knowledge base agent
    kb_agent = KnowledgeBaseAgent(
        market_kb=kb,
        id="ai_ethics_kb_agent"
    )
    
    # Create persona
    persona = Persona(
        role="AI Ethics Advisor",
        persona="I am an AI ethics advisor specializing in responsible AI development and deployment. I provide guidance on ethical considerations, best practices, and potential risks in AI systems.",
        objectives=[
            "Provide ethical guidance for AI development",
            "Identify potential ethical risks in AI systems",
            "Recommend best practices for responsible AI",
            "Promote human-centered AI design principles"
        ]
    )
    
    # Create agent with knowledge base integration
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="ethics_advisor",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona,
        knowledge_agent=kb_agent
    )
    
    print(f"Created ethics agent with ID: {agent.id}")
    print(f"Role: {agent.role}")
    print(f"Knowledge base documents: {doc_count}")
    
    return agent

async def ask_ethics_question(agent, question):
    """
    Ask the ethics agent a question, leveraging its knowledge base.
    """
    print(f"\nQuestion: {question}")
    print("-" * 80)
    
    # First, query the knowledge base for relevant information
    kb_results = await agent.knowledge_agent.query(
        query=question,
        top_k=2
    )
    
    # Format the knowledge base results as context
    context = ""
    if kb_results:
        context = "Relevant information from my knowledge base:\n\n"
        for i, result in enumerate(kb_results, 1):
            context += f"Source {i}:\n{result.content}\n\n"
    
    # Generate response using the agent's LLM with knowledge base context
    prompt = f"""
    You are {agent.role}. {agent.persona}
    
    Please answer the following question about AI ethics:
    
    {question}
    
    {context}
    
    Based on the above information and your knowledge, provide a comprehensive and nuanced response.
    Include specific recommendations or best practices where appropriate.
    """
    
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[{"role": "system", "content": prompt}]
    )
    
    print(f"Response: {response.content}")
    return response.content

async def main():
    # Create an ethics agent with knowledge base
    agent = await create_ethics_agent()
    
    # Ask the agent some ethics questions
    questions = [
        "How can we ensure AI systems are transparent to users?",
        "What are the best practices for mitigating bias in AI systems?",
        "How should companies handle privacy concerns when developing AI?",
        "What accountability mechanisms should be in place for AI systems?",
        "How can we design AI systems that augment human capabilities rather than replace humans?"
    ]
    
    for question in questions:
        await ask_ethics_question(agent, question)

if __name__ == "__main__":
    asyncio.run(main())
