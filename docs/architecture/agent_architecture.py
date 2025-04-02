#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

# Create the agent architecture diagram
with Diagram("MarketAgent Architecture", show=False, filename="agent_architecture", outformat="png", direction="TB"):
    
    # Core agent components
    with Cluster("MarketAgent"):
        agent = Python("MarketAgent")
        
        with Cluster("Memory Systems"):
            stm = Storage("Short-Term Memory")
            ltm = Storage("Long-Term Memory")
            
        with Cluster("Cognitive Components"):
            prompter = Python("PromptManager")
            cognitive_steps = Python("CognitiveSteps")
            
        with Cluster("Specialized Components"):
            econ_agent = Python("EconomicAgent")
            kb_agent = Python("KnowledgeBaseAgent")
            rl_agent = Python("VerbalRLAgent")
            
        with Cluster("Communication"):
            protocol = Python("Protocol")
            
        with Cluster("Inference"):
            llm_orchestrator = Python("InferenceOrchestrator")
            llm_config = Python("LLMConfig")
    
    # External components
    llm_api = Rack("LLM APIs")
    vector_db = PostgreSQL("Vector Database")
    storage_api = Server("Storage API")
    environments = Python("Environments")
    
    # Connect components
    agent >> stm
    agent >> ltm
    
    stm >> storage_api
    ltm >> storage_api
    storage_api >> vector_db
    
    agent >> prompter
    agent >> cognitive_steps
    
    agent >> econ_agent
    agent >> kb_agent
    agent >> rl_agent
    
    agent >> protocol
    
    agent >> llm_orchestrator
    llm_orchestrator >> llm_config
    llm_orchestrator >> llm_api
    
    agent >> environments
    
    kb_agent >> vector_db
