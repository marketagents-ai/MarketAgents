#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server
from diagrams.onprem.client import User
from diagrams.programming.language import Python
from diagrams.programming.framework import Flask
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

# Create the orchestration flow diagram
with Diagram("Orchestration Flow Architecture", show=False, filename="orchestration_flow", outformat="png", direction="TB"):
    
    # Core orchestration components
    with Cluster("Orchestration Components"):
        meta_orchestrator = Python("MetaOrchestrator")
        orchestrator = Python("Orchestrator")
        team = Python("MarketAgentTeam")
        
        with Cluster("Configuration"):
            config = Storage("OrchestratorConfig")
            env_order = Storage("EnvironmentOrder")
            llm_configs = Storage("LLMConfigs")
        
        with Cluster("Execution Flow"):
            init = Python("Initialize")
            run = Python("RunOrchestration")
            step = Python("ExecuteStep")
            collect = Python("CollectResults")
    
    # External components
    with Cluster("Agents"):
        agent1 = Python("Agent1")
        agent2 = Python("Agent2")
        agent3 = Python("Agent3")
    
    with Cluster("Environments"):
        env1 = Python("GroupChatEnv")
        env2 = Python("ResearchEnv")
        env3 = Python("StockMarketEnv")
        env4 = Python("MCPServerEnv")
    
    # API Services
    with Cluster("API Services"):
        storage_api = Flask("Storage API")
        groupchat_api = Flask("GroupChat API")
        mcp_api = Server("MCP Server API")
    
    # User interaction
    user = User("Developer")
    dashboard = Flask("Dashboard")
    
    # Connect components
    user >> dashboard
    user >> Edge(label="configures") >> config
    
    config >> meta_orchestrator
    config >> orchestrator
    
    meta_orchestrator >> orchestrator
    orchestrator >> team
    team >> agent1
    team >> agent2
    team >> agent3
    
    meta_orchestrator >> env_order
    meta_orchestrator >> llm_configs
    
    orchestrator >> init
    init >> run
    run >> step
    step >> collect
    
    step >> env1
    step >> env2
    step >> env3
    step >> env4
    
    agent1 >> env1
    agent2 >> env1
    agent3 >> env1
    
    agent1 >> env2
    agent2 >> env2
    agent3 >> env2
    
    env1 >> groupchat_api
    env4 >> mcp_api
    agent1 >> storage_api
    agent2 >> storage_api
    agent3 >> storage_api
    
    dashboard >> Edge(label="monitors") >> meta_orchestrator
    dashboard >> Edge(label="monitors") >> orchestrator
    
    # Execution flow
    with Cluster("Execution Sequence"):
        flow1 = Python("1. Load Configuration")
        flow2 = Python("2. Initialize Agents")
        flow3 = Python("3. Setup Environments")
        flow4 = Python("4. For Each Environment:")
        flow5 = Python("5.   Run Environment Steps")
        flow6 = Python("6.   Collect Results")
        flow7 = Python("7. Return Final Results")
        
        flow1 >> flow2 >> flow3 >> flow4 >> flow5 >> flow6 >> flow7
