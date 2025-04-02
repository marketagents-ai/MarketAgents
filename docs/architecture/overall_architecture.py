#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.database import PostgreSQL
from diagrams.onprem.queue import Kafka
from diagrams.programming.framework import Flask
from diagrams.programming.language import Python
from diagrams.generic.compute import Rack
from diagrams.generic.storage import Storage
from diagrams.generic.place import Datacenter

# Create the overall architecture diagram
with Diagram("MarketAgents Framework Architecture", show=False, filename="overall_architecture", outformat="png", direction="TB"):
    
    # External services and APIs
    with Cluster("External Services"):
        llm_api = Rack("LLM APIs")
        data_sources = Datacenter("External Data Sources")
    
    # Core infrastructure
    with Cluster("Infrastructure Services"):
        postgres = PostgreSQL("Vector Database")
        storage_api = Flask("Storage API")
        groupchat_api = Flask("GroupChat API")
        mcp_servers = Server("MCP Servers")
    
    # Core framework components
    with Cluster("MarketAgents Core"):
        with Cluster("Agent Components"):
            market_agent = Python("MarketAgent")
            memory = Storage("Memory Systems")
            econ_agent = Python("EconomicAgent")
            kb_agent = Python("KnowledgeBaseAgent")
        
        with Cluster("Environment Mechanisms"):
            group_chat = Python("GroupChat")
            research = Python("Research")
            stock_market = Python("StockMarket")
            mcp_env = Python("MCPServer")
        
        with Cluster("Orchestration"):
            orchestrator = Python("Orchestrator")
            meta_orchestrator = Python("MetaOrchestrator")
            team = Python("MarketAgentTeam")
    
    # User interaction
    user = User("Developer")
    dashboard = Flask("Dashboard")
    
    # Connect components
    user >> dashboard
    user >> Edge(label="creates") >> market_agent
    
    market_agent >> memory
    market_agent >> econ_agent
    market_agent >> kb_agent
    
    memory >> storage_api
    storage_api >> postgres
    
    market_agent >> group_chat
    market_agent >> research
    market_agent >> stock_market
    market_agent >> mcp_env
    
    group_chat >> groupchat_api
    
    team >> market_agent
    orchestrator >> team
    meta_orchestrator >> orchestrator
    
    mcp_env >> mcp_servers
    mcp_servers >> data_sources
    
    market_agent >> llm_api
