#!/usr/bin/env python3
from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python
from diagrams.programming.framework import Flask
from diagrams.generic.storage import Storage
from diagrams.generic.compute import Rack

# Create the environment mechanisms diagram
with Diagram("Environment Mechanisms Architecture", show=False, filename="environment_mechanisms", outformat="png", direction="TB"):
    
    # Core environment components
    with Cluster("MultiAgentEnvironment"):
        environment = Python("MultiAgentEnvironment")
        
        with Cluster("Environment Mechanisms"):
            group_chat = Python("GroupChat")
            research = Python("Research")
            stock_market = Python("StockMarket")
            auction = Python("Auction")
            info_board = Python("InformationBoard")
            mcp_server = Python("MCPServer")
            
        with Cluster("Action & Observation Spaces"):
            action_space = Python("ActionSpace")
            observation_space = Python("ObservationSpace")
            
        with Cluster("State Management"):
            global_state = Storage("GlobalState")
            agent_state = Storage("AgentState")
    
    # External components
    agents = Python("MarketAgents")
    orchestrator = Python("Orchestrator")
    
    # API Services
    with Cluster("API Services"):
        groupchat_api = Flask("GroupChat API")
        mcp_api = Server("MCP Server API")
        finance_tools = Server("Finance Tools")
    
    # Connect components
    environment >> group_chat
    environment >> research
    environment >> stock_market
    environment >> auction
    environment >> info_board
    environment >> mcp_server
    
    environment >> action_space
    environment >> observation_space
    
    environment >> global_state
    environment >> agent_state
    
    agents >> environment
    orchestrator >> environment
    
    group_chat >> groupchat_api
    mcp_server >> mcp_api
    mcp_api >> finance_tools
    
    # Environment interactions
    group_chat >> Edge(label="messages") >> agents
    research >> Edge(label="findings") >> agents
    stock_market >> Edge(label="market data") >> agents
    auction >> Edge(label="bids/offers") >> agents
    info_board >> Edge(label="information") >> agents
    mcp_server >> Edge(label="tool results") >> agents
