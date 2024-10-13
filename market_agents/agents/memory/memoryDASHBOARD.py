# maSPRINTBOARD.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
from datetime import datetime, timedelta
from memory import MemoryManager, EmbeddingModel, ChunkingStrategy
import colorama
from tqdm import tqdm
import time
import statistics

# Initialize colorama for colored output
colorama.init()

# Import Dash components and other required libraries
import dash
from dash import html, dcc, Input, Output, State, dash_table
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import re

#import nltk
#nltk.download('punkt_tab')

app = FastAPI()

# Initialize Dash app
dash_app = dash.Dash(__name__)

# Serve static files (for the GUI)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set the path to the folder containing JSON files
JSON_FOLDER = "test_jsonl"

# Initialize MemoryManager
memory_manager = MemoryManager()

# Load data from JSON files
def load_data():
    print(colorama.Fore.CYAN + "Loading and embedding data:" + colorama.Fore.RESET)
    for filename in tqdm(os.listdir(JSON_FOLDER), desc="Processing files"):
        if filename.endswith(".jsonl"):
            with open(os.path.join(JSON_FOLDER, filename), "r") as f:
                for line in f:
                    data = json.loads(line)
                    agent_id = re.search(r'agent_(\d+)', filename)
                    if agent_id:
                        agent_id = f"agent_{agent_id.group(1)}"
                        memory_id = f"{agent_id}_{data['id']}"
                        memory_manager.add_memory(agent_id, json.dumps(data), {"type": "interaction", "id": memory_id})
                    else:
                        print(f"Warning: Could not determine agent ID from filename {filename}")
    print(colorama.Fore.GREEN + "Data loading and embedding complete!" + colorama.Fore.RESET)

# Load data
load_data()

from pydantic import BaseModel
from typing import List, Dict, Union, Optional
import json

class SystemInfo(BaseModel):
    role: str
    content: str

class BaseAgentData(BaseModel):
    id: str
    name: str
    round: int
    system: SystemInfo
    task: str
    timestamp: str

class Trade(BaseModel):
    # Define trade attributes as needed
    buyer: str
    seller: str
    price: float
    quantity: int

class Order(BaseModel):
    # Define order attributes as needed
    agent_id: str
    price: float
    quantity: int

class MarketSummary(BaseModel):
    # Define market summary attributes as needed
    average_price: float
    total_volume: int

class PerceptionData(BaseAgentData):
    response: Dict[str, str]

class ActionData(BaseAgentData):
    response: Dict[str, float]

class ReflectionData(BaseAgentData):
    response: Dict[str, str]

class EnvironmentStateData(BaseModel):
    current_round: int
    trades: List[Trade]
    waiting_bids: List[Order]
    waiting_asks: List[Order]

class ObservationData(BaseModel):
    trades: List[Trade]
    market_summary: MarketSummary
    waiting_orders: List[Order]

DataModel = Union[PerceptionData, ActionData, ReflectionData, EnvironmentStateData, ObservationData]

def parse_json_to_model(json_data: Dict) -> DataModel:
    if "response" in json_data:
        if "monologue" in json_data["response"] and "strategy" in json_data["response"]:
            return PerceptionData(**json_data)
        elif "price" in json_data["response"] and "quantity" in json_data["response"]:
            return ActionData(**json_data)
        elif "reflection" in json_data["response"] and "strategy_update" in json_data["response"]:
            return ReflectionData(**json_data)
    elif "current_round" in json_data and "trades" in json_data:
        return EnvironmentStateData(**json_data)
    elif "trades" in json_data and "market_summary" in json_data:
        return ObservationData(**json_data)
    else:
        raise ValueError("Unknown data schema")

def json_to_markdown(data: Union[DataModel, Dict]) -> str:
    if isinstance(data, dict):
        data = parse_json_to_model(data)

    def format_perception(data: PerceptionData) -> str:
        md = f"# Perception (Round {data.round})\n\n"
        md += f"**Agent ID:** {data.id}\n\n"
        md += f"**Agent Name:** {data.name}\n\n"
        md += "## System\n"
        md += f"**Role:** {data.system.role}\n\n"
        md += f"**Content:** {data.system.content}\n\n"
        md += f"**Task:** {data.task}\n\n"
        md += "## Response\n"
        md += f"**Monologue:** {data.response['monologue']}\n\n"
        md += f"**Strategy:** {data.response['strategy']}\n\n"
        md += f"**Timestamp:** {data.timestamp}\n\n"
        return md

    def format_action(data: ActionData) -> str:
        md = f"# Action (Round {data.round})\n\n"
        md += f"**Agent ID:** {data.id}\n\n"
        md += f"**Agent Name:** {data.name}\n\n"
        md += "## System\n"
        md += f"**Role:** {data.system.role}\n\n"
        md += f"**Content:** {data.system.content}\n\n"
        md += f"**Task:** {data.task}\n\n"
        md += "## Response\n"
        md += f"**Price:** {data.response['price']}\n\n"
        md += f"**Quantity:** {data.response['quantity']}\n\n"
        md += f"**Timestamp:** {data.timestamp}\n\n"
        return md

    def format_reflection(data: ReflectionData) -> str:
        md = f"# Reflection (Round {data.round})\n\n"
        md += f"**Agent ID:** {data.id}\n\n"
        md += f"**Agent Name:** {data.name}\n\n"
        md += "## System\n"
        md += f"**Role:** {data.system.role}\n\n"
        md += f"**Content:** {data.system.content}\n\n"
        md += f"**Task:** {data.task}\n\n"
        md += "## Response\n"
        md += f"**Reflection:** {data.response['reflection']}\n\n"
        md += f"**Strategy Update:** {data.response['strategy_update']}\n\n"
        md += f"**Timestamp:** {data.timestamp}\n\n"
        return md

    def format_environment_state(data: EnvironmentStateData) -> str:
        md = f"# Environment State (Round {data.current_round})\n\n"
        md += "## Trades\n"
        for trade in data.trades:
            md += "- **Trade**\n"
            md += f"  - Buyer: {trade.buyer}\n"
            md += f"  - Seller: {trade.seller}\n"
            md += f"  - Price: {trade.price}\n"
            md += f"  - Quantity: {trade.quantity}\n"
            md += "\n"
        md += "## Waiting Bids\n"
        for bid in data.waiting_bids:
            md += "- **Bid**\n"
            md += f"  - Agent ID: {bid.agent_id}\n"
            md += f"  - Price: {bid.price}\n"
            md += f"  - Quantity: {bid.quantity}\n"
            md += "\n"
        md += "## Waiting Asks\n"
        for ask in data.waiting_asks:
            md += "- **Ask**\n"
            md += f"  - Agent ID: {ask.agent_id}\n"
            md += f"  - Price: {ask.price}\n"
            md += f"  - Quantity: {ask.quantity}\n"
            md += "\n"
        return md

    def format_observation(data: ObservationData) -> str:
        md = "# Observation\n\n"
        md += "## Trades\n"
        for trade in data.trades:
            md += "- **Trade**\n"
            md += f"  - Buyer: {trade.buyer}\n"
            md += f"  - Seller: {trade.seller}\n"
            md += f"  - Price: {trade.price}\n"
            md += f"  - Quantity: {trade.quantity}\n"
            md += "\n"
        md += "## Market Summary\n"
        md += f"- **Average Price:** {data.market_summary.average_price}\n"
        md += f"- **Total Volume:** {data.market_summary.total_volume}\n"
        md += "\n## Waiting Orders\n"
        for order in data.waiting_orders:
            md += "- **Order**\n"
            md += f"  - Agent ID: {order.agent_id}\n"
            md += f"  - Price: {order.price}\n"
            md += f"  - Quantity: {order.quantity}\n"
            md += "\n"
        return md

    if isinstance(data, PerceptionData):
        return format_perception(data)
    elif isinstance(data, ActionData):
        return format_action(data)
    elif isinstance(data, ReflectionData):
        return format_reflection(data)
    elif isinstance(data, EnvironmentStateData):
        return format_environment_state(data)
    elif isinstance(data, ObservationData):
        return format_observation(data)
    else:
        return json.dumps(data.dict(), indent=2)  # Fallback to simple JSON formatting


@app.get("/agents")
async def get_agents():
    return list(memory_manager.memories.keys())

@app.get("/search")
async def search(query: str = Query(..., min_length=1), agent: str = Query("all"), top_k: int = Query(10, gt=0)):
    try:
        if agent == "all":
            results = []
            for agent_id in memory_manager.memories.keys():
                agent_results = memory_manager.search(agent_id, query, top_k=top_k)
                results.extend(agent_results)
            results.sort(key=lambda x: x.score, reverse=True)
            results = results[:top_k]
        else:
            results = memory_manager.search(agent, query, top_k=top_k)

        formatted_results = [
            {
                'id': result.id,
                'agent_id': result.agent_id,
                'content': json_to_markdown(json.loads(result.content)),
                'score': result.score,
            }
            for result in results
        ]
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings_graph")
async def get_embeddings_graph(search_results: List[Dict[str, Any]]):
    embeddings = []
    texts = []
    ids = []
    for result in search_results:
        memory = next((m for m in memory_manager.memories[result['agent_id']] if m.id == result['id']), None)
        if memory:
            embedding = memory_manager.embedding_model.embed(memory.content)
            embeddings.append(embedding)
            texts.append(memory.content)
            ids.append(memory.id)

    embeddings = np.array(embeddings)

    if embeddings.shape[0] < 2:
        return {"error": "Not enough data points for visualization."}
    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings).tolist()

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter([point[0] for point in reduced_embeddings], [point[1] for point in reduced_embeddings])
    ax.set_title('Embeddings Visualization')
    #ax.set_xlabel('Dimension 1')
    #ax.set_ylabel('Dimension 2')

    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)

    img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')

    plt.close()

    return {
        "image_data": f"data:image/png;base64,{img_base64}",
        "reduced_embeddings": reduced_embeddings,
        "texts": texts,
        "ids": ids
    }

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

# Create Dash app layout
dash_app.layout = html.Div([
    html.H1("Interactive Memory and Market Dashboard"),
    
    # Search Section
    html.Div([
        html.H2("Search Memories"),
        dcc.Dropdown(
            id='agent-select',
            options=[{'label': 'All Agents', 'value': 'all'}],
            value='all',
            style={'width': '100%', 'marginBottom': '10px'}
        ),
        dcc.Input(id='search-input', type='text', placeholder='Enter your query', style={'width': '70%'}),
        html.Button('Search', id='search-button', style={'width': '30%'}),
    ], style={'margin': '20px'}),
    
    # Search Results
    html.Div([
        html.H3("Search Results"),
        dash_table.DataTable(
            id='search-results-table',
            columns=[
                {'name': 'Agent ID', 'id': 'agent_id'},
                {'name': 'Content', 'id': 'content'},
                {'name': 'Score', 'id': 'score'}
            ],
            data=[],
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '180px', 'maxWidth': '300px'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            markdown_options={'html': True}  # Enable HTML rendering for markdown content
        )
    ], style={'margin': '20px'}),
    
    # Embeddings Visualization
    html.Div([
        html.H3("Embeddings Visualization"),
        dcc.Graph(id='embeddings-graph')
    ], style={'margin': '20px'}),
])

# Callbacks for Dash app
@dash_app.callback(
    [Output('search-results-table', 'data'),
     Output('embeddings-graph', 'figure')],
    [Input('search-button', 'n_clicks')],
    [State('search-input', 'value'),
     State('agent-select', 'value')]
)
def update_results(n_clicks, query, agent):
    if n_clicks is None or not query:
        return [], {}

    # Perform search
    results = app.search(query=query, agent=agent, top_k=10)

    # Update search results table
    table_data = [
        {
            'agent_id': result['agent_id'],
            'content': result['content'],  # This is now markdown
            'score': round(result['score'], 4)
        }
        for result in results
    ]

    # Update embeddings visualization
    embeddings_data = app.get_embeddings_graph(results)
    
    if 'error' in embeddings_data:
        return table_data, {}

    figure = {
        'data': [{
            'x': [point[0] for point in embeddings_data['reduced_embeddings']],
            'y': [point[1] for point in embeddings_data['reduced_embeddings']],
            'mode': 'markers',
            'type': 'scatter',
            'text': embeddings_data['texts'],
            'hoverinfo': 'text'
        }],
        'layout': {
            'title': 'Embeddings Visualization',
            'xaxis': {'title': 'Dimension 1'},
            'yaxis': {'title': 'Dimension 2'}
        }
    }

    return table_data, figure

@dash_app.callback(
    Output('agent-select', 'options'),
    [Input('agent-select', 'search_value')]
)
def update_agent_options(search_value):
    agents = app.get_agents()
    options = [{'label': 'All Agents', 'value': 'all'}] + [{'label': agent, 'value': agent} for agent in agents]
    return options

if __name__ == "__main__":
    print(colorama.Fore.CYAN + "Starting server..." + colorama.Fore.RESET)
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)