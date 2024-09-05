import json
import os
import dash
import argparse
import logging

from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np
from datetime import datetime

from llm_agents.environments.auction.auction_environment import Environment, generate_llm_market_agents
from llm_agents.environments.auction.auction import DoubleAuction

from base_agent.utils import setup_logger  # Import the setup_logger function
from base_agent.aiutilities import LLMConfig  # Import LLMConfig from aiutilities

# Parse command line arguments
parser = argparse.ArgumentParser(description='Market Simulation Dashboard')
parser.add_argument('--max_rounds', type=int, default=5, help='Maximum number of rounds for the double auction simulation')
parser.add_argument('--log_level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                    help='Set the logging level for simulation logs')
args = parser.parse_args()

# Create log directory
log_dir = "logs/sim_logs"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(log_dir, f"simulation_{datetime.now().strftime('%Y%m%d%H%M%S')}.log")
setup_logger(log_file, log_level=args.log_level)

# Get a logger for this file
logger = logging.getLogger(__name__)

# Initialize environment and auction
num_buyers = 5
num_sellers = 5
spread = 0.5

llm_config = LLMConfig(
    client="openai",
    model="gpt-4o-mini",
    temperature=0.5,
    response_format="json_object",
    max_tokens=4096,
    use_cache=True
)

agents = generate_llm_market_agents(
    num_agents=num_buyers + num_sellers, 
    num_units=5, 
    buyer_base_value=100, 
    seller_base_value=80, 
    spread=spread, 
    use_llm=True,
    llm_config=llm_config,
    initial_cash=1000,
    initial_goods=0,
    noise_factor=0.1
)

env = Environment(agents=agents)
auction = DoubleAuction(environment=env, max_rounds=args.max_rounds)

# Global variables to store data
data = {
    'trade_numbers': [],
    'prices': [],
    'cumulative_quantities': [],
    'cumulative_surplus': [],
    'price_history': [],  # New for price history
}

# Calculate equilibrium values
ce_price, ce_quantity, _, _, theoretical_total_surplus = env.calculate_equilibrium(initial=True)

app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

app.layout = html.Div([
    html.H1("Market Simulation Live Dashboard", className="text-center mb-4"),
    html.Div([
        html.Button('Start Simulation', id='start-button', n_clicks=0, className="btn btn-primary mb-3 mr-2"),
        html.Button('Stop Simulation', id='stop-button', n_clicks=0, className="btn btn-danger mb-3 ml-2", disabled=True),
    ], className="text-center"),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0,
        disabled=True
    ),
    html.Div([
        # First Column: Tables (4 tables in a 2x2 grid)
        html.Div([
            # Market State Table
            html.Div([
                html.H3("Market State", className="card-header"),
                html.Div(id='market-state-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
            ], className="card mb-4"),
            
            # Order Book Table
            html.Div([
                html.H3("Order Book", className="card-header"),
                html.Div(id='order-book-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
            ], className="card mb-4"),

            # Trade History Table
            html.Div([
                html.H3("Trade History", className="card-header"),
                html.Div(id='trade-history-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
            ], className="card mb-4"),
            
        ], className="col-md-6"),  # Half of the screen
        
        # Second Column: Charts (4 charts in a 2x2 grid)
        html.Div([
            # Supply and Demand Chart
            html.Div([
                dcc.Graph(id='supply-demand-chart', config={'displayModeBar': False})
            ], className="card mb-4"),

            # Price vs Trade Chart
            html.Div([
                dcc.Graph(id='price-vs-trade-chart', config={'displayModeBar': False})
            ], className="card mb-4"),
            
            # Cumulative Quantity vs Surplus Chart
            html.Div([
                dcc.Graph(id='cumulative-quantity-surplus-chart', config={'displayModeBar': False})
            ], className="card mb-4"),
            
        ], className="col-md-6"),  # Half of the screen
    ], className="row"),
    
], className="container-fluid")

# Global variable to track if the auction has been run
auction_completed = False

@app.callback(
    [Output('interval-component', 'disabled'),
     Output('start-button', 'disabled'),
     Output('stop-button', 'disabled')],
    [Input('start-button', 'n_clicks'),
     Input('stop-button', 'n_clicks')],
    [State('interval-component', 'disabled')]
)
def toggle_simulation(start_clicks, stop_clicks, interval_disabled):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True, False, True
    else:
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'start-button' and start_clicks > 0:
        return False, True, False
    elif button_id == 'stop-button' and stop_clicks > 0:
        return True, False, True
    else:
        return interval_disabled, not interval_disabled, interval_disabled

@app.callback(
    [Output('market-state-table', 'children'),
     Output('order-book-table', 'children'),
     Output('trade-history-table', 'children'),
     Output('supply-demand-chart', 'figure'),
     Output('price-vs-trade-chart', 'figure'),
     Output('cumulative-quantity-surplus-chart', 'figure')],
    Input('interval-component', 'n_intervals'),
    State('start-button', 'n_clicks')
)
def update_dashboard(n, start_clicks):
    global auction_completed
    
    if start_clicks == 0:
        raise PreventUpdate

    if not auction_completed:
        logger.info("Starting auction simulation")
        # Run the entire auction
        while auction.current_round < auction.max_rounds:
            auction.run_auction()
            logger.info(f"Auction round {auction.current_round} completed")

            # save agent interactions after each round
            for agent in env.agents:
                logger.info(f"Saving logs for agent {agent.zi_agent.id}, round {auction.current_round}")
                if hasattr(agent, 'memory'):
                    save_llama_logs(agent.memory, agent.zi_agent.id, auction.current_round)
                else:
                    logger.warning(f"Agent {agent.zi_agent.id} does not have memory attribute")

            # Update data storage
            update_data_storage()

        auction_completed = True
        logger.info("Auction simulation completed")

    # Generate tables and charts
    market_state_table = generate_market_state_table(env)
    order_book_table = generate_order_book_table(auction)
    trade_history_table = generate_trade_history_table(auction)
    supply_demand_chart = generate_supply_demand_chart(env)
    price_vs_trade_chart = generate_price_vs_trade_chart()
    cumulative_quantity_surplus_chart = generate_cumulative_quantity_surplus_chart()

    return (market_state_table, order_book_table, trade_history_table, supply_demand_chart, 
            price_vs_trade_chart, cumulative_quantity_surplus_chart)

def update_data_storage():
    data['trade_numbers'].append(len(auction.successful_trades))
    if auction.successful_trades:
        last_trade = auction.successful_trades[-1]
        data['prices'].append(last_trade.price)
        data['cumulative_quantities'].append(sum(trade.quantity for trade in auction.successful_trades))
        data['cumulative_surplus'].append(auction.total_surplus_extracted)
        data['price_history'].append(last_trade.price)
    else:
        # To keep lengths consistent, append `None` or the last known value
        data['prices'].append(None)
        data['cumulative_quantities'].append(data['cumulative_quantities'][-1] if data['cumulative_quantities'] else 0)
        data['cumulative_surplus'].append(data['cumulative_surplus'][-1] if data['cumulative_surplus'] else 0)
        data['price_history'].append(data['price_history'][-1] if data['price_history'] else None)

def generate_market_state_table(env):
    headers = ['Agent ID', 'Role', 'Goods', 'Cash', 'Individual Surplus']
    rows = []
    for agent in env.agents:
        ziagent = agent.zi_agent
        role = "Buyer" if ziagent.is_buyer else "Seller"
        individual_surplus = ziagent.individual_surplus  # Use the individual_surplus property
        rows.append([
            ziagent.id,
            role,
            ziagent.allocation.goods,
            f"${ziagent.allocation.cash:.2f}",
            f"{individual_surplus:.2f}"
        ])
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_order_book_table(auction):
    headers = ['Type', 'Agent ID', 'Price']
    rows = []
    
    # Add bids to the table
    for bid in auction.order_book.bids:
        rows.append([
            'Bid',
            bid.agent_id,
            bid.market_action.price
        ])
    
    # Add asks to the table
    for ask in auction.order_book.asks:
        rows.append([
            'Ask',
            ask.agent_id,
            ask.market_action.price
        ])
    
    # Sort rows by price (descending for bids, ascending for asks)
    rows.sort(key=lambda x: (-float(x[2]) if x[0] == 'Bid' else float(x[2])))
    
    # Format price as string after sorting
    rows = [[row[0], row[1], f"${float(row[2]):.2f}"] for row in rows]
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_trade_history_table(auction):
    headers = ['Trade ID', 'Buyer ID', 'Seller ID', 'Price', 'Quantity']
    rows = []
    for trade in auction.trade_history:
        rows.append([
            trade.trade_id,
            trade.bid.agent_id,
            trade.ask.agent_id,
            f"${trade.price:.2f}",
            trade.quantity
        ])
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_supply_demand_chart(env):
    # Get the initial supply and demand curves
    initial_demand_curve = env.initial_demand_curve
    initial_supply_curve = env.initial_supply_curve

    # Get x and y values for demand and supply curves
    demand_x, demand_y = initial_demand_curve.get_x_y_values()
    supply_x, supply_y = initial_supply_curve.get_x_y_values()

    # Calculate equilibrium values
    ce_price, ce_quantity, _, _, _ = env.calculate_equilibrium(initial=True)

    fig = go.Figure()
    
    # Plot demand curve
    fig.add_trace(go.Scatter(x=demand_x, y=demand_y, mode='lines', name='Demand', line=dict(color='blue')))
    
    # Plot supply curve
    fig.add_trace(go.Scatter(x=supply_x, y=supply_y, mode='lines', name='Supply', line=dict(color='red')))
    
    # Plot equilibrium lines
    fig.add_trace(go.Scatter(x=[ce_quantity, ce_quantity], y=[0, ce_price], mode='lines', name='CE Quantity', line=dict(color='green', dash='dash')))
    fig.add_trace(go.Scatter(x=[0, ce_quantity], y=[ce_price, ce_price], mode='lines', name='CE Price', line=dict(color='green', dash='dash')))

    fig.update_layout(
        title="Supply and Demand Curve",
        xaxis_title="Quantity",
        yaxis_title="Price",
        template="plotly_white",
        height=350
    )
    
    return fig

def generate_price_vs_trade_chart():
    fig = go.Figure()
    
    # Fetch actual prices from auction's successful trades
    trade_numbers = list(range(1, len(auction.successful_trades) + 1))
    prices = [trade.price for trade in auction.successful_trades]

    fig.add_trace(go.Scatter(x=trade_numbers, y=prices, mode='lines+markers', name='Price per Trade', line=dict(color='blue')))

    fig.update_layout(
        title="Price vs Trade",
        xaxis_title="Trade Number",
        yaxis_title="Price",
        template="plotly_white",
        height=350
    )
    
    return fig

def generate_cumulative_quantity_surplus_chart():
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Calculate cumulative quantities and surplus
    cumulative_quantities = []
    cumulative_surplus = []
    total_quantity = 0
    total_surplus = 0
    
    for round_num in range(1, auction.max_rounds + 1):
        trades_in_round = [trade for trade in auction.successful_trades if trade.round == round_num]
        total_quantity += sum(trade.quantity for trade in trades_in_round)
        total_surplus += sum((trade.buyer_value - trade.price) + (trade.price - trade.seller_cost) for trade in trades_in_round)
        
        cumulative_quantities.append(total_quantity)
        cumulative_surplus.append(total_surplus)

    trade_numbers = list(range(1, len(cumulative_quantities) + 1))

    fig.add_trace(go.Scatter(x=trade_numbers, y=cumulative_quantities, mode='lines+markers', name='Cumulative Quantity', line=dict(color='red')), secondary_y=False)
    fig.add_trace(go.Scatter(x=trade_numbers, y=cumulative_surplus, mode='lines+markers', name='Cumulative Surplus', line=dict(color='green')), secondary_y=True)

    fig.update_layout(
        title="Cumulative Quantity vs Surplus",
        xaxis_title="Trade Number",
        template="plotly_white",
        height=350
    )
    fig.update_yaxes(title_text="Cumulative Quantity", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Surplus", secondary_y=True)
    
    return fig

def save_llama_logs(interactions, agent_id, round_number):
    log_path = "logs"
    qa_interactions_path = os.path.join(log_path, "qa_interactions")
    qa_interaction_file = f"qa_interactions_agent_{agent_id}.json"
    qa_interaction_path = os.path.join(qa_interactions_path, qa_interaction_file)

    logger.debug(f"Preparing to save logs for agent {agent_id}, round {round_number}")

    # Ensure the directory exists
    os.makedirs(qa_interactions_path, exist_ok=True)

    # Load existing data if file exists, or create an empty list
    if os.path.exists(qa_interaction_path):
        with open(qa_interaction_path, "r") as file:
            existing_data = json.load(file)
    else:
        existing_data = []

    # Append new interactions with round information
    new_entry = {
        "round": round_number,
        "interactions": interactions
    }
    existing_data.append(new_entry)

    # Write updated data back to file
    try:
        with open(qa_interaction_path, "w") as file:
            json.dump(existing_data, file, indent=2)
        logger.debug(f"Successfully saved logs to {qa_interaction_path}")
    except Exception as e:
        logger.error(f"Failed to save logs: {str(e)}")

if __name__ == '__main__':
    os.makedirs("logs/qa_interactions", exist_ok=True)

    logger.info("Starting the Dash server")
    app.run_server(debug=True)
