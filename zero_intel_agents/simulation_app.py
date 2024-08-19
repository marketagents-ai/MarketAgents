import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dash.exceptions import PreventUpdate

from environment import Environment, generate_agents
from auction import DoubleAuction
from ziagents import ZIAgent, PreferenceSchedule, Allocation, Order, Trade

# Initialize environment and auction
num_buyers = 1000
num_sellers = 1000
spread = 0.5
agents = generate_agents(num_agents=num_buyers + num_sellers, num_units=5, buyer_base_value=100, seller_base_value=80, spread=spread)
env = Environment(agents=agents)
auction = DoubleAuction(environment=env, max_rounds=25)

# Global variables to store data
data = {
    'trade_numbers': [],
    'prices': [],
    'cumulative_quantities': [],
    'cumulative_surplus': [],
    'price_history': [],  # New for price history
}

# Calculate equilibrium values
ce_price, ce_quantity, _, _, theoretical_total_surplus = env.calculate_equilibrium()

app = dash.Dash(__name__, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

app.layout = html.Div([
    html.H1("Market Simulation Live Dashboard", className="text-center mb-4"),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
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

@app.callback(
    [Output('market-state-table', 'children'),
     Output('order-book-table', 'children'),
     Output('trade-history-table', 'children'),
     Output('supply-demand-chart', 'figure'),
     Output('price-vs-trade-chart', 'figure'),
     Output('cumulative-quantity-surplus-chart', 'figure')],
    Input('interval-component', 'n_intervals')
)
def update_dashboard(n):
    if n is None:
        raise PreventUpdate

    if auction.current_round < auction.max_rounds:
        # Run one step of the auction
        auction.run_auction()

        # Update data storage only if there are successful trades
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

    # Generate tables
    market_state_table = generate_market_state_table(env)
    order_book_table = generate_order_book_table(auction)
    trade_history_table = generate_trade_history_table(auction)

    # Generate charts
    supply_demand_chart = generate_supply_demand_chart(env)
    price_vs_trade_chart = generate_price_vs_trade_chart()
    cumulative_quantity_surplus_chart = generate_cumulative_quantity_surplus_chart()

    return (market_state_table, order_book_table, trade_history_table, supply_demand_chart, 
            price_vs_trade_chart, cumulative_quantity_surplus_chart)

def generate_market_state_table(env):
    headers = ['Agent ID', 'Role', 'Goods', 'Cash', 'Utility']
    rows = []
    for agent in env.agents:
        role = "Buyer" if agent.preference_schedule.is_buyer else "Seller"
        rows.append([
            agent.id,
            role,
            agent.allocation.goods,
            f"${agent.allocation.cash:.2f}",
            f"{env.get_agent_utility(agent):.2f}"
        ])
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_order_book_table(auction):
    headers = ['Price', 'Shares', 'Total']
    rows = []
    for order in auction.get_order_book():
        rows.append([
            f"${order['price']:.2f}",
            order['shares'],
            f"${order['total']:.2f}"
        ])
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_trade_history_table(auction):
    headers = ['Trade ID', 'Buyer ID', 'Seller ID', 'Price', 'Quantity']
    rows = []
    for trade in auction.get_trade_history():
        rows.append([
            trade.trade_id,
            trade.buyer_id,
            trade.seller_id,
            f"${trade.price:.2f}",
            trade.quantity
        ])
    
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in headers]), className="thead-light"),
        html.Tbody([html.Tr([html.Td(cell) for cell in row]) for row in rows])
    ], className="table table-striped table-hover")

def generate_supply_demand_chart(env):
    # Calculate theoretical supply and demand
    demand_x, demand_y, supply_x, supply_y = env.calculate_theoretical_supply_demand()

    # Calculate equilibrium values
    ce_price, ce_quantity, _, _, _ = env.calculate_equilibrium()

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

if __name__ == '__main__':
    app.run_server(debug=True)
