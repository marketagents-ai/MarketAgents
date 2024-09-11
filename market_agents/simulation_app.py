import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import psutil
import os
import signal

def kill_process_on_port(port):
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                process = psutil.Process(conn.pid)
                process.terminate()
                process.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
            except psutil.TimeoutExpired:
                os.kill(conn.pid, signal.SIGKILL)

def create_dashboard(data_source):
    port = 8050  # Default Dash port
    kill_process_on_port(port)
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
            # First Column: Tables
            html.Div([
                html.Div([
                    html.H3("Market State", className="card-header"),
                    html.Div(id='market-state-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
                ], className="card mb-4"),
                html.Div([
                    html.H3("Order Book", className="card-header"),
                    html.Div(id='order-book-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
                ], className="card mb-4"),
                html.Div([
                    html.H3("Trade History", className="card-header"),
                    html.Div(id='trade-history-table', className="card-body table-responsive", style={'max-height': '300px', 'overflow-y': 'auto'})
                ], className="card mb-4"),
            ], className="col-md-6"),
            
            # Second Column: Charts
            html.Div([
                html.Div([
                    dcc.Graph(id='supply-demand-chart', config={'displayModeBar': False})
                ], className="card mb-4"),
                html.Div([
                    dcc.Graph(id='price-vs-trade-chart', config={'displayModeBar': False})
                ], className="card mb-4"),
                html.Div([
                    dcc.Graph(id='cumulative-quantity-surplus-chart', config={'displayModeBar': False})
                ], className="card mb-4"),
            ], className="col-md-6"),
        ], className="row"),
    ], className="container-fluid")

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
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'start-button':
            return False, True, False
        elif button_id == 'stop-button':
            return True, False, True
        return interval_disabled, False, True
    
    @app.callback(
        [Output('market-state-table', 'children'),
        Output('order-book-table', 'children'),
        Output('trade-history-table', 'children'),
        Output('supply-demand-chart', 'figure'),
        Output('price-vs-trade-chart', 'figure'),
        Output('cumulative-quantity-surplus-chart', 'figure')],
        Input('interval-component', 'n_intervals')
    )
    def update_dashboard(n_intervals):
        # Fetch the latest data from the data source
        latest_data = data_source()
        
        # Generate tables and charts using the latest data
        market_state_table = generate_market_state_table(latest_data)
        order_book_table = generate_order_book_table(latest_data['order_book'])
        trade_history_table = generate_trade_history_table(latest_data['trade_history'])
        supply_demand_chart = generate_supply_demand_chart(latest_data)
        price_vs_trade_chart = generate_price_vs_trade_chart(latest_data['successful_trades'])
        cumulative_quantity_surplus_chart = generate_cumulative_quantity_surplus_chart(latest_data)

        return (market_state_table, order_book_table, trade_history_table, supply_demand_chart, 
                price_vs_trade_chart, cumulative_quantity_surplus_chart)

    return app

def generate_market_state_table(latest_data):
    market_state = {
        'Current Step': latest_data['current_step'],
        'Max Steps': latest_data['max_steps'],
        'CE Price': latest_data['ce_price'],
        'CE Quantity': latest_data['ce_quantity'],
        'Efficiency': latest_data['efficiency'],
        'Total Utility': latest_data['total_utility'],
    }
    df = pd.DataFrame([market_state])
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in df.columns])),
        html.Tbody([
            html.Tr([html.Td(str(df.iloc[i][col])) for col in df.columns])
            for i in range(len(df))
        ])
    ], className="table table-striped table-bordered table-hover")

def generate_order_book_table(order_book):
    bids = pd.DataFrame([(b['agent_id'], b['price'], b['quantity']) for b in order_book['bids']],
                        columns=['Agent ID', 'Price', 'Quantity'])
    asks = pd.DataFrame([(a['agent_id'], a['price'], a['quantity']) for a in order_book['asks']],
                        columns=['Agent ID', 'Price', 'Quantity'])
    return html.Div([
        html.H4("Bids"),
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in bids.columns])),
            html.Tbody([
                html.Tr([html.Td(str(bids.iloc[i][col])) for col in bids.columns])
                for i in range(len(bids))
            ])
        ], className="table table-striped table-bordered table-hover"),
        html.H4("Asks"),
        html.Table([
            html.Thead(html.Tr([html.Th(col) for col in asks.columns])),
            html.Tbody([
                html.Tr([html.Td(str(asks.iloc[i][col])) for col in asks.columns])
                for i in range(len(asks))
            ])
        ], className="table table-striped table-bordered table-hover")
    ])

def generate_trade_history_table(trade_history):
    trades = pd.DataFrame([(t['trade_id'], t['buyer_id'], t['seller_id'], t['price'], t['quantity'], t['round'])
                           for t in trade_history],
                          columns=['Trade ID', 'Buyer ID', 'Seller ID', 'Price', 'Quantity', 'Round'])
    return html.Table([
        html.Thead(html.Tr([html.Th(col) for col in trades.columns])),
        html.Tbody([
            html.Tr([html.Td(str(trades.iloc[i][col])) for col in trades.columns])
            for i in range(len(trades))
        ])
    ], className="table table-striped table-bordered table-hover")

def generate_supply_demand_chart(latest_data):
    supply_curve = latest_data.get('current_supply_curve', {})
    demand_curve = latest_data.get('current_demand_curve', {})
    
    supply_points = supply_curve.get('points', [])
    demand_points = demand_curve.get('points', [])
    
    if not supply_points or not demand_points:
        # Return an empty figure with a message if there's no data
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Supply and Demand Curve", height=350)
        return fig
    
    supply_x, supply_y = zip(*[(point.get('quantity', 0), point.get('price', 0)) for point in supply_points])
    demand_x, demand_y = zip(*[(point.get('quantity', 0), point.get('price', 0)) for point in demand_points])
    
    ce_price = latest_data.get('ce_price', 0)
    ce_quantity = latest_data.get('ce_quantity', 0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=supply_x, y=supply_y, mode='lines', name='Supply', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=demand_x, y=demand_y, mode='lines', name='Demand', line=dict(color='blue')))
    
    if ce_price and ce_quantity:
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

def generate_price_vs_trade_chart(successful_trades):
    trade_numbers = list(range(1, len(successful_trades) + 1))
    prices = [trade['price'] for trade in successful_trades]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trade_numbers, y=prices, mode='lines+markers', name='Price per Trade', line=dict(color='blue')))

    fig.update_layout(
        title="Price vs Trade",
        xaxis_title="Trade Number",
        yaxis_title="Price",
        template="plotly_white",
        height=350
    )
    
    return fig

def generate_cumulative_quantity_surplus_chart(latest_data):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    cumulative_quantities = []
    cumulative_surplus = []
    total_quantity = 0
    total_surplus = 0
    
    for trade in latest_data['successful_trades']:
        total_quantity += trade['quantity']
        total_surplus += trade['total_surplus']
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
    # This part should be handled by the Orchestrator
    pass