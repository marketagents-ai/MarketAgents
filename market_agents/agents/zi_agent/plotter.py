import os
import datetime
import matplotlib.pyplot as plt
from market_agents.environments.environment import Environment
from ziagents import Trade, ZIAgent

def create_report_folder():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_folder = os.path.join("reports", f"auction_report_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)
    return report_folder

def save_figure(fig, report_folder, filename):
    filepath = os.path.join(report_folder, filename)
    fig.savefig(filepath)
    plt.close(fig)
    return f"./{filename}"

def plot_price_vs_trade(trade_numbers, prices, ce_price):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Price')
    ax.plot(trade_numbers, prices, marker='o', linestyle='-', color='blue', label='Trade Prices')
    ax.axhline(y=ce_price, color='red', linestyle='--', label=f'CE Price: {ce_price:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True)
    return fig

def plot_cumulative_quantity_and_surplus(cumulative_quantities, cumulative_surplus, equilibrium_quantity, equilibrium_surplus, final_efficiency):
    # Increase the figure size to accommodate the title
    fig, ax1 = plt.subplots(figsize=(12, 8))

    color = 'tab:blue'
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Cumulative Quantity', color=color)
    
    # Plot cumulative quantity
    qty_line, = ax1.plot(cumulative_quantities, color=color, label='Cumulative Quantity')
    ax1.tick_params(axis='y', labelcolor=color)

    # Add equilibrium quantity line
    eq_qty_line = ax1.axhline(y=equilibrium_quantity, color='red', linestyle='--', label=f'Equilibrium Quantity: {equilibrium_quantity:.2f}')

    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Cumulative Surplus', color=color)
    
    # Plot cumulative surplus
    surp_line, = ax2.plot(cumulative_surplus, color=color, label='Cumulative Surplus')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add equilibrium surplus line
    eq_surp_line = ax2.axhline(y=equilibrium_surplus, color='orange', linestyle='--', label=f'Equilibrium Surplus: {equilibrium_surplus:.2f}')

    # Ensure both axes show all data points
    ax1.set_ylim(0, max(max(cumulative_quantities), equilibrium_quantity) * 1.15)
    ax2.set_ylim(0, max(max(cumulative_surplus), equilibrium_surplus) * 1.15)

    # Combine legends from both axes
    lines = [qty_line, eq_qty_line, surp_line, eq_surp_line]
    labels = [str(line.get_label()) for line in lines]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Set the title with padding
    plt.title(f'Cumulative Quantity and Surplus (Efficiency: {final_efficiency:.2f}%)', pad=20)

    # Adjust the layout to prevent cutting off the title and legend
    plt.tight_layout()
    
    # Fine-tune the layout after tight_layout
    plt.subplots_adjust(top=0.9, right=0.85)  # Adjust top and right margins

    return fig

def generate_agent_allocation_table(env: Environment, max_agents_display=20):
    """
    Generate a markdown table of the initial and final allocations of the agents.
    """
    table_header = "| Agent ID | Role   | Initial Goods | Initial Cash | Final Goods | Final Cash | Surplus |\n"
    table_header += "|----------|--------|---------------|--------------|-------------|------------|---------|\n"
    table_rows = []

    # Select up to the first `max_agents_display` agents
    for agent in env.agents[:max_agents_display]:
        role = "Buyer" if agent.is_buyer else "Seller"
        initial_goods = agent.allocation.initial_goods
        initial_cash = agent.allocation.initial_cash
        final_goods = agent.allocation.goods
        final_cash = agent.allocation.cash
        surplus = agent.individual_surplus

        row = f"| {agent.id} | {role} | {initial_goods} | {initial_cash:.2f} | {final_goods} | {final_cash:.2f} | {surplus:.2f} |"
        table_rows.append(row)

    return table_header + "\n".join(table_rows)