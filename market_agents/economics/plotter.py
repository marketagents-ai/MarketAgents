import os
import datetime
import matplotlib.pyplot as plt
from typing import List

def create_report_folder():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_folder = os.path.join("reports", f"market_report_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)
    return report_folder

def save_figure(fig, report_folder, filename):
    filepath = os.path.join(report_folder, filename)
    fig.savefig(filepath)
    plt.close(fig)
    # Return the filename relative to the report folder
    return filename

def plot_price_vs_trade(trade_numbers: List[int], prices: List[float], ce_price: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Price')
    ax.plot(trade_numbers, prices, marker='o', linestyle='-', color='blue', label='Trade Prices')
    ax.axhline(y=ce_price, color='red', linestyle='--', label=f'Equilibrium Price: {ce_price:.2f}')
    ax.legend(loc='upper right')
    ax.grid(True)
    ax.set_title('Price vs Trade Number')
    fig.tight_layout()
    return fig

def plot_cumulative_quantity_and_surplus(cumulative_quantities: List[int], cumulative_surplus: List[float], 
                                         equilibrium_quantity: int, equilibrium_surplus: float, final_efficiency: float):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot cumulative quantity
    color = 'tab:blue'
    ax1.set_xlabel('Trade Number')
    ax1.set_ylabel('Cumulative Quantity', color=color)
    qty_line, = ax1.plot(range(1, len(cumulative_quantities) + 1), cumulative_quantities, color=color, marker='o', label='Cumulative Quantity')
    ax1.tick_params(axis='y', labelcolor=color)
    eq_qty_line = ax1.axhline(y=equilibrium_quantity, color='navy', linestyle='--', label=f'Equilibrium Quantity: {equilibrium_quantity}')

    # Plot cumulative surplus on a secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Cumulative Surplus', color=color)
    surp_line, = ax2.plot(range(1, len(cumulative_surplus) + 1), cumulative_surplus, color=color, marker='x', label='Cumulative Surplus')
    ax2.tick_params(axis='y', labelcolor=color)
    eq_surp_line = ax2.axhline(y=equilibrium_surplus, color='darkgreen', linestyle='--', label=f'Equilibrium Surplus: {equilibrium_surplus:.2f}')

    # Adjust the axes limits
    ax1.set_ylim(0, max(max(cumulative_quantities), equilibrium_quantity) * 1.1)
    ax2.set_ylim(0, max(max(cumulative_surplus), equilibrium_surplus) * 1.1)

    # Combine legends from both axes
    lines = [qty_line, eq_qty_line, surp_line, eq_surp_line]
    labels = [str(line.get_label()) for line in lines]
    ax1.legend(lines, labels, loc='upper left')

    plt.title(f'Cumulative Quantity and Surplus (Efficiency: {final_efficiency:.2f}%)', pad=20)
    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    return fig