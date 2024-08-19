import os
import datetime
import matplotlib.pyplot as plt
from environment import Environment
from ziagents import Trade

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
    labels = [line.get_label() for line in lines]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # Set the title with padding
    plt.title(f'Cumulative Quantity and Surplus (Efficiency: {final_efficiency:.2f}%)', pad=20)

    # Adjust the layout to prevent cutting off the title and legend
    plt.tight_layout()
    
    # Fine-tune the layout after tight_layout
    plt.subplots_adjust(top=0.9, right=0.85)  # Adjust top and right margins

    return fig

def generate_agent_allocation_table(env, max_agents_display=20):
    """
    Generate a markdown table of the initial and final allocations of the agents.
    """
    table_header = "| Agent ID | Role   | Initial Goods | Initial Cash | Final Goods | Final Cash | Surplus |\n"
    table_header += "|----------|--------|---------------|--------------|-------------|------------|---------|\n"
    table_rows = []

    # Select up to the first `max_agents_display` agents
    for agent in env.agents[:max_agents_display]:
        role = "Buyer" if agent.preference_schedule.is_buyer else "Seller"
        initial_goods = agent.allocation.initial_goods
        initial_cash = agent.allocation.initial_cash
        final_goods = agent.allocation.goods
        final_cash = agent.allocation.cash
        surplus = agent.calculate_individual_surplus()

        row = f"| {agent.id} | {role} | {initial_goods} | {initial_cash:.2f} | {final_goods} | {final_cash:.2f} | {surplus:.2f} |"
        table_rows.append(row)

    return table_header + "\n".join(table_rows)

def analyze_and_plot_auction_results(auction, env, pdf_filename="auction_report.pdf"):
    report_folder = create_report_folder()
    
    ce_price, ce_quantity, theoretical_buyer_surplus, theoretical_seller_surplus, theoretical_total_surplus = env.calculate_equilibrium()
    practical_total_surplus = auction.total_surplus_extracted
    surplus_difference = practical_total_surplus - theoretical_total_surplus
    final_efficiency = (practical_total_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0

    num_buyers = sum(1 for agent in env.agents if agent.preference_schedule.is_buyer)
    num_sellers = sum(1 for agent in env.agents if not agent.preference_schedule.is_buyer)

    summary_text = f"""
# Auction Report

## Environment Summary
- **Number of Buyers**: {num_buyers}
- **Number of Sellers**: {num_sellers}
- **Total Rounds**: {auction.max_rounds}

## Auction Summary
- **Total Successful Trades**: {len(auction.successful_trades)}
- **Total Surplus Extracted**: {practical_total_surplus:.2f}
- **Average Price**: {sum(auction.average_prices) / len(auction.average_prices):.2f}
- **Competitive Equilibrium Price**: {ce_price:.2f}
- **Competitive Equilibrium Quantity**: {ce_quantity}
- **Theoretical Total Surplus**: {theoretical_total_surplus:.2f}
- **Practical Total Surplus**: {practical_total_surplus:.2f}
- **Difference (Practical - Theoretical)**: {surplus_difference:.2f}
- **Final Efficiency**: {final_efficiency:.2f}%
"""

    # Save summary text to markdown file
    report_markdown = os.path.join(report_folder, "report.md")
    with open(report_markdown, "w") as f:
        f.write(summary_text)

    # Plot and save the theoretical supply and demand curve
    fig = env.plot_theoretical_supply_demand(save_location=report_folder)
    img_path = save_figure(fig, report_folder, "equilibrium_supply_demand.png")
    with open(report_markdown, "a") as f:
        f.write(f"\n\n## Theoretical Supply and Demand Curves\n\n![Theoretical Supply and Demand Curves]({img_path})")


    # Plot price vs trade number
    trade_numbers = list(range(1, len(auction.successful_trades) + 1))
    prices = [trade.price for trade in auction.successful_trades]

    fig = plot_price_vs_trade(trade_numbers, prices, ce_price)
    img_path = save_figure(fig, report_folder, "price_vs_trade.png")
    with open(report_markdown, "a") as f:
        f.write(f"\n\n## Price vs Trade Number\n\n![Price vs Trade Number]({img_path})")

    # Plot cumulative quantity and surplus with complete legend and horizontal line for equilibrium quantity
    cumulative_quantities = []
    cumulative_surplus = []
    total_quantity = 0
    total_surplus = 0
    
    for round_num in range(auction.max_rounds):
        trades_in_round = [trade for trade in auction.successful_trades if trade.round == round_num + 1]
        total_quantity += sum(trade.quantity for trade in trades_in_round)
        total_surplus += sum((trade.buyer_value - trade.price) + (trade.price - trade.seller_cost) for trade in trades_in_round)
        
        cumulative_quantities.append(total_quantity)
        cumulative_surplus.append(total_surplus)

    fig = plot_cumulative_quantity_and_surplus(cumulative_quantities, cumulative_surplus, equilibrium_quantity=ce_quantity, equilibrium_surplus=theoretical_total_surplus, final_efficiency=final_efficiency)
    img_path = save_figure(fig, report_folder, "cumulative_quantity_surplus.png")
    with open(report_markdown, "a") as f:
        f.write(f"\n\n## Cumulative Quantity and Surplus\n\n![Cumulative Quantity and Surplus]({img_path})")


    # Final Allocation Table
    agent_summary = generate_agent_allocation_table(env)
    
    with open(report_markdown, "a") as f:
        f.write(f"\n\n## Final Allocation of Agents\n\n{agent_summary}")

    print(f"Markdown report saved as {report_markdown}")
