import os
from typing import List, Union
from datetime import datetime
from market_agents.economics.plotter import create_report_folder, save_figure, plot_price_vs_trade, plot_cumulative_quantity_and_surplus
from market_agents.economics.econ_agent import EconomicAgent
from market_agents.economics.equilibrium import Equilibrium
from market_agents.simple_agent import SimpleAgent

def analyze_and_plot_market_results(trades: List, agents: List[Union[EconomicAgent, SimpleAgent]], equilibrium: Equilibrium, goods: List[str], max_rounds: int,
                                    cumulative_quantities: List[int], cumulative_surplus: List[float]):
    # Create a timestamped folder for this specific market report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_folder = os.path.join("outputs", "reports", f"market_report_{timestamp}")
    os.makedirs(report_folder, exist_ok=True)

    # Assuming we only have one good for simplicity
    good = goods[0]

    # Get theoretical equilibrium results
    equilibrium_results = equilibrium.calculate_equilibrium()[good]
    ce_price, ce_quantity = equilibrium_results.price, int(equilibrium_results.quantity)
    theoretical_total_surplus = equilibrium_results.total_surplus

    # Calculate total practical surplus
    total_buyer_surplus = sum(agent.calculate_individual_surplus() for agent in agents if agent.is_buyer(good))
    total_seller_surplus = sum(agent.calculate_individual_surplus() for agent in agents if agent.is_seller(good))
    practical_total_surplus = total_buyer_surplus + total_seller_surplus
    surplus_difference = practical_total_surplus - theoretical_total_surplus
    final_efficiency = (practical_total_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0

    num_buyers = len([agent for agent in agents if agent.is_buyer(good)])
    num_sellers = len([agent for agent in agents if agent.is_seller(good)])

    average_price = sum(trade.price for trade in trades) / len(trades) if trades else 0

    summary_text = f"""
# Market Report

## Environment Summary
- **Number of Buyers**: {num_buyers}
- **Number of Sellers**: {num_sellers}
- **Total Rounds**: {max_rounds}

## Market Summary
- **Total Successful Trades**: {len(trades)}
- **Total Buyer Surplus**: {total_buyer_surplus:.2f}
- **Total Seller Surplus**: {total_seller_surplus:.2f}
- **Total Surplus Extracted**: {practical_total_surplus:.2f}
- **Average Price**: {average_price:.2f}
- **Competitive Equilibrium Price**: {ce_price:.2f}
- **Competitive Equilibrium Quantity**: {ce_quantity}
- **Theoretical Total Surplus**: {theoretical_total_surplus:.2f}
- **Practical Total Surplus**: {practical_total_surplus:.2f}
- **Difference (Practical - Theoretical)**: {surplus_difference:.2f}
- **Final Efficiency**: {final_efficiency:.2f}%"""

    report_markdown = "market_report.md"
    with open(os.path.join(report_folder, report_markdown), "w") as f:
        f.write(summary_text)

    # Plot equilibrium supply and demand curves and save the figure
    fig = equilibrium.plot_supply_demand(good)
    if fig is not None:
        img_path = save_figure(fig, report_folder, "equilibrium_plot.png")
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Equilibrium Supply and Demand Curves\n\n![Equilibrium Plot]({img_path})")
    else:
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Equilibrium Supply and Demand Curves\n\nUnable to generate plot.")

    # Plot price vs trade number
    trade_numbers = list(range(1, len(trades) + 1))
    prices = [trade.price for trade in trades]

    if prices:
        fig = plot_price_vs_trade(trade_numbers, prices, ce_price)
        img_path = save_figure(fig, report_folder, "price_vs_trade.png")
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Price vs Trade Number\n\n![Price vs Trade Number]({img_path})")
    else:
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Price vs Trade Number\n\nNo trades occurred.")
    
    
    if cumulative_quantities and cumulative_surplus:
        fig = plot_cumulative_quantity_and_surplus(
            cumulative_quantities,
            cumulative_surplus,
            equilibrium_quantity=ce_quantity,
            equilibrium_surplus=theoretical_total_surplus,
            final_efficiency=final_efficiency)
        img_path = save_figure(fig, report_folder, "cumulative_quantity_surplus.png")
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Cumulative Quantity and Surplus\n\n![Cumulative Quantity and Surplus]({img_path})")
    else:
        with open(os.path.join(report_folder, report_markdown), "a") as f:
            f.write(f"\n\n## Cumulative Quantity and Surplus\n\nNo trades occurred.")

    # Final Allocation Table
    agent_summary = generate_agent_allocation_table(agents, good)
    
    with open(os.path.join(report_folder, report_markdown), "a") as f:
        f.write(f"\n\n## Final Allocation of Agents\n\n{agent_summary}")

    print(f"Markdown report saved as {os.path.join(report_folder, report_markdown)}")

def generate_agent_allocation_table(agents: List[Union[EconomicAgent, SimpleAgent]], good: str, max_agents_display=50):
    table_header = "| Agent ID | Agent Type | Role   | Initial Goods | Initial Cash | Final Goods | Final Cash | Surplus |\n"
    table_header += "|----------|------------|--------|---------------|--------------|-------------|------------|---------|\n"
    table_rows = []

    for agent in agents[:max_agents_display]:
        role = "Buyer" if agent.is_buyer(good) else "Seller"
        agent_type = "SimpleAgent" if isinstance(agent, SimpleAgent) else "EconomicAgent"
        initial_goods = agent.endowment.initial_basket.get_good_quantity(good)
        initial_cash = agent.endowment.initial_basket.cash
        final_goods = agent.endowment.current_basket.get_good_quantity(good)
        final_cash = agent.endowment.current_basket.cash
        surplus = agent.calculate_individual_surplus()

        row = f"| {agent.id} | {agent_type} | {role} | {initial_goods} | {initial_cash:.2f} | {final_goods} | {final_cash:.2f} | {surplus:.2f} |"
        table_rows.append(row)

    return table_header + "\n".join(table_rows)