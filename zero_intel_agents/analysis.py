from plotter import create_report_folder, save_figure, plot_price_vs_trade, plot_cumulative_quantity_and_surplus, generate_agent_allocation_table
from environment import Environment
from auction import DoubleAuction
import os

def analyze_and_plot_auction_results(auction: DoubleAuction, env: Environment):
    report_folder = create_report_folder()
    
    ce_price, ce_quantity, theoretical_buyer_surplus, theoretical_seller_surplus, theoretical_total_surplus = env.calculate_equilibrium()
    practical_total_surplus = auction.total_surplus_extracted
    surplus_difference = practical_total_surplus - theoretical_total_surplus
    final_efficiency = (practical_total_surplus / theoretical_total_surplus) * 100 if theoretical_total_surplus > 0 else 0

    num_buyers = len(env.buyers)
    num_sellers = len(env.sellers)

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
- **Final Efficiency**: {final_efficiency:.2f}%"""

    # Save summary text to markdown file
    report_markdown = os.path.join(report_folder, "report.md")
    with open(report_markdown, "w") as f:
        f.write(summary_text)

    # Plot and save the theoretical supply and demand curve
    fig = env.plot_supply_demand_curves(initial=True, save_location=report_folder)
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
        total_quantity += len(trades_in_round)  # Each trade is for 1 unit
        total_surplus += sum(trade.total_surplus for trade in trades_in_round)
        
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