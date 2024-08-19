import datetime
import os
import tkinter as tk
from tkinter import ttk
from auction import DoubleAuction

class SimulationUI:
    def __init__(self, master, environment, auction):
        self.master = master
        self.auction = auction
        self.environment = environment
        self.master.title("Double Auction Simulation")
        self.master.geometry("1000x600")
        self.master.configure(bg='#1e2329')

        self.report_folder = self.create_report_folder()

        self.create_widgets()
        self.update_ui()  # Initial UI update
        self.refresh_interval = 1000  # Refresh every 1000 ms (1 second)
        self.master.after(self.refresh_interval, self.update_ui_periodically)

    def update_ui_periodically(self):

        # Check if the auction has ended
        if self.auction.current_round >= self.auction.max_rounds:
            print("Simulation has ended")
            return
        
        self.auction.run_auction()  # Run the auction step
        self.update_ui()  # Update the UI
        self.master.after(self.refresh_interval, self.update_ui_periodically)  # Schedule next update

    def create_widgets(self):
        # Market state
        market_frame = tk.Frame(self.master, bg='#1e2329')
        market_frame.pack(pady=10)
        market_label = tk.Label(market_frame, text="Market State", font=("Arial", 16, "bold"), fg="white", bg='#1e2329')
        market_label.pack()

        # Agents table
        agents_frame = tk.Frame(self.master, bg='#1e2329')
        agents_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        columns = ('Agent ID', 'Role', 'Goods', 'Cash', 'Utility')
        self.agents_tree = ttk.Treeview(agents_frame, columns=columns, show='headings', height=6)

        for col in columns:
            self.agents_tree.heading(col, text=col)
            self.agents_tree.column(col, width=100, anchor='center')

        self.agents_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(agents_frame, orient=tk.VERTICAL, command=self.agents_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.agents_tree.configure(yscrollcommand=scrollbar.set)

        # Total utility display
        self.total_utility_label = tk.Label(self.master, text="", font=("Arial", 14), fg="white", bg='#1e2329')
        self.total_utility_label.pack(pady=10)

        # Order book
        order_book_frame = tk.Frame(self.master, bg='#1e2329')
        order_book_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        order_book_label = tk.Label(order_book_frame, text="Order Book", font=("Arial", 14, "bold"), fg="white", bg='#1e2329')
        order_book_label.pack()

        order_columns = ('Price', 'Shares', 'Total')
        self.order_book_tree = ttk.Treeview(order_book_frame, columns=order_columns, show='headings', height=6)

        for col in order_columns:
            self.order_book_tree.heading(col, text=col)
            self.order_book_tree.column(col, width=100, anchor='center')

        self.order_book_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        order_scrollbar = ttk.Scrollbar(order_book_frame, orient=tk.VERTICAL, command=self.order_book_tree.yview)
        order_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.order_book_tree.configure(yscrollcommand=order_scrollbar.set)

        # Trade history
        trade_history_frame = tk.Frame(self.master, bg='#1e2329')
        trade_history_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        trade_history_label = tk.Label(trade_history_frame, text="Trade History", font=("Arial", 14, "bold"), fg="white", bg='#1e2329')
        trade_history_label.pack()

        trade_columns = ('Trade ID', 'Buyer ID', 'Seller ID', 'Price', 'Quantity')
        self.trade_history_tree = ttk.Treeview(trade_history_frame, columns=trade_columns, show='headings', height=6)

        for col in trade_columns:
            self.trade_history_tree.heading(col, text=col)
            self.trade_history_tree.column(col, width=100, anchor='center')

        self.trade_history_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        trade_scrollbar = ttk.Scrollbar(trade_history_frame, orient=tk.VERTICAL, command=self.trade_history_tree.yview)
        trade_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trade_history_tree.configure(yscrollcommand=trade_scrollbar.set)

    def update_ui(self):
        # Clear existing items
        for item in self.agents_tree.get_children():
            self.agents_tree.delete(item)

        # Update agents
        for agent in self.environment.agents:
            role = "Buyer" if agent.preference_schedule.is_buyer else "Seller"
            self.agents_tree.insert('', tk.END, values=(agent.id, role, agent.allocation.goods, f"${agent.allocation.cash:.2f}", f"{self.environment.get_agent_utility(agent):.2f}"))

        # Update total utility
        total_utility = self.environment.get_total_utility()
        self.total_utility_label.config(text=f"Total Market Utility: ${total_utility:.2f}")

        # Update order book
        self.update_order_book()

        # Update trade history
        self.update_trade_history()
    
    def create_report_folder(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_folder = os.path.join("reports", f"auction_report_{timestamp}")
        os.makedirs(report_folder, exist_ok=True)
        return report_folder

    def update_order_book(self):
        # Clear existing order book items
        for item in self.order_book_tree.get_children():
            self.order_book_tree.delete(item)

        # Get the current order book from the auction
        order_book = self.auction.get_order_book()  # Adjust this line based on your structure
        for order in order_book:
            self.order_book_tree.insert('', tk.END, values=(f"{order['price']:.2f}", order['shares'], f"${order['total']:.2f}"))

    def update_trade_history(self):
        # Clear existing trade history items
        for item in self.trade_history_tree.get_children():
            self.trade_history_tree.delete(item)

        # Get the trade history from the auction
        trades = self.auction.get_trade_history()  # Adjust this line based on your structure
        for trade in trades:
            self.trade_history_tree.insert('', tk.END, values=(trade.trade_id, trade.buyer_id, trade.seller_id, f"{trade.price:.2f}", trade.quantity))

def run_ui(environment, auction):
    root = tk.Tk()
    app = SimulationUI(root, environment, auction)
    root.mainloop()

if __name__ == "__main__":
    from environment import Environment, generate_agents  # Adjust the import based on your structure
    from auction import DoubleAuction
    # Generate test agents
    num_buyers = 5
    num_sellers = 5
    spread = 0.5
    agents = generate_agents(num_agents=num_buyers + num_sellers, num_units=5, buyer_base_value=100, seller_base_value=80, spread=spread)
    
    # Create the environment
    env = Environment(agents=agents)
    auction = DoubleAuction(env, max_rounds=5)

    # Run the UI
    run_ui(env, auction)