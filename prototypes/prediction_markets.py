import os
import json
import random
from typing import List, Dict, Any
from pydantic import BaseModel
import openai
import tkinter as tk
from tkinter import ttk
import threading
import uuid
from openai import AzureOpenAI
from dotenv import load_dotenv

import tkinter as tk
from tkinter import ttk
import random

import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

class ACLMessage(BaseModel):
    performative: str
    sender: str
    receiver: str
    content: Dict
    reply_with: str = ""
    in_reply_to: str = ""
    language: str = "JSON"
    ontology: str = "PredictionMarketOntology"
    protocol: str = "PredictionMarketProtocol"
    conversation_id: str = ""

class Agent:
    def __init__(self, agent_id: str, initial_belief: Dict[str, float], cash: float, client):
        self.agent_id = agent_id
        self.belief = initial_belief
        self.cash = cash
        self.shares = {outcome: 0 for outcome in initial_belief.keys()}
        self.client = client

    def place_order(self, market_question: str, outcomes: List[str]) -> ACLMessage:
        context = f"You are an AI agent participating in a prediction market. The market question is: '{market_question}'. The possible outcomes are: {outcomes}. Your current beliefs are: {self.belief}. Your available cash is ${self.cash:.2f}."
        question = f"""Based on your beliefs and available cash, which outcome would you like to place an order for? If you want to buy, what's the maximum price you're willing to pay? If you want to sell, what's the minimum price you'd accept? How many shares? 
        
        Respond in the format below with ACL JSON within <acl> XML tags. Do not use ```json markdown block.
        Record your reasoning steps with GOAP framework within <scratch_pad> tags.
        <scratch_pad>
        Goal: <state goal>
        Observation: <report observations made>
        Actions: <list actions to perform>
        Reflection: <evaluate actions based on goal and observations>
        </scratch_pad>
        <acl>
        {{
            "action": "<ACTION>",
            "outcome": "<OUTCOME>", 
            "price": <PRICE>, 
            "quantity": <QUANTITY>
        }}
        </acl>
        """

        response = call_openai_api(context, question, self.client)
        
        print(Fore.CYAN + f"Agent ID: {self.agent_id}")
        print(Fore.GREEN + f"- Raw Response from LLM (Place Order): {response}")

        parsed_response = parse_acl_response(response)
        
        print(Fore.BLUE + f"- Parsed ACL Message (Place Order): {parsed_response}")

        action, outcome, price, quantity = parsed_response["action"], parsed_response["outcome"], parsed_response["price"], parsed_response["quantity"]

        acl_message = ACLMessage(
            performative="PROPOSE",
            sender=self.agent_id,
            receiver="Market",
            content={
                "action": action.lower(),
                "outcome": outcome,
                "price": float(price),
                "quantity": int(quantity)
            },
            conversation_id=str(uuid.uuid4())
        )
        
        print(Fore.MAGENTA + f"- ACLMessage JSON: {acl_message.json()}")

        return acl_message

    def update_belief(self, new_information: str, market_question: str, outcomes: List[str]):
        context = f"You are an AI agent participating in a prediction market. The market question is: '{market_question}'. The possible outcomes are: {outcomes}. Your current beliefs are: {self.belief}."
        question = f"""Given the new information: '{new_information}', how would you update your beliefs about the probabilities of each outcome? Respond with updated probabilities for each outcome, separated by commas, in the order of the outcomes listed above.
        
        Respond in the format below with ACL JSON within <acl> XML tags. Do not use ```json markdown block.
        Record your reasoning steps with GOAP framework within <scratch_pad> tags.
        <scratch_pad>
        Goal: <state goal>
        Observation: <report observations made>
        Actions: <list actions to perform>
        Reflection: <evaluate actions based on goal and observations>
        </scratch_pad>
        <acl>
        {{
            "outcome_probabilities": [0.0, 0.0, 0.0, 0.0] 
        }}
        </acl>
        """

        response = call_openai_api(context, question, self.client)
        
        print(Fore.CYAN + f"Agent ID: {self.agent_id}")
        print(Fore.YELLOW + f"- Raw Response from LLM (Update Belief): {response}")

        parsed_response = parse_acl_response(response)
        
        print(Fore.MAGENTA + f"- ACL Message (Update Belief): {parsed_response}")

        updated_probabilities = parsed_response["outcome_probabilities"]
        acl_message = ACLMessage(
            performative="UPDATE",
            sender=self.agent_id,
            receiver="Market",
            content={'outcome_probabilities': updated_probabilities},
            conversation_id=str(uuid.uuid4())
        )
        
        print(Fore.BLUE + f"- ACLMessage JSON: {acl_message.json()}")

        self.belief = dict(zip(outcomes, updated_probabilities))


def parse_acl_response(response: str) -> Dict[str, Any]:
    try:
        # Locate <acl> tags
        start_idx = response.find("<acl>")
        end_idx = response.find("</acl>")
        
        # Check if both tags are present
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid response format: Missing <acl> tags")
        
        # Extract JSON between <acl> and </acl>
        start_idx += len("<acl>")
        acl_json = response[start_idx:end_idx].strip()
        
        # Check if JSON is empty
        if not acl_json:
            raise ValueError("Empty JSON content in <acl> tags")

        # Clean up potential issues with the JSON string
        acl_json = acl_json.replace('\n', '').replace('\r', '')
        
        # Attempt to parse JSON data
        message_data = json.loads(acl_json)
        
        return message_data
    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing LLM response: {e}")
        print(f"Raw Response: {response}")
        # Print the extracted JSON for debugging
        if 'acl_json' in locals():
            print(f"Extracted JSON: {acl_json}")
        return None

class Market:
    def __init__(self, question: str, outcomes: List[str]):
        self.question = question
        self.outcomes = outcomes
        self.order_books = {outcome: [] for outcome in outcomes}
        self.transaction_history = []

    def process_message(self, message: ACLMessage) -> List[ACLMessage]:
        if message.performative == "PROPOSE":
            return self.add_order(message)
        return []

    def add_order(self, message: ACLMessage) -> List[ACLMessage]:
        order = message.content
        order['sender'] = message.sender  # Add this line to include the sender
        self.order_books[order['outcome']].append(order)
        return self.match_orders(order['outcome'])

    def match_orders(self, outcome: str) -> List[ACLMessage]:
        orders = self.order_books[outcome]
        buy_orders = sorted([o for o in orders if o['action'] == "buy"], key=lambda x: x['price'], reverse=True)
        sell_orders = sorted([o for o in orders if o['action'] == "sell"], key=lambda x: x['price'])

        transactions = []
        while buy_orders and sell_orders and buy_orders[0]['price'] >= sell_orders[0]['price']:
            buy = buy_orders.pop(0)
            sell = sell_orders.pop(0)
            quantity = min(buy['quantity'], sell['quantity'])
            price = (buy['price'] + sell['price']) / 2
            
            transaction = ACLMessage(
                performative="CONFIRM",
                sender="Market",
                receiver=f"{buy['sender']},{sell['sender']}",
                content={
                    "outcome": outcome,
                    "quantity": quantity,
                    "price": price
                },
                conversation_id=str(uuid.uuid4())
            )
            transactions.append(transaction)
            self.transaction_history.append((buy['sender'], sell['sender'], outcome, quantity, price))

            if buy['quantity'] > quantity:
                buy_orders.insert(0, {**buy, 'quantity': buy['quantity'] - quantity})
            if sell['quantity'] > quantity:
                sell_orders.insert(0, {**sell, 'quantity': sell['quantity'] - quantity})

        self.order_books[outcome] = buy_orders + sell_orders
        return transactions

    def get_market_prices(self) -> Dict[str, float]:
        return {outcome: self.order_books[outcome][0]['price'] if self.order_books[outcome] else 0.5 for outcome in self.outcomes}

def call_openai_api(context, question, client):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": context},
            {"role": "user", "content": question}
        ],
        max_tokens=420
    )
    return response.choices[0].message.content

class PredictionMarketSimulation:
    def __init__(self, agents: List[Agent], market: Market, max_rounds: int, news_events: List[str]):
        self.agents = agents
        self.market = market
        self.max_rounds = max_rounds
        self.news_events = news_events
        self.current_round = 0
        self.simulation_results = []

    def run_simulation_step(self):
        if self.current_round >= self.max_rounds:
            return self.get_results()

        new_info = self.news_events[self.current_round] if self.current_round < len(self.news_events) else None
        
        if new_info:
            for agent in self.agents:
                agent.update_belief(new_info, self.market.question, self.market.outcomes)
        
        for agent in self.agents:
            order_message = agent.place_order(self.market.question, self.market.outcomes)
            transactions = self.market.process_message(order_message)
            self.simulation_results.append({
                "round": self.current_round + 1,
                "agent": agent.agent_id,
                "order": order_message.dict(),
                "transactions": [t.dict() for t in transactions]
            })

        market_prices = self.market.get_market_prices()
        self.simulation_results.append({
            "round": self.current_round + 1,
            "market_prices": market_prices
        })

        self.current_round += 1
        return market_prices

    def get_results(self):
        return self.simulation_results

# Entry point for starting simulation
def run_simulation():
    market_question = "How many Fed rate cuts this year?"
    outcomes = ["0 (0 bps)", "1 (25 bps)", "2 (50 bps)", "3 (75 bps)", "4 (100 bps)", "5 (125 bps)"]

    market = Market(question=market_question, outcomes=outcomes)
    agents = [
        Agent(f"Agent_{i}", initial_belief={o: 1/len(outcomes) for o in outcomes}, cash=10000, client=client)
        for i in range(1, 11)
    ]

    news_events = [
        "Inflation rate drops to 2.1%, lower than expected.",
        "Fed Chair hints at potential rate cuts in upcoming FOMC meeting.",
        "Unemployment rate rises unexpectedly to 4.2%.",
        "GDP growth slows to 1.5% in Q2.",
        "Major bank predicts 3 rate cuts by year-end."
    ]

    simulation = PredictionMarketSimulation(agents, market, max_rounds=4, news_events=news_events)
    
    return simulation

class PredictionMarketUI:
    def __init__(self, master, simulation):
        self.master = master
        self.simulation = simulation
        self.master.title("Prediction Market Simulation")
        self.master.geometry("800x600")
        self.master.configure(bg='#1e2329')

        self.create_widgets()
        self.update_ui()  # Initial UI update
        self.refresh_interval = 1000  # Refresh every 1000 ms (1 second)
        self.master.after(self.refresh_interval, self.update_ui_periodically)

    def update_ui_periodically(self):
        self.simulation.run_simulation_step()  # Run the simulation step
        self.update_ui()  # Update the UI
        self.master.after(self.refresh_interval, self.update_ui_periodically)  # Schedule next update

    def create_widgets(self):
        # Market question
        question_frame = tk.Frame(self.master, bg='#1e2329')
        question_frame.pack(pady=10)

        question_label = tk.Label(question_frame, text=self.simulation.market.question, font=("Arial", 16, "bold"), fg="white", bg='#1e2329')
        question_label.pack()

        # Outcomes table
        outcomes_frame = tk.Frame(self.master, bg='#1e2329')
        outcomes_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        columns = ('Outcome', 'Probability', 'Bet Yes', 'Bet No')
        self.tree = ttk.Treeview(outcomes_frame, columns=columns, show='headings', height=6)

        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=150, anchor='center')

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(outcomes_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scrollbar.set)

        # Order book
        order_book_frame = tk.Frame(self.master, bg='#1e2329')
        order_book_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)

        order_book_label = tk.Label(order_book_frame, text="Order Book", font=("Arial", 14, "bold"), fg="white", bg='#1e2329')
        order_book_label.pack()

        columns = ('Price', 'Shares', 'Total')
        self.order_book_tree = ttk.Treeview(order_book_frame, columns=columns, show='headings', height=6)

        for col in columns:
            self.order_book_tree.heading(col, text=col)
            self.order_book_tree.column(col, width=150, anchor='center')

        self.order_book_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(order_book_frame, orient=tk.VERTICAL, command=self.order_book_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.order_book_tree.configure(yscrollcommand=scrollbar.set)

        # Next round button
        next_round_button = tk.Button(self.master, text="Next Round", command=self.next_round, bg='#3498db', fg='white')
        next_round_button.pack(pady=10)

        self.update_ui()

    def update_ui(self):
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)

        for item in self.order_book_tree.get_children():
            self.order_book_tree.delete(item)

        # Update outcomes
        market_prices = self.simulation.market.get_market_prices()
        for outcome, price in market_prices.items():
            probability = f"{price * 100:.1f}%"
            bet_yes = f"Bet Yes {price:.1f}¢"
            bet_no = f"Bet No {(1 - price) * 100:.1f}¢"
            self.tree.insert('', tk.END, values=(outcome, probability, bet_yes, bet_no))

        # Update order book (simplified version with random data)
        for _ in range(5):
            price = random.uniform(0.05, 0.95)
            shares = random.randint(100, 10000)
            total = price * shares
            self.order_book_tree.insert('', tk.END, values=(f"{price:.2f}¢", f"{shares:,}", f"${total:.2f}"))

    def next_round(self):
        if self.simulation.current_round < self.simulation.max_rounds:
            self.simulation.run_simulation_step()
            self.update_ui()
        else:
            print("Simulation complete!")

def run_ui(simulation):
    root = tk.Tk()
    app = PredictionMarketUI(root, simulation)
    root.mainloop()

if __name__ == "__main__":
    simulation = run_simulation()
    run_ui(simulation)