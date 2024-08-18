import requests
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass
from colorama import Fore, Style, init
from pydantic import BaseModel
import json

# Initialize colorama
init(autoreset=True)

class ACLPerformative(str, Enum):
    PROPOSE = "PROPOSE"
    CHALLENGE = "CHALLENGE"
    VERIFY = "VERIFY"
    CONFIRM = "CONFIRM"

class ACLMessage(BaseModel):
    performative: ACLPerformative
    sender: str
    receiver: str
    content: str
    reply_with: str
    in_reply_to: str = None
    language: str = "ACL"
    ontology: str = "MarketPrediction"
    protocol: str = "Debate"
    conversation_id: str = "debate1"

class Agent:
    def __init__(self, name: str):
        self.name = name

    def generate_message(self, context: List[ACLMessage], llm: 'LLM') -> ACLMessage:
        # Create a prompt for the LLM
        context_summary = "\n".join([f"{msg.sender} ({msg.performative.value}): {msg.content}" for msg in context])
        acl_message_schema = ACLMessage.schema_json(indent=2)
        prompt = f"""Generate an ACL message for a debate within <acl> tags in JSON format. The current context is as follows:
    {context_summary}

    The JSON schema for the ACL message is:
    {acl_message_schema}

    Example:
    Respond in the format below with ACL JSON within <acl></acl> XML tags. Do not use ```json markdown block.
    <acl>
    {{
        "performative": "<PERFORMATIVE_TYPE>",
        "sender": "{self.name}",
        "receiver": "<RECEIVER_TYPE>",
        "content": "<CONTENT_MESSAGE>",
        "reply_with": "<REPLY_MESSAGE>",
        "language": "<LANGUAGE_TYPE>",
        "ontology": "<ONTOLOGY_TYPE>",
        "protocol": "<PROTOCOL_TYPE>",
        "conversation_id": "<CONVERSATION_ID>"
    }}
    </acl>
    """
        response = llm.request(prompt)
        message_data = parse_acl_response(response)
        if message_data:
            acl_msg = ACLMessage(**message_data)
            print_colored_message(acl_msg)
            return acl_msg
        return None

class LLM:
    def __init__(self, api_key: str, model_url: str):
        self.api_key = api_key
        self.model_url = model_url

    def request(self, prompt: str) -> str:
        resp = requests.post(
            self.model_url,
            headers={"Authorization": f"Api-Key {self.api_key}"},
            json={'prompt': prompt, 'max_tokens': 4096}
        )
        # Debugging: Print the raw response from the LLM
        try:
            resp_json = resp.text
            print(f"LLM Response: {resp_json}")
            return resp_json
        except (ValueError, KeyError) as e:
            print(f"Error decoding LLM response: {e}")
            print(f"Raw Response: {resp.text}")
            return ""

class VeraAgent(Agent):
    def __init__(self, name: str, llm: LLM):
        super().__init__(name)
        self.llm = llm

    def generate_message(self, context: List[ACLMessage], llm: LLM) -> ACLMessage:
        debate_summary = "\n".join([f"{msg.sender}: {msg.content}" for msg in context])
        acl_message_schema = ACLMessage.schema_json(indent=2)
        prompt = f"""Based on the following debate about a market prediction, evaluate the arguments and evidence presented. Then, provide a judgment on the validity of the original prediction within <acl> tags in JSON format.

Debate summary:
{debate_summary}

The JSON schema for the ACL message is:
{acl_message_schema}

Example:
Respond in the format below with ACL JSON within XML tags. Do not use ```json markdown block.
<acl>
{{
  "performative": "<PERFORMATIVE_TYPE>",
  "sender": "{self.name}",
  "receiver": "<RECEIVER_TYPE>",
  "content": "<CONTENT_MESSAGE>",
  "reply_with": "<REPLY_MESSAGE>",
  "language": "<LANGUAGE_TYPE>",
  "ontology": "<ONTOLOGY_TYPE>",
  "protocol": "<PROTOCOL_TYPE>",
  "conversation_id": "<CONVERSATION_ID>"
}}
</acl>
"""
        response = llm.request(prompt)
        message_data = parse_acl_response(response)
        if message_data:
            acl_msg = ACLMessage(**message_data)
            print_colored_message(acl_msg)
            return acl_msg
        return None

class DebateSimulation:
    def __init__(self, agents: List[Agent], max_rounds: int, llm: LLM, initial_claim: str = "", context: str = ""):
        self.agents = agents
        self.max_rounds = max_rounds
        self.conversation: List[ACLMessage] = []
        self.llm = llm
        self.initial_claim = initial_claim
        self.context = context
        
        # Add initial claim and context to the conversation
        if initial_claim:
            initial_message = ACLMessage(
                performative=ACLPerformative.PROPOSE,
                sender="User",
                receiver="ALL",
                content=initial_claim,
                reply_with="msg1"
            )
            self.conversation.append(initial_message)
        
        if context:
            context_message = ACLMessage(
                performative=ACLPerformative.PROPOSE,
                sender="User",
                receiver="ALL",
                content=context,
                reply_with="msg2"
            )
            self.conversation.append(context_message)

    def run_debate(self) -> List[ACLMessage]:
        for _ in range(self.max_rounds):
            for agent in self.agents:
                if not isinstance(agent, VeraAgent):
                    message = agent.generate_message(self.conversation, self.llm)
                    if message:
                        self.conversation.append(message)
            
            # Perform verification at the end by VeraAgent
            vera_agent = next(agent for agent in self.agents if isinstance(agent, VeraAgent))
            vera_message = vera_agent.generate_message(self.conversation, self.llm)
            if vera_message:
                self.conversation.append(vera_message)
        
        return self.conversation
    
    def print_initial_claim_and_context(self):
        print(f"{Fore.GREEN}Initial Claim:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}User (PROPOSE): {self.initial_claim}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}\nContext:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}User (PROPOSE): {self.context}{Style.RESET_ALL}")
        print("\nDebate Transcript:")
        print("------------------")

def print_colored_message(message: ACLMessage):
    color_map = {
        ACLPerformative.PROPOSE: Fore.GREEN,
        ACLPerformative.CHALLENGE: Fore.YELLOW,
        ACLPerformative.VERIFY: Fore.CYAN,
        ACLPerformative.CONFIRM: Fore.MAGENTA
    }
    color = color_map.get(message.performative, Fore.WHITE)
    print(f"{color}{message.sender} ({message.performative.value}): {message.content}{Style.RESET_ALL}")

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


def main():
    api_key =  ""
    model_url = "https://model-6wgo95v3.api.baseten.co/production/predict"

    llm = LLM(api_key, model_url)

    agents = [
        Agent("Agent_1"),
        Agent("Agent_2"),
        Agent("Agent_3"),
        Agent("Agent_4"),
        VeraAgent("VERA", llm)
    ]

    initial_claim = "Tesla's price will exceed $250 in 2 weeks."
    context = """
Tesla's current price is $207, and recent innovations and strong Q2 results will drive the price up.

News Summary 1:
Tesla stock was lower to start a new week of trading, falling as investors worry about global growth. Shares of the electric-vehicle giant were down 7.3% in premarket trading Monday at $192.33. Stocks around the world were falling as investors fretted that weak economic data signal a recession ahead. Despite positive comments from CEO Elon Musk about Tesla’s sales, the stock has fallen about 16% this year and is struggling to overcome negative global investor sentiment.

News Summary 2:
Tesla faces growing competition and softening demand, impacting its stock price which is trading 43% below its all-time high. The company’s profitability is declining, with earnings per share shrinking 46% year-over-year in Q2 2024. Despite recent price cuts and a plan to produce a low-cost EV model, sales growth has decelerated. Tesla is also involved in autonomous self-driving software, humanoid robots, and solar energy, but these segments may take years to significantly impact revenue.
"""


    debate = DebateSimulation(agents, max_rounds=10, llm=llm, initial_claim=initial_claim, context=context)
    result = debate.run_debate()

    print("Debate Transcript:")
    print("------------------")
    for msg in result:
        print_colored_message(msg)
    
    final_judgment = next(msg for msg in reversed(result) if msg.sender == "VERA")
    print("\nFinal Judgment:")
    print("---------------")
    print(f"{Fore.MAGENTA}{final_judgment.content}{Style.RESET_ALL}")

if __name__ == "__main__":
    main()
