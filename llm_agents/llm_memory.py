import json
import os

class LLMMemory:
    def __init__(self, agent_id, log_dir="logs/qa_interactions"):
        self.agent_id = agent_id
        self.log_dir = log_dir
        self.interactions = []
        self.load_interactions()

    def load_interactions(self):
        qa_interaction_file = f"qa_interactions_agent_{self.agent_id}.json"
        qa_interaction_path = os.path.join(self.log_dir, qa_interaction_file)
        if os.path.exists(qa_interaction_path):
            with open(qa_interaction_path, "r") as file:
                self.interactions = json.load(file)
        else:
            self.interactions = []

    def save_interactions(self):
        qa_interaction_file = f"qa_interactions_agent_{self.agent_id}.json"
        qa_interaction_path = os.path.join(self.log_dir, qa_interaction_file)
        os.makedirs(self.log_dir, exist_ok=True)
        with open(qa_interaction_path, "w") as file:
            json.dump(self.interactions, file, indent=2)

    def add_interaction(self, interaction, round_number):
        self.interactions.append({
            "round": round_number,
            "interaction": interaction
        })
        if len(self.interactions) % 10 == 0:  # Save every 10 interactions
            self.save_interactions()