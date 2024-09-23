# GroupChat Orchestrator Design Specification

## 1. Overview
This document outlines the design for a GroupChat Orchestrator, which will manage a multi-agent simulation focused on group chat interactions. The orchestrator will facilitate communication rounds where market agents participate in debates and market speculation.

## 2. Key Components

### 2.1 GroupChatOrchestrator Class
- Attributes:
  - `agents`: List of MarketAgent instances
  - `environment`: GroupChat MultiAgentEnvironment instance
  - `config`: Configuration dictionary for the orchestrator
  - `timeline`: List to store chat history and topics

- Methods:
  - `__init__(self, config: Dict[str, Any])`: Initialize the orchestrator with configuration
  - `setup_environment(self)`: Set up the GroupChat environment
  - `setup_agents(self)`: Create and initialize market agents
  - `run_simulation(self)`: Main loop to run the group chat simulation
  - `process_round(self)`: Process a single round of group chat
  - `update_timeline(self, messages: List[GroupChatMessage], topic: str)`: Update the chat timeline
  - `generate_report(self)`: Generate a summary report of the simulation

### 2.2 GroupChatConfig Class
- Attributes:
  - `max_rounds`: Maximum number of chat rounds
  - `num_agents`: Number of agents in the chat
  - `initial_topic`: Initial discussion topic
  - `agent_roles`: List of agent roles (e.g., "analyst", "trader")
  - `llm_config`: Configuration for the language model

### 2.3 GroupChatEnvironment Class (extends MultiAgentEnvironment)
- Additional Attributes:
  - `current_topic`: Current discussion topic
  - `speaker_order`: List determining the order of speakers

- Additional Methods:
  - `set_topic(self, topic: str)`: Set a new discussion topic
  - `get_next_speaker(self) -> str`: Determine the next speaker in the chat

### 2.4 MarketAgent Class (extends existing MarketAgent)
- Additional Methods:
  - `generate_chat_message(self, topic: str, context: Dict[str, Any]) -> str`: Generate a chat message based on the current topic and context
  - `propose_topic(self) -> str`: Propose a new topic for discussion

## 3. Simulation Flow

1. Initialize GroupChatOrchestrator with configuration
2. Setup GroupChat environment
3. Create and initialize market agents
4. For each round in the simulation:
   a. Get the current topic and speaker
   b. Generate chat message for the current speaker
   c. Process the chat message in the environment
   d. Update the timeline with new messages and topic
   e. Allow agents to reflect on the chat
   f. Optionally propose a new topic
5. Generate a summary report of the simulation

## 4. Helper Functions

- `create_agent_config(role: str, persona: str) -> Dict[str, Any]`: Create configuration for a single agent
- `parse_chat_message(message: str) -> Dict[str, Any]`: Parse a chat message into structured data
- `calculate_sentiment(message: str) -> float`: Calculate sentiment score for a message
- `identify_key_topics(messages: List[str]) -> List[str]`: Identify key topics from a list of messages

## 5. Timeline Structure

The timeline will be a list of dictionaries, each containing:
- `round`: The round number
- `topic`: The current discussion topic
- `messages`: List of GroupChatMessage objects
- `proposed_topics`: Any new topics proposed in this round

## 6. Report Generation

The final report should include:
- Summary of all topics discussed
- Key insights or decisions made
- Most active participants
- Sentiment analysis of the overall chat
- Proposed topics and their reception

## 7. Next Steps

1. Implement the GroupChatOrchestrator class
2. Extend the MarketAgent class with chat-specific methods
3. Implement the GroupChatEnvironment class
4. Develop helper functions for parsing and analyzing chat messages
5. Create a config file structure for easy simulation setup
6. Implement the report generation functionality
7. Test the orchestrator with a small number of agents and rounds
8. Refine and optimize based on initial results
