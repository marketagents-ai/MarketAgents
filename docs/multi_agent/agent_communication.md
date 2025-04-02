# Agent Communication in GroupChat

## Overview

In the MarketAgents framework, agents communicate with each other through structured messages within the GroupChat environment. This document explains how agent communication works, including message structure, communication protocols, and interaction patterns.

## Message Structure

Agent messages in a GroupChat environment follow a standardized structure:

```python
message = {
    "sender_id": "agent_1",           # ID of the sending agent
    "content": "I believe tech stocks will outperform in Q3 due to AI advancements.",
    "recipient_id": "all",            # "all" or specific agent ID
    "timestamp": "2025-04-02T12:45:00Z",
    "metadata": {                     # Optional additional information
        "sentiment": "bullish",
        "confidence": 0.85,
        "sources": ["market_analysis", "earnings_reports"]
    }
}
```

## Communication Protocols

The MarketAgents framework uses the Agent Communication Language (ACL) protocol for structured agent communication:

```python
from market_agents.agents.protocols.acl_message import ACLMessage

# Create a message using ACL protocol
message = ACLMessage(
    sender="agent_1",
    receiver="agent_2",
    content="What's your analysis of recent tech earnings?",
    performative="query",
    conversation_id="tech_discussion_123",
    reply_with="query_1"
)

# Send the message
await agent.send_message(message)

# Receive and process messages
received_messages = await agent.receive_messages()
for msg in received_messages:
    if msg.performative == "query":
        # Formulate a response
        response = ACLMessage(
            sender=agent.id,
            receiver=msg.sender,
            content="Based on my analysis, tech earnings are exceeding expectations.",
            performative="inform",
            conversation_id=msg.conversation_id,
            in_reply_to=msg.reply_with
        )
        await agent.send_message(response)
```

### ACL Performatives

The ACL protocol supports various performatives (message types):

- `inform`: Provide information
- `query`: Request information
- `propose`: Suggest an action or idea
- `accept`: Accept a proposal
- `reject`: Reject a proposal
- `request`: Ask for an action to be performed
- `agree`: Agree to perform a requested action
- `refuse`: Refuse to perform a requested action

## Sending Messages in GroupChat

Agents send messages to the GroupChat environment through the environment's step function:

```python
# Agent sending a message to the group chat
action = {
    "message": "I've analyzed recent market data and believe tech stocks are undervalued.",
    "recipient_id": "all",  # Send to all agents
    "metadata": {
        "analysis_type": "valuation",
        "confidence": 0.8
    }
}

# Execute the action in the environment
observation = agent.environments["group_chat"].step(action)
```

## Receiving Messages

Agents receive messages as part of the environment's observations:

```python
# Get the current state of the group chat
chat_state = agent.environments["group_chat"].get_global_state(agent_id=agent.id)

# Access received messages
received_messages = chat_state.get("round_messages", [])

# Process messages
for message in received_messages:
    sender = message.get("sender_id")
    content = message.get("content")
    # Process the message content
    # ...
```

## Direct vs. Broadcast Communication

Agents can communicate directly with specific agents or broadcast to all participants:

```python
# Direct message to a specific agent
direct_message = {
    "message": "What's your specific view on Apple's stock?",
    "recipient_id": "agent_2",  # Specific recipient
    "metadata": {"private": True}
}

# Broadcast message to all agents
broadcast_message = {
    "message": "I'd like to hear everyone's thoughts on the tech sector outlook.",
    "recipient_id": "all",  # All participants
    "metadata": {"topic": "tech_outlook"}
}
```

## Communication in Cognitive Steps

Agent communication is typically handled within cognitive steps, particularly the ActionStep:

```python
class GroupChatActionStep(ActionStep):
    async def execute(self, agent):
        # Get current chat state
        chat_state = agent.environments[self.environment_name].get_global_state(agent_id=agent.id)
        
        # Process received messages
        received_messages = chat_state.get("round_messages", [])
        
        # Generate response based on received messages
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[
                {"role": "system", "content": self.construct_prompt(agent, received_messages)}
            ]
        )
        
        # Format and send the response
        action = {
            "message": response.content,
            "recipient_id": "all",
            "metadata": {"round": chat_state.get("current_round", 0)}
        }
        
        # Execute the action
        observation = agent.environments[self.environment_name].step(action)
        
        return response.content
```

## Practical Example: Agent Communication

Here's a complete example demonstrating agent communication in a group chat:

```python
import asyncio
from market_agents.agents.market_agent import MarketAgent
from market_agents.agents.cognitive_steps import ActionStep
from market_agents.environments.mechanisms.group_chat import GroupChat, GroupChatEnvironment

class GroupChatResponseStep(ActionStep):
    async def execute(self, agent):
        # Get current chat state
        chat_state = agent.environments[self.environment_name].get_global_state(agent_id=agent.id)
        
        # Process received messages
        received_messages = chat_state.get("round_messages", [])
        messages_text = "\n".join([
            f"{msg.get('sender_id')}: {msg.get('content')}" 
            for msg in received_messages
        ])
        
        # Generate response
        prompt = f"""
        You are {agent.role}. {agent.persona}
        
        Current discussion topic: {chat_state.get('current_topic')}
        Current round: {chat_state.get('current_round')} of {chat_state.get('max_rounds')}
        
        Recent messages:
        {messages_text}
        
        Based on your role and the conversation so far, provide your response to the discussion.
        """
        
        response = await agent.llm_orchestrator.generate(
            model=agent.llm_config.model,
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Send the response
        action = {
            "message": response.content,
            "recipient_id": "all"
        }
        
        observation = agent.environments[self.environment_name].step(action)
        
        return response.content

async def run_group_chat_communication(agents, group_chat_env):
    # Run a single round of communication
    results = []
    
    for agent in agents:
        # Run the communication step for each agent
        response = await agent.run_step(
            step=GroupChatResponseStep,
            environment_name="group_chat"
        )
        results.append({
            "agent_id": agent.id,
            "role": agent.role,
            "response": response
        })
    
    return results

# Run the communication example
communication_results = asyncio.run(run_group_chat_communication(agents, group_chat_env))
```

In the next section, we'll explore how to configure and use the GroupChat orchestrator for managing multi-agent discussions.
