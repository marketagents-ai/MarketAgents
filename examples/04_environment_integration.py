import asyncio
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

from market_agents.agents.market_agent import MarketAgent
from market_agents.memory.agent_storage.agent_storage_api_utils import AgentStorageAPIUtils
from market_agents.memory.config import AgentStorageConfig
from market_agents.agents.personas.persona import Persona
from market_agents.environments.mechanisms.environment_mechanism import EnvironmentMechanism
from market_agents.environments.multi_agent_environment import MultiAgentEnvironment
from minference.lite.models import LLMConfig, ResponseFormat
from pydantic import Field, BaseModel

# Define a custom environment mechanism for a simple simulation
class SimulationEnvironment(EnvironmentMechanism):
    """A simple simulation environment for agent interaction."""
    
    name: str = Field(default="simulation", description="Name of the simulation environment")
    state: Dict[str, Any] = Field(default_factory=dict, description="Current state of the simulation")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="History of simulation actions")
    
    def initialize(self):
        """Initialize the simulation environment."""
        self.state = {
            "current_step": 0,
            "resources": {
                "energy": 100,
                "materials": 100,
                "information": 100
            },
            "events": [],
            "agent_states": {}
        }
        self.history = []
        print(f"Initialized simulation environment: {self.name}")
    
    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Process an action in the simulation environment."""
        agent_id = action.get("agent_id", "unknown")
        action_type = action.get("action_type", "")
        
        # Update step counter
        self.state["current_step"] += 1
        
        # Initialize agent state if not exists
        if agent_id not in self.state["agent_states"]:
            self.state["agent_states"][agent_id] = {
                "energy": 50,
                "materials": 50,
                "information": 50,
                "actions_taken": 0
            }
        
        # Process different action types
        result = {"success": False, "message": "Unknown action type"}
        
        if action_type == "gather":
            resource = action.get("resource", "")
            amount = action.get("amount", 10)
            result = self._gather_resource(agent_id, resource, amount)
        
        elif action_type == "process":
            resource = action.get("resource", "")
            amount = action.get("amount", 10)
            result = self._process_resource(agent_id, resource, amount)
        
        elif action_type == "share":
            resource = action.get("resource", "")
            amount = action.get("amount", 10)
            target_agent = action.get("target_agent", "")
            result = self._share_resource(agent_id, target_agent, resource, amount)
        
        elif action_type == "observe":
            result = self._observe_environment(agent_id)
        
        # Record action in history
        history_entry = {
            "step": self.state["current_step"],
            "agent_id": agent_id,
            "action": action,
            "result": result
        }
        self.history.append(history_entry)
        
        # Increment actions taken
        self.state["agent_states"][agent_id]["actions_taken"] += 1
        
        # Add random events occasionally
        if self.state["current_step"] % 3 == 0:
            event = self._generate_random_event()
            self.state["events"].append(event)
            result["event"] = event
        
        return result
    
    def _gather_resource(self, agent_id: str, resource: str, amount: int) -> Dict[str, Any]:
        """Gather a resource from the environment."""
        if resource not in self.state["resources"]:
            return {"success": False, "message": f"Resource {resource} not found in environment"}
        
        # Check if enough resources available
        available = self.state["resources"][resource]
        if available < amount:
            amount = available  # Take what's available
        
        # Update environment and agent resources
        self.state["resources"][resource] -= amount
        self.state["agent_states"][agent_id][resource] += amount
        
        return {
            "success": True,
            "message": f"Gathered {amount} units of {resource}",
            "resource": resource,
            "amount": amount,
            "agent_state": self.state["agent_states"][agent_id]
        }
    
    def _process_resource(self, agent_id: str, resource: str, amount: int) -> Dict[str, Any]:
        """Process a resource to increase its value."""
        if resource not in self.state["agent_states"][agent_id]:
            return {"success": False, "message": f"Resource {resource} not found in agent inventory"}
        
        # Check if agent has enough resources
        available = self.state["agent_states"][agent_id][resource]
        if available < amount:
            return {"success": False, "message": f"Not enough {resource} available (have {available}, need {amount})"}
        
        # Process the resource (convert to information at 2:1 ratio)
        self.state["agent_states"][agent_id][resource] -= amount
        information_gain = amount * 2
        self.state["agent_states"][agent_id]["information"] += information_gain
        
        return {
            "success": True,
            "message": f"Processed {amount} units of {resource} into {information_gain} units of information",
            "resource": resource,
            "amount": amount,
            "information_gain": information_gain,
            "agent_state": self.state["agent_states"][agent_id]
        }
    
    def _share_resource(self, agent_id: str, target_agent: str, resource: str, amount: int) -> Dict[str, Any]:
        """Share a resource with another agent."""
        if resource not in self.state["agent_states"][agent_id]:
            return {"success": False, "message": f"Resource {resource} not found in agent inventory"}
        
        if target_agent not in self.state["agent_states"]:
            return {"success": False, "message": f"Target agent {target_agent} not found"}
        
        # Check if agent has enough resources
        available = self.state["agent_states"][agent_id][resource]
        if available < amount:
            return {"success": False, "message": f"Not enough {resource} available (have {available}, need {amount})"}
        
        # Transfer the resource
        self.state["agent_states"][agent_id][resource] -= amount
        self.state["agent_states"][target_agent][resource] += amount
        
        return {
            "success": True,
            "message": f"Shared {amount} units of {resource} with agent {target_agent}",
            "resource": resource,
            "amount": amount,
            "target_agent": target_agent,
            "agent_state": self.state["agent_states"][agent_id],
            "target_state": self.state["agent_states"][target_agent]
        }
    
    def _observe_environment(self, agent_id: str) -> Dict[str, Any]:
        """Observe the current state of the environment."""
        return {
            "success": True,
            "message": "Observed environment state",
            "environment_state": {
                "current_step": self.state["current_step"],
                "resources": self.state["resources"],
                "events": self.state["events"][-3:] if self.state["events"] else []
            },
            "agent_state": self.state["agent_states"][agent_id]
        }
    
    def _generate_random_event(self) -> Dict[str, Any]:
        """Generate a random event in the environment."""
        events = [
            {"type": "discovery", "description": "New resource deposit discovered", "effect": {"resource": "materials", "amount": 20}},
            {"type": "innovation", "description": "Processing efficiency improved", "effect": {"resource": "information", "amount": 15}},
            {"type": "depletion", "description": "Energy source depleted", "effect": {"resource": "energy", "amount": -10}},
            {"type": "cooperation", "description": "Agents sharing information", "effect": {"resource": "information", "amount": 10}}
        ]
        
        import random
        event = random.choice(events)
        
        # Apply event effect to environment
        resource = event["effect"]["resource"]
        amount = event["effect"]["amount"]
        
        if resource in self.state["resources"]:
            self.state["resources"][resource] += amount
            # Ensure resources don't go below zero
            self.state["resources"][resource] = max(0, self.state["resources"][resource])
        
        return {
            "step": self.state["current_step"],
            "type": event["type"],
            "description": event["description"],
            "effect": event["effect"]
        }
    
    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get the global state of the environment."""
        state = {
            "current_step": self.state["current_step"],
            "resources": self.state["resources"],
            "events": self.state["events"][-5:] if self.state["events"] else [],
            "agents": list(self.state["agent_states"].keys())
        }
        
        if agent_id:
            state["agent_state"] = self.state["agent_states"].get(agent_id, {})
        
        return state
    
    def get_agent_state(self, agent_id: str) -> Dict[str, Any]:
        """Get the state for a specific agent."""
        return self.state["agent_states"].get(agent_id, {})

# Create a wrapper for the simulation environment
class SimulationMultiAgentEnvironment(MultiAgentEnvironment):
    """Wrapper for the simulation environment mechanism."""
    
    name: str
    address: str
    max_steps: int = Field(default=20)
    mechanism: SimulationEnvironment

async def create_resource_manager_agent():
    """Create a resource manager agent."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Resource Manager",
        persona="I am a resource manager responsible for efficiently gathering and allocating resources. I monitor resource levels and make strategic decisions to optimize resource utilization.",
        objectives=[
            "Maintain adequate resource levels",
            "Optimize resource allocation",
            "Respond to environmental changes",
            "Coordinate with other agents"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="resource_manager",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
    print(f"Created Resource Manager agent with ID: {agent.id}")
    
    return agent

async def create_information_analyst_agent():
    """Create an information analyst agent."""
    
    # Configure storage
    storage_config = AgentStorageConfig(
        api_url="http://localhost:8001",
        embedding_model="text-embedding-ada-002",
        vector_dimension=1536
    )
    
    # Initialize storage utilities
    storage_utils = AgentStorageAPIUtils(config=storage_config)
    
    # Create persona
    persona = Persona(
        role="Information Analyst",
        persona="I am an information analyst specializing in processing and interpreting data. I transform raw resources into valuable information and insights.",
        objectives=[
            "Process resources into information",
            "Analyze environmental patterns",
            "Share insights with other agents",
            "Adapt to changing conditions"
        ]
    )
    
    # Create agent
    agent = await MarketAgent.create(
        storage_utils=storage_utils,
        agent_id="information_analyst",
        use_llm=True,
        llm_config=LLMConfig(
            model="gpt-4o",
            client="openai",
            temperature=0.7
        ),
        persona=persona
    )
    
    print(f"Created Information Analyst agent with ID: {agent.id}")
    
    return agent

async def run_simulation():
    """Run a simulation with agents interacting in the environment."""
    
    # Create agents
    resource_manager = await create_resource_manager_agent()
    information_analyst = await create_information_analyst_agent()
    
    # Create and initialize simulation environment
    sim_mechanism = SimulationEnvironment(name="resource_simulation")
    sim_mechanism.initialize()
    
    # Create environment wrapper
    sim_env = SimulationMultiAgentEnvironment(
        name="resource_simulation",
        address="resource_sim_env",
        mechanism=sim_mechanism
    )
    
    # Add environment to agents
    resource_manager.add_environment("simulation", sim_env)
    information_analyst.add_environment("simulation", sim_env)
    
    # Run simulation for multiple steps
    print("\nStarting simulation...")
    print("-" * 80)
    
    for step in range(1, 11):
        print(f"\nSimulation Step {step}")
        print("-" * 40)
        
        # Resource Manager's turn
        await resource_manager_action(resource_manager, sim_env, step)
        
        # Information Analyst's turn
        await information_analyst_action(information_analyst, sim_env, step)
        
        # Print current environment state
        env_state = sim_env.get_global_state()
        print(f"\nEnvironment State after Step {step}:")
        print(f"Resources: {env_state['resources']}")
        if env_state['events']:
            print(f"Recent Events: {env_state['events'][-1]['description']}")
    
    # Print final simulation results
    print("\nSimulation Complete")
    print("-" * 80)
    
    rm_state = sim_env.get_agent_state(resource_manager.id)
    ia_state = sim_env.get_agent_state(information_analyst.id)
    
    print(f"Resource Manager final state: {rm_state}")
    print(f"Information Analyst final state: {ia_state}")
    
    # Calculate total resources and information generated
    total_resources = sum(rm_state.get(r, 0) + ia_state.get(r, 0) for r in ["energy", "materials"])
    total_information = rm_state.get("information", 0) + ia_state.get("information", 0)
    
    print(f"\nTotal resources gathered: {total_resources}")
    print(f"Total information generated: {total_information}")
    
    return {
        "steps": step,
        "resource_manager_state": rm_state,
        "information_analyst_state": ia_state,
        "total_resources": total_resources,
        "total_information": total_information
    }

async def resource_manager_action(agent, environment, step):
    """Determine and execute the Resource Manager's action."""
    
    # Observe the environment
    observe_action = {
        "agent_id": agent.id,
        "action_type": "observe"
    }
    observation = environment.step(observe_action)
    
    # Create a prompt for the agent to decide on an action
    env_state = observation["environment_state"]
    agent_state = observation["agent_state"]
    
    prompt = f"""
    You are {agent.role}. {agent.persona}
    
    Current simulation step: {step}/10
    
    Environment state:
    - Available resources: {env_state['resources']}
    - Recent events: {env_state['events']}
    
    Your current state:
    - Energy: {agent_state.get('energy', 0)}
    - Materials: {agent_state.get('materials', 0)}
    - Information: {agent_state.get('information', 0)}
    - Actions taken: {agent_state.get('actions_taken', 0)}
    
    As the Resource Manager, decide on your next action. You can:
    1. Gather a resource (energy, materials, information)
    2. Process a resource into information
    3. Share a resource with the Information Analyst
    
    Choose the action that best aligns with your objectives of maintaining adequate resource levels
    and optimizing resource allocation.
    
    Respond with a JSON object containing your chosen action:
    For gathering: {{"action": "gather", "resource": "[resource_name]", "amount": [number]}}
    For processing: {{"action": "process", "resource": "[resource_name]", "amount": [number]}}
    For sharing: {{"action": "share", "resource": "[resource_name]", "amount": [number], "target": "information_analyst"}}
    """
    
    # Generate response using the agent's LLM with JSON response format
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    # Parse the decision
    try:
        decision = json.loads(response.content)
        print(f"Resource Manager decided to: {decision}")
        
        # Execute the action
        action = {
            "agent_id": agent.id,
            "action_type": decision.get("action")
        }
        
        if decision.get("action") == "gather":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
        
        elif decision.get("action") == "process":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
        
        elif decision.get("action") == "share":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
            action["target_agent"] = "information_analyst"
        
        result = environment.step(action)
        print(f"Result: {result.get('message')}")
        
        return result
    
    except Exception as e:
        print(f"Error parsing Resource Manager decision: {str(e)}")
        print(f"Raw response: {response.content}")
        
        # Default action if parsing fails
        default_action = {
            "agent_id": agent.id,
            "action_type": "gather",
            "resource": "energy",
            "amount": 10
        }
        result = environment.step(default_action)
        print(f"Executed default action. Result: {result.get('message')}")
        
        return result

async def information_analyst_action(agent, environment, step):
    """Determine and execute the Information Analyst's action."""
    
    # Observe the environment
    observe_action = {
        "agent_id": agent.id,
        "action_type": "observe"
    }
    observation = environment.step(observe_action)
    
    # Create a prompt for the agent to decide on an action
    env_state = observation["environment_state"]
    agent_state = observation["agent_state"]
    
    prompt = f"""
    You are {agent.role}. {agent.persona}
    
    Current simulation step: {step}/10
    
    Environment state:
    - Available resources: {env_state['resources']}
    - Recent events: {env_state['events']}
    
    Your current state:
    - Energy: {agent_state.get('energy', 0)}
    - Materials: {agent_state.get('materials', 0)}
    - Information: {agent_state.get('information', 0)}
    - Actions taken: {agent_state.get('actions_taken', 0)}
    
    As the Information Analyst, decide on your next action. You can:
    1. Gather a resource (energy, materials, information)
    2. Process a resource into information (your specialty)
    3. Share a resource with the Resource Manager
    
    Choose the action that best aligns with your objectives of processing resources into information
    and analyzing environmental patterns.
    
    Respond with a JSON object containing your chosen action:
    For gathering: {{"action": "gather", "resource": "[resource_name]", "amount": [number]}}
    For processing: {{"action": "process", "resource": "[resource_name]", "amount": [number]}}
    For sharing: {{"action": "share", "resource": "[resource_name]", "amount": [number], "target": "resource_manager"}}
    """
    
    # Generate response using the agent's LLM with JSON response format
    response = await agent.llm_orchestrator.generate(
        model=agent.llm_config.model,
        messages=[{"role": "system", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    # Parse the decision
    try:
        decision = json.loads(response.content)
        print(f"Information Analyst decided to: {decision}")
        
        # Execute the action
        action = {
            "agent_id": agent.id,
            "action_type": decision.get("action")
        }
        
        if decision.get("action") == "gather":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
        
        elif decision.get("action") == "process":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
        
        elif decision.get("action") == "share":
            action["resource"] = decision.get("resource")
            action["amount"] = decision.get("amount", 10)
            action["target_agent"] = "resource_manager"
        
        result = environment.step(action)
        print(f"Result: {result.get('message')}")
        
        return result
    
    except Exception as e:
        print(f"Error parsing Information Analyst decision: {str(e)}")
        print(f"Raw response: {response.content}")
        
        # Default action if parsing fails
        default_action = {
            "agent_id": agent.id,
            "action_type": "process",
            "resource": "information",
            "amount": 10
        }
        result = environment.step(default_action)
        print(f"Executed default action. Result: {result.get('message')}")
        
        return result

async def main():
    # Run the simulation
    results = await run_simulation()
    
    print("\nSimulation Summary:")
    print(f"Completed {results['steps']} steps")
    print(f"Total resources gathered: {results['total_resources']}")
    print(f"Total information generated: {results['total_information']}")

if __name__ == "__main__":
    import json
    asyncio.run(main())
