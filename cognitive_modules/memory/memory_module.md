# Memory

The current agent paradigm is defined by a `Perceive <-> Reflection <-> Action` framework based on FIPA (http://www.fipa.org/specs/fipa00023/SC00023J.html). The `Memory Module` should ingest relevent information contextually into these prompts. 

For a separate memory module, we would need to track the following key elements:

`memory: List[Dict[str, Any]]` This is the main memory storage, containing a list of dictionaries representing various memory entries.

`last_action: Optional[Dict[str, Any]]` Stores the most recent action taken by the agent.

Implement the following functionalities:

1. `add_memory(entry: Dict[str, Any]) -> None`
   Add a new memory entry to the storage.

2. `get_recent_memories(n: int) -> List[Dict[str, Any]]`
   Retrieve the n most recent memory entries.

3. `update_last_action(action: Dict[str, Any]) -> None`
   Update the last action taken by the agent.

4. `get_last_action() -> Optional[Dict[str, Any]]`
   Retrieve the last action taken by the agent.

5. `get_memory_summary() -> str`
   Generate a summary of recent memories for use in prompts.

6. `clear_memory() -> None`
   Clear all stored memories (if needed for resets or new episodes).

The memory entries should include:
   - Type of memory (e.g., "reflection", "action", "perception")
   - Content (specific to the type of memory)
   - Timestamp
   - Associated environment (if applicable)
   - Reward (if applicable)
   - Strategy updates (for reflections)

Integrate a new `memory_module` class to work with the `generate_actions` def in `agent_module.py`. 

The `memory_module` should have the following features:

**Database Integration and Retrieval Modes**

1. **Database Integration**:
   - Collaborate with @QuantumQuester to integrate the memory module with the PostgreSQL database they are building.
   - Explore integration with a vector database, ensuring seamless interaction between the two systems.
   - Assign @Bexboy and @QuantumQuester to work together on the integration and testing of the database integration.

2. **Retrieval Modes**:
   - **BM25 Index**: Implement a BM25 index for efficient text-based retrieval. This should be accompanied by a separate script for testing purposes. The system should allow for toggling between BM25 and BM25+cosine vector search modes, with or without embeddings.
   - **BM25+Cosine Vector Search**: This mode combines the BM25 index with cosine vector search for more advanced retrieval capabilities.
   - **Hyde**: Implement the Hyde retrieval mode, which summarizes chunks of data and uses cosine similarity to identify the most relevant chunks.
   - **Raptor**: The Raptor mode utilizes a graph database and clusters generated summaries to find the most relevant chunks by isolating the search space.

3. **Chunking Strategy**:
   - Develop an appropriate chunking strategy for each file format and content type to ensure efficient data processing and retrieval.

**Current target prompt-schema for poking the memory variables into the agent prompts:**

```yaml
# Market Agent Prompts
perception: |
  Perceive the current state of the {environment_name} environment:

  Environment State: {environment_info}
  Recent Memories: {recent_memories}

  Generate a brief monologue about your current perception of this environment.

action: |
  Generate an action for the {environment_name} environment based on the following:

  Perception: {perception}
  Environment State: {environment_info}
  Recent Memories: {recent_memories}
  Available Actions: {action_space}

  Choose an appropriate action for this environment.

reflection: |
  Reflect on this observation from the {environment_name} environment:

  Observation: {observation}
  Environment State: {environment_info}
  Last Action: {last_action}
  Reward: {reward}

  Actions:
  1. Reflect on the observation and surplus based on your last action
  2. Update strategy based on this reflection, the surplus, and your previous strategy

  Previous strategy: {previous_strategy}
```
