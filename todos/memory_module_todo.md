# Memory Module TODO

## Objective
Implement a basic memory module for the LLMAgent class to store and utilize agent interactions from previous rounds.

## Tasks

1. Modify LLMAgent class in `agents.py`:
   - [ ] Add a `memory` attribute to store interactions
   - [ ] Update the `log_interaction` method to append to `memory`
   - [ ] Implement a method to retrieve recent memories (e.g., `get_recent_memories(n)`)

2. Update MarketAgent class in `market_agents.py`:
   - [ ] Add a method to access LLMAgent's memory
   - [ ] Modify `generate_bid` method to include recent memories in market info

3. Adjust simulation logic in `simulation_app.py`:
   - [ ] Remove or modify the current JSON file logging for agent interactions
   - [ ] Update the auction loop to utilize the new memory feature

4. Implement memory utilization:
   - [ ] Modify the prompt generation in `agents.py` to include recent memories
   - [ ] Add a function in `prompter.py` to format memories into text prompt
   - [ ] Experiment with including memories from n-k rounds as the simulation progresses

5. (Optional) Implement memory indexing:
   - [ ] Research and choose an appropriate indexing method for agent memories
   - [ ] Implement indexing as the number of rounds increases

6. Testing and validation:
   - [ ] Create unit tests for the new memory-related methods
   - [ ] Run simulations to verify the impact of memory on agent behavior

7. Documentation:
   - [ ] Update relevant documentation to reflect the new memory feature
   - [ ] Add comments explaining the memory utilization in the code

## Notes
- Start with a simple implementation of appending all interactions to the memory
- As the project evolves, consider more sophisticated memory management techniques
- Monitor performance and adjust the number of remembered rounds (n-k) as needed
