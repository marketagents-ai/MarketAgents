# Market Agent Refactoring TODO List

## 1. Update Imports
- [ ] Update imports in `market_agents.py` to reflect new folder structure
  - [ ] Import `ZIAgent`, `Order`, `Trade` from `ziagents.py`
  - [ ] Import `Agent as LLMAgent` from `base_agent.agent`
  - [ ] Import `MarketActionSchema`, `BidSchema` from appropriate schema file

## 2. Refactor MarketAgent Class
- [ ] Update `MarketAgent` class definition
  - [ ] Use `ZIAgent` instead of directly using `PreferenceSchedule`
  - [ ] Use `LLMAgent` for LLM-based decision making
  - [ ] Update `create` class method to initialize both `ZIAgent` and `LLMAgent`

## 3. Update generate_bid Method
- [ ] Refactor `generate_bid` method
  - [ ] Use `ZIAgent`'s `generate_bid` method when not using LLM
  - [ ] Update LLM-based bid generation to use new `LLMAgent` execute method

## 4. Adapt _generate_llm_bid Method
- [ ] Update method to use new `LLMAgent` and `MarketActionSchema`
- [ ] Ensure compatibility with new `Order` class from `ziagents.py`

## 5. Update Utility Methods
- [ ] Refactor `finalize_trade` to use `ZIAgent`'s method
- [ ] Update `respond_to_order` if necessary
- [ ] Adapt `calculate_trade_surplus` and `calculate_individual_surplus` to use `ZIAgent` methods

## 6. Memory and Logging
- [ ] Review and update memory storage mechanism
- [ ] Ensure logging is compatible with new agent structure

## 7. Market Information Handling
- [ ] Update `_get_market_info` method to reflect any changes in market information structure

## 8. Testing
- [ ] Update existing tests to work with refactored `MarketAgent`
- [ ] Add new tests for LLM integration if not already present

## 9. Documentation
- [ ] Update docstrings and comments to reflect new structure and dependencies

## 10. Performance Optimization
- [ ] Review the refactored code for any potential performance improvements

## 11. Error Handling
- [ ] Ensure robust error handling, especially for LLM-related operations

## 12. Configuration
- [ ] Update any configuration files or environment variables needed for the refactored agents

## 13. Integration
- [ ] Test the refactored `MarketAgent` in the broader simulation environment
- [ ] Resolve any integration issues that arise

## 14. Code Cleanup
- [ ] Remove any obsolete code or commented-out sections
- [ ] Ensure consistent coding style throughout the refactored files
