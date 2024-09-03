# Base Agent Integration with AI Utilities TODO List

1. Update Agent class initialization:
   - [ ] Modify the `__init__` method to create an instance of `AIUtilities` and store it as `self.ai_utilities`.

2. Refactor execute method:
   - [ ] Replace the current AI inference logic with calls to `self.ai_utilities.run_ai_completion()`.
   - [ ] Update the method to handle both text and JSON responses based on the `output_format` configuration.

3. Implement tool support:
   - [ ] Add a method to convert the agent's `tools` to the format expected by `run_ai_tool_completion()`.
   - [ ] Update the `execute` method to use `run_ai_tool_completion()` when tools are present.

4. Enhance LLM configuration:
   - [ ] Create a `LLMConfig` instance in the `execute` method based on the agent's `llm_config`.
   - [ ] Ensure all relevant parameters (model, max_tokens, temperature, etc.) are properly set.

5. Improve error handling:
   - [ ] Implement more robust error handling for AI completion calls.
   - [ ] Add appropriate logging for errors and unexpected responses.

6. Update output processing:
   - [ ] Implement logic to parse and validate JSON responses when `output_format` is set to "json" or "json_object".
   - [ ] Ensure the output adheres to the specified `output_format` schema if provided.

7. Optimize message handling:
   - [ ] Refactor the message preparation logic to work with both OpenAI and Anthropic message formats.
   - [ ] Implement a method to convert between different message formats if necessary.

8. Add support for streaming responses:
   - [ ] Investigate and implement support for streaming responses from AI completions if applicable.

9. Implement caching mechanism:
   - [ ] Add a caching layer to store and retrieve frequent AI completion results for improved performance.

10. Update documentation:
    - [ ] Update the Agent class docstrings to reflect the new integration with AIUtilities.
    - [ ] Add examples and usage instructions for the new functionality.

11. Write tests:
    - [ ] Create unit tests for the updated Agent class methods.
    - [ ] Implement integration tests to ensure proper interaction between Agent and AIUtilities.

12. Refactor PromptManager integration:
    - [ ] Update how PromptManager is used in conjunction with the new AIUtilities integration.

13. Performance optimization:
    - [ ] Profile the updated Agent class to identify any performance bottlenecks.
    - [ ] Optimize code where necessary for improved efficiency.

14. Compatibility checks:
    - [ ] Ensure backwards compatibility with any existing code using the Agent class.
    - [ ] Create a migration guide for updating from the old implementation to the new one.
