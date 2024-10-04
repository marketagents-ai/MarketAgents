# TODO: Implement Pydantic models for agent message roles

## 1. BaseAgentMessage

- [ ] Create a base Pydantic model for all agent messages
- [ ] Include common fields like `content`, `additional_kwargs`, `type`, etc.
- [ ] Implement `get_lc_namespace` method

## 2. RoleMessage

- [ ] Create a Pydantic model that inherits from BaseAgentMessage
- [ ] Include fields for `role`, `persona`, and `objectives`
- [ ] Override `type` with Literal["role"]
- [ ] Implement custom validation for role-specific content

## 3. TaskMessage

- [ ] Create a Pydantic model that inherits from BaseAgentMessage
- [ ] Include fields for `task`, `context`, and `resources`
- [ ] Override `type` with Literal["task"]
- [ ] Implement custom validation for task-specific content

## 4. ExecutionMessage

- [ ] Create a Pydantic model that inherits from BaseAgentMessage
- [ ] Include fields for `output`, `reasoning`, and `metadata`
- [ ] Override `type` with Literal["execution"]
- [ ] Implement custom validation for execution-specific content

## 5. ToolMessage

- [ ] Create a Pydantic model that inherits from BaseAgentMessage
- [ ] Include fields for `tool_name`, `input_parameters`, and `output`
- [ ] Override `type` with Literal["tool"]
- [ ] Implement custom validation for tool-specific content

## 6. MessageArray

- [ ] Create a Pydantic model to represent an array of agent messages
- [ ] Use Union type to allow different message types in the array
- [ ] Implement validation to ensure proper message order (e.g., Role -> Task -> Execution)

## Implementation Notes

- Use Pydantic's `Field` for additional validation and metadata
- Implement custom validators where necessary
- Ensure all models are compatible with JSON serialization
- Consider implementing `to_dict` and `from_dict` methods for easy conversion
- Update `agent.py` and `prompter.py` to use these new message models
- Implement conversion methods to/from ChatML format for compatibility

## Testing

- [ ] Write unit tests for each message type
- [ ] Test serialization and deserialization
- [ ] Test conversion to/from ChatML format
- [ ] Test message array validation

## Documentation

- [ ] Add docstrings to all models and methods
- [ ] Provide usage examples in the module-level docstring
- [ ] Update existing documentation to reflect the new message structure


