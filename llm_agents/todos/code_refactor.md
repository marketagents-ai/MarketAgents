# Python Code Formatting and Refactoring Guidelines

## General Guidelines

1. Follow PEP 8 style guide for Python code.
2. Use Google Python Style Guide for additional recommendations.
3. Maintain consistency throughout the codebase.

## Formatting

1. Use 4 spaces for indentation (not tabs).
2. Limit all lines to a maximum of 79 characters for code, 72 for docstrings/comments.
3. Use blank lines to separate functions and classes, and larger blocks of code inside functions.
4. Use inline comments sparingly, and write them at least two spaces away from the code.
5. Use docstrings for all public modules, functions, classes, and methods.

## Naming Conventions

1. Use `snake_case` for function and variable names.
2. Use `PascalCase` for class names.
3. Use `UPPER_CASE` for constants.
4. Prefix private attributes with a single underscore.

## Code Structure

1. Place imports at the top of the file, grouped in the following order:
   - Standard library imports
   - Related third-party imports
   - Local application/library specific imports
2. Use absolute imports when possible.
3. Avoid wildcard imports (`from module import *`).

## Function and Method Guidelines

1. Keep functions and methods short and focused on a single task.
2. Use type hints for function arguments and return values.
3. Use default parameter values instead of overloading methods.

## Class Guidelines

1. Use the `@property` decorator for getters and setters.
2. Implement `__str__` and `__repr__` methods for classes.
3. Use `@classmethod` for alternative constructors.

## Error Handling

1. Use exceptions for error handling, not return codes.
2. Be specific with exception types.
3. Use context managers (`with` statements) for resource management.

## Testing

1. Write unit tests for all functions and methods.
2. Use pytest for testing framework.
3. Aim for high test coverage, especially for critical components.

## Market Simulation Specific Guidelines

1. Ensure clear separation between agent logic, market mechanisms, and simulation control.
2. Use descriptive variable names that reflect economic concepts (e.g., `price`, `quantity`, `utility`).
3. Implement proper encapsulation for agent attributes to prevent direct manipulation.
4. Use appropriate data structures for order books, trade history, and agent memories.
5. Ensure that random number generation is controllable and reproducible for experiments.
6. Implement proper logging for important events and agent decisions.
7. Use design patterns appropriate for agent-based simulations (e.g., Observer pattern for market updates).
8. Optimize performance-critical sections, especially in market clearing algorithms.
9. Implement clear interfaces between different components (e.g., agents, market, institution).
10. Use appropriate numerical libraries (e.g., NumPy) for mathematical operations.

## Refactoring Tips

1. Regularly review and refactor code to improve readability and maintainability.
2. Look for repeated code and extract it into reusable functions or methods.
3. Use meaningful names for variables, functions, and classes that describe their purpose or behavior.
4. Break down large functions or methods into smaller, more manageable pieces.
5. Use appropriate design patterns to solve common problems and improve code structure.
6. Regularly update dependencies and use modern Python features when appropriate.
7. Remove dead code and unused imports.
8. Use profiling tools to identify and optimize performance bottlenecks.

Remember, the goal is to write clean, readable, and maintainable code that accurately models the economic system we're simulating. Always consider the balance between code elegance and performance, especially in computationally intensive simulations.
