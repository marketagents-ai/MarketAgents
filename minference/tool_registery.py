import asyncio
import inspect
import logging
from typing import Dict, Callable, Optional, Any
from uuid import UUID

from minference.enregistry import EntityRegistry
from minference.lite.models import Entity, CallableTool, StructuredTool


class ToolRegistry(EntityRegistry):
    _callable_registry: Dict[str, Callable] = {}

    def register_callable(self, name: str, func: Callable) -> None:
        """Registers a callable function."""
        if name in self._callable_registry:
            raise ValueError(f"Callable '{name}' already registered.")
        self._callable_registry[name] = func
        self._logger.debug(f"Registered callable: {name}")

    def get_callable(self, name: str) -> Optional[Callable]:
        """Retrieves a callable function by name."""
        return self._callable_registry.get(name)

    def register_callable_from_text(self, name: str, source_code: str) -> None:
        """Registers a callable from source code."""
        try:
            local_vars: Dict[str, Any] = {}
            exec(source_code, {}, local_vars)
            func = local_vars.get(name)
            if not func:
                raise ValueError(f"Function '{name}' not found in provided source code.")
            self.register_callable(name, func)
        except Exception as e:
            self._logger.error(f"Error registering callable from text: {e}")
            raise

    def execute_callable(self, name: str, input_data: Dict[str, Any]) -> Any:
        """Executes a registered callable."""
        func = self.get_callable(name)
        if not func:
            raise ValueError(f"Callable '{name}' not found in registry.")
        try:
            return func(**input_data)
        except TypeError as e:
            self._logger.error(f"Error executing callable '{name}': {e}")
            raise ValueError(f"Input data does not match the signature of callable '{name}': {e}") from e

    async def aexecute_callable(self, name: str, input_data: Dict[str, Any]) -> Any:
        """Asynchronously executes a registered callable."""
        func = self.get_callable(name)
        if not func:
            raise ValueError(f"Callable '{name}' not found in registry.")

        if inspect.iscoroutinefunction(func):
            return await func(**input_data)
        else:
            # Wrap synchronous function in a thread
            return await asyncio.to_thread(func, **input_data)

    def get_tool_definition(self, entity: Entity) -> Optional[Dict[str, Any]]:
        """
        Gets the OpenAI function calling tool definition for a given entity.
        Returns None if the entity doesn't support OpenAI function calling.
        """
        if isinstance(entity, (CallableTool, StructuredTool)):
            return entity.get_openai_tool()
        return None