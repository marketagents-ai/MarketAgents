"""
Registry module for managing callable tools and their schemas.

This module provides a global registry for managing callable functions along with 
their input/output schemas. It includes functionality for registering functions,
deriving schemas from type hints, and executing registered functions safely both
synchronously and asynchronously.

Main components:
- CallableRegistry: Singleton registry for managing functions
- Schema helpers: Functions for deriving and validating JSON schemas
- Registration helpers: Functions for safely registering callables
- Execution helpers: Functions for safely executing registered callables
"""
from typing import Dict, Any, List, Optional, Literal, Union, Tuple, Callable, TypeAlias, Protocol, TypeVar, Awaitable, get_type_hints
from pydantic import ValidationError, create_model, BaseModel
import json
from ast import literal_eval
import sys
import libcst as cst
from inspect import signature, iscoroutinefunction
from dataclasses import dataclass
import asyncio

from .base_registry import BaseRegistry

# Type aliases and protocols
JsonDict: TypeAlias = Dict[str, Any]
SchemaType: TypeAlias = Dict[str, Any]
RegistryType: TypeAlias = Dict[str, Callable]

class AsyncCallable(Protocol):
    """Protocol for async callable objects"""
    async def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class SyncCallable(Protocol):
    """Protocol for sync callable objects"""
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

AnyCallable = Union[AsyncCallable, SyncCallable]

@dataclass
class RegistryInfo:
    """Information about a registered function"""
    name: str
    signature: str
    doc: Optional[str]
    input_schema: SchemaType
    output_schema: SchemaType
    is_async: bool

def is_async_callable(func: Callable) -> bool:
    """Check if a callable is async."""
    return iscoroutinefunction(func) or hasattr(func, '__await__')

async def ensure_async(func: AnyCallable, *args: Any, **kwargs: Any) -> Any:
    """
    Ensure a function is executed asynchronously.
    Wraps sync functions in asyncio.to_thread.
    """
    if is_async_callable(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)

class CallableRegistry(BaseRegistry[Callable]):
    """Global registry for tool callables"""
    _registry: Dict[str, Callable] = {}  # Add explicit type annotation
    _timestamps: Dict[str, float] = {}   # Add explicit type annotation
    
    @classmethod
    def register(cls, name: str, func: Callable) -> None:
        """Register a new callable with validation."""
        cls._logger.info(f"Attempting to register function: {name}")
        
        if name in cls._registry:
            cls._logger.error(f"Registration failed: Function '{name}' already registered")
            raise ValueError(f"Function '{name}' already registered. Use update() to replace.")
        
        try:
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._record_timestamp(name)
            cls._logger.info(f"Successfully registered function: {name}")
        except Exception as e:
            cls._logger.error(f"Registration failed for {name}: {str(e)}")
            raise
    
    @classmethod
    def _validate_type_hints(cls, name: str, func: Callable) -> None:
        """Validate that a function has proper type hints."""
        cls._logger.debug(f"Validating type hints for function: {name}")
        
        try:
            type_hints = get_type_hints(func)
        except (NameError, TypeError) as e:
            cls._logger.error(f"Validation failed: Function '{name}' has invalid type hints")
            raise ValueError(f"Function '{name}' must have valid type hints") from e
        
        if not type_hints:
            cls._logger.error(f"Validation failed: Function '{name}' has no type hints")
            raise ValueError(f"Function '{name}' must have type hints")
        
        if 'return' not in type_hints:
            cls._logger.error(f"Validation failed: Function '{name}' has no return type hint")
            raise ValueError(f"Function '{name}' must have a return type hint")
        
        cls._logger.debug(f"Type hints valid for function: {name}")

    @classmethod
    def register_from_text(cls, name: str, func_text: str) -> None:
        """Register a function from its text representation with safety checks."""
        cls._logger.info(f"Attempting to register function from text: {name}")
        
        if name in cls._registry:
            cls._logger.error(f"Registration failed: Function '{name}' already registered")
            raise ValueError(f"Function '{name}' already registered. Use update().")
        
        try:
            func = cls._parse_function_text(name, func_text)
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._record_timestamp(name)
            cls._logger.info(f"Successfully registered function from text: {name}")
        except Exception as e:
            cls._logger.error(f"Failed to register function from text: {str(e)}")
            raise ValueError(f"Failed to parse function: {str(e)}")

    @staticmethod
    def _parse_function_text(name: str, func_text: str) -> Callable:
        """Parse function text into a callable with safety checks."""
        # Handle lambdas
        if func_text.strip().startswith('lambda'):
            wrapper_text = f"""
def {name}(x: float) -> float:
    \"\"\"Wrapped lambda function\"\"\"
    func = {func_text}
    return func(x)
"""
            func_text = wrapper_text

        try:
            module = cst.parse_module(func_text)
            namespace = {
                'float': float, 'int': int, 'str': str, 'bool': bool,
                'list': list, 'dict': dict, 'tuple': tuple,
                'List': List, 'Dict': Dict, 'Tuple': Tuple,
                'Optional': Optional, 'Union': Union, 'Any': Any,
                'BaseModel': BaseModel
            }
            
            exec(module.code, namespace)
            
            if func_text.strip().startswith('lambda'):
                return namespace[name]
            return namespace[func_text.split('def ')[1].split('(')[0].strip()]
            
        except Exception as e:
            raise ValueError(f"Failed to parse function: {str(e)}")

    @classmethod
    def update(cls, name: str, func: Callable) -> None:
        """Update an existing callable with validation."""
        cls._logger.info(f"Attempting to update function: {name}")
        try:
            cls._validate_type_hints(name, func)
            cls._registry[name] = func
            cls._record_timestamp(name)
            cls._logger.info(f"Successfully updated function: {name}")
        except Exception as e:
            cls._logger.error(f"Update failed for {name}: {str(e)}")
            raise
    
    @classmethod
    def delete(cls, name: str) -> None:
        """Delete a callable from registry."""
        cls._logger.info(f"Attempting to delete function: {name}")
        if name not in cls._registry:
            cls._logger.error(f"Deletion failed: Function '{name}' not found")
            raise ValueError(f"Function '{name}' not found in registry.")
        del cls._registry[name]
        del cls._timestamps[name]
        cls._logger.info(f"Successfully deleted function: {name}")
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get a registered callable by name."""
        cls._logger.debug(f"Retrieving function: {name}")
        return cls._registry.get(name)
    
    @classmethod
    def get_info(cls, name: str) -> Optional[RegistryInfo]:
        """Get detailed information about a registered function."""
        cls._logger.debug(f"Retrieving info for function: {name}")
        
        func = cls.get(name)
        if not func:
            cls._logger.debug(f"No info available: Function '{name}' not found")
            return None
            
        return RegistryInfo(
            name=name,
            signature=str(signature(func)),
            doc=func.__doc__,
            input_schema=derive_input_schema(func),
            output_schema=derive_output_schema(func),
            is_async=is_async_callable(func)
        )

    @classmethod
    def get_registry_status(cls) -> JsonDict:
        """Get current status of the registry."""
        cls._logger.debug("Retrieving registry status")
        return {
            "total_functions": len(cls._registry),
            "registered_functions": list(cls._registry.keys()),
            "function_signatures": {
                name: str(signature(func))
                for name, func in cls._registry.items()
            }
        }
    
    @classmethod
    def execute(cls, name: str, input_data: JsonDict) -> JsonDict:
        """
        Execute a registered callable synchronously.
        
        Args:
            name: Name of function to execute
            input_data: Input data for function
            
        Returns:
            JsonDict containing the execution result
            
        Raises:
            ValueError: If function not found or execution fails
        """
        cls._logger.info(f"Attempting sync execution of function: {name}")
        try:
            result = execute_callable(name, input_data, registry=cls)
            cls._logger.info(f"Successfully executed {name} synchronously")
            return result
        except Exception as e:
            cls._logger.error(f"Sync execution failed for {name}: {str(e)}")
            raise
        
    @classmethod
    async def aexecute(cls, name: str, input_data: JsonDict) -> JsonDict:
        """Execute a registered callable asynchronously."""
        cls._logger.info(f"Attempting async execution of function: {name}")
        try:
            result = await aexecute_callable(name, input_data, registry=cls)
            cls._logger.info(f"Successfully executed {name} asynchronously")
            return result
        except Exception as e:
            cls._logger.error(f"Async execution failed for {name}: {str(e)}")
            raise

# Helper function definitions exactly as before
def derive_input_schema(func: Callable) -> SchemaType:
    """Derive JSON schema from function type hints."""
    type_hints = get_type_hints(func)
    sig = signature(func)
    
    if 'return' not in type_hints:
        raise ValueError(f"Function {func.__name__} must have a return type hint")
    
    first_param = next(iter(sig.parameters.values()))
    first_param_type = type_hints.get(first_param.name)
    
    if (isinstance(first_param_type, type) and 
        issubclass(first_param_type, BaseModel)):
        return first_param_type.model_json_schema()
    
    input_fields = {}
    for param_name, param in sig.parameters.items():
        if param_name not in type_hints:
            raise ValueError(f"Parameter {param_name} must have a type hint")
        
        if param.default is param.empty:
            input_fields[param_name] = (type_hints[param_name], ...)
        else:
            input_fields[param_name] = (type_hints[param_name], param.default)

    InputModel = create_model(f"{func.__name__}Input", **input_fields)
    return InputModel.model_json_schema()

def derive_output_schema(func: Callable) -> SchemaType:
    """Derive output JSON schema from function return type."""
    type_hints = get_type_hints(func)
    
    if 'return' not in type_hints:
        raise ValueError(f"Function {func.__name__} must have a return type hint")
    
    output_type = type_hints['return']
    if isinstance(output_type, type) and issubclass(output_type, BaseModel):
        OutputModel = output_type
    else:
        OutputModel = create_model(f"{func.__name__}Output", result=(output_type, ...))
    
    return OutputModel.model_json_schema()

def validate_schema_compatibility(
    derived_schema: SchemaType,
    provided_schema: SchemaType
) -> None:
    """Validate that provided schema matches derived schema."""
    def get_ref_schema(schema: SchemaType, ref: str) -> SchemaType:
        """Resolve a $ref to its actual schema."""
        if not ref.startswith("#/$defs/"):
            raise ValueError(f"Invalid $ref format: {ref}")
        model_name = ref.split("/")[-1]
        return schema.get("$defs", {}).get(model_name, {})

    def compare_props(
        name: str,
        derived_prop: SchemaType,
        provided_prop: SchemaType,
        derived_root: SchemaType,
        provided_root: SchemaType
    ) -> None:
        """Compare two property schemas, handling $refs."""
        # Resolve refs if present
        if "$ref" in derived_prop:
            derived_prop = get_ref_schema(derived_root, derived_prop["$ref"])
        if "$ref" in provided_prop:
            provided_prop = get_ref_schema(provided_root, provided_prop["$ref"])

        # Handle array types
        if derived_prop.get("type") == "array":
            if provided_prop.get("type") != "array":
                raise ValueError(f"Property '{name}' type mismatch: Expected array")
                
            derived_items = derived_prop["items"]
            provided_items = provided_prop["items"]
            
            # Handle refs in array items
            if "$ref" in derived_items:
                derived_items = get_ref_schema(derived_root, derived_items["$ref"])
            if "$ref" in provided_items:
                provided_items = get_ref_schema(provided_root, provided_items["$ref"])
                
            if derived_items.get("type") != provided_items.get("type"):
                raise ValueError(
                    f"Array items type mismatch for '{name}'.\n"
                    f"Derived: {derived_items.get('type')}\n"
                    f"Provided: {provided_items.get('type')}"
                )
                
        # Handle direct type comparison
        elif derived_prop.get("type") != provided_prop.get("type"):
            raise ValueError(
                f"Property '{name}' type mismatch.\n"
                f"Derived: {derived_prop.get('type')}\n"
                f"Provided: {provided_prop.get('type')}"
            )
    
    # First check for missing properties
    derived_props = derived_schema.get("properties", {})
    provided_props = provided_schema.get("properties", {})
    
    for prop_name, prop_schema in derived_props.items():
        if prop_name not in provided_props:
            raise ValueError(f"Missing property '{prop_name}' in provided schema")
        provided_type = provided_props[prop_name].get("type")
        derived_type = prop_schema.get("type")
        if provided_type != derived_type:
            raise ValueError(
                f"Property '{prop_name}' type mismatch.\n"
                f"Derived: {derived_type}\n"
                f"Provided: {provided_type}"
            )

    # Then check for extra properties
    extra_props = set(provided_props.keys()) - set(derived_props.keys())
    if extra_props:
        raise ValueError(f"Extra properties in provided schema: {extra_props}")
        
    # Finally check required properties
    derived_required = set(derived_schema.get("required", []))
    provided_required = set(provided_schema.get("required", []))
    if derived_required != provided_required:
        raise ValueError(
            f"Schema mismatch: Required properties don't match.\n"
            f"Derived: {derived_required}\n"
            f"Provided: {provided_required}"
        )
def execute_callable(
    name: str,
    input_data: JsonDict,
    registry: Optional[type[CallableRegistry]] = None
) -> JsonDict:
    """Execute a registered callable with input data."""
    if registry is None:
        registry = CallableRegistry
    
    # Add type assertion to help type checker
    assert isinstance(registry, type)
        
    callable_func = registry.get(name)
    if not callable_func:
        raise ValueError(
            f"Function '{name}' not found in registry. "
            f"Available: {list(registry._registry.keys())}"
        )
    
    try:
        sig = signature(callable_func)
        type_hints = get_type_hints(callable_func)
        first_param = next(iter(sig.parameters.values()))
        param_type = type_hints.get(first_param.name)
        
        # Handle input based on parameter type
        if (isinstance(param_type, type) and 
            issubclass(param_type, BaseModel)):
            model_input = param_type.model_validate(input_data)
            response = callable_func(model_input)
        else:
            response = callable_func(**input_data)
        
        # Handle response serialization
        if isinstance(response, BaseModel):
            return json.loads(response.model_dump_json())
        return {"result": response}
        
    except Exception as e:
        raise ValueError(f"Error executing {name}: {str(e)}") from e





async def aexecute_callable(
    name: str,
    input_data: JsonDict,
    registry: Optional[type[CallableRegistry]] = None
) -> JsonDict:
    """
    Execute a registered callable asynchronously with input data.
    Handles both async and sync functions.
    
    Args:
        name: Name of function to execute
        input_data: Input data for function
        registry: Optional registry instance (uses global if None)
        
    Returns:
        JsonDict containing the execution result
        
    Raises:
        ValueError: If function not found or execution fails
    """
    if registry is None:
        registry = CallableRegistry
    
    # Add type assertion to help type checker
    assert isinstance(registry, type)
        
    callable_func = registry.get(name)
    if not callable_func:
        raise ValueError(
            f"Function '{name}' not found in registry. "
            f"Available: {list(registry._registry.keys())}"
        )
    
    try:
        sig = signature(callable_func)
        
        # Check for empty parameters FIRST
        if not sig.parameters:
            # Execute with no parameters
            if iscoroutinefunction(callable_func):
                response = await callable_func()
            else:
                response = await asyncio.to_thread(callable_func)
        else:
            type_hints = get_type_hints(callable_func)
            first_param = next(iter(sig.parameters.values()))
            param_type = type_hints.get(first_param.name)
            
            # Prepare input
            if (isinstance(param_type, type) and 
                issubclass(param_type, BaseModel)):
                model_input = param_type.model_validate(input_data)
                input_arg = model_input
            else:
                input_arg = input_data
                
            # Execute function based on its type
            if iscoroutinefunction(callable_func):
                if isinstance(input_arg, BaseModel):
                    response = await callable_func(input_arg)
                else:
                    response = await callable_func(**input_arg)
            else:
                # Run sync function in executor to avoid blocking
                if isinstance(input_arg, BaseModel):
                    response = await asyncio.to_thread(callable_func, input_arg)
                else:
                    response = await asyncio.to_thread(callable_func, **input_arg)
        
        # Handle response serialization
        if isinstance(response, BaseModel):
            return json.loads(response.model_dump_json())
        return {"result": response}
        
    except Exception as e:
        raise ValueError(f"Error executing {name}: {str(e)}") from e

__all__ = [
    'CallableRegistry',
    'RegistryInfo',
    'JsonDict',
    'SchemaType',
    'RegistryType',
    'AsyncCallable',
    'SyncCallable',
    'AnyCallable',

    'derive_input_schema',
    'derive_output_schema',
    'validate_schema_compatibility',
    'execute_callable',
    'aexecute_callable',
    'is_async_callable',
    'ensure_async'
]