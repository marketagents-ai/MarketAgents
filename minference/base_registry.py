"""
Base registry implementation providing common functionality for specialized registries.

This module defines the base registry class that implements common features like
logging, timestamps, and basic registry operations.
"""
from typing import Dict, Any, Optional, Generic, TypeVar, Union
from datetime import datetime
import logging
from io import StringIO
from uuid import UUID

T = TypeVar('T')

class BaseRegistry(Generic[T]):
    """
    Base class for registry implementations.
    
    Provides common functionality for registry operations including:
    - Singleton pattern
    - Logging setup and management
    - Timestamped operations
    - Basic registry operations
    
    Type Args:
        T: Type of items stored in registry
    """
    _instance = None
    _registry: Dict[Union[str, UUID], T] = {}
    _timestamps: Dict[Union[str, UUID], datetime] = {}
    
    def __new__(cls) -> 'BaseRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._setup_logging()
        return cls._instance
    
    @classmethod
    def _setup_logging(cls) -> None:
        """Initialize logging for this registry instance."""
        registry_name = cls.__name__
        cls._log_stream = StringIO()
        cls._logger = logging.getLogger(registry_name)
        
        if not cls._logger.handlers:
            handler = logging.StreamHandler(cls._log_stream)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            ))
            cls._logger.addHandler(handler)
            cls._logger.setLevel(logging.INFO)
    
    @classmethod
    def get_logs(cls) -> str:
        """Get all logs as text."""
        return cls._log_stream.getvalue()
    
    @classmethod
    def clear_logs(cls) -> None:
        """Clear all logs."""
        cls._log_stream.truncate(0)
        cls._log_stream.seek(0)
        cls._logger.info("Logs cleared")

    @classmethod
    def set_log_level(cls, level: Union[int, str]) -> None:
        """Set the logging level."""
        cls._logger.setLevel(level)
        cls._logger.info(f"Log level set to {level}")

    @classmethod
    def _record_timestamp(cls, key: Union[str, UUID]) -> None:
        """Record timestamp for an operation."""
        cls._timestamps[key] = datetime.utcnow()

    @classmethod
    def get_timestamp(cls, key: Union[str, UUID]) -> Optional[datetime]:
        """Get timestamp for when an item was last modified."""
        return cls._timestamps.get(key)

    @classmethod
    def clear(cls) -> None:
        """Clear all items from registry."""
        cls._logger.info(f"{cls.__name__}: Clearing all items")
        cls._registry.clear()
        cls._timestamps.clear()

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Get current status of the registry."""
        cls._logger.debug(f"{cls.__name__}: Retrieving registry status")
        return {
            "total_items": len(cls._registry),
            "oldest_item": min(cls._timestamps.values()) if cls._timestamps else None,
            "newest_item": max(cls._timestamps.values()) if cls._timestamps else None
        }