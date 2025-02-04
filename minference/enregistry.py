"""
Registry implementation for managing immutable Pydantic model instances.

This module provides specialized registry functionality for managing immutable
Pydantic model instances with UUID-based identification and version tracking.
Entities are treated as immutable - any modifications require creating new instances
with new UUIDs.

Main components:
- EntityRegistry: Singleton registry for managing immutable Pydantic models
- Entity Type Validation: Runtime type checking for entity instances
- Entity Lineage Tracking: Functions for tracking entity versions
"""
import logging
from typing import Dict, Any, Optional, Type, TypeVar, List, Generic, Protocol, runtime_checkable
from pydantic import BaseModel
from uuid import UUID
import json
from datetime import datetime

from minference.base_registry import BaseRegistry

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@runtime_checkable
class HasID(Protocol):
    id: UUID

EntityType = TypeVar('EntityType', bound='HasID')

class EntityRegistry(BaseRegistry[EntityType]):
    """
    Registry for managing immutable Pydantic model instances.
    
    Extends BaseRegistry to provide specialized handling for immutable Pydantic models.
    All entities are treated as immutable - any modifications require creating
    new instances with new UUIDs.
    
    Type Args:
        EntityType: Type of Pydantic models to store
    """
    _logger = logger
    _registry: Dict[UUID, EntityType] = {}
    _timestamps: Dict[UUID, datetime] = {}

    @classmethod
    def register(cls, entity: EntityType) -> None:
        """
        Register a new entity instance or verify reference to existing entity.
        
        Args:
            entity: Pydantic model instance to register
            
        Raises:
            ValueError: If entity validation fails or if existing entity doesn't match
        """
        if not isinstance(entity, BaseModel):
            cls._logger.error(f"Invalid entity type: {type(entity)}")
            raise ValueError("Entity must be a Pydantic model instance")
            
        if not isinstance(entity, HasID):
            cls._logger.error(f"Entity missing ID field: {type(entity)}")
            raise ValueError("Entity must have an 'id' field")
            
        entity_id = entity.id
        cls._logger.debug(f"Attempting to register {entity.__class__.__name__}({entity_id})")
        
        # If entity already exists, verify it's the same
        if entity_id in cls._registry:
            existing_entity = cls._registry[entity_id]
            # Compare entity types
            if type(existing_entity) != type(entity):
                cls._logger.error(
                    f"Type mismatch for {entity_id}:\n"
                    f"Existing: {type(existing_entity)}\n"
                    f"New: {type(entity)}"
                )
                raise ValueError(f"Entity type mismatch for {entity_id}")
            
            # Compare entity contents (excluding id and created_at)
            new_dict = entity.model_dump(exclude={'id', 'created_at'})
            existing_dict = existing_entity.model_dump(exclude={'id', 'created_at'})
            
            if new_dict != existing_dict:
                # Find and log the differences
                diffs = []
                for key in set(new_dict.keys()) | set(existing_dict.keys()):
                    if key not in new_dict:
                        diffs.append(f"Key '{key}' missing in new entity")
                    elif key not in existing_dict:
                        diffs.append(f"Key '{key}' missing in existing entity")
                    elif new_dict[key] != existing_dict[key]:
                        diffs.append(
                            f"Value mismatch for '{key}':\n"
                            f"  Existing: {existing_dict[key]}\n"
                            f"  New: {new_dict[key]}"
                        )
                
                cls._logger.error(
                    f"Content mismatch for {entity.__class__.__name__}({entity_id}):\n"
                    f"Differences:\n" + "\n".join(diffs)
                )
                raise ValueError(
                    f"Entity content mismatch for {entity_id}. "
                    "Create a new instance with a new UUID for modifications."
                )
                
            # Entity exists and matches
            cls._logger.debug(f"{entity.__class__.__name__}({entity_id}) already registered and matches")
            return
            
        try:
            # Register new entity
            cls._registry[entity_id] = entity
            cls._timestamps[entity_id] = datetime.utcnow()
            cls._logger.info(f"Successfully registered {entity.__class__.__name__}({entity_id})")
        except Exception as e:
            cls._logger.error(f"Failed to register {entity.__class__.__name__}({entity_id}): {str(e)}")
            raise ValueError(f"Entity registration failed: {str(e)}") from e

    @classmethod
    def get(
        cls, 
        entity_id: UUID, 
        expected_type: Optional[Type[EntityType]] = None
    ) -> Optional[EntityType]:
        """
        Retrieve an immutable entity by ID with optional type checking.
        
        Args:
            entity_id: UUID of the entity to retrieve
            expected_type: Optional type to validate against
            
        Returns:
            The immutable entity instance if found and type matches, None otherwise
        """
        cls._logger.debug(f"Retrieving entity {entity_id}")
        entity = cls._registry.get(entity_id)
        
        if entity is None:
            cls._logger.debug(f"Entity {entity_id} not found")
            return None
            
        if expected_type and not isinstance(entity, expected_type):
            cls._logger.error(
                f"Type mismatch for {entity_id}. "
                f"Expected {expected_type.__name__}, got {type(entity).__name__}"
            )
            return None
            
        return entity

    @classmethod
    def list_by_type(cls, entity_type: Type[EntityType]) -> List[EntityType]:
        """
        List all entities of a specific type.
        
        Args:
            entity_type: Type of entities to list
            
        Returns:
            List of immutable entities matching the specified type
        """
        cls._logger.debug(f"Listing entities of type {entity_type.__name__}")
        return [
            entity for entity in cls._registry.values()
            if isinstance(entity, entity_type)
        ]

    @classmethod
    def get_many(
        cls,
        entity_ids: List[UUID],
        expected_type: Optional[Type[EntityType]] = None
    ) -> List[EntityType]:
        """
        Retrieve multiple immutable entities by their IDs.
        
        Args:
            entity_ids: List of UUIDs to retrieve
            expected_type: Optional type to validate against
            
        Returns:
            List of found immutable entities matching the type (if specified)
        """
        cls._logger.debug(f"Retrieving {len(entity_ids)} entities")
        return [
            entity for entity_id in entity_ids
            if (entity := cls.get(entity_id, expected_type)) is not None
        ]

    @classmethod
    def get_registry_status(cls) -> Dict[str, Any]:
        """Get detailed status of the registry."""
        base_status = super().get_registry_status()
        
        # Add entity-specific stats
        type_counts = {}
        for entity in cls._registry.values():
            type_name = type(entity).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
        # Add timestamp-based versioning info
        timestamps = sorted(cls._timestamps.values())
        
        return {
            **base_status,
            "entities_by_type": type_counts,
            "version_history": {
                "first_version": timestamps[0].isoformat() if timestamps else None,
                "latest_version": timestamps[-1].isoformat() if timestamps else None,
                "version_count": len(timestamps)
            }
        }