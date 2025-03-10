import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import uuid

from market_agents.memory.agent_storage.storage_service import StorageService

class OrchestrationDataInserter:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        self.db = storage_service.db
        self.logger = logging.getLogger("orchestration_data_inserter")
        self.NAMESPACE_AGENTS = uuid.uuid5(uuid.NAMESPACE_DNS, 'market_agents.agents')

    async def insert_agents(self, agents_data: List[Dict[str, Any]]) -> Dict[str, uuid.UUID]:
        """Insert agent data and return mapping of agent IDs."""
        query = """
            INSERT INTO agents (id, role, persona, is_llm, max_iter, llm_config, economic_agent)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            ON CONFLICT (id) DO UPDATE SET
                role = EXCLUDED.role,
                persona = EXCLUDED.persona,
                is_llm = EXCLUDED.is_llm,
                max_iter = EXCLUDED.max_iter,
                llm_config = EXCLUDED.llm_config,
                economic_agent = EXCLUDED.economic_agent
            RETURNING id
        """
        agent_id_map = {}
        
        async with self.db.transaction() as txn:
            for agent in agents_data:
                try:
                    string_id = str(agent['id'])
                    agent_uuid = self._get_agent_uuid(string_id)

                    result = await self.db.fetch_one(
                        query,
                        agent_uuid,
                        agent['role'],
                        json.dumps(agent.get('persona', {})),
                        agent.get('is_llm', True),
                        agent.get('max_iter', 0),
                        json.dumps(agent.get('llm_config', {})),
                        json.dumps(agent.get('economic_agent', {}))
                    )
                    if result:
                        agent_id_map[string_id] = result[0]
                except Exception as e:
                    self.logger.error(f"Error inserting agent {string_id}: {e}")
                    raise
        return agent_id_map

    async def insert_actions(self, actions_data: List[Dict[str, Any]], agent_id_map: Optional[Dict[str, uuid.UUID]] = None):
        query = """
            INSERT INTO actions (agent_id, environment_name, round, sub_round, action_data, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        async with self.db.transaction() as txn:
            for action in actions_data:
                try:
                    agent_id_str = str(action['agent_id'])
                    
                    agent_uuid = self._get_agent_uuid(agent_id_str)

                    action_data = {
                        'content': action.get('content') or action.get('action'),
                        'type': action.get('type', 'default'),
                        'cohort_id': action.get('cohort_id'),
                        'topic': action.get('topic')
                    }

                    metadata = {
                        'timestamp': action.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        'message_id': str(action.get('message_id', uuid.uuid4())),
                        **{k: v for k, v in action.items() if k not in ['agent_id', 'environment_name', 'round', 'sub_round', 'content', 'action']}
                    }

                    await self.db.execute(
                        query,
                        agent_uuid,
                        action['environment_name'],
                        action['round'],
                        action.get('sub_round'),
                        json.dumps(action_data),
                        json.dumps(metadata)
                    )
                except Exception as e:
                    self.logger.error(f"Error inserting action: {e}")
                    raise

    async def insert_environment_state(
        self,
        environment_name: str,
        round_num: int,
        state_data: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Insert environment state data."""
        query = """
            INSERT INTO environment_states (environment_name, round, state_data, metadata)
            VALUES ($1, $2, $3, $4)
        """
        try:
            await self.db.execute(
                query,
                environment_name,
                round_num,
                json.dumps(state_data),
                json.dumps(metadata or {})
            )
        except Exception as e:
            self.logger.error(f"Error inserting environment state: {e}")
            raise

    async def insert_round_data(
        self,
        round_num: int,
        agents: List[Any],
        environment: Any,
        config: Any,
        environment_name: str = 'default'
    ):
        """Insert environment state and actions for a specific round."""
        try:
            # Process environment state if available
            if hasattr(environment, 'get_global_state'):
                env_state = environment.get_global_state()
                config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
                metadata = {
                    'config': config_dict,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'num_agents': len(agents)
                }
                await self.insert_environment_state(environment_name, round_num, env_state, metadata)

            # Process actions
            actions_data = []
            for agent in agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    actions_data.append({
                        'agent_id': agent.id,
                        'environment_name': environment_name,
                        'round': round_num,
                        'action': agent.last_action
                    })
                
            if actions_data:
                # Use existing agent IDs since they're already in the database
                agent_id_map = {str(agent.id): agent.id for agent in agents}
                await self.insert_actions(actions_data, agent_id_map)

        except Exception as e:
            self.logger.error(f"Error inserting round data: {e}")
            raise

    def _get_agent_uuid(self, agent_id: str) -> uuid.UUID:
        """Convert string agent ID to UUID consistently."""
        try:
            if isinstance(agent_id, uuid.UUID):
                return agent_id
                
            namespace_uuid = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
            return uuid.uuid5(namespace_uuid, str(agent_id))
        except Exception as e:
            self.logger.error(f"Error converting agent ID {agent_id} to UUID: {e}")
            raise