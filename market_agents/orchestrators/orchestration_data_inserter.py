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

    async def insert_agents(self, agents_data: List[Dict[str, Any]]) -> Dict[str, uuid.UUID]:
        """Insert agent data and return mapping of agent IDs."""
        query = """
            INSERT INTO agents (id, role, persona, is_llm, max_iter, llm_config)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO UPDATE SET
                role = EXCLUDED.role,
                persona = EXCLUDED.persona,
                is_llm = EXCLUDED.is_llm,
                max_iter = EXCLUDED.max_iter,
                llm_config = EXCLUDED.llm_config
            RETURNING id
        """
        agent_id_map = {}
        async with self.db.transaction() as txn:
            for agent in agents_data:
                try:
                    agent_id = uuid.UUID(str(agent['id'])) if isinstance(agent['id'], (str, int)) else agent['id']
                    result = await self.db.fetch_one(
                        query,
                        agent_id,
                        agent['role'],
                        json.dumps(agent.get('persona', {})),
                        agent.get('is_llm', True),
                        agent.get('max_iter', 0),
                        json.dumps(agent.get('llm_config', {}))
                    )
                    if result:
                        agent_id_map[str(agent['id'])] = result[0]
                except Exception as e:
                    self.logger.error(f"Error inserting agent: {e}")
                    raise
        return agent_id_map

    async def insert_actions(self, actions_data: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        """Insert actions (both group chat messages and research actions)."""
        query = """
            INSERT INTO actions (agent_id, environment_name, round, sub_round, action_data, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
        """
        async with self.db.transaction() as txn:
            for action in actions_data:
                try:
                    agent_id = action['agent_id']
                    if not isinstance(agent_id, uuid.UUID):
                        self.logger.error(f"Invalid agent_id type: {type(agent_id)}")
                        continue

                    action_data = {
                        'content': action.get('content') or action.get('action'),
                        'type': action.get('type', 'default'),
                        'cohort_id': action.get('cohort_id'),
                        'topic': action.get('topic')
                    }

                    metadata = {
                        'timestamp': action.get('timestamp', datetime.now(timezone.utc).isoformat()),
                        'message_id': str(action.get('message_id', uuid.uuid4())),
                        **{k: v for k, v in action.items() if k not in [
                            'agent_id', 'environment_name', 'round', 'sub_round', 
                            'content', 'action'
                        ]}
                    }

                    await self.db.execute(
                        query,
                        agent_id,
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
        """Insert all orchestration data for a specific round."""
        try:
            agents_data = []
            for agent in agents:
                if not isinstance(agent.id, uuid.UUID):
                    NAMESPACE_AGENTS = uuid.uuid5(uuid.NAMESPACE_DNS, 'market_agents.agents')
                    agent_uuid = uuid.uuid5(NAMESPACE_AGENTS, str(agent.id))
                else:
                    agent_uuid = agent.id

                llm_config = getattr(agent, 'llm_config', {})
                if hasattr(llm_config, 'model_dump'):
                    llm_config = llm_config.model_dump()

                agents_data.append({
                    'id': agent_uuid,
                    'role': getattr(agent, 'role', 'default'),
                    'persona': getattr(agent, 'persona', {}),
                    'is_llm': getattr(agent, 'use_llm', True),
                    'max_iter': getattr(agent, 'max_iter', 0),
                    'llm_config': llm_config 
                })

            agent_id_map = await self.insert_agents(agents_data)

            if hasattr(environment, 'get_global_state'):
                env_state = environment.get_global_state()
                config_dict = config.model_dump() if hasattr(config, 'model_dump') else config
                metadata = {
                    'config': config_dict,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'num_agents': len(agents)
                }
                await self.insert_environment_state(environment_name, round_num, env_state, metadata)

            actions_data = []
            for agent in agents:
                if hasattr(agent, 'last_action') and agent.last_action:
                    agent_uuid = agent_id_map.get(str(agent.id))
                    if not agent_uuid:
                        self.logger.warning(f"No UUID mapping found for agent {agent.id}, skipping action")
                        continue
                        
                    actions_data.append({
                        'agent_id': agent_uuid,
                        'environment_name': environment_name,
                        'round': round_num,
                        'action': agent.last_action
                    })
            
            if actions_data:
                await self.insert_actions(actions_data, agent_id_map)

        except Exception as e:
            self.logger.error(f"Error inserting round data: {e}")
            raise