import psycopg2
import psycopg2.extras
import os
import uuid
from typing import List, Dict, Any
from psycopg2.extensions import register_adapter, AsIs
import json
import logging
import uuid
from datetime import datetime
from market_agents.economics.econ_models import Bid, BuyerPreferenceSchedule, SellerPreferenceSchedule
from .setup_orchestrator_db import create_database, setup_orchestrator_tables

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (BuyerPreferenceSchedule, SellerPreferenceSchedule)):
        return obj.dict()
    elif hasattr(obj, 'model_dump'):
        return obj.model_dump()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    raise TypeError(f"Type {type(obj)} not serializable")

def serialize_memory_data(memory_data):
    if isinstance(memory_data, dict):
        return {k: serialize_memory_data(v) for k, v in memory_data.items()}
    elif isinstance(memory_data, list):
        return [serialize_memory_data(v) for v in memory_data]
    elif isinstance(memory_data, datetime):
        return memory_data.isoformat()
    elif hasattr(memory_data, 'model_dump'):
        return serialize_memory_data(memory_data.model_dump())
    elif hasattr(memory_data, '__dict__'):
        return serialize_memory_data(vars(memory_data))
    elif isinstance(memory_data, (str, int, float, bool, type(None))):
        return memory_data
    else:
        return str(memory_data)
    
def validate_json(data):
    """Validate if data is JSON or can be parsed as JSON."""
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None
    return None

class SimulationDataInserter:
    def __init__(self, db_params):
        create_database(db_params)
        
        # Connect to the database
        self.conn = psycopg2.connect(**db_params)
        self.cursor = self.conn.cursor()
        
        setup_orchestrator_tables(db_params)

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def insert_agents(self, agents_data):
        query = """
            INSERT INTO agents (id, role, persona, is_llm, max_iter, llm_config)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                role = EXCLUDED.role,
                persona = EXCLUDED.persona,
                is_llm = EXCLUDED.is_llm,
                max_iter = EXCLUDED.max_iter,
                llm_config = EXCLUDED.llm_config
            RETURNING id
        """
        agent_id_map = {}
        for agent in agents_data:
            try:
                agent_id = uuid.UUID(str(agent['id'])) if isinstance(agent['id'], (str, int)) else agent['id']
                with self.conn.cursor() as cur:
                    cur.execute(query, (
                        agent_id,
                        agent['role'],
                        json.dumps(agent.get('persona', {})),
                        agent['is_llm'],
                        agent['max_iter'],
                        json.dumps(agent['llm_config']),
                    ))
                    inserted_id = cur.fetchone()
                    if inserted_id:
                        agent_id_map[str(agent['id'])] = inserted_id[0]
                    else:
                        logging.warning(f"No id returned for agent: {agent['id']}")
                self.conn.commit()
            except Exception as e:
                logging.error(f"Error inserting agent: {str(e)}")
                self.conn.rollback()
        return agent_id_map

    def insert_agent_memories(self, memories: List[Dict[str, Any]]):
        for memory in memories:
            try:
                agent_id = uuid.UUID(str(memory['agent_id'])) if isinstance(memory['agent_id'], (str, int)) else memory['agent_id']
                with self.conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO agent_memories (agent_id, step_id, memory_data)
                    VALUES (%s, %s, %s)
                    """, (agent_id, memory['step_id'], psycopg2.extras.Json(memory['memory_data'])))
                self.conn.commit()
            except Exception as e:
                self.conn.rollback()
                logging.error(f"Error inserting agent memory: {e}")
        logging.info(f"Inserted {len(memories)} agent memories into the database")

    def insert_groupchat_messages(self, messages: List[Dict[str, Any]], round_num: int, agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO groupchat (message_id, agent_id, round, sub_round, cohort_id, content, timestamp, topic)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for message in messages:
                    # Get the mapped agent ID from the database
                    agent_id = agent_id_map.get(str(message['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {message['agent_id']}")
                        continue

                    cur.execute(query, (
                        message['message_id'],
                        agent_id,  # Use the mapped agent_id
                        round_num,
                        message['sub_round'],
                        message['cohort_id'],
                        message['content'],
                        message['timestamp'],
                        message.get('topic')
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(messages)} group chat messages")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting group chat messages: {str(e)}")
            raise

    def insert_interactions(self, interactions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO interactions (agent_id, round, task, response)
        VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for interaction in interactions:
                    agent_id = agent_id_map.get(str(interaction['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {interaction['agent_id']}")
                        continue
                    cur.execute(query, (
                        agent_id,
                        interaction['round'],
                        interaction['task'],
                        interaction['response']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(interactions)} interactions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting interactions: {str(e)}")
            raise

    def insert_observations(self, observations: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO observations (agent_id, environment_name, round, observation)
        VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for observation in observations:
                    agent_id = agent_id_map.get(str(observation['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {observation['agent_id']}")
                        continue
                    cur.execute(query, (
                        agent_id,
                        observation['environment_name'],
                        observation['round'],
                        psycopg2.extras.Json(observation['observation'])
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(observations)} observations into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting observations: {str(e)}")
            raise

    def insert_perceptions(self, perceptions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO perceptions (memory_id, agent_id, environment_name, round, observation)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for perception in perceptions:
                    agent_id = agent_id_map.get(str(perception['agent_id']))
                    memory_id = perception.get('memory_id', uuid.uuid4())  # Generate UUID if not provided
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {perception['agent_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        agent_id,
                        perception['environment_name'],
                        perception['round'],
                        psycopg2.extras.Json(perception['observation'])
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(perceptions)} perceptions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting perceptions: {str(e)}")
            raise

    def insert_actions(self, actions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO actions (
            memory_id,
            agent_id, 
            environment_name,
            round,
            action
        )
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for action in actions:
                    agent_id = agent_id_map.get(str(action['agent_id']))
                    memory_id = action.get('memory_id', uuid.uuid4())  # Generate UUID if not provided
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {action['agent_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        agent_id,
                        action['environment_name'],
                        action['round'],
                        json.dumps(action['action'], default=str)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(actions)} actions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting actions: {str(e)}")
            raise

    def insert_reflections(self, reflections: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO reflections (
            memory_id,
            agent_id,
            environment_name,
            round,
            reflection,
            self_reward,
            environment_reward,
            total_reward,
            strategy_update
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for reflection in reflections:
                    agent_id = agent_id_map.get(str(reflection['agent_id']))
                    memory_id = reflection.get('memory_id', uuid.uuid4())  # Generate UUID if not provided
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {reflection['agent_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        agent_id,
                        reflection['environment_name'],
                        reflection['round'],
                        reflection['reflection'],
                        reflection.get('self_reward'),
                        reflection.get('environment_reward'),
                        reflection.get('total_reward'),
                        reflection.get('strategy_update')
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(reflections)} reflections into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting reflections: {str(e)}")
            raise

    def insert_ai_requests(self, ai_requests):
        requests_data = []
        for request in ai_requests:
            start_time = request.start_time
            end_time = request.end_time
            if isinstance(start_time, (float, int)):
                start_time = datetime.fromtimestamp(start_time)
            if isinstance(end_time, (float, int)):
                end_time = datetime.fromtimestamp(end_time)

            total_time = (end_time - start_time).total_seconds()

            # Extract system message
            system_message = next((msg['content'] for msg in request.completion_kwargs.get('messages', []) if msg['role'] == 'system'), None)

            requests_data.append({
                'prompt_context_id': str(request.source_id),
                'start_time': start_time,
                'end_time': end_time,
                'total_time': total_time,
                'model': request.completion_kwargs.get('model', ''),
                'max_tokens': request.completion_kwargs.get('max_tokens', None),
                'temperature': request.completion_kwargs.get('temperature', None),
                'messages': request.completion_kwargs.get('messages', []),
                'system': system_message,
                'tools': request.completion_kwargs.get('tools', []),
                'tool_choice': request.completion_kwargs.get('tool_choice', {}),
                'raw_response': request.raw_result,
                'completion_tokens': request.usage.completion_tokens if request.usage else None,
                'prompt_tokens': request.usage.prompt_tokens if request.usage else None,
                'total_tokens': request.usage.total_tokens if request.usage else None
            })

        if requests_data:
            try:
                self._insert_ai_requests_to_db(requests_data)
            except Exception as e:
                logging.error(f"Error inserting AI requests: {e}")

    def _insert_ai_requests_to_db(self, requests_data):
        query = """
        INSERT INTO requests 
        (prompt_context_id, start_time, end_time, total_time, model, 
        max_tokens, temperature, messages, system, tools, tool_choice,
        raw_response, completion_tokens, prompt_tokens, total_tokens)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for request in requests_data:
                    cur.execute(query, (
                        request['prompt_context_id'],
                        request['start_time'],
                        request['end_time'],
                        request['total_time'],
                        request['model'],
                        request['max_tokens'],
                        request['temperature'],
                        json.dumps(request['messages']),
                        request['system'],
                        json.dumps(request.get('tools', [])),
                        json.dumps(request.get('tool_choice', {})),
                        json.dumps(request['raw_response']),
                        request['completion_tokens'],
                        request['prompt_tokens'],
                        request['total_tokens']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(requests_data)} AI requests")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting AI requests: {str(e)}")
            raise

    def insert_round_data(
        self, 
        round_num: int, 
        agents: List[Any], 
        environment: Any,
        config: Any, 
        tracker: Any,
        environment_name: str = 'auction'
    ):
        """Insert simulation data for a specific round."""
        try:
            # Insert agent data
            agents_data = [
                {
                    'id': str(agent.id),
                    'role': agent.role,
                    'persona': agent.persona.dict() if hasattr(agent.persona, 'dict') else agent.persona,
                    'is_llm': agent.use_llm,
                    'max_iter': config.max_rounds,
                    'llm_config': agent.llm_config if isinstance(agent.llm_config, dict) else agent.llm_config.dict()
                }
                for agent in agents
            ]
            agent_id_map = self.insert_agents(agents_data)

            # Perceptions data
            logging.info("Preparing perceptions data")
            perceptions_data = []
            for agent in agents:
                if agent.last_perception is not None:
                    perception = validate_json(agent.last_perception)
                    if perception is not None:
                        perceptions_data.append({
                            'agent_id': str(agent.id),  # Changed from memory_id to agent_id
                            'environment_name': environment_name,
                            'round': round_num,
                            'observation': perception  # Changed to include full perception
                        })
                    else:
                        logging.warning(f"Invalid JSON perception data for agent {agent.id}: {agent.last_perception}")

            if perceptions_data:
                self.insert_perceptions(perceptions_data, agent_id_map)

            # Actions data
            logging.info("Preparing actions data")
            actions_data = [
                {
                    'agent_id': str(agent.id),  # Changed from memory_id to agent_id
                    'environment_name': environment_name,
                    'round': round_num,
                    'action': agent.last_action
                }
                for agent in agents
                if hasattr(agent, 'last_action') and agent.last_action
            ]
            if actions_data:
                self.insert_actions(actions_data, agent_id_map)

            # Observations data
            logging.info("Preparing observations data")
            observations_data = []
            for agent in agents:
                if hasattr(agent, 'last_observation') and agent.last_observation:
                    observations_data.append({
                        'agent_id': str(agent.id),
                        'environment_name': environment_name,
                        'round': round_num,
                        'observation': serialize_memory_data(agent.last_observation)
                    })
                else:
                    logging.debug(f"Agent {agent.id} has no last_observation")

            if observations_data:
                self.insert_observations(observations_data, agent_id_map)
            else:
                logging.warning("No observations data to insert")

            # Reflections data (if available)
            logging.info("Preparing reflections data")
            reflections_data = [
                {
                    'agent_id': str(agent.id),  # Changed from memory_id to agent_id
                    'environment_name': environment_name,
                    'round': round_num,
                    'reflection': str(agent.last_reflection) if hasattr(agent, 'last_reflection') else None,
                    'self_reward': getattr(agent, 'self_reward', 0.0),
                    'environment_reward': getattr(agent, 'environment_reward', 0.0),
                    'total_reward': getattr(agent, 'total_reward', 0.0),
                    'strategy_update': getattr(agent, 'strategy_update', '')
                }
                for agent in agents
                if hasattr(agent, 'last_reflection') and agent.last_reflection
            ]
            if reflections_data:
                self.insert_reflections(reflections_data, agent_id_map)

            # Group chat data
            groupchat_data = []
            if hasattr(environment, 'mechanism') and hasattr(environment.mechanism, 'topics'):
                for message in environment.mechanism.messages:
                    groupchat_data.append({
                        'message_id': str(uuid.uuid4()),
                        'agent_id': str(message.agent_id),
                        'round': round_num,
                        'sub_round': getattr(message, 'sub_round', None),
                        'cohort_id': message.cohort_id,
                        'content': message.content,
                        'timestamp': message.timestamp if hasattr(message, 'timestamp') else datetime.now(),
                        'topic': environment.mechanism.topics.get(message.cohort_id, '')
                    })
            
            if groupchat_data:
                self.insert_groupchat_messages(groupchat_data, round_num, agent_id_map)

        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting data for round {round_num}: {str(e)}")
            logging.exception("Exception details:")

    def check_tables_exist(self):
        cursor = self.conn.cursor()
        tables = ['agents', 'agent_memories', 'groupchat', 'perceptions', 'actions', 'reflections']
        for table in tables:
            cursor.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = '{table}')")
            exists = cursor.fetchone()[0]
            if not exists:
                cursor.close()
                return False
        cursor.close()
        return True

def addapt_uuid(uuid_value):
    return AsIs(f"'{uuid_value}'")

register_adapter(uuid.UUID, addapt_uuid)

if __name__ == "__main__":
    pass