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
from market_agents.agents.db.setup_database import create_database, create_tables

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
        
        create_tables(db_params)

    def __del__(self):
        if hasattr(self, 'cursor') and self.cursor:
            self.cursor.close()
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def insert_agents(self, agents_data):
        query = """
            INSERT INTO agents (id, role, is_llm, max_iter, llm_config)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                role = EXCLUDED.role,
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
                        agent['is_llm'],
                        agent['max_iter'],
                        json.dumps(agent['llm_config'])
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

    def insert_allocations(self, allocations: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO allocations (agent_id, goods, cash, locked_goods, locked_cash, initial_goods, initial_cash)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for allocation in allocations:
                    agent_id = agent_id_map.get(str(allocation['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {allocation['agent_id']}")
                        continue
                    cur.execute(query, (
                        agent_id,
                        allocation['goods'],
                        allocation['cash'],
                        allocation['locked_goods'],
                        allocation['locked_cash'],
                        allocation['initial_goods'],
                        allocation['initial_cash']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(allocations)} allocations into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting allocations: {str(e)}")
            raise
    
    def insert_schedules(self, schedules_data: List[Dict[str, Any]]):
        """
        Inserts schedule data into the preference_schedules table without handling conflicts.

        Args:
            schedules_data (List[Dict[str, Any]]): List of schedule dictionaries containing:
                - agent_id (UUID)
                - is_buyer (bool)
                - values (BuyerPreferenceSchedule or None)
                - costs (SellerPreferenceSchedule or None)
                - initial_endowment (Basket)
        """
        query = """
        INSERT INTO preference_schedules (agent_id, is_buyer, values, costs, initial_endowment)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for schedule in schedules_data:
                    logging.debug(f"Inserting schedule: {schedule}")

                    # Handle 'values' and 'costs' based on is_buyer
                    if schedule['is_buyer']:
                        values_json = json.dumps(schedule['values'], default=json_serial) if schedule['values'] else None
                        costs_json = None
                    else:
                        values_json = None
                        costs_json = json.dumps(schedule['costs'], default=json_serial) if schedule['costs'] else None

                    # Handle 'initial_endowment'
                    initial_endowment = schedule['initial_endowment']
                    if hasattr(initial_endowment, 'dict'):
                        # Convert Basket instance to dict
                        initial_endowment_serializable = initial_endowment.dict()
                    else:
                        # Assume it's already a dict or another serializable type
                        initial_endowment_serializable = initial_endowment

                    try:
                        initial_endowment_json = json.dumps(initial_endowment_serializable)
                    except TypeError as e:
                        logging.error(f"Error serializing 'initial_endowment' for agent_id {schedule['agent_id']}: {e}")
                        raise

                    # Execute the SQL Insert without ON CONFLICT
                    try:
                        cur.execute(query, (
                            schedule['agent_id'],
                            schedule['is_buyer'],
                            values_json,
                            costs_json,
                            initial_endowment_json
                        ))
                    except psycopg2.Error as db_err:
                        logging.error(f"Database error inserting schedule for agent_id {schedule['agent_id']}: {db_err}")
                        raise
            self.conn.commit()
            logging.info(f"Successfully inserted {len(schedules_data)} schedules")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting schedules: {str(e)}")
            logging.exception("Exception details:")
            raise

    def insert_orders(self, orders: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO orders (agent_id, is_buy, quantity, price, base_value, base_cost)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for order in orders:
                    agent_id = agent_id_map.get(str(order['agent_id']))
                    if agent_id is None:
                        logging.error(f"No matching UUID found for agent_id: {order['agent_id']}")
                        continue
                    cur.execute(query, (
                        agent_id,
                        order['is_buy'],
                        order['quantity'],
                        order['price'],
                        order['base_value'],
                        order['base_cost']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(orders)} orders into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting orders: {str(e)}")
            raise

    def insert_trades(self, trades_data: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        # Add logging to debug the incoming data
        logging.info(f"Attempting to insert trades: {trades_data}")
        logging.info(f"Agent ID map: {agent_id_map}")
        
        query = """
        INSERT INTO trades (buyer_id, seller_id, quantity, price, buyer_surplus, seller_surplus, total_surplus, round)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        try:
            with self.conn.cursor() as cur:
                for trade in trades_data:
                    buyer_id = agent_id_map.get(str(trade['buyer_id']))
                    seller_id = agent_id_map.get(str(trade['seller_id']))
                    
                    if buyer_id is None or seller_id is None:
                        logging.error(f"Missing agent mapping - buyer_id: {trade['buyer_id']}, seller_id: {trade['seller_id']}")
                        continue
                    values = (
                        buyer_id,
                        seller_id,
                        trade['quantity'],
                        trade['price'],
                        trade['buyer_surplus'],
                        trade['seller_surplus'],
                        trade['total_surplus'],
                        trade['round']
                    )
                    logging.info(f"Inserting trade with values: {values}")
                    
                    cur.execute(query, values)
                    trade_id = cur.fetchone()[0]
                    logging.info(f"Successfully inserted trade with ID: {trade_id}")
                    
            self.conn.commit()
            logging.info(f"Successfully committed {len(trades_data)} trades to database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting trades: {str(e)}")
            logging.exception("Full exception details:")
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
        INSERT INTO observations (memory_id, environment_name, observation)
        VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for observation in observations:
                    memory_id = agent_id_map.get(str(observation['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {observation['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        observation['environment_name'],
                        psycopg2.extras.Json(observation['observation'])
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(observations)} observations into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting observations: {str(e)}")
            raise

    def insert_reflections(self, reflections: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO reflections (memory_id, environment_name, reflection, self_reward, environment_reward, total_reward, strategy_update)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for reflection in reflections:
                    memory_id = agent_id_map.get(str(reflection['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {reflection['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        reflection['environment_name'],
                        reflection['reflection'],
                        reflection['self_reward'],
                        reflection['environment_reward'],
                        reflection['total_reward'],
                        reflection['strategy_update']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(reflections)} reflections into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting reflections: {str(e)}")
            raise

    def insert_perceptions(self, perceptions: List[Dict[str, Any]], agent_id_map: Dict[str, uuid.UUID]):
        query = """
        INSERT INTO perceptions (memory_id, environment_name, monologue, strategy, confidence)
        VALUES (%s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for perception in perceptions:
                    memory_id = agent_id_map.get(str(perception['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {perception['memory_id']}")
                        continue
                    cur.execute(query, (
                        memory_id,
                        perception['environment_name'],
                        perception['monologue'],
                        perception['strategy'],
                        perception['confidence']
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
            environment_name, 
            action
        )
        VALUES (%s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for action in actions:
                    memory_id = agent_id_map.get(str(action['memory_id']))
                    if memory_id is None:
                        logging.error(f"No matching UUID found for memory_id: {action['memory_id']}")
                        continue
                    environment_name = action['environment_name']
                    action_data = action['action']
                    
                    cur.execute(query, (
                        memory_id,
                        environment_name,
                        json.dumps(action_data, default=str)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(actions)} actions into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting actions: {str(e)}")
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
        """
        Insert simulation data for a specific round.
        
        Args:
            round_num (int): The current round number
            agents (List[Any]): List of agent objects
            environment (Any): The environment object
            config (Any): Configuration object
            tracker (Any): The tracker object for this environment
            environment_name (str): Name of the environment (defaults to 'auction')
        """
        try:
            # Insert agent data
            agents_data = [
                {
                    'id': str(agent.id),
                    'role': agent.role,
                    'is_llm': agent.use_llm,
                    'max_iter': config.max_rounds,
                    'llm_config': agent.llm_config if isinstance(agent.llm_config, dict) else agent.llm_config.dict()
                }
                for agent in agents
            ]
            agent_id_map = self.insert_agents(agents_data)

            # Allocations data
            logging.info("Preparing allocations data")
            allocations_data = [
                {
                    'agent_id': str(agent.id),
                    'goods': agent.economic_agent.endowment.current_basket.goods_dict.get(config.agent_config.good_name, 0),
                    'cash': agent.economic_agent.endowment.current_basket.cash,
                    'locked_goods': getattr(agent.economic_agent, 'locked_goods', {}).get(config.agent_config.good_name, 0),
                    'locked_cash': getattr(agent.economic_agent, 'locked_cash', 0),
                    'initial_goods': agent.economic_agent.endowment.initial_basket.goods_dict.get(config.agent_config.good_name, 0),
                    'initial_cash': agent.economic_agent.endowment.initial_basket.cash
                }
                for agent in agents
            ]
            self.insert_allocations(allocations_data, agent_id_map)

            # Schedules data
            logging.info("Preparing schedules data")
            schedules_data = [
                {
                    'agent_id': str(agent.id),
                    'is_buyer': agent.role == "buyer",
                    'values': agent.economic_agent.value_schedules.get(config.agent_config.good_name, {}),
                    'costs': agent.economic_agent.cost_schedules.get(config.agent_config.good_name, {}),
                    'initial_endowment': agent.economic_agent.endowment.initial_basket
                }
                for agent in agents
            ]
            self.insert_schedules(schedules_data)

            # Orders data
            logging.info("Preparing orders data")
            orders_data = [
                {
                    'agent_id': str(agent.id),
                    'is_buy': isinstance(order, Bid),
                    'quantity': order.quantity,
                    'price': order.price,
                    'base_value': getattr(order, 'base_value', None),
                    'base_cost': getattr(order, 'base_cost', None)
                }
                for agent in agents
                for order in agent.economic_agent.pending_orders.get(config.agent_config.good_name, [])
            ]
            self.insert_orders(orders_data, agent_id_map)

            # Interactions data
            logging.info("Preparing interactions data")
            interactions_data = [
                {
                    'agent_id': str(agent.id),
                    'round': round_num,
                    'task': interaction['type'],
                    'response': serialize_memory_data(interaction['content'])
                }
                for agent in agents
                for interaction in agent.interactions
            ]
            self.insert_interactions(interactions_data, agent_id_map)

            # Perceptions data
            logging.info("Preparing perceptions data")
            perceptions_data = []
            for agent in agents:
                if agent.last_perception is not None:
                    perception = validate_json(agent.last_perception)
                    if perception is not None:
                        perceptions_data.append({
                            'memory_id': str(agent.id),
                            'environment_name': environment_name,
                            'monologue': str(perception.get('monologue', '')),
                            'strategy': str(perception.get('strategy', '')),
                            'confidence': perception.get('confidence', 0)
                        })
                    else:
                        logging.warning(f"Invalid JSON perception data for agent {agent.id}: {agent.last_perception}")

            if perceptions_data:
                self.insert_perceptions(perceptions_data, agent_id_map)

            # Actions data
            logging.info("Preparing actions data")
            actions_data = [
                {
                    'memory_id': str(agent.id),
                    'environment_name': environment_name,
                    'action': agent.last_action
                }
                for agent in agents
                if hasattr(agent, 'last_action') and agent.last_action
            ]
            self.insert_actions(actions_data, agent_id_map)

            # Observations and Reflections data
            logging.info("Preparing observations and reflections data")
            observations_data = [
                {
                    'memory_id': str(agent.id),
                    'environment_name': environment_name,
                    'observation': serialize_memory_data(agent.last_observation)
                }
                for agent in agents
                if agent.last_observation
            ]

            self.insert_observations(observations_data, agent_id_map)

            # Trades data
            logging.info("Preparing trades data")
            trades_data = []
            if hasattr(tracker, 'all_trades'):
                logging.info(f"Found {len(tracker.all_trades)} trades in tracker")
                for trade_info in tracker.all_trades:
                    # The trade_info is now a dictionary from the AuctionTracker
                    try:
                        trades_data.append({
                            'buyer_id': trade_info['buyer_id'],
                            'seller_id': trade_info['seller_id'],
                            'quantity': trade_info['quantity'],
                            'price': trade_info['price'],
                            'buyer_surplus': trade_info['buyer_surplus'],
                            'seller_surplus': trade_info['seller_surplus'],
                            'total_surplus': trade_info['total_surplus'],
                            'round': trade_info['round']
                        })
                        logging.info(f"Processed trade: {trades_data[-1]}")
                    except Exception as e:
                        logging.error(f"Error processing trade {trade_info}: {str(e)}")

            if trades_data:
                logging.info(f"Attempting to insert {len(trades_data)} trades")
                self.insert_trades(trades_data, agent_id_map)
            else:
                logging.warning("No trades data found to insert")

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
        tables = ['agents', 'agent_memories', 'allocations', 'groupchat', 'trades']
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

# Register the UUID adapter
register_adapter(uuid.UUID, addapt_uuid)

if __name__ == "__main__":
    # This section can be used for testing or standalone execution
    pass