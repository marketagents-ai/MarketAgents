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
from market_agents.economics.econ_models import BuyerPreferenceSchedule, SellerPreferenceSchedule


def json_serial(self, obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, (BuyerPreferenceSchedule, SellerPreferenceSchedule)):
        return obj.dict()
    raise TypeError(f"Type {type(obj)} not serializable")

class SimulationDataInserter:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.environ.get('DB_NAME', 'market_simulation'),
            user=os.environ.get('DB_USER', 'db_user'),
            password=os.environ.get('DB_PASSWORD', 'db_pwd@123'),
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5433')
        )
        self.cursor = self.conn.cursor()

    def __del__(self):
        if self.cursor:
            self.cursor.close()
        if self.conn:
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
                # Generate a new UUID if the id is not a valid UUID
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
            except ValueError as e:
                agent_id = uuid.uuid4()
                logging.warning(f"Invalid UUID format for agent: {agent['id']}. Generated new UUID: {agent_id}")
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
    
    def json_serial(self, obj):
        """JSON serializer for objects not serializable by default json code"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, (BuyerPreferenceSchedule, SellerPreferenceSchedule)):
            return obj.dict()

        raise TypeError(f"Type {type(obj)} not serializable")

    def insert_agent_memories(self, memories: List[Dict[str, Any]]):
        for memory in memories:
            try:
                # Convert agent_id to UUID if it's not already one
                agent_id = uuid.UUID(str(memory['agent_id'])) if isinstance(memory['agent_id'], (str, int)) else memory['agent_id']
                with self.conn.cursor() as cur:
                    cur.execute("""
                    INSERT INTO agent_memories (agent_id, step_id, memory_data)
                    VALUES (%s, %s, %s)
                    """, (agent_id, memory['step_id'], psycopg2.extras.Json(memory['memory_data'])))
                self.conn.commit()
            except ValueError as e:
                logging.error(f"Invalid UUID format for agent memory: {memory['agent_id']}. Error: {str(e)}")
            except Exception as e:
                self.conn.rollback()
                logging.error(f"Error inserting agent memory: {e}")
        logging.info(f"Inserted {len(memories)} agent memories into the database")


    def insert_schedules(self, schedules_data: List[Dict[str, Any]]):
        """
        Inserts schedule data into the preference_schedules table without handling conflicts.

        Args:
            schedules_data (List[Dict[str, Any]]): List of schedule dictionaries containing:
                - agent_id (UUID)
                - is_buyer (bool)
                - values (BuyerPreferenceSchedule or SellerPreferenceSchedule)
                - initial_endowment (Basket)
        """
        query = """
        INSERT INTO preference_schedules (agent_id, is_buyer, values, initial_endowment)
        VALUES (%s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for schedule in schedules_data:
                    logging.debug(f"Inserting schedule: {schedule}")

                    # Serialize 'values' using json_serial
                    try:
                        values_json = json.dumps(schedule['values'], default=self.json_serial)
                    except TypeError as e:
                        logging.error(f"Error serializing 'values' for agent_id {schedule['agent_id']}: {e}")
                        raise

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
        query = """
        INSERT INTO trades (buyer_id, seller_id, quantity, price, buyer_surplus, seller_surplus, total_surplus, round)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                for trade in trades_data:
                    buyer_id = agent_id_map.get(str(trade['buyer_id']))
                    seller_id = agent_id_map.get(str(trade['seller_id']))
                    
                    if buyer_id is None or seller_id is None:
                        logging.error(f"No matching UUID found for buyer_id: {trade['buyer_id']} or seller_id: {trade['seller_id']}")
                        continue

                    cur.execute(query, (
                        buyer_id,
                        seller_id,
                        trade['quantity'],
                        trade['price'],
                        trade['buyer_surplus'],
                        trade['seller_surplus'],
                        trade['total_surplus'],
                        trade['round']
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(trades_data)} trades into the database")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting trades: {str(e)}")
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

    def insert_auctions(self, auctions: List[Dict[str, Any]]):
        for auction in auctions:
            self.cursor.execute("""
            INSERT INTO auctions (max_rounds, current_round, total_surplus_extracted, average_prices, order_book, trade_history)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (auction['max_rounds'], auction['current_round'], auction['total_surplus_extracted'],
                  psycopg2.extras.Json(auction['average_prices']), psycopg2.extras.Json(auction['order_book']),
                  psycopg2.extras.Json(auction['trade_history'])))
        self.conn.commit()

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
        INSERT INTO perceptions (memory_id, environment_name, monologue, strategy)
        VALUES (%s, %s, %s, %s)
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
                        perception['strategy']
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

    def insert_ai_requests(self, requests_data):
        query = """
        INSERT INTO requests 
        (prompt_context_id, start_time, end_time, total_time, model, max_tokens, temperature, messages, system, tools, tool_choice)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
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
                        json.dumps(request['messages'], default=str),
                        request['system'],
                        json.dumps(request['tools'], default=str),
                        json.dumps(request['tool_choice'], default=str)
                    ))
            self.conn.commit()
            logging.info(f"Inserted {len(requests_data)} AI requests")
        except Exception as e:
            self.conn.rollback()
            logging.error(f"Error inserting AI requests: {str(e)}")
            raise

def addapt_uuid(uuid_value):
    return AsIs(f"'{uuid_value}'")

# Register the UUID adapter
register_adapter(uuid.UUID, addapt_uuid)

def main():
    inserter = SimulationDataInserter()

    # Example data (replace with actual simulation data)
    agents = [
        {'role': 'buyer', 'is_llm': True, 'max_iter': 10, 'llm_config': {'model': 'gpt-3.5-turbo'}},
        {'role': 'seller', 'is_llm': True, 'max_iter': 10, 'llm_config': {'model': 'gpt-3.5-turbo'}}
    ]
    agent_ids = inserter.insert_agents(agents)

    memories = [
        {'agent_id': agent_ids[0], 'step_id': 1, 'memory_data': {'text': 'Memory 1'}},
        {'agent_id': agent_ids[1], 'step_id': 1, 'memory_data': {'text': 'Memory 2'}}
    ]
    inserter.insert_agent_memories(memories)

    # Insert other data similarly...

    print("Data inserted successfully.")

if __name__ == "__main__":
    main()