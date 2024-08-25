import json
import ast
import logging

agent_logger = logging.getLogger(__name__)

def setup_logger(log_file=None, log_level=logging.INFO):
    # Configure the root logger
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # If a log file is specified, add a file handler to the root logger
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)

def extract_json_from_response(response_string):
    try:
        # Extract JSON data from the response string
        start_index = response_string.find('{')
        end_index = response_string.rfind('}') + 1
        
        if start_index == -1 or end_index == 0:
            return None
        
        json_data = response_string[start_index:end_index]
        
        # Attempt to parse the extracted JSON data
        try:
            return json.loads(json_data)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(json_data)
            except (ValueError, SyntaxError):
                return None
    except Exception as e:
        agent_logger.error(f"Error extracting JSON: {e}")
        return None

def extract_and_save_results(role, file_path, completion):
    try:
        json_object = None

        try:
            json_object = json.loads(completion)
            agent_logger.info("Parsed with json.loads")
        except json.JSONDecodeError:
            try:
                json_object = ast.literal_eval(completion)
                agent_logger.info("Parsed with ast.literal_eval")
            except (ValueError, SyntaxError):
                json_object = extract_json_from_response(completion)
                agent_logger.info("Extracted JSON manually")

        if not json_object:
            raise ValueError("Completion contains an invalid JSON object")

        with open(file_path, 'w') as json_file:
            json.dump(json_object, json_file, indent=2)

        agent_logger.debug(f"Successfully saved results for {role}")
        return json_object

    except Exception as e:
        agent_logger.debug(f"Error extracting and saving results for {role}: {str(e)}")