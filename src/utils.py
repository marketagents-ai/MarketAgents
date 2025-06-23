# utility functions
import ast
import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import requests
import yaml
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from unstructured.partition.html import partition_html

agent_logger = logging.getLogger("agent-orchestrator")

def setup_logger(log_file_path: str):
    logging.basicConfig(
        format="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )
    # Use RotatingFileHandler from the logging.handlers module
    file_handler = RotatingFileHandler(log_file_path, maxBytes=0, backupCount=0)
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s", datefmt="%Y-%m-%d:%H:%M:%S")
    file_handler.setFormatter(formatter)

    agent_logger = logging.getLogger("agent-orchestrator")
    agent_logger.addHandler(file_handler)

    return agent_logger

def get_source_dir(starting_dir=None):
    """Get the source directory of data-genie-agents, either by traversing upwards or checking subdirectories."""
    # If no starting directory is provided, use the current working directory
    if starting_dir is None:
        starting_dir = os.getcwd()
    current_path = os.path.abspath(starting_dir)

    # Check upwards in the directory hierarchy
    while current_path != os.path.dirname(current_path):
        if os.path.basename(current_path) == "data-genie-agents":
            return current_path
        current_path = os.path.dirname(current_path)

    # If "data-genie-agents" is not found upwards, check if it is a subdirectory
    sub_dir_path = os.path.join(starting_dir, "data-genie-agents")
    if os.path.isdir(sub_dir_path):
        return os.path.abspath(sub_dir_path)

    raise FileNotFoundError("The 'data-genie-agents' root folder could not be found.")

def create_log_directories():
    """Create the log directories for the agent orchestrator."""
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "orchestrator_logs"), exist_ok=True)
    os.makedirs(os.path.join(logs_dir, "qa_interactions"), exist_ok=True)

def is_valid_json(message):
    try:
        json.dumps(message)
        return True
    except (TypeError, ValueError):
        return False
    
def load_yaml(file_path):
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def generate_query(task, generation_type=None):
    if generation_type == "function_calling":
        return f'"{task.Task}" "{task.SubCategory}" "{task.Category}" AND (documentation OR functions OR examples OR code)'
    elif generation_type == "financial_rag":
        return f'"{task.Category}" "{task.SubCategory}" "{task.Task}" AND (news OR articles OR "press release" OR research)'
    else:
        return f'"{task.Category}" "{task.SubCategory}" "{task.Task}"'


def combine_search_results(search_results, char_limit):
    combined_text = ''
    character_count = 0

    for item in search_results:
        if item is not None and "url" in item and "content" in item:
            url = item.get("url", "")
            content = item.get("content", "")

            # Remove special characters from content
            cleaned_content = remove_special_characters(content)

            #if "tables" in item:
            #    cleaned_content += f"\n{convert_tables_to_markdown(item['tables'])}"

            # Check if appending the current document exceeds the token limit
            if character_count + len(cleaned_content) > char_limit:
                print(f"Character limit reached. Stopping further document append.")
                break

            # Update the character count with the characters from the current document
            character_count += len(cleaned_content)

            combined_text += f'<doc index="{url}">\n'
            combined_text += f'{cleaned_content}'

            combined_text += '\n</doc>\n'
            agent_logger.info(f"Document from {url} added to the combined text")
    return combined_text

def combine_retrieved_documents(docs, type=None):
    documents = ""
    for doc in docs:
        if type == "examples":
            documents += f"<example index={os.path.basename(doc.metadata['source'])}>\n"
            documents += f"{doc.page_content}"
            documents += f"\n</example>\n"
        elif type == "documents":
            documents += f"<doc index={os.path.basename(doc.metadata['source'])}>\n"
            documents += f"{doc.page_content}"
            documents += f"\n</doc>\n"
    return documents

def convert_tables_to_markdown(tables):
    markdown = ""
    for table in tables:
        markdown += "|"
        for header in table[0]:
            markdown += f" {header} |"
        markdown += "\n|"
        for _ in table[0]:
            markdown += " --- |"
        markdown += "\n"
        for row in table[1:]:
            markdown += "|"
            for cell in row:
                markdown += f" {cell} |"
            markdown += "\n"
    return markdown

def read_documents_from_folder(folder_path, num_results):
    search_results = []

    if os.path.exists(folder_path) and os.listdir(folder_path):
        # Read from existing JSON files
        for i, filename in enumerate(os.listdir(folder_path)):
            if i < num_results and filename.endswith(".json"):
                file_path = os.path.join(folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as file:
                    result_data = json.load(file)
                    search_results.append(result_data)
        return search_results
    else:
        return "No files in the directory"

def save_search_results(folder_path, search_results):
    os.makedirs(folder_path, exist_ok=True)

    for i, item in enumerate(search_results):
        if item is not None and "url" in item and "content" in item:
            file_path = os.path.join(folder_path, f"result_{i}.json")
            result_data = {"url": item["url"], "content": item["content"], "tables": item["tables"]}

            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(result_data, file, ensure_ascii=False, indent=2)
            
            agent_logger.info(f"Websearch results from {item['url']} saved")

def remove_special_characters(input_string):
    # Use a regular expression to remove non-alphanumeric characters (excluding spaces)
    cleaned_string = re.sub(r'[^a-zA-Z0-9\s]', '', input_string)
    return cleaned_string

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

def create_folder_path(task, folder_type):
    data_genie_path = get_source_dir()
    max_length = 65
    trunc_task = task.Task
    if len(trunc_task) > max_length:
        trunc_task = trunc_task[:max_length - 10] 
    trunc_task = clean_file_path(trunc_task)
    today_date = datetime.date.today()

    results_path = f"{folder_type}/{today_date}"
    results_path = os.path.join(data_genie_path, results_path, task.Category, task.SubCategory, trunc_task)
    results_path = results_path.replace(' ', '_')
    os.makedirs(results_path, exist_ok=True)
    
    agent_logger.info(f"Results path: {results_path}")

    return results_path


def create_folder_path_taskid(task, folder_type):
    exists = False
    data_genie_path = get_source_dir()
    trunc_task = task.Task
    today_date = datetime.date.today()

    results_path = f"{folder_type}/{today_date}"
    task_id = f"task_id_{task.Meta_Data['id']}"
    results_path = os.path.join(data_genie_path, results_path, task.Category, task.SubCategory, task_id)
    results_path = results_path.replace(' ', '_')

    if os.path.exists(results_path):
        exists = True
        return exists, results_path
    
    os.makedirs(results_path, exist_ok=True)
    agent_logger.info(f"Results path: {results_path}")

    return exists, results_path

def create_task_file_path(task, agent, folder_type, file_type="txt"):
    results_path = create_folder_path(task, folder_type)
    
    if file_type == "json":
        file_path = os.path.join(results_path, f"{agent}.json")
    elif file_type == "txt":
        file_path = os.path.join(results_path, f"{agent}.txt")
    file_path = file_path.replace(' ', '_')
    return file_path

def convert_enum_to_list(prop_data):
    if "enum" in prop_data:
        enum_value = prop_data["enum"]
        if isinstance(enum_value, dict):
            prop_data["enum"] = list(enum_value.keys())
        elif not isinstance(enum_value, list):
            prop_data["enum"] = [enum_value]  # Convert to a list
            
def fix_tools_format(tool):

    if "type" not in tool or tool["type"] != "function":
        parameters = tool.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        for prop_name, prop_data in properties.items():
            convert_enum_to_list(prop_data)

        return {
            "type": "function",
            "function": {
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    else:
        parameters = tool.get("function", {}).get("parameters", {})
        properties = parameters.get("properties", {})
        for prop_name, prop_data in properties.items():
            convert_enum_to_list(prop_data)

        return tool

def get_assistant_message(completion):
    message  = json.loads(completion)['choices'][0]['message']
    if message['tool_calls']:
        tool_calls = []
        for tool_call in message['tool_calls']:
            tool_calls.append(tool_call['function'])
        return tool_calls
    else:
        return message['content']
    
def extract_toolcall_code_blocks(content):
    # Define the pattern to find all tool_call blocks
    pattern = r"```tool_call\s*({.*?})\s*```"

    # Find all matches
    matches = re.findall(pattern, content, re.DOTALL)

    # Process the matches
    result = []
    for match in matches:
        try:
            # Load as JSON
            json_data = ast.literal_eval(match)
            result.append(json_data)
        except Exception as e:
            print(f"Error processing block {match}: {e}")
    return result

def extract_tool_code_block(content):
     # Define the pattern to find all tool_call blocks
    pattern = r"```tools\s*({.*?})\s*```"

    # Find all matches
    match = re.search(pattern, content, re.DOTALL)

    # Process the matches
    result = None
    if match:
        try:
            # Load as JSON
            json_data = ast.literal_eval(match.group(1))
            result = json_data
        except Exception as e:
            print(f"Error processing block {match.group(0)}: {e}")
    return result

def strip_incomplete_text(text):
    # Find the last occurrence of a full stop
    last_full_stop_index = text.rfind('.')
    
    if last_full_stop_index != -1:  # If a full stop is found
        # Return the text up to the last full stop
        return text[:last_full_stop_index+1]  # Include the full stop
    else:
        # If no full stop is found, return the original text
        return text

def clean_file_path(file_path):
    # Remove special characters
    cleaned_file_path = re.sub(r'[^\w\s-]', '', file_path)
    
    # Replace spaces with underscores
    cleaned_file_path = cleaned_file_path.replace(' ', '_')
    
    # Shorten the file path if it exceeds 255 characters
    if len(cleaned_file_path) > 255:
        base_path, file_name = os.path.split(cleaned_file_path)
        file_name = file_name[:255 - len(base_path) - 1]  # Subtract 1 for the separator
        cleaned_file_path = os.path.join(base_path, file_name)
    
    return cleaned_file_path

def embedding_search(url, query):
    text = download_form_html(url)
    elements = partition_html(text=text)
    content = "\n".join([str(el) for el in elements])
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len,
        is_separator_regex=False,
    )
    docs = text_splitter.create_documents([content])

    # Load a pre-trained sentence transformer model
    #embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model = OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL"))

    # Create FAISS index and retriever
    index = FAISS.from_documents(docs, embedding_model)
    retriever = index.as_retriever()

    answers = retriever.invoke(query, top_k=4)
    chunks = []
    for i, doc in enumerate(answers):
        chunk = f"\n<chunk index={i}>\n{doc.page_content}\n</chunk>\n"
        chunks.append(chunk)

    result = "".join(chunks)
    return f"<documents>\n{result}</documents>"

def download_form_html(url):
    headers = {
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept-Language': 'en-US,en;q=0.9,pt-BR;q=0.8,pt;q=0.7',
      'Cache-Control': 'max-age=0',
      'Dnt': '1',
      'Sec-Ch-Ua': '"Not_A Brand";v="8", "Chromium";v="120"',
      'Sec-Ch-Ua-Mobile': '?0',
      'Sec-Ch-Ua-Platform': '"macOS"',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    response = requests.get(url, headers=headers)
    return response.text