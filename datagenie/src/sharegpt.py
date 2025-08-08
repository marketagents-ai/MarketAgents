import argparse
import json
import os
import uuid
from datasets import Dataset
from tool_formatter import ChatCompletionToolParam

class ShareGPTFormatter:
    @staticmethod
    def validate_function_signatures(tools):
        validated_tools = []
        for signature_dict in tools:
            try:
                if "function" in signature_dict and "name" in signature_dict["function"]:
                    function_signature = ChatCompletionToolParam(**signature_dict)
                    validated_tools.append(function_signature)
                else:
                    print(f"Missing required fields in function signature: {signature_dict}")
            except Exception as e:
                print(f"Validation error for function signature: {e}")
        return validated_tools

    @staticmethod
    def process_conversation(conversations):
        processed = []
        current_tool_response = ""
        
        for i, message in enumerate(conversations):
            if message['from'] == 'tool':
                current_tool_response += message['value']
                
                # Check if next message is not a tool or if this is the last message
                if i == len(conversations) - 1 or conversations[i+1]['from'] != 'tool':
                    processed.append({
                        'from': 'tool',
                        'value': current_tool_response.strip()
                    })
                    current_tool_response = ""
            else:
                processed.append(message)
        
        return processed

    @staticmethod
    def convert_to_hermes_format(conversation):
        converted_conversation = []
        task = None
        failed = False
        sys_prompt = "You are a function calling AI model. You are provided with function signatures within <tools> </tools> XML tags. You may call one or more functions to assist with the user query. If available tools are not relevant in assisting with user query, just respond in natural conversational language. Don't make assumptions about what values to plug into functions. After calling & executing the functions, you will be provided with function results within <tool_response> </tool_response> XML tags."
        sys_prompt += f'\n<tools>\n{conversation["tools"]}\n</tools>\n'
        sys_prompt += "For each function call return a JSON object, with the following pydantic model json schema for each:\n{'title': 'FunctionCall', 'type': 'object', 'properties': {'arguments': {'title': 'Arguments', 'type': 'object'}, 'name': {'title': 'Name', 'type': 'string'}}, 'required': ['arguments', 'name']}\n"
        sys_prompt += "Each function call should be enclosed within <tool_call> </tool_call> XML tags. You must use <scratch_pad> </scratch_pad> XML tags to record your reasoning and planning before you call the functions as follows.\nExample:\n<scratch_pad>\n{Goal -> Action -> Observation -> Reflection cycle}\n</scratch_pad>\n<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>"
        system_message = {
            "from": "system",
            "value": sys_prompt
        }
        converted_conversation.append(system_message)
        
        for message in conversation['messages']:
            role = message['role']
            content = message.get('content', '')
            
            if role == 'user':
                user_message = {'from': 'human', 'value': content}
                converted_conversation.append(user_message)
                if task is None:
                    task = content
            
            elif role == 'assistant':
                if 'tool_calls' in message and message['tool_calls'] is not None:
                    if message['content']:
                        scratch_pad = message['content']
                        if '<scratch_pad>' in scratch_pad:
                            scratch_pad = scratch_pad[scratch_pad.index('<scratch_pad>') + len('<scratch_pad>'):scratch_pad.index('</scratch_pad>')]
                            gpt_value = f'<scratch_pad>{scratch_pad}</scratch_pad>\n'
                        else:
                            gpt_value = f'<scratch_pad>\n{scratch_pad}\n</scratch_pad>\n'
                    else:
                        gpt_value = ''
                        failed = True
                    for tool_call in message['tool_calls']:
                        function_json = json.dumps(tool_call['function'])
                        gpt_value += f'<tool_call>\n{function_json}\n</tool_call>\n'
                    tool_call_message = {'from': 'gpt', 'value': gpt_value}
                    converted_conversation.append(tool_call_message)           
                else:
                    summary_message = {'from': 'gpt', 'value': content}
                    converted_conversation.append(summary_message)     
            
            elif role == 'tool':
                function_name = message['name']
                tool_call_id = message['tool_call_id']
                function_content = message['content']
                
                combined_value = json.dumps({
                    'tool_call_id': tool_call_id,
                    'name': function_name,
                    'content': function_content
                })
                
                tool_results = f'<tool_response>\n{combined_value}\n</tool_response>\n'
                tool_message = {'from': 'tool', 'value': tool_results}
                converted_conversation.append(tool_message)
        
        converted_conversation = ShareGPTFormatter.process_conversation(converted_conversation)
        return converted_conversation, task, failed

    @staticmethod
    def prepare_sharegpt_dataset(folder_path):
        raw_data = []
        output_data = []
        failed_counter = 0
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith("json"):
                    unique_id = str(uuid.uuid4())
                    file_path = os.path.join(root, file)
                    print(f"File Path: {file_path}")
                    with open(file_path) as file:
                        json_data = json.load(file)
                    
                    raw_data.append(json_data)
                    ShareGPTFormatter.validate_function_signatures(json_data["tools"])
                    conversation = json_data
                    converted_conversation, task, failed = ShareGPTFormatter.convert_to_hermes_format(conversation)
                    if failed:
                        failed_counter += 1
                        print(f"Failed Converstaion\n{converted_conversation}")
                    if converted_conversation and not failed:
                        output_data.append({
                            "id": unique_id,
                            "conversations": converted_conversation,
                            "task": task,
                            "tools": json_data["tools"],
                            "category": "API Calls",
                            "source": "Salesforce"
                        })
        print(failed_counter)
        return output_data, raw_data

    @staticmethod
    def format_and_upload_to_hub(folder_path, upload=False, dataset_path="interstellarninja/salesforce_hermes_tools"):
        sharegpt_format_data, raw_data = ShareGPTFormatter.prepare_sharegpt_dataset(folder_path)
        
        raw_dataset = Dataset.from_list(raw_data)
        with open("./raw_data.json", 'w') as file:
            json.dump(raw_data, file)
        sharegpt_dataset = Dataset.from_list(sharegpt_format_data)
        with open("./sharegpt.json", 'w') as file:
            json.dump(sharegpt_format_data, file)
        if upload:
            sharegpt_dataset.push_to_hub(
                dataset_path,
                #commit_message="Upload ShareGPT-formatted dataset"
            )
            raw_dataset.push_to_hub(
                f"{dataset_path}_raw"
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format and upload ShareGPT dataset.")
    parser.add_argument("folder_path", help="Path to the folder containing the Salesforce results")
    parser.add_argument("--upload", action="store_true", help="Upload the dataset to Hugging Face Hub")
    parser.add_argument("--dataset_path", default="interstellarninja/salesforce_hermes_tools", 
                        help="Hugging Face dataset path (default: interstellarninja/salesforce_hermes_tools)")

    args = parser.parse_args()

    ShareGPTFormatter.format_and_upload_to_hub(
        folder_path=args.folder_path,
        upload=args.upload,
        dataset_path=args.dataset_path
    )

    print(f"Processing complete. Data saved to sharegpt.json")
    if args.upload:
        print(f"Dataset uploaded to Hugging Face Hub: {args.dataset_path}")