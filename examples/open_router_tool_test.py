import requests
import json
import dotenv
import os

dotenv.load_dotenv()

url = "https://openrouter.ai/api/v1/chat/completions"
api_key = str(os.getenv("VLLM_API_KEY"))
model = "qwen/qwen-2.5-72b-instruct"

def send_chat_request():
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Analyze the following sentence: llamas are cute and helpful assistants. Provide a summary and sentiment."}
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "analyze_text",
                    "description": "Analyze the given text and provide a summary and sentiment",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string"},
                            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
                        },
                        "required": ["summary", "sentiment"]
                    }
                }
            }
        ],
        #"tool_choice": "required",
        #"tool_choice": {"type": "function", "function": {"name": "analyze_text"}},
        "provider": {
            "require_parameters": True
        }
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    send_chat_request()
