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
            {"role": "user", "content": "Generate a json object for taxonomic information of a llama"}
        ],
        "provider": {
            "require_parameters": True
        },
        "response_format": { 
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "kingdom": {"type": "string"},
                        "phylum": {"type": "string"},
                        "class": {"type": "string"},
                        "order": {"type": "string"},
                        "family": {"type": "string"},
                        "genus": {"type": "string"},
                        "species": {"type": "string"}
                    },
                    "required": ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
                }
            }
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