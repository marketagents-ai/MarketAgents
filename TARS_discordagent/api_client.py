import logging
import os
from functools import partial
import asyncio
import yaml
import json
from openai import OpenAI
from openai import AzureOpenAI

# Load system prompts
with open('system_prompts.yaml', 'r') as file:
    system_prompts = yaml.safe_load(file)

# API configuration
DEFAULT_AZURE_MODEL = 'gpt-4o-mini'
DEFAULT_OLLAMA_MODEL = 'vanilj/hermes-3-llama-3.1-8b:latest'
DEFAULT_OPENROUTER_MODEL = 'nousresearch/hermes-3-llama-3.1-405b'

class APIClient:
    def __init__(self, args):
        self.api = args.api
        self.model = args.model
        
        if self.api == 'azure':
            self.azure_config()
        elif self.api == 'openrouter':
            self.openrouter_config()
        else:  # ollama
            self.ollama_config()

    def azure_config(self):
        self.AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
        self.AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
        self.AZURE_OPENAI_DEPLOYMENT = self.model if self.model else DEFAULT_AZURE_MODEL
        self.client = AzureOpenAI(
            api_key=self.AZURE_OPENAI_API_KEY,
            api_version=self.AZURE_OPENAI_API_VERSION,
            azure_endpoint=self.AZURE_OPENAI_ENDPOINT
        )
        logging.info(f"Using Azure OpenAI with model: {self.AZURE_OPENAI_DEPLOYMENT}")

    def ollama_config(self):
        self.OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434/v1')
        self.OLLAMA_MODEL = self.model if self.model else DEFAULT_OLLAMA_MODEL
        self.client = OpenAI(
            base_url=self.OLLAMA_BASE_URL,
            api_key='ollama'
        )
        logging.info(f"Using Ollama with model: {self.OLLAMA_MODEL}")

    def openrouter_config(self):
        self.OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        self.OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
        self.OPENROUTER_MODEL = self.model if self.model else DEFAULT_OPENROUTER_MODEL
        self.YOUR_SITE_URL = os.getenv('YOUR_SITE_URL', '')
        self.YOUR_APP_NAME = os.getenv('YOUR_APP_NAME', '')
        self.client = OpenAI(
            base_url=self.OPENROUTER_BASE_URL,
            api_key=self.OPENROUTER_API_KEY
        )
        logging.info(f"Using OpenRouter with model: {self.OPENROUTER_MODEL}")

    async def call_api(self, prompt, context="", system_prompt_key="default"):
        full_prompt = f"Context:\n{context}\n\nPrompt:\n{prompt}" if context else prompt
        logging.info(f"Prompt sent to API:\n{full_prompt}")
        
        system_content = system_prompts.get(system_prompt_key, system_prompts['default'])
        
        try:
            loop = asyncio.get_running_loop()
            
            if self.api == 'azure':
                response = await loop.run_in_executor(
                    None,
                    partial(
                        self.client.chat.completions.create,
                        model=self.AZURE_OPENAI_DEPLOYMENT,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                )
            elif self.api == 'openrouter':
                response = await loop.run_in_executor(
                    None,
                    partial(
                        self.client.chat.completions.create,
                        extra_headers={
                            "HTTP-Referer": self.YOUR_SITE_URL,
                            "X-Title": self.YOUR_APP_NAME,
                        },
                        model=self.OPENROUTER_MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                )
            else:  # ollama
                response = await loop.run_in_executor(
                    None,
                    partial(
                        self.client.chat.completions.create,
                        model=self.OLLAMA_MODEL,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": full_prompt}
                        ]
                    )
                )
            
            answer = response.choices[0].message.content
            logging.info(f"Response from API:\n{answer}")
            
            # Save the interaction
            self.save_interaction(system_content, full_prompt, answer)
            
            return answer
        except Exception as e:
            logging.error(f"Error calling API: {str(e)}")
            if "DeploymentNotFound" in str(e):
                return f"Error: The specified Azure OpenAI deployment '{self.AZURE_OPENAI_DEPLOYMENT}' was not found. Please check your configuration."
            return f"An error occurred while calling the API: {str(e)}"

    def save_interaction(self, system_content, user_content, assistant_content):
        interaction = {
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ]
        }
        with open("agent_interactions.jsonl", "a") as f:
            f.write(json.dumps(interaction) + "\n")

# Initialize the API client
api_client = None

def initialize_api_client(args):
    global api_client
    api_client = APIClient(args)

async def call_api(prompt, context="", system_prompt_key="default"):
    global api_client
    if api_client is None:
        raise ValueError("API client not initialized. Call initialize_api_client first.")
    return await api_client.call_api(prompt, context, system_prompt_key)