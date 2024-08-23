import os
import time
from dotenv import load_dotenv

#import together
from openai import OpenAI, AzureOpenAI
from anthropic import Anthropic

class AIUtilities:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file
        
        # openai credentials
        self.openai_key = os.getenv("OPENAI_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL")

        # azure credentials
        self.azure_openai_key = os.getenv("AZURE_OPENAI_KEY")
        self.api_version = os.getenv("API_VERSION")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_model = os.getenv("AZURE_OPENAI_MODEL")

        # anthropic credentials
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_model = os.getenv("ANTHROPIC_MODEL")

    def run_ai_completion(self, prompt, llm_config):
        client = None
        
        if llm_config["client"] == "openai":
            client = OpenAI(api_key=self.openai_key)
            return self.run_openai_completion(client, prompt, llm_config)
        
        elif llm_config["client"] == "azure_openai":
            client = AzureOpenAI(
                api_key=self.azure_openai_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_openai_endpoint
            )
            return self.run_azure_openai_completion(client, prompt, llm_config)
        
        elif llm_config["client"] == "anthropic":
            anthropic = Anthropic(api_key=self.anthropic_api_key)
            return self.run_anthropic_completion(anthropic, prompt, llm_config)
        
        else:
            return "Invalid AI vendor"
    
    def get_ai_context_length(self, ai_vendor):
        if ai_vendor == "openai":
            return os.getenv("OPENAI_CONTEXT_LENGTH")
        if ai_vendor == "azure_openai":
            return os.getenv("AZURE_OPENAI_CONTEXT_LENGTH")
        elif ai_vendor == "anthropic":
            return os.getenv("ANTHROPIC_CONTEXT_LENGTH")
        else:
            return "Invalid AI vendor"
        
    def run_ai_tool_completion(self, prompt, tools, llm_config, tool_choice="auto"):
        
        if llm_config["client"] == "openai":
            client = OpenAI(
                api_key=self.openai_key,
            )
            model = llm_config.get("model", self.openai_model)
        
        elif llm_config["client"] == "azure_openai":
            client = AzureOpenAI(
                api_key=self.azure_openai_key,
                api_version=self.api_version,
                azure_endpoint=self.azure_openai_endpoint
            )
            model = llm_config.get("model", self.azure_openai_model)
        
        try:    
            response = client.chat.completions.create(
                model=model,
                messages=prompt,
                tools=tools,
                tool_choice=tool_choice if tools is not None else None,
                max_tokens=llm_config.get("max_tokens", 4096),
                temperature=llm_config.get("temperature", 0)
                #response_format=llm_config.get("response_format"),
            )
            completion = response.choices[0].message
            print(completion)
            return completion
        except Exception as e:
            return str(e)
        
    def run_openai_completion(self, client, prompt, llm_config):
        try:
            response = client.chat.completions.create(
                model=llm_config.get("model", self.openai_model),
                messages=prompt,
                #messages=[
                #    {"role": "system", "content": f"You are a helpful assistant designed to output JSON."},
                #    {"role": "user", "content": prompt}
                #],
                response_format=llm_config.get("response_format"),
                max_tokens=llm_config.get("max_tokens", 1024),
                temperature=llm_config.get("temperature", 0),
            )
            return response.choices[0].message.content
        except Exception as e:
            return str(e)

    def run_azure_openai_completion(self, client, prompt, llm_config):
        try:
            response = client.chat.completions.create(
                model=llm_config.get("model", self.azure_openai_model),
                messages=prompt,
                #messages=[
                #    {"role": "system", "content": f"You are a helpful assistant designed to output JSON."},
                #    {"role": "user", "content": prompt}
                #],
                response_format=llm_config.get("response_format"),
                max_tokens=llm_config.get("max_tokens", 4096),
                temperature=llm_config.get("temperature", 0),
                frequency_penalty=1,
                presence_penalty=0.5
            )
            completion = response.choices[0].message.content
            return completion
        except Exception as e:
            return str(e)

    def run_anthropic_completion(self, anthropic, prompt, llm_config):
        try:
            messages=[
                    {"role": "user", "content": prompt[1]["content"]}
                ]
            if llm_config.get("response_format") == "json":
                    messages.append(
                        {"role": "assistant", "content": "Here's the valid JSON object response:```json"}
                    )
            
            response = anthropic.messages.create(
                model=llm_config.get("model", self.anthropic_model),
                max_tokens=llm_config.get("max_tokens", 1024),
                temperature=llm_config.get("temperature", 0),
                system=prompt[0]["content"],
                messages=messages
            )
            
            return response.content[0].text
        except Exception as e:
            return str(e)

def main():
    load_dotenv()  # Load environment variables from .env file

    ai_utilities = AIUtilities()

    # Example usage
    prompt = "Tell me a programmer joke"
    ai_vendor = "openai"  # Change this to the desired AI vendor

    # Run AI completion
    result = ai_utilities.run_ai_completion(prompt, ai_vendor)
    print(f"AI Completion Result:\n{result}")

if __name__ == "__main__":
    main()