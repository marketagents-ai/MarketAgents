from aiutilities import AIUtilities, LLMConfig, HistoPrompt, StructuredTool
from dotenv import load_dotenv

def test_ai_utilities():
    load_dotenv()  # Load environment variables from .env file

    ai_utilities = AIUtilities()

    # JSON schema for structured responses
    json_schema = {
        "type": "object",
        "properties": {
            "joke": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["joke", "explanation"],
        "additionalProperties": False
    }

    structured_tool = StructuredTool(
        json_schema=json_schema,
        schema_name="tell_joke",
        schema_description="Generate a programmer joke with explanation"
    )

    print("Schema: ", json_schema)
    
    # OpenAI examples
    print("OpenAI Examples:")
    
    # 1. OpenAI with text response
    prompt = HistoPrompt(
        system_string="You are a helpful pirate that tells jokes in pirate language.",
        history=[{"role": "user", "content": "What game could we play?"},
                 {"role": "assistant", "content": "I would suggest playing jokes to learn languages!'."}],
        new_message="Fammi uno scherzo da programmatore in italiano",
        llm_config=LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="text")
    )
    result = ai_utilities.run_ai_completion(prompt)
    print("\n1. OpenAI Completion Result (Text):")
    print(f"{result}\n")

    # 2. OpenAI with JSON response (no schema)
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="json"))
    
    result = ai_utilities.run_ai_completion(prompt)
    print("\n2. OpenAI Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 3. OpenAI with JSON response (with schema)
    prompt.structured_output = structured_tool
    prompt.use_schema_instruction = True
    result = ai_utilities.run_ai_completion(prompt)
    print("\n3. OpenAI Completion Result (JSON, with schema):")
    print(f"{result}\n")

    # 4. OpenAI with tool completion 
    prompt =  prompt.update_llm_config(LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="tool"))
    prompt.use_schema_instruction = False
    result = ai_utilities.run_ai_tool_completion(prompt)

    print("\n4. OpenAI Completion Result (tool completion):")
    print(f"{result}\n")

    # # Anthropic examples
    # print("\nAnthropic Examples:")

    # 5. Anthropic with text response
    prompt =  prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="text"))
    result = ai_utilities.run_ai_completion(prompt)
    print("\n5. Anthropic Completion Result (Text):")
    print(f"{result}\n")

    # 6. Anthropic with JSON response (no schema)
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_object"))
    result = ai_utilities.run_ai_completion(prompt)
    print("\n6. Anthropic Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 7. Anthropic with JSON response (schema)
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_object"))
    prompt.use_schema_instruction = True
    result = ai_utilities.run_ai_completion(prompt)
    
    print("\n7. Anthropic Completion Result (JSON, with schema):")
    print(f"{result}\n")

    # 8. Anthropic with tool completion
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool"))
    result = ai_utilities.run_ai_tool_completion(prompt)
    prompt.use_schema_instruction = True
    print("\n7. Anthropic Completion Result (tool completion):")
    print(f"{result}\n")

if __name__ == "__main__":
    test_ai_utilities()