from aiutilities import AIUtilities, LLMConfig, Prompt, StructuredTool
from dotenv import load_dotenv

def test_ai_utilities():
    load_dotenv()  # Load environment variables from .env file

    ai_utilities = AIUtilities()

    # Example prompt
    prompt = Prompt(
        system="You are a helpful pirate that tells jokes in pirate language.",
        history=[{"role": "user", "content": "What game could we play?"},
                 {"role": "assistant", "content": "I would suggest playing jokes to learn languages!'."}],
        user_message="Tell me a programmer joke in italiano"
    )
    
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
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="text")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n1. OpenAI Completion Result (Text):")
    print(f"{result}\n")

    # 2. OpenAI with JSON response (no schema)
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="json_object")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n2. OpenAI Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 3. OpenAI with JSON response (with schema)
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="json", tool=structured_tool)
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n3. OpenAI Completion Result (JSON, with schema):")
    print(f"{result}\n")

    # 4. OpenAI with tool completion
    llm_config = LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="tool", tool=structured_tool)
    result = ai_utilities.run_ai_tool_completion(prompt, llm_config)
    print("\n4. OpenAI Completion Result (tool completion):")
    print(f"{result}\n")

    # Anthropic examples
    print("\nAnthropic Examples:")

    # 5. Anthropic with text response
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="text")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n5. Anthropic Completion Result (Text):")
    print(f"{result}\n")

    # 6. Anthropic with JSON response (no schema)
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_object")
    result = ai_utilities.run_ai_completion(prompt, llm_config)
    print("\n6. Anthropic Completion Result (JSON, no schema):")
    print(f"{result}\n")

    # 7. Anthropic with tool completion
    llm_config = LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool", tool=structured_tool)
    result = ai_utilities.run_ai_tool_completion(prompt, llm_config)
    print("\n7. Anthropic Completion Result (tool completion):")
    print(f"{result}\n")

if __name__ == "__main__":
    test_ai_utilities()