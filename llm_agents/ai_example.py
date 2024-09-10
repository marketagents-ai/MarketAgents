from aiutilities import AIUtilities, LLMConfig, HistoPrompt, StructuredTool, LLMOutput
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
    print("\n1. OpenAI Completion Result (Text):")
    prompt = HistoPrompt(
        system_string="You are a helpful pirate that tells jokes in pirate language.",
        history=[{"role": "user", "content": "What game could we play?"},
                 {"role": "assistant", "content": "I would suggest playing jokes to learn languages!'."}],
        new_message="Fammi uno scherzo da programmatore in italiano",
        llm_config=LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="text"),
        structured_output=structured_tool
    )
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    

    # 2. OpenAI with JSON response (no schema no json_mode)
    print("\n2. OpenAI Completion Result (json_beg, no schema in system message):")
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="json_beg"))
    
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    

    # 3. OpenAI with JSON response (with schema no json_mode)
    print("\n3. OpenAI Completion Result (json_beg, with schema in system message):")
    prompt.use_schema_instruction = True
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
   


    # 4. OpenAI with JSON response (with  no schema and json_mode)
    print("\n4. OpenAI Completion Result (json_object no schema and json_mode):")
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="json_object"))
    prompt.use_schema_instruction = False

    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    

    # 5. OpenAI with JSON response (with  schema in system message and json_mode)
    print("\n5. OpenAI Completion Result (json_object, with schema and json_mode):")
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="json_object"))
    prompt.use_schema_instruction = True

    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    


    # 6. OpenAI with Structured output response (with  schema in system message and json_mode)
    print("\n6 OpenAI Completion Result (Structured output - strict mode False):")
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="structured_output"))
    prompt.use_schema_instruction = False
    assert prompt.structured_output is not None, "Structured output is not set"
    prompt.structured_output.strict_schema = False
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")

    # 7. OpenAI with Structured output response (with  schema in system message and json_mode)

    print("\n7 OpenAI Completion Result (Structured output - strict mode True):")
    prompt = prompt.update_llm_config(LLMConfig(client="openai", model="gpt-4o-2024-08-06", response_format="structured_output"))
    prompt.use_schema_instruction = True
    assert prompt.structured_output is not None, "Structured output is not set"
    prompt.structured_output.strict_schema = False
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    
    # 8. OpenAI with tool completion
    print("\n8. OpenAI Completion Result (tool completion):")
    prompt =  prompt.update_llm_config(LLMConfig(client="openai", model="gpt-3.5-turbo", response_format="tool"))
    prompt.use_schema_instruction = False
    result = ai_utilities.run_ai_tool_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")

    

    # # Anthropic examples
    # print("\nAnthropic Examples:")
    print("\n9. Anthropic Completion Result (Text):")
    # 9. Anthropic with text response
    prompt =  prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="text"))
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")

    
    
    print("\n10. Anthropic Completion Result (json_beg, no schema in system message):")
    # 10. Anthropic with JSON response (no schema)
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_beg"))
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    
    print("\n11. Anthropic Completion Result (json_beg, with schema in system message):")
    # 11. Anthropic with JSON response (schema in system message)
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="json_beg"))
    prompt.use_schema_instruction = True
    
    result = ai_utilities.run_ai_completion(prompt)
    if isinstance(result,LLMOutput):
       print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")

    # 12. Anthropic with tool completion
    print("\n12. Anthropic Completion Result (tool completion):")
    prompt = prompt.update_llm_config(LLMConfig(client="anthropic", model="claude-3-5-sonnet-20240620", response_format="tool"))
    prompt.use_schema_instruction = True
    result = ai_utilities.run_ai_tool_completion(prompt)
    if isinstance(result,LLMOutput):
        print(f"detected json_object:\n {result.json_object}") if result.contains_object else print(f"detected string:\n {result.str_content}")
    
    

if __name__ == "__main__":
    test_ai_utilities()