import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMPromptContext, LLMConfig, StructuredTool
from typing import Literal, List
import time
import os

async def main():
    load_dotenv()
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=40000)
    vllm_request_limits = RequestLimits(max_requests_per_minute=100, max_tokens_per_minute=100000)
    parallel_ai = ParallelAIUtilities(oai_request_limits=oai_request_limits, anthropic_request_limits=anthropic_request_limits, vllm_request_limits=vllm_request_limits)

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

    # Create prompts for different JSON modes and tool usage
    def create_prompts(client, model, response_formats : List[Literal["json_beg", "text","json_object","structured_output","tool"]]= ["text"], count=1):
        prompts = []
        for response_format in response_formats:
            for i in range(count):
                prompts.append(
                    LLMPromptContext(
                        id=f"{client}_{model}_{response_format}_{i}",
                        system_string="You are a helpful assistant that tells programmer jokes.",
                        new_message=f"Tell me a programmer joke about the number {i}.",
                        llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=400),
                        structured_output=structured_tool,
                        
                    )
                )
        return prompts

    # OpenAI prompts
    openai_prompts = create_prompts("openai", "gpt-4o-mini",["text","json_beg","json_object","structured_output","tool"],1)
    vllm_model = os.getenv("VLLM_MODEL")
    vllm_prompts = []
    if vllm_model is not None:
        vllm_prompts = create_prompts("vllm", vllm_model, ["text","json_beg","json_object","structured_output","tool"],1)

    # Anthropic prompts
    anthropic_prompts = create_prompts("anthropic", "claude-3-5-sonnet-20240620", ["text","json_beg","tool"],1)
    # Run parallel completions
    print("Running parallel completions...")
    all_prompts = anthropic_prompts + openai_prompts + vllm_prompts
    # all_prompts=anthropic_prompts
    start_time = time.time()
    completion_results = await parallel_ai.run_parallel_ai_completion(all_prompts)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    num_text = 0
    num_json = 0
    total_calls = 0
    for prompt, result in zip(all_prompts, completion_results):
        print(f"\nClient: {prompt.llm_config.client}, Response Format: {prompt.llm_config.response_format}")
        print(f"Prompt: {prompt.new_message}")
        if result.contains_object:
            print(f"Response (JSON): {result.json_object}")
            num_json += 1
        else:
            print(f"Response (Text): {result.str_content}")
            print("result:", result)
            num_text += 1
        print(f"Usage: {result.usage}")
        print("-" * 50)
        total_calls += 1

    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Request limits oai: {oai_request_limits.max_requests_per_minute} requests/min, {oai_request_limits.max_tokens_per_minute} tokens/min")
    print(f"Request limits anthropic: {anthropic_request_limits.max_requests_per_minute} requests/min, {anthropic_request_limits.max_tokens_per_minute} tokens/min")
    print(f"Number of OpenAI requests: {len(openai_prompts)}")
    print(f"Number of Anthropic requests: {len(anthropic_prompts)}")
    print(f"Number of text responses: {num_text}")
    print(f"Number of JSON responses: {num_json}")
    print(f"Total number of responses: {total_calls}")

if __name__ == "__main__":
    asyncio.run(main())