import asyncio
from dotenv import load_dotenv
from market_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.message_models import LLMPromptContext, LLMConfig, StructuredTool
from typing import List, Literal
import time
import os

async def main():
    load_dotenv()
    vllm_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)
    parallel_ai = ParallelAIUtilities(vllm_request_limits=vllm_request_limits)

    # Define a simple JSON schema for structured output and tool use
    json_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
        },
        "required": ["summary", "sentiment"]
    }

    structured_tool = StructuredTool(
        json_schema=json_schema,
        schema_name="analyze_text",
        schema_description="Analyze the given text and provide a summary and sentiment"
    )

    # Create prompts for text, structured output, and tool use
    def create_prompts(model: str, topics: List[str], response_formats: List[Literal["text", "structured_output","json_object", "tool"]]):
        prompts = []
        for i, topic in enumerate(topics):
            for response_format in response_formats:
                system_string = "You are an AI that analyzes text and provides summaries with sentiment."
                new_message = f"Analyze the following topic: {topic}"
                
                if response_format == "text":
                    system_string = "You are a helpful assistant that provides information on various topics."
                    new_message = f"Tell me about {topic}."

                prompts.append(
                    LLMPromptContext(
                        id=f"{response_format}_{i}",
                        system_string=system_string,
                        new_message=new_message,
                        llm_config=LLMConfig(client="vllm", model=model, response_format=response_format),
                        structured_output=structured_tool if response_format in ["structured_output", "tool"] else None
                    )
                )
        return prompts

    # Set up prompts
    model = os.getenv("VLLM_MODEL")
    if model is None:
        raise ValueError("VLLM_MODEL is not set")
    topics = ["artificial intelligence", "climate change", "space exploration"]
    response_formats : List[Literal["text", "structured_output","json_object", "tool"]] = ["text", "structured_output", "tool"]
    all_prompts = create_prompts(model, topics, response_formats)

    # Run parallel completions
    print("Running parallel completions...")
    start_time = time.time()
    completion_results = await parallel_ai.run_parallel_ai_completion(all_prompts)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    for result in completion_results:
        print(f"\nPrompt ID: {result.source_id}")
        assert result.completion_kwargs is not None

        
        if result.contains_object:
            print(f"Response (Structured): {result.json_object}")
        else:
            print(f"Response (Text): {result.str_content}")
        print("-" * 50)

    print(f"\nTotal time taken: {total_time:.2f} seconds")
    print(f"Number of prompts processed: {len(all_prompts)}")

if __name__ == "__main__":
    asyncio.run(main())