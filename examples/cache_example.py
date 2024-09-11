import asyncio
from llm_agents.inference.parallel_inference import ParallelAIUtilities, RequestLimits
from llm_agents.inference.message_models import LLMPromptContext, LLMConfig, StructuredTool
import time
from typing import Literal, List
from pydantic import BaseModel, Field
import os

class PromptCachingResponse(BaseModel):
    definition: str = Field(..., description="Definition of prompt caching")
    benefits: list[str] = Field(..., description="List of benefits of using prompt caching")
    best_practices: list[str] = Field(..., description="List of best practices for implementing prompt caching")

class CacheExamples:
    def __init__(self):
        anthropic_request_limits = RequestLimits(max_requests_per_minute=40, max_tokens_per_minute=40000, provider="anthropic")
        self.parallel_ai = ParallelAIUtilities(anthropic_request_limits=anthropic_request_limits)
        self.big_file_content = self.read_big_file()
        self.structured_tool = StructuredTool(
            json_schema=PromptCachingResponse.model_json_schema(),
            schema_name="prompt_caching_info",
            schema_description="Generate information about prompt caching"
        )

    def read_big_file(self):
        # examples\anthropic_caching.md

        try:
            with open(r"examples\anthropic_caching.md", "r", encoding="utf-8") as file:
                return file.read()
        except UnicodeDecodeError:
            with open(r"examples\anthropic_caching.md", "r", encoding="iso-8859-1") as file:
                return file.read()

    async def run_and_print_result(self, prompt, description):
        start_time = time.time()
        result = await self.parallel_ai.run_parallel_ai_completion([prompt])
        end_time = time.time()

        print(f"\n{description}")
        print(f"Request time: {end_time - start_time:.2f} seconds")
        print(f"Usage: {result[0].usage}")
        print("Response:")
        if result[0].json_object:
            print("json")
        elif result[0].str_content:
           print("str")
        else:
            print("No content found in response")
        return result[0]

    async def test_system_prompt_caching(self, response_format: Literal["json_beg", "text", "json_object", "structured_output", "tool"] = "text"):
        prompt = LLMPromptContext(
            system_string=f"<file_contents>{self.big_file_content}</file_contents>",
            new_message="What is prompt caching, how does it work, and what are its benefits and best practices?" + 
                        (" Respond in JSON format." if response_format in ["json_beg", "json_object"] else ""),
            llm_config=LLMConfig(
                client="anthropic",
                model="claude-3-5-sonnet-20240620",
                response_format=response_format,
                use_cache=True,
                max_tokens=1000
            ),
            structured_output=self.structured_tool if response_format in ["structured_output", "tool"] else None
        )

        await self.run_and_print_result(prompt, f"Non-cached request ({response_format})")
        await self.run_and_print_result(prompt, f"Cached request ({response_format})")

    async def test_growing_history(self, response_format: Literal["json_beg", "text", "json_object", "structured_output", "tool"] = "text"):
        base_prompt = LLMPromptContext(
            system_string=f"You are a helpful AI assistant. Here's some context about prompt caching: <file_contents>{self.big_file_content}</file_contents>",
            new_message="Let's have a conversation about prompt caching. What is it and how does it work?" + 
                        (" Respond in JSON format." if response_format in ["json_beg", "json_object"] else ""),
            llm_config=LLMConfig(
                client="anthropic",
                model="claude-3-5-sonnet-20240620",
                response_format=response_format,
                use_cache=True,
                max_tokens=1000
            ),
            structured_output=self.structured_tool if response_format in ["structured_output", "tool"] else None
        )

        # First turn
        result1 = await self.run_and_print_result(base_prompt, f"Turn 1 (System content cached) ({response_format})")

        # Second turn
        prompt2 = base_prompt.add_chat_turn_history(result1)
        prompt2.new_message = "How does prompt caching improve performance?" 
        result2 = await self.run_and_print_result(prompt2, f"Turn 2 (System content + partial history cached) ({response_format})")

        # Third turn
        prompt3 = prompt2.add_chat_turn_history(result2)
        prompt3.new_message = "What are some best practices for using prompt caching?" 
        await self.run_and_print_result(prompt3, f"Turn 3 (System content + more history cached) ({response_format})")

async def main():
    examples = CacheExamples()
    
    response_formats :List[Literal["text", "json_beg", "json_object", "structured_output", "tool"]] = ["text","json_beg", "tool"]
    
    for format in response_formats:
        print(f"\n\nTesting system prompt caching ({format}):")
        await examples.test_system_prompt_caching(format)
        
        print(f"\n\nTesting growing history caching ({format}):")
        await examples.test_growing_history(format)

if __name__ == "__main__":
    asyncio.run(main())