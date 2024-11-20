import asyncio
from dotenv import load_dotenv
from abstractions.inference.sql_inference import ParallelAIUtilities, RequestLimits
from abstractions.inference.sql_models import ChatThread, LLMConfig, Tool, LLMClient, ResponseFormat, SystemStr
from typing import List, Optional
from sqlmodel import create_engine, SQLModel, Session
from pydantic import BaseModel
import statistics

# Example BaseModel for inputs/outputs
class NumbersInput(BaseModel):
    numbers: List[float]
    round_to: int = 2

class Stats(BaseModel):
    mean: float
    std: float

class FilterInput(BaseModel):
    numbers: List[float]
    threshold: float

class FilterOutput(BaseModel):
    filtered_numbers: List[float]
    count_removed: int

class SortInput(BaseModel):
    numbers: List[float]
    ascending: bool = True

class SortOutput(BaseModel):
    sorted_numbers: List[float]

class GoalInput(BaseModel):
    description: str
    final_result: List[float]
    initial_numbers: List[float]

class GoalOutput(BaseModel):
    goal_achieved: bool
    explanation: str

# Tool functions
def calculate_stats(input_data: NumbersInput) -> Stats:
    """Calculate mean and standard deviation of numbers."""
    return Stats(
        mean=round(statistics.mean(input_data.numbers), input_data.round_to),
        std=round(statistics.stdev(input_data.numbers), input_data.round_to)
    )

def filter_numbers(input_data: FilterInput) -> FilterOutput:
    """Filter numbers above a threshold."""
    filtered = [n for n in input_data.numbers if n <= input_data.threshold]
    return FilterOutput(
        filtered_numbers=filtered,
        count_removed=len(input_data.numbers) - len(filtered)
    )

def sort_numbers(input_data: SortInput) -> SortOutput:
    """Sort numbers in ascending or descending order."""
    return SortOutput(
        sorted_numbers=sorted(input_data.numbers, reverse=not input_data.ascending)
    )

def check_goal_achieved(input_data: GoalInput) -> GoalOutput:
    """Verify if the goal was achieved and explain the results."""
    return GoalOutput(
        goal_achieved=True,
        explanation=f"Successfully processed the numbers. Started with {input_data.initial_numbers} and ended with {input_data.final_result}"
    )

async def run_sequential_steps(parallel_ai: ParallelAIUtilities, initial_chat: ChatThread, engine) -> None:
    """Run sequential steps, properly managing database sessions."""
    max_steps = 10
    step = 0
    chat_id = initial_chat.id
    
    while step < max_steps:
        print(f"\nExecuting step {step + 1}...")
        
        with Session(engine) as session:
            chat = session.get(ChatThread, chat_id)
            assert chat is not None, "Chat not found in database"
        
        completion_results = await parallel_ai.run_parallel_ai_completion([chat])
        
        with Session(engine) as session:
            chat = session.get(ChatThread, chat_id)
            assert chat is not None, "Chat not found in database"
            
            if not chat.history:
                break
                
            latest_message = chat.history[-1]
            print(f"\nStep {step + 1} Result:")
            print(f"Role: {latest_message.role}")
            print(f"Content: {latest_message.content}")
            print(f"Tool Name: {latest_message.tool_name}")
            print(f"Tool Call: {latest_message.tool_call}")
            
            if latest_message.tool_name == "check_goal_achieved_response":
                print("\nGoal achieved, stopping sequence.")
                break
                
            # If no tool was called or the assistant is done, break the loop
            # if not latest_message.tool_call and latest_message.role != "tool":
            #     print("\nNo more tool calls, sequence complete.")
            #     break
            
            step += 1  # Increment step for all cases except breaks

async def main():
    load_dotenv()
    
    # Initialize database
    sqlite_file_name = "chat.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"
    engine = create_engine(sqlite_url, echo=False)
    SQLModel.metadata.create_all(engine)

    # Create tools
    tools = [
        Tool.from_callable(calculate_stats),
        Tool.from_callable(filter_numbers),
        Tool.from_callable(sort_numbers),
        Tool.from_callable(check_goal_achieved)
    ]

    # Example data
    example_numbers = [15, 7, 32, 9, 21, 6, 18, 25, 13, 28]
    
    system_prompt = SystemStr(
        content="""You are a helpful assistant that processes numerical data using available tools. 
        You have access to these tools:
        - calculate_stats: Calculate mean and standard deviation
        - filter_numbers: Filter numbers above a threshold
        - sort_numbers: Sort numbers in ascending/descending order
        - check_goal_achieved: Verify and explain the achieved results
        
        Please break down tasks into appropriate steps and use tools sequentially to achieve the goals.
        After completing all necessary calculations, use check_goal_achieved to summarize the results.
        Explain your reasoning at each step with normal text. I expect at each response both text content and tool calls. The process will not stop until you use the check_goal_achieved tool.""",
        name="sequential_tools_system"
    )

    # Initialize AI utilities
    parallel_ai = ParallelAIUtilities(
        oai_request_limits=RequestLimits(
            max_requests_per_minute=500,
            max_tokens_per_minute=200000
        ),
        engine=engine
    )

    # Create and store initial chat
    with Session(engine) as session:
        chat = ChatThread(
            system_prompt=system_prompt,
            new_message=f"Using the numbers {example_numbers}, please filter out numbers above 20, then sort the remaining numbers in ascending order, and calculate their statistics.",
            llm_config=LLMConfig(
                client=LLMClient.openai,
                model="gpt-4o",
                response_format=ResponseFormat.auto_tools,
                max_tokens=500
            ),
            tools=tools
        )
        session.add(chat)
        session.commit()
        session.refresh(chat)

    print("Starting sequential tool inference...")
    await run_sequential_steps(parallel_ai, chat, engine)
    
    # Print final chat history
    with Session(engine) as session:
        chat = session.get(ChatThread, chat.id)
        if chat:
            print("\nFinal Chat History:")
            for message in chat.history:
                print(f"\n{message.role}: {message.content}")
                if message.tool_call:
                    print(f"Tool Call: {message.tool_name}")
                    print(f"Tool Input: {message.tool_call}")

if __name__ == "__main__":
    asyncio.run(main())