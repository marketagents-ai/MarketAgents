import asyncio
from dotenv import load_dotenv
from market_agents.inference.sql_inference import ParallelAIUtilities, RequestLimits
from market_agents.inference.sql_models import ChatThread, LLMConfig, Tool, LLMClient,ResponseFormat
from typing import Literal, List
import time
import os
from sqlmodel import create_engine, SQLModel, Session, select
from sqlalchemy.engine import Engine

async def main():
    load_dotenv()
    oai_request_limits = RequestLimits(max_requests_per_minute=500, max_tokens_per_minute=200000)

    sqlite_file_name = "inference_example.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)
    SQLModel.metadata.create_all(engine)


    parallel_ai = ParallelAIUtilities(oai_request_limits=oai_request_limits, engine=engine)

    json_schema = {
        "type": "object",
        "properties": {
            "joke": {"type": "string"},
            "explanation": {"type": "string"}
        },
        "required": ["joke", "explanation"],
        "additionalProperties": False
    }

    structured_tool = Tool(
        json_schema=json_schema,
        schema_name="tell_joke",
        schema_description="Generate a programmer joke with explanation"
    )
    
    # Create chats for different JSON modes and tool usage
    def create_chats(engine:Engine,client:LLMClient, model, response_formats : List[ResponseFormat]= [ResponseFormat.text], count=1) -> List[ChatThread]:
        with Session(engine) as session:
            chats : List[ChatThread] = []
            for response_format in response_formats:
                llm_config=LLMConfig(client=client, model=model, response_format=response_format,max_tokens=250)
                for i in range(count):
                    chats.append(
                        ChatThread(
                        
                        system_string="You are a helpful assistant that tells programmer jokes.",
                        new_message=f"Tell me a programmer joke about the number {i}.",
                        llm_config=llm_config,
                        structured_output=structured_tool,
                        
                    )
                )
            session.add_all(chats)
            session.commit()
            for prompt in chats:
                session.refresh(prompt)
        return chats
    
    def get_chats_from_session(session:Session) -> List[ChatThread]:
        return list(session.exec(select(ChatThread)).unique().all())


    # OpenAI chats
    # openai_chats = create_chats("openai", "gpt-4o-mini",[ResponseFormat.text,ResponseFormat.json_beg,ResponseFormat.json_object,ResponseFormat.structured_output,ResponseFormat.tool],1)
    openai_chats = create_chats(engine,LLMClient.openai, "gpt-4o-mini",[ResponseFormat.tool],5)
    # with Session(engine) as session:
    #     chats = get_warm_chats_within_session(session)
    
    chats_id = [chat.id for chat in openai_chats]
        

    # print(chats[0].llm_config)
    print("Running parallel completions...")
    all_chats = openai_chats
    start_time = time.time()
    # with Session(engine) as session:
    completion_results = await parallel_ai.run_parallel_ai_completion(openai_chats)
    with Session(engine) as session:
        chats = get_chats_from_session(session)
        chats_filtered = [chat for chat in chats if chat.id in chats_id]
        for chat in chats_filtered:
            print("latest_message:",chat.new_message)
            print("history:",chat.history)
            chat.new_message = "And why is it funny?"
            session.add(chat)
        session.commit()
        for chat in chats_filtered:
            session.refresh(chat)
            print("latest_message:",chat.new_message)
    second_step_completion_results = await parallel_ai.run_parallel_ai_completion(chats_filtered)
    end_time = time.time()
    total_time = end_time - start_time

    # Print results
    num_text = 0
    num_json = 0
    total_calls = 0


if __name__ == "__main__":
    asyncio.run(main())