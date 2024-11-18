from abstractions.inference.sql_models import ProcessedOutput, Usage, GeneratedJsonObject, RawOutput, ChatThread, LLMClient, ResponseFormat, LLMConfig, Tool, ChatMessage, SQLModel
from sqlmodel import Session, create_engine, select
from sqlalchemy.engine import Engine

if __name__ == "__main__":
    def create_processed_output(engine:Engine) -> ProcessedOutput:
        with Session(engine) as session:
            first_chat = session.exec(select(ChatThread)).first()
            if first_chat is None:
                raise ValueError("No chat thread found, can not create processed output")
            elif first_chat.id is None:
                raise ValueError("Chat thread id is not set, can not create processed output")
            dummy_usage = Usage(prompt_tokens=69, completion_tokens=420, total_tokens=69420)
            dummy_json_object = GeneratedJsonObject(name="dummy_json_object", object={"dummy": "object"})
            dummy_raw_output = RawOutput(raw_result="dummy_raw_output", client=LLMClient.openai, start_time=10, end_time=20, chat_thread_id=first_chat.id)
            dummy_processed_output = ProcessedOutput(usage=dummy_usage, json_object=dummy_json_object, raw_output=dummy_raw_output, time_taken=10, llm_client=LLMClient.openai, chat_thread=first_chat, 
                                                     chat_thread_id=first_chat.id    )
            session.add(dummy_processed_output)
            session.commit()
        return dummy_processed_output



    def create_chat(engine: Engine):
        with Session(engine) as session:
            oai_config = LLMConfig(client=LLMClient.openai, model="gpt-4o", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
            edit_tool = Tool(schema_name="edit_tool",
                            schema_description="Edit the provided JSON schema.",
                            instruction_string="Please follow this JSON schema for your response:",
                            strict_schema=True,
                            json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}})
            chat = ChatThread (new_message="Hello, how are you?", structured_output=edit_tool, llm_config=oai_config)      
            chat_italian = ChatThread (new_message="Ciao, come stai?", structured_output=edit_tool, llm_config=oai_config)
            anthropic_config = LLMConfig(client=LLMClient.anthropic, model="claude-3-5-sonnet-20240620", max_tokens=4000, temperature=0, response_format=ResponseFormat.tool)
            chat_french_anthropic = ChatThread (new_message="Bonjour, comment ça va?", structured_output=Tool(schema_name="outil_edition",
                                schema_description="Éditer le schéma JSON fourni.",
                                instruction_string="Veuillez suivre ce schéma JSON pour votre réponse:",
                                strict_schema=True,
                                json_schema={"type": "object", "properties": {"original_text": {"type": "string"}, "edited_text": {"type": "string"}}}),
                                llm_config=anthropic_config)
            session.add(chat)
            session.add(chat_italian)
            session.add(chat_french_anthropic)
            session.commit()

    def create_all_snapshots(engine: Engine):
        """ we get all the chats and create snapshots for each one """
        with Session(engine) as session:
            statement = select(ChatThread)
            result = session.exec(statement).unique()
            for chat in result:
                snapshot = chat.create_snapshot()
                session.add(snapshot)
            session.commit()

    def add_history_to_all_chats(engine: Engine):
        test_history = [{"role":"user","content":"Hello, how are you?"},{"role":"assistant","content":"I'm fine, thank you!"}]
        with Session(engine) as session:
            statement = select(ChatThread)
            result = session.exec(statement).unique()
            for chat in result:
                chat.history = [ChatMessage.from_dict(message) for message in test_history]
                chat.update_db_from_session(session)

    def select_openai_config(engine: Engine):
        with Session(engine) as session:
            statement = select(ChatThread , LLMConfig).join(LLMConfig).where(LLMConfig.client == LLMClient.openai)
            result = session.exec(statement)
            for chat, config in result:
                print(chat.new_message)

    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    engine = create_engine(sqlite_url, echo=True)

    SQLModel.metadata.create_all(engine)
    create_chat(engine)
    # select_openai_config(engine)
    create_all_snapshots(engine)
    add_history_to_all_chats(engine)
    create_processed_output(engine)
