from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, asc, col
from typing import List, Optional
from market_agents.inference.sql_models import (
    ChatThread, 
    LLMConfig, 
    Tool, 
    LLMClient, 
    ResponseFormat,
    ChatMessage,
    MessageRole
)
from market_agents.inference.sql_inference import ParallelAIUtilities
from app.api.deps import DatabaseDep, get_ai_utils
from pydantic import BaseModel
from uuid import UUID

router = APIRouter(prefix="/chats", tags=["chats"])

# --- Pydantic models for requests/responses ---
class MessageCreate(BaseModel):
    content: str

class ChatMessageResponse(BaseModel):
    role: MessageRole
    content: str
    author_name: Optional[str] = None

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    id: int
    uuid: UUID
    new_message: Optional[str]
    history: List[ChatMessageResponse]
    system_string: Optional[str]

    class Config:
        from_attributes = True

    @classmethod
    def from_chat_thread(cls, chat: ChatThread) -> "ChatResponse":
        if chat.id is None:
            raise ValueError("Chat ID is required")
        
        # Convert ChatMessage objects to ChatMessageResponse
        history = [
            ChatMessageResponse(
                role=msg.role,
                content=msg.content,
                author_name=msg.author_name
            ) for msg in chat.history
        ] if chat.history else []

        return cls(
            id=chat.id,
            uuid=chat.uuid,
            new_message=chat.new_message,
            history=history,
            system_string=chat.system_string
        )

# --- Default configurations ---
DEFAULT_LLM_CONFIG = LLMConfig(
    client=LLMClient.openai,
    model="gpt-4o-mini",
    response_format=ResponseFormat.tool,
    temperature=0,
    max_tokens=2500
)

CHAIN_OF_THOUGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "thought": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["step", "thought", "reasoning"]
            }
        },
        "final_answer": {"type": "string"}
    },
    "required": ["thought_process", "final_answer"]
}

@router.get("/", response_model=List[ChatResponse])
async def list_chats(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
) -> List[ChatResponse]:
    """List recent chats"""
    statement = (
        select(ChatThread)
        .offset(skip)
        .limit(limit)
        .order_by(col(ChatThread.id).desc())
    )
    chats = db.exec(statement).unique().all()
    return [ChatResponse.from_chat_thread(chat) for chat in chats]

@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: int,
    db: DatabaseDep
) -> ChatResponse:
    """Get a specific chat by ID"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    return ChatResponse.from_chat_thread(chat)

@router.delete("/{chat_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_chat(
    chat_id: int,
    db: DatabaseDep
) -> None:
    """Delete a specific chat"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    db.delete(chat)
    db.commit()

@router.post("/", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat(
    db: DatabaseDep
) -> ChatResponse:
    """Create a new chat with default configuration"""
    
    # Try to find existing tool
    statement = select(Tool).where(
        Tool.schema_name == "reasoning_steps",
        Tool.json_schema == CHAIN_OF_THOUGHT_SCHEMA
    )
    tool = db.exec(statement).first()
    
    # Create tool if it doesn't exist
    if not tool:
        tool = Tool(
            schema_name="reasoning_steps",
            schema_description="Break down the reasoning process step by step",
            instruction_string="Please follow this JSON schema for your response:",
            json_schema=CHAIN_OF_THOUGHT_SCHEMA
        )
        db.add(tool)
        db.flush()

    # Try to find existing config
    statement = select(LLMConfig).where(
        LLMConfig.client == DEFAULT_LLM_CONFIG.client,
        LLMConfig.model == DEFAULT_LLM_CONFIG.model,
        LLMConfig.response_format == DEFAULT_LLM_CONFIG.response_format,
        LLMConfig.temperature == DEFAULT_LLM_CONFIG.temperature
    )
    llm_config = db.exec(statement).first()
    
    # Create config if it doesn't exist
    if not llm_config:
        llm_config = DEFAULT_LLM_CONFIG
        db.add(llm_config)
        db.flush()
    
    # Create new chat with existing or new tool and config
    chat = ChatThread(
        new_message=None,  # Initialize with no message
        llm_config=llm_config,
        structured_output=tool,
        system_string="You are a helpful assistant that thinks step by step."
    )
    
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.post("/{chat_id}/messages/", response_model=ChatResponse)
async def send_message(
    message: MessageCreate,
    chat_id: int,
    ai_utils: ParallelAIUtilities = Depends(get_ai_utils)
) -> ChatResponse:
    """Send a new message in a chat"""
    # Get chat using a dedicated session
    with Session(ai_utils.engine) as db:
        statement = select(ChatThread).where(ChatThread.id == chat_id)
        chat = db.exec(statement).first()
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found"
            )
        
        # Update chat with new message
        chat.new_message = message.content
        db.add(chat)
        db.commit()
        db.refresh(chat)
    
    try:
        # Process with AI outside any session context
        results = await ai_utils.run_parallel_ai_completion([chat])
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get AI response"
            )
        
        # Get fresh chat after AI processing
        with Session(ai_utils.engine) as db:
            chat = db.get(ChatThread, chat_id)
            if not chat:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat {chat_id} not found after processing"
                )
            return ChatResponse.from_chat_thread(chat)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message: {str(e)}"
        )

@router.post("/{chat_id}/clear", response_model=ChatResponse)
async def clear_chat_history(
    chat_id: int,
    db: DatabaseDep
) -> ChatResponse:
    """Clear the history of a specific chat"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    chat.history = []  # Updated to use empty list instead of None
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return ChatResponse.from_chat_thread(chat)