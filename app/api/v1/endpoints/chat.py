#app\api\v1\endpoints\chat.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, asc, col
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel

from abstractions.inference.sql_models import (
    ChatThread,
    Tool,
    ChatMessage,
    LLMConfig,
    SystemStr,
    MessageRole,
    LLMClient,
    ResponseFormat
)
from abstractions.inference.sql_inference import ParallelAIUtilities
from abstractions.hub.tools import DEFAULT_TOOLS
from abstractions.hub.system_prompts import SYSTEM_PROMPTS
from app.api.deps import DatabaseDep, get_ai_utils

router = APIRouter(prefix="/chats", tags=["chats"])

# --- Response Models ---
class ChatMessageResponse(BaseModel):
    role: MessageRole
    content: str
    author_name: Optional[str] = None
    uuid: UUID
    parent_message_uuid: Optional[UUID] = None
    tool_name: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_json_schema: Optional[Dict[str, Any]] = None
    timestamp: datetime

    class Config:
        from_attributes = True

class SystemStrResponse(BaseModel):
    id: Optional[int] = None
    uuid: UUID
    name: str
    content: str

    class Config:
        from_attributes = True

class ChatResponse(BaseModel):
    id: Optional[int] = None
    uuid: UUID
    new_message: Optional[str]
    history: List[ChatMessageResponse]
    system_prompt: Optional[str] = None
    active_tool_id: Optional[int] = None

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
                author_name=msg.author_name,
                uuid=msg.uuid,
                parent_message_uuid=msg.parent_message_uuid,
                tool_name=msg.tool_name,
                tool_call_id=msg.tool_call_id,
                tool_json_schema=msg.tool_json_schema,
                timestamp=msg.timestamp
            ) for msg in chat.history
        ] if chat.history else []

        return cls(
            id=chat.id,
            uuid=chat.uuid,
            new_message=chat.new_message,
            history=history,
            system_prompt=chat.system_prompt.content if chat.system_prompt else None,
            active_tool_id=chat.structured_output_id if chat.structured_output_id else None
        )

class MessageCreate(BaseModel):
    content: str

class ToolCreate(BaseModel):
    schema_name: str
    schema_description: str
    instruction_string: str = "Please follow this JSON schema for your response:"
    json_schema: Dict[str, Any]
    strict_schema: bool = True

class ToolResponse(BaseModel):
    id: int
    schema_name: str
    schema_description: str
    instruction_string: str
    json_schema: Dict[str, Any]
    strict_schema: bool

    class Config:
        from_attributes = True

class ToolUpdate(BaseModel):
    schema_description: Optional[str] = None
    instruction_string: Optional[str] = None
    json_schema: Optional[Dict[str, Any]] = None
    strict_schema: Optional[bool] = None

class LLMConfigCreate(BaseModel):
    client: LLMClient
    model: Optional[str] = None
    max_tokens: int = 2500
    temperature: float = 0
    response_format: ResponseFormat = ResponseFormat.tool
    use_cache: bool = True

    class Config:
        from_attributes = True

class LLMConfigUpdate(BaseModel):
    client: Optional[LLMClient] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    response_format: Optional[ResponseFormat] = None
    use_cache: Optional[bool] = None

    class Config:
        from_attributes = True

class LLMConfigResponse(BaseModel):
    id: int
    client: LLMClient
    model: Optional[str]
    max_tokens: int
    temperature: float
    response_format: ResponseFormat
    use_cache: bool

    class Config:
        from_attributes = True

class SystemStrCreate(BaseModel):
    name: str
    content: str

class ChatCreateWithSystem(BaseModel):
    system_prompt_uuid: Optional[UUID] = None
    system_prompt_name: Optional[str] = None
    llm_config: Optional[LLMConfigCreate] = None

class ChatNameUpdate(BaseModel):
    name: str

# --- Default configurations ---
DEFAULT_LLM_CONFIG = LLMConfigCreate(
    client=LLMClient.openai,
    model="gpt-4o",
    response_format=ResponseFormat.tool,
    temperature=0,
    max_tokens=2500,
    use_cache=True
)

def _create_default_tools(db: Session) -> List[Tool]:
    tools = []
    for name, config in DEFAULT_TOOLS.items():
        tool = Tool(
            schema_name=name,
            schema_description=config["description"],
            instruction_string=config["instruction"],
            json_schema=config["schema"],
            strict_schema=True
        )
        db.add(tool)
        tools.append(tool)
    db.commit()  # Add this line to commit the tools
    return tools

def _create_default_system_prompts(db: Session) -> List[SystemStr]:
    """Create default system prompts if they don't exist."""
    prompts = []
    for name, content in SYSTEM_PROMPTS.items():
        prompt = SystemStr(
            name=name,
            content=content
        )
        db.add(prompt)
        prompts.append(prompt)
    db.commit()
    return prompts

# --- Tool management endpoints ---
@router.get("/tools/", response_model=List[ToolResponse])
async def list_tools(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
) -> List[ToolResponse]:
    """List available tools"""
    # First check if we have any tools
    statement = select(Tool)
    existing_tools = db.exec(statement).unique().all()
    
    # If no tools exist, create default ones
    if not existing_tools:
        existing_tools = _create_default_tools(db)
    
    # Now get the requested page of tools
    statement = select(Tool).offset(skip).limit(limit)
    tools = db.exec(statement).unique().all()
    return [ToolResponse.from_orm(tool) for tool in tools]

@router.post("/tools/", response_model=ToolResponse, status_code=status.HTTP_201_CREATED)
async def create_tool(
    tool_data: ToolCreate,
    db: DatabaseDep
) -> ToolResponse:
    """Create a new tool"""
    # Check if tool with same name exists
    existing = db.exec(
        select(Tool).where(Tool.schema_name == tool_data.schema_name)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool with name '{tool_data.schema_name}' already exists"
        )
    
    # Create new tool
    tool = Tool(**tool_data.dict())
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return ToolResponse.from_orm(tool)

@router.get("/tools/{tool_id}", response_model=ToolResponse)
async def get_tool(
    tool_id: int,
    db: DatabaseDep
) -> ToolResponse:
    """Get a specific tool"""
    tool = db.get(Tool, tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found"
        )
    return ToolResponse.from_orm(tool)

@router.delete("/tools/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_tool(
    tool_id: int,
    db: DatabaseDep
) -> None:
    """Delete a specific tool"""
    tool = db.get(Tool, tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found"
        )
    
    # Check if tool is in use by any chat threads
    chats_using_tool = db.exec(
        select(ChatThread).where(ChatThread.structured_output_id == tool_id)
    ).first()
    
    if chats_using_tool:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool {tool_id} is in use by chat threads and cannot be deleted"
        )
    
    db.delete(tool)
    db.commit()

@router.patch("/tools/{tool_id}", response_model=ToolResponse)
async def update_tool(
    tool_id: int,
    tool_update: ToolUpdate,
    db: DatabaseDep
) -> ToolResponse:
    """Update a specific tool"""
    tool = db.get(Tool, tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found"
        )
    
    # Update only provided fields
    update_data = tool_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(tool, key, value)
    
    db.add(tool)
    db.commit()
    db.refresh(tool)
    return ToolResponse.from_orm(tool)

# --- Chat thread tool assignment endpoint ---
@router.put("/{chat_id}/tool/{tool_id}", response_model=ChatResponse)
async def assign_tool_to_chat(
    chat_id: int,
    tool_id: int,
    db: DatabaseDep
) -> ChatResponse:
    """Assign a different tool to a chat thread"""
    # Get chat
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Get tool
    tool = db.get(Tool, tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool {tool_id} not found"
        )
    
    # Update chat's tool
    chat.structured_output = tool
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.get("/", response_model=List[ChatResponse])
async def list_chats(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
) -> List[ChatResponse]:
    """List recent chats"""
    # Get chats with explicit joins
    statement = (
        select(ChatThread)
        .offset(skip)
        .limit(limit)
        .order_by(col(ChatThread.id).desc())
    )
    chats = db.exec(statement).unique().all()
    
    # Convert each chat to response model
    return [ChatResponse.from_chat_thread(chat) for chat in chats]

@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: int,
    db: DatabaseDep
) -> ChatResponse:
    """Get a specific chat by ID"""
    chat = db.exec(select(ChatThread).where(ChatThread.id == chat_id)).first()
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
    # Create with default config
    chat = await _create_chat_with_config(db, DEFAULT_LLM_CONFIG)
    
    # Explicitly load relationships for response
    db.refresh(chat)
    statement = select(ChatThread).where(ChatThread.id == chat.id)
    chat = db.exec(statement).unique().first()
    
    return ChatResponse.from_chat_thread(chat)

@router.post("/with-config", response_model=ChatResponse, status_code=status.HTTP_201_CREATED)
async def create_chat_with_config(
    chat_data: ChatCreateWithSystem,
    db: DatabaseDep
) -> ChatResponse:
    """Create a new chat with custom LLM configuration"""
    system_prompt = None
    
    if chat_data.system_prompt_uuid:
        query = select(SystemStr).where(SystemStr.uuid == chat_data.system_prompt_uuid)
        system_prompt = db.exec(query).first()
    elif chat_data.system_prompt_name:
        query = select(SystemStr).where(SystemStr.name == chat_data.system_prompt_name)
        system_prompt = db.exec(query).first()
    
    if chat_data.system_prompt_uuid or chat_data.system_prompt_name:
        if not system_prompt:
            raise HTTPException(status_code=404, detail="System prompt not found")
    
    if chat_data.llm_config:
        chat = await _create_chat_with_config(db, chat_data.llm_config)
    else:
        chat = await _create_chat_with_config(db, DEFAULT_LLM_CONFIG)
    
    if system_prompt:
        chat.system_string = system_prompt.content
        db.add(chat)
        db.commit()
        db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.post("/with-system/", response_model=ChatResponse)
async def create_chat_with_system(
    chat_data: ChatCreateWithSystem,
    db: DatabaseDep
):
    """Create a new chat with specified system prompt."""
    system_prompt = None
    
    if chat_data.system_prompt_uuid:
        query = select(SystemStr).where(SystemStr.uuid == chat_data.system_prompt_uuid)
        system_prompt = db.exec(query).first()
    elif chat_data.system_prompt_name:
        query = select(SystemStr).where(SystemStr.name == chat_data.system_prompt_name)
        system_prompt = db.exec(query).first()
    
    if chat_data.system_prompt_uuid or chat_data.system_prompt_name:
        if not system_prompt:
            raise HTTPException(status_code=404, detail="System prompt not found")
    
    if chat_data.llm_config:
        chat = await _create_chat_with_config(db, chat_data.llm_config)
    else:
        chat = await _create_chat_with_config(db, DEFAULT_LLM_CONFIG)
    
    if system_prompt:
        chat.system_prompt = system_prompt
        db.add(chat)
        db.commit()
        db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

async def _create_chat_with_config(
    db: Session,
    config: LLMConfigCreate
) -> ChatThread:
    """Create a new chat thread with the given LLM configuration."""
    # Create LLM config
    db_config = LLMConfig.from_orm(config)
    db.add(db_config)
    db.flush()  # Flush to get the ID
    
    # Get or create default tools
    tools_query = select(Tool)
    existing_tools = db.exec(tools_query).unique().all()
    
    if not existing_tools:
        tools = _create_default_tools(db)
        db.flush()  # Flush to ensure tools have IDs
    else:
        tools = existing_tools

    # Get or create default system prompts
    prompts_query = select(SystemStr)
    existing_prompts = db.exec(prompts_query).unique().all()
    
    if not existing_prompts:
        prompts = _create_default_system_prompts(db)
        db.flush()  # Flush to ensure prompts have IDs
    else:
        prompts = existing_prompts
    
    # Select a random system prompt
    import random
    system_prompt = random.choice(prompts)
    
    # Create chat thread with explicit relationships
    chat = ChatThread(
        llm_config_id=db_config.id,  # Use ID instead of relationship
        structured_output_id=tools[0].id if tools else None  # Use ID instead of relationship
    )
    db.add(chat)
    db.flush()  # Flush to get chat ID
    
    # Add system prompt through the linkage table
    if system_prompt:
        from abstractions.inference.sql_models import ThreadSystemLinkage
        linkage = ThreadSystemLinkage(
            chat_thread_id=chat.id,
            system_str_id=system_prompt.id
        )
        db.add(linkage)
    
    # Commit all changes
    db.commit()
    db.refresh(chat)
    
    return chat

# --- LLM Config management endpoints ---
@router.get("/llm-configs/", response_model=List[LLMConfigResponse])
async def list_llm_configs(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
) -> List[LLMConfigResponse]:
    """List available LLM configurations"""
    statement = select(LLMConfig).offset(skip).limit(limit)
    configs = db.exec(statement).unique().all()
    return [LLMConfigResponse.from_orm(config) for config in configs]

@router.post("/llm-configs/", response_model=LLMConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_llm_config(
    config_data: LLMConfigCreate,
    db: DatabaseDep
) -> LLMConfigResponse:
    """Create a new LLM configuration"""
    config = LLMConfig(**config_data.dict())
    db.add(config)
    db.commit()
    db.refresh(config)
    return LLMConfigResponse.from_orm(config)

@router.get("/llm-configs/{config_id}", response_model=LLMConfigResponse)
async def get_llm_config(
    config_id: int,
    db: DatabaseDep
) -> LLMConfigResponse:
    """Get a specific LLM configuration"""
    config = db.get(LLMConfig, config_id)
    if not config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"LLM config {config_id} not found"
        )
    return LLMConfigResponse.from_orm(config)

@router.put("/{chat_id}/llm-config", response_model=ChatResponse)
async def update_chat_llm_config(
    chat_id: int,
    config_update: LLMConfigUpdate,
    db: DatabaseDep
) -> ChatResponse:
    """Update LLM configuration for a specific chat"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Update only provided fields
    update_data = config_update.dict(exclude_unset=True)
    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields to update"
        )
    
    # Create new config or update existing
    if chat.llm_config:
        for key, value in update_data.items():
            setattr(chat.llm_config, key, value)
        config = chat.llm_config
    else:
        config = LLMConfig(**update_data)
        db.add(config)
        db.flush()  # Get ID without committing
        chat.llm_config = config
    
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
    # Get chat and setup message
    with Session(ai_utils.engine) as db:
        chat = db.get(ChatThread, chat_id)
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
        # Process with AI and WAIT for the results
        results = await ai_utils.run_parallel_ai_completion([chat])
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get AI response"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing message during parallel completion: {str(e)}"
        )
    try:
        # Now that we have results, get final state
        with Session(ai_utils.engine) as db:
            final_chat = db.get(ChatThread, chat_id)
            if not final_chat:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat {chat_id} not found after processing"
                )
            return ChatResponse.from_chat_thread(final_chat)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting final chat state after parallel completion: {str(e)}"
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

@router.get("/by-name/{chat_name}", response_model=ChatResponse)
async def get_chat_by_name(
    chat_name: str,
    db: DatabaseDep
) -> ChatResponse:
    """Get a specific chat by its name"""
    chat = db.exec(select(ChatThread).where(ChatThread.name == chat_name)).first()
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat with name '{chat_name}' not found"
        )
    return ChatResponse.from_chat_thread(chat)

@router.put("/{chat_id}/name", response_model=ChatResponse)
async def update_chat_name(
    chat_id: int,
    name_update: ChatNameUpdate,
    db: DatabaseDep
) -> ChatResponse:
    """Update or set the name of a specific chat"""
    # Check if chat exists
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # If trying to remove name (empty string), just set to None
    if not name_update.name.strip():
        chat.name = None
        db.add(chat)
        db.commit()
        db.refresh(chat)
        return ChatResponse.from_chat_thread(chat)
    
    # Check if name is already taken by another chat
    existing_chat = db.exec(
        select(ChatThread)
        .where(
            ChatThread.name == name_update.name,
            ChatThread.id != chat_id  # Exclude current chat
        )
    ).first()
    
    if existing_chat:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Chat with name '{name_update.name}' already exists"
        )
    
    # Update chat name
    chat.name = name_update.name
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

# --- System prompt management endpoints ---
@router.get("/system-prompts/", response_model=List[SystemStrResponse])
async def list_system_prompts(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
):
    """List available system prompts."""
    query = select(SystemStr).offset(skip).limit(limit)
    return db.exec(query).unique().all()

@router.post("/system-prompts/", response_model=SystemStrResponse)
async def create_system_prompt(
    prompt_data: SystemStrCreate,
    db: DatabaseDep
):
    """Create a new system prompt."""
    db_prompt = SystemStr(**prompt_data.dict())
    db.add(db_prompt)
    db.commit()
    db.refresh(db_prompt)
    return db_prompt

@router.get("/system-prompts/{prompt_id}", response_model=SystemStrResponse)
async def get_system_prompt(
    prompt_id: int,
    db: DatabaseDep
):
    """Get a specific system prompt."""
    prompt = db.get(SystemStr, prompt_id)
    if not prompt:
        raise HTTPException(status_code=404, detail="System prompt not found")
    return prompt

@router.get("/system-prompts/by-uuid/{prompt_uuid}", response_model=SystemStrResponse)
async def get_system_prompt_by_uuid(
    prompt_uuid: UUID,
    db: DatabaseDep
):
    """Get a specific system prompt by UUID."""
    query = select(SystemStr).where(SystemStr.uuid == prompt_uuid)
    prompt = db.exec(query).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="System prompt not found")
    return prompt

@router.get("/system-prompts/by-name/{prompt_name}", response_model=SystemStrResponse)
async def get_system_prompt_by_name(
    prompt_name: str,
    db: DatabaseDep
):
    """Get a specific system prompt by name."""
    query = select(SystemStr).where(SystemStr.name == prompt_name)
    prompt = db.exec(query).first()
    if not prompt:
        raise HTTPException(status_code=404, detail="System prompt not found")
    return prompt