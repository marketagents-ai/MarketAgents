#app\api\v1\endpoints\chat.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select, asc, col
from typing import List, Optional, Dict, Any, Callable
from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, Field

from abstractions.inference.sql_models import (
    ChatThread,
    Tool,
    ChatMessage,
    LLMConfig,
    SystemStr,
    MessageRole,
    LLMClient,
    ResponseFormat,
    CallableRegistry
)
from abstractions.inference.sql_inference import ParallelAIUtilities
from abstractions.hub.tools import DEFAULT_TOOLS
from abstractions.hub.system_prompts import SYSTEM_PROMPTS
from abstractions.hub.callable_tools import DEFAULT_CALLABLE_TOOLS
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
    tool_call: Optional[Dict[str, Any]] = None
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
    name: Optional[str] = None
    new_message: Optional[str]
    history: List[ChatMessageResponse]
    system_prompt: Optional[str] = None
    system_prompt_id: Optional[int] = None
    active_tool_id: Optional[int] = None
    auto_tools_ids: List[int] = []

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
                tool_call=msg.tool_call,
                timestamp=msg.timestamp
            ) for msg in chat.history
        ] if chat.history else []

        # Get IDs of active tools
        auto_tools_ids = [tool.id for tool in chat.tools if tool.id is not None]

        return cls(
            id=chat.id,
            uuid=chat.uuid,
            name=chat.name,
            new_message=chat.new_message,
            history=history,
            system_prompt=chat.system_prompt.content if chat.system_prompt else None,
            system_prompt_id=chat.system_prompt.id if chat.system_prompt else None,
            active_tool_id=chat.structured_output_id if chat.structured_output_id else None,
            auto_tools_ids=auto_tools_ids
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
    is_callable: bool = False

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
    """Create default tools if they don't exist."""
    tools = []
    
    # Check existing tools to avoid duplicates
    existing_tools = {
        tool.schema_name: tool 
        for tool in db.exec(select(Tool)).unique().all()
    }
    
    # Create regular tools
    for name, config in DEFAULT_TOOLS.items():
        if name not in existing_tools:
            tool = Tool(
                schema_name=name,
                schema_description=config["description"],
                instruction_string=config["instruction"],
                json_schema=config["schema"],
                strict_schema=True
            )
            db.add(tool)
            tools.append(tool)
    
    # Create callable tools
    for name, config in DEFAULT_CALLABLE_TOOLS.items():
        if name not in existing_tools:
            try:
                tool = Tool.from_callable(
                    func=config["function"],
                    schema_name=name,
                    schema_description=config["description"]
                )
                tool.allow_literal_eval = config.get("allow_literal_eval", False)
                db.add(tool)
                tools.append(tool)
            except Exception as e:
                print(f"Failed to create callable tool {name}: {str(e)}")
    
    if tools:  # Only commit if we created new tools
        db.commit()
        for tool in tools:
            db.refresh(tool)
    
    # Return all tools, both existing and newly created
    return list(existing_tools.values()) + tools

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
    """List available regular (non-callable) tools, creating defaults if missing"""
    # First check if we have any tools
    statement = select(Tool).where(Tool.callable == False)  # Only get regular tools
    existing_tools: List[Tool] = list(db.exec(statement).unique().all())
    
    # Get existing tool names
    existing_tool_names = {tool.schema_name for tool in existing_tools}
    
    # Check for missing default tools (only regular tools)
    missing_regular_tools = set(DEFAULT_TOOLS.keys()) - existing_tool_names
    
    if missing_regular_tools:
        new_tools: List[Tool] = []
        
        # Create missing regular tools
        for name in missing_regular_tools:
            config = DEFAULT_TOOLS[name]
            tool = Tool(
                schema_name=name,
                schema_description=config["description"],
                instruction_string=config["instruction"],
                json_schema=config["schema"],
                strict_schema=True,
                callable=False
            )
            db.add(tool)
            new_tools.append(tool)
        
        db.commit()
        for tool in new_tools:
            db.refresh(tool)
        existing_tools.extend(new_tools)
    
    # Now get the requested page of regular tools
    statement = select(Tool).where(Tool.callable == False).offset(skip).limit(limit)
    tools = db.exec(statement).unique().all()
    
    responses: List[ToolResponse] = []
    for tool in tools:
        if tool.id is not None:  # Only include tools with valid IDs
            responses.append(ToolResponse(
                id=tool.id,  # Now guaranteed to be non-None
                schema_name=tool.schema_name,
                schema_description=tool.schema_description,
                instruction_string=tool.instruction_string,
                json_schema=tool.json_schema,
                strict_schema=tool.strict_schema,
                is_callable=tool.callable
            ))
    return responses

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
    chat  = db.exec(statement).unique().first()
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
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
        structured_output_id=tools[0].id if tools else None , # Use ID instead of relationship
        system_prompt=system_prompt
    )
    db.add(chat)
    db.flush()  # Flush to get chat ID
    
   
    
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

@router.put("/{chat_id}/system-prompt/by-uuid/{prompt_uuid}", response_model=ChatResponse)
async def update_chat_system_prompt_by_uuid(
    chat_id: int,
    prompt_uuid: UUID,
    db: DatabaseDep
) -> ChatResponse:
    """Update the system prompt for a specific chat using a system prompt UUID"""
    # Get chat
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Get system prompt
    query = select(SystemStr).where(SystemStr.uuid == prompt_uuid)
    system_prompt = db.exec(query).first()
    if not system_prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System prompt with UUID {prompt_uuid} not found"
        )
    
    # Update chat's system prompt
    chat.system_prompt = system_prompt
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.put("/{chat_id}/system-prompt/by-name/{prompt_name}", response_model=ChatResponse)
async def update_chat_system_prompt_by_name(
    chat_id: int,
    prompt_name: str,
    db: DatabaseDep
) -> ChatResponse:
    """Update the system prompt for a specific chat using a system prompt name"""
    # Get chat
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Get system prompt
    query = select(SystemStr).where(SystemStr.name == prompt_name)
    system_prompt = db.exec(query).first()
    if not system_prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System prompt with name '{prompt_name}' not found"
        )
    
    # Update chat's system prompt
    chat.system_prompt = system_prompt
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.put("/{chat_id}/system-prompt/by-id/{prompt_id}", response_model=ChatResponse)
async def update_chat_system_prompt_by_id(
    chat_id: int,
    prompt_id: int,
    db: DatabaseDep
) -> ChatResponse:
    """Update the system prompt for a specific chat using a system prompt ID"""
    # Get chat
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Get system prompt
    system_prompt = db.get(SystemStr, prompt_id)
    if not system_prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System prompt with ID {prompt_id} not found"
        )
    
    # Update chat's system prompt
    chat.system_prompt = system_prompt
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.delete("/system-prompts/{prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_system_prompt(
    prompt_id: int,
    db: DatabaseDep
) -> None:
    """Delete a specific system prompt"""
    # Get the system prompt
    prompt = db.get(SystemStr, prompt_id)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"System prompt {prompt_id} not found"
        )
    
    # Check if prompt is in use by any chat threads
    chats_using_prompt = db.exec(
        select(ChatThread).where(ChatThread.system_prompt == prompt)
    ).first()
    
    if chats_using_prompt:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"System prompt {prompt_id} is in use by chat threads and cannot be deleted"
        )
    
    db.delete(prompt)
    db.commit()

# Add new response/create models for callable tools
class CallableToolCreate(BaseModel):
    name: str
    description: str
    function_text: str
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    allow_literal_eval: bool = False

class CallableToolResponse(BaseModel):
    id: int
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    is_registered: bool

    class Config:
        from_attributes = True

# Add new endpoints for callable tools
@router.post("/callable-tools/", response_model=CallableToolResponse, status_code=status.HTTP_201_CREATED)
async def create_callable_tool(
    tool_data: CallableToolCreate,
    db: DatabaseDep
) -> CallableToolResponse:
    """Create a new callable tool"""
    # Check if tool with same name exists
    existing = db.exec(
        select(Tool).where(Tool.schema_name == tool_data.name)
    ).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool with name '{tool_data.name}' already exists"
        )
    
    # Create tool from callable
    try:
        # First register the function from text
        CallableRegistry().register_from_text(
            name=tool_data.name,
            func_text=tool_data.function_text
        )
        
        # Get the registered callable
        func = CallableRegistry().get(tool_data.name)
        if func is None:
            raise ValueError("Failed to register callable function")
        
        # Now create the tool using the callable
        tool = Tool.from_callable(
            func=func,
            schema_name=tool_data.name,
            schema_description=tool_data.description,
            json_schema=tool_data.input_schema
        )
        
        db.add(tool)
        db.commit()
        db.refresh(tool)
        
        if tool.id is None:
            raise ValueError("Tool ID was not generated")
            
        return CallableToolResponse(
            id=tool.id,
            name=tool.schema_name,
            description=tool.schema_description,
            input_schema=tool.json_schema,
            output_schema=tool.callable_output_schema or {},
            is_registered=tool.callable
        )
    except Exception as e:
        # Clean up registry if tool creation fails
        try:
            CallableRegistry().delete(tool_data.name)
        except:
            pass
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to create callable tool: {str(e)}"
        )

@router.get("/callable-tools/", response_model=List[CallableToolResponse])
async def list_callable_tools(
    db: DatabaseDep,
    skip: int = 0,
    limit: int = 10
) -> List[CallableToolResponse]:
    """List available callable tools, creating defaults if missing"""
    # First check if we have any callable tools
    statement = select(Tool).where(Tool.callable == True)
    existing_tools: List[Tool] = list(db.exec(statement).unique().all())
    
    # Get existing callable tool names
    existing_tool_names = {tool.schema_name for tool in existing_tools}
    
    # Check for missing default callable tools
    missing_callable_tools = set(DEFAULT_CALLABLE_TOOLS.keys()) - existing_tool_names
    
    if missing_callable_tools:
        new_tools: List[Tool] = []
        
        # Create missing callable tools
        for name in missing_callable_tools:
            config = DEFAULT_CALLABLE_TOOLS[name]
            try:
                tool = Tool.from_callable(
                    func=config["function"],
                    schema_name=name,
                    schema_description=config["description"]
                )
                tool.allow_literal_eval = config.get("allow_literal_eval", False)
                db.add(tool)
                new_tools.append(tool)
            except Exception as e:
                print(f"Failed to create callable tool {name}: {str(e)}")
        
        db.commit()
        for tool in new_tools:
            db.refresh(tool)
        existing_tools.extend(new_tools)
    
    # Now get the requested page of callable tools
    statement = select(Tool).where(Tool.callable == True).offset(skip).limit(limit)
    tools = db.exec(statement).unique().all()
    
    responses: List[CallableToolResponse] = []
    for tool in tools:
        if tool.id is not None:  # Only include tools with valid IDs
            responses.append(CallableToolResponse(
                id=tool.id,  # Now guaranteed to be non-None
                name=tool.schema_name,
                description=tool.schema_description,
                input_schema=tool.json_schema,
                output_schema=tool.callable_output_schema or {},
                is_registered=tool.callable
            ))
    return responses

@router.get("/callable-tools/{tool_id}", response_model=CallableToolResponse)
async def get_callable_tool(
    tool_id: int,
    db: DatabaseDep
) -> CallableToolResponse:
    """Get a specific callable tool"""
    tool = db.get(Tool, tool_id)
    if not tool or not tool.callable or tool.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Callable tool {tool_id} not found"
        )
    
    return CallableToolResponse(
        id=tool.id,
        name=tool.schema_name,
        description=tool.schema_description,
        input_schema=tool.json_schema,
        output_schema=tool.callable_output_schema or {},
        is_registered=tool.callable
    )

@router.delete("/callable-tools/{tool_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_callable_tool(
    tool_id: int,
    db: DatabaseDep
) -> None:
    """Delete a specific callable tool"""
    tool = db.get(Tool, tool_id)
    if not tool or not tool.callable or tool.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Callable tool {tool_id} not found"
        )
    
    # Check if tool is in use
    chats_using_tool = db.exec(
        select(ChatThread).where(ChatThread.structured_output_id == tool_id)
    ).first()
    
    if chats_using_tool:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Tool {tool_id} is in use by chat threads and cannot be deleted"
        )
    
    # Remove from registry first
    CallableRegistry().delete(tool.schema_name)
    
    db.delete(tool)
    db.commit()

# Add new endpoint to test callable tools
@router.post("/callable-tools/{tool_id}/test", response_model=Dict[str, Any])
async def test_callable_tool(
    tool_id: int,
    input_data: Dict[str, Any],
    db: DatabaseDep
) -> Dict[str, Any]:
    """Test a callable tool with sample input"""
    tool = db.get(Tool, tool_id)
    if not tool or not tool.callable or tool.id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Callable tool {tool_id} not found"
        )
    
    try:
        result = tool.execute(input=input_data)
        return {"result": result.content}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to execute tool: {str(e)}"
        )

# --- Auto tools management endpoints ---
@router.put("/{chat_id}/auto-tools", response_model=ChatResponse)
async def update_chat_auto_tools(
    chat_id: int,
    tool_ids: List[int],
    db: DatabaseDep
) -> ChatResponse:
    """Update the list of active tools for auto tools mode"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    # Verify all tools exist and get them
    tools = []
    for tool_id in tool_ids:
        tool = db.get(Tool, tool_id)
        if not tool:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Tool {tool_id} not found"
            )
        tools.append(tool)
    
    # Update chat's tools without modifying response format
    chat.tools = tools
    db.add(chat)
    db.commit()
    db.refresh(chat)
    
    return ChatResponse.from_chat_thread(chat)

@router.post("/{chat_id}/assistant-response", response_model=ChatResponse)
async def trigger_assistant_response(
    chat_id: int,
    ai_utils: ParallelAIUtilities = Depends(get_ai_utils)
) -> ChatResponse:
    """Trigger an assistant response without a new user message"""
    with Session(ai_utils.engine) as db:
        chat = db.get(ChatThread, chat_id)
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found"
            )
        
        if not chat.tools:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Chat must have active tools to trigger assistant response"
            )
    
    try:
        results = await ai_utils.run_parallel_ai_completion([chat])
        if not results:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get AI response"
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing assistant response: {str(e)}"
        )

    with Session(ai_utils.engine) as db:
        final_chat = db.get(ChatThread, chat_id)
        if not final_chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat {chat_id} not found after processing"
            )
        return ChatResponse.from_chat_thread(final_chat)

@router.get("/{chat_id}/llm-config", response_model=LLMConfigResponse)
async def get_chat_llm_config(
    chat_id: int,
    db: DatabaseDep
) -> LLMConfigResponse:
    """Get the LLM configuration for a specific chat"""
    chat = db.get(ChatThread, chat_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat {chat_id} not found"
        )
    
    if not chat.llm_config:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No LLM configuration found for chat {chat_id}"
        )
    
    return LLMConfigResponse.from_orm(chat.llm_config)