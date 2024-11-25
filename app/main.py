from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.v1.endpoints import chat
from app.core.config import settings
from app.db.session import engine
from sqlmodel import SQLModel, Session, select
from typing import Union
from abstractions.inference.sql_models import (
    Tool, 
    CallableRegistry, 
    SystemStr, 
    ChatThread, 
    LLMConfig,
    LLMClient,
    ResponseFormat
)
from abstractions.hub.callable_tools import DEFAULT_CALLABLE_TOOLS
from abstractions.hub.angels import (
    DEFAULT_TOOLS,
    tier1_programs,
    tier2_programs,
    tier3_programs,
    TIER1_SYSTEM_PROMPT,
    TIER2_SYSTEM_PROMPT,
    TIER3_SYSTEM_PROMPT,
    BASIC_METADATA_SCHEMA,
    SUMMARY_SCHEMA,
    THEME_ANALYSIS_SCHEMA,
    CHARACTER_ANALYSIS_SCHEMA,
    SETTING_ANALYSIS_SCHEMA,
    HISTORICAL_CONTEXT_SCHEMA,
    LITERARY_DEVICES_SCHEMA,
    CRITICAL_ANALYSIS_SCHEMA
)
from abstractions.hub.forge import forge_tool, call_forge_api
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def register_default_tools(db: Session):
    """Register default tools in the CallableRegistry during startup"""
    logger.info("=== Starting Tool Registration Process ===")
    registry = CallableRegistry()
    
    try:
        # First, register all default callable tools
        logger.info("Registering default callable tools...")
        for name, config in DEFAULT_CALLABLE_TOOLS.items():
            try:
                registry.register(name, config["function"])
                logger.info(f"✓ Registered default tool: {name}")
            except ValueError as e:
                logger.warning(f"! Skipped default tool {name}: {str(e)}")
        
        # Get all callable tools from database
        statement = select(Tool).where(Tool.callable == True)
        db_tools = db.exec(statement).unique().all()
        logger.info(f"Found {len(db_tools)} callable tools in database")
        
        # Compare and register missing tools
        for tool in db_tools:
            logger.info(f"\nProcessing database tool: {tool.schema_name}")
            
            # Skip if already in registry
            if registry.get(tool.schema_name):
                logger.info(f"→ Tool {tool.schema_name} already registered")
                continue
            
            try:
                if tool.schema_name in DEFAULT_CALLABLE_TOOLS:
                    # Register from default tools if available
                    func = DEFAULT_CALLABLE_TOOLS[tool.schema_name]["function"]
                    registry.register(tool.schema_name, func)
                    logger.info(f"✓ Registered from DEFAULT_CALLABLE_TOOLS: {tool.schema_name}")
                    
                elif tool.callable_function and tool.allow_literal_eval:
                    # Register from stored function text
                    registry.register_from_text(
                        name=tool.schema_name,
                        func_text=tool.callable_function
                    )
                    logger.info(f"✓ Registered from stored text: {tool.schema_name}")
                    
                else:
                    logger.warning(
                        f"! Cannot register {tool.schema_name}: "
                        f"No default implementation and literal_eval not allowed"
                    )
                    
            except Exception as e:
                logger.error(f"! Failed to register {tool.schema_name}: {str(e)}")
        
        # Final registry status
        registered_tools = list(registry._registry.keys())
        logger.info("\n=== Tool Registration Summary ===")
        logger.info(f"Total tools in registry: {len(registered_tools)}")
        logger.info(f"Registered tools: {registered_tools}")
        
    except Exception as e:
        logger.error(f"Critical error during tool registration: {str(e)}", exc_info=True)
        raise

def register_literary_analysis_tools(db: Session):
    """Register literary analysis tools and system prompts during startup"""
    logger.info("=== Starting Literary Analysis Tools Registration ===")
    
    try:
        # Define all tools
        tier1_tools = [
            Tool(
                schema_name="extract_basic_metadata",
                schema_description="Extracts basic metadata like title and author",
                json_schema=BASIC_METADATA_SCHEMA,
                instruction_string="Extract the basic metadata from the text"
            ),
            Tool(
                schema_name="generate_summary",
                schema_description="Generates a brief summary of the text",
                json_schema=SUMMARY_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Generate a summary of the text"
            ),
            Tool(
                schema_name="extract_keywords",
                schema_description="Extracts key terms and concepts",
                json_schema={
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of key terms and concepts"
                        }
                    },
                    "required": ["keywords"]
                },
                instruction_string="Extract key terms and concepts from the text"
            )
        ]

        # Create and register Tier 2 tools
        tier2_tools = [
            Tool(
                schema_name="analyze_themes",
                schema_description="Analyzes major themes in the text",
                json_schema=THEME_ANALYSIS_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Analyze the major themes in the text"
            ),
            Tool(
                schema_name="analyze_characters",
                schema_description="Analyzes character details and relationships",
                json_schema=CHARACTER_ANALYSIS_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Analyze the characters and their relationships"
            ),
            Tool(
                schema_name="analyze_setting",
                schema_description="Analyzes the setting and time period",
                json_schema=SETTING_ANALYSIS_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Analyze the setting of the text"
            )
        ]

        # Create and register Tier 3 tools
        tier3_tools = [
            Tool(
                schema_name="analyze_historical_context",
                schema_description="Analyzes historical and cultural background",
                json_schema=HISTORICAL_CONTEXT_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Analyze the historical context of the text"
            ),
            Tool(
                schema_name="identify_literary_devices",
                schema_description="Identifies and analyzes literary techniques",
                json_schema=LITERARY_DEVICES_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Identify and analyze literary devices in the text"
            ),
            Tool(
                schema_name="perform_critical_analysis",
                schema_description="Provides interpretive analysis",
                json_schema=CRITICAL_ANALYSIS_SCHEMA,  # This is the output constraint for the LLM
                instruction_string="Perform a critical analysis of the text"
            )
        ]
        

        all_tools = tier1_tools + tier2_tools + tier3_tools
        
        # Check and add tools only if they don't exist
        for tool in all_tools:
            existing_tool = db.exec(
                select(Tool).where(Tool.schema_name == tool.schema_name)
            ).first()
            
            if not existing_tool:
                db.add(tool)
                logger.info(f"✓ Created new literary analysis tool: {tool.schema_name}")
            else:
                logger.info(f"→ Tool already exists: {tool.schema_name}")

        # Check and add system prompts only if they don't exist
        system_prompts = {
            "Tier 1 Literary Analysis": TIER1_SYSTEM_PROMPT,
            "Tier 2 Literary Analysis": TIER2_SYSTEM_PROMPT,
            "Tier 3 Literary Analysis": TIER3_SYSTEM_PROMPT
        }
        
        for name, content in system_prompts.items():
            existing_prompt = db.exec(
                select(SystemStr).where(SystemStr.name == name)
            ).first()
            
            if not existing_prompt:
                system_str = SystemStr(name=name, content=content)
                db.add(system_str)
                logger.info(f"✓ Created new system prompt: {name}")
            else:
                logger.info(f"→ System prompt already exists: {name}")
        
        # Commit changes
        db.commit()
        logger.info("\n=== Literary Analysis Registration Summary ===")
        logger.info(f"Total tools checked: {len(all_tools)}")
        logger.info(f"Total system prompts checked: {len(system_prompts)}")
        
    except Exception as e:
        logger.error(f"Critical error during literary analysis registration: {str(e)}", exc_info=True)
        raise

class SQLAlchemyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except SQLAlchemyError as e:
            logger.error(f"Database error: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Database error: {str(e)}"}
            )

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger.info(f"=== Starting {request.method} {request.url.path} ===")
        try:
            response = await call_next(request)
            logger.info(f"✓ Completed {request.method} {request.url.path}")
            return response
        except Exception as e:
            logger.error(f"✕ Error processing {request.method} {request.url.path}: {str(e)}")
            raise

def create_application() -> FastAPI:
    logger.info("Creating FastAPI application...")
    
    app = FastAPI(
        title="Chat API",
        description="API for managing AI chat interactions",
        version="0.0.1",
        default_response_class=JSONResponse
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"]
    )
    
    app.add_middleware(SQLAlchemyMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize database, tools, and default chat threads."""
        logger.info("Running startup tasks...")
        try:
            # Create database tables
            logger.info("Creating database tables...")
            SQLModel.metadata.create_all(engine)
            
            # Register tools and system prompts
            with Session(engine) as db:
                register_default_tools(db)
                register_literary_analysis_tools(db)
                
                # Initialize default chat threads
                logger.info("Checking/Initializing default chat threads...")
                
                # Get system prompts
                tier1_prompt = db.exec(
                    select(SystemStr).where(SystemStr.name == "Tier 1 Literary Analysis")
                ).first()
                tier2_prompt = db.exec(
                    select(SystemStr).where(SystemStr.name == "Tier 2 Literary Analysis")
                ).first()
                tier3_prompt = db.exec(
                    select(SystemStr).where(SystemStr.name == "Tier 3 Literary Analysis")
                ).first()
                # Get tools for each tier based on their names in the system prompts
                tier1_tools = db.exec(
                    select(Tool).where(Tool.schema_name.in_([
                        "extract_basic_metadata",
                        "generate_summary", 
                        "extract_keywords"
                    ]))
                ).all()
                tier2_tools = db.exec(
                    select(Tool).where(Tool.schema_name.in_([
                        "analyze_themes",
                        "analyze_characters",
                        "analyze_setting",
                        "correlate_themes_setting"
                    ]))
                ).all()
                tier3_tools = db.exec(
                    select(Tool).where(Tool.schema_name.in_([
                        "analyze_historical_context",
                        "identify_literary_devices",
                        "perform_critical_analysis",
                        "analyze_philosophy",
                        "analyze_psychology",
                        "analyze_structure",
                        "analyze_linguistics",
                        "perform_comparative"
                    ]))
                ).all()
                # Check for forge tool in database and registry
                forge_tool_db = db.exec(
                    select(Tool).where(Tool.schema_name == forge_tool.schema_name)
                ).first()
                
                if forge_tool_db:
                    logger.info("Found forge tool in database, re-registering to registry...")
                    try:
                        CallableRegistry().register(forge_tool.schema_name, call_forge_api)
                        logger.info("✓ Re-registered forge tool to registry")
                    except ValueError:
                        logger.info("→ Forge tool already in registry")
                
                # Create forge system prompt if needed
                forge_prompt = db.exec(
                    select(SystemStr).where(SystemStr.name == "Forge Agent")
                ).first()
                
                if not forge_prompt:
                    forge_prompt = SystemStr(
                        name="Forge Agent",
                        content="You are a forge agent capable of creating and modifying tools. Use the call_forge tool to handle tool-related operations."
                    )
                    db.add(forge_prompt)
                    logger.info("✓ Created forge system prompt")
                
                # Get default LLM config
                default_config = db.exec(
                    select(LLMConfig)
                    .where(LLMConfig.model == "gpt-4o")
                    .where(LLMConfig.response_format == ResponseFormat.auto_tools)
                ).first()
                
                if not default_config:
                    default_config = LLMConfig(
                        client=LLMClient.openai,
                        model="gpt-4o",
                        response_format=ResponseFormat.auto_tools,
                        temperature=0,
                        max_tokens=2500,
                        use_cache=True
                    )
                    db.add(default_config)
                    db.commit()
                    logger.info("✓ Created default LLM config")
                
                # Define default chats configuration with their specific tools
                default_chats_config = [
                    ("Dante", tier1_prompt, tier1_tools),
                    ("Shakespeare", tier2_prompt, tier2_tools),
                    ("Virgil", tier3_prompt, tier3_tools),
                    ("Forge", forge_prompt, [forge_tool_db] if forge_tool_db else [forge_tool])  # Add Forge agent
                ]
                
                # Check and create each chat if it doesn't exist
                for chat_name, system_prompt, tools in default_chats_config:
                    existing_chat = db.exec(
                        select(ChatThread).where(ChatThread.name == chat_name)
                    ).first()
                    
                    if not existing_chat:
                        logger.info(f"Creating new chat thread: {chat_name} with tools: {[t.schema_name for t in tools]}")
                        new_chat = ChatThread(
                            name=chat_name,
                            system_prompt=system_prompt,
                            tools=tools,  # Assign tier-specific tools
                            llm_config=default_config,
                            stop_tool=tools[-1] if tools and chat_name != "Forge" else None  # Set last tool as stop tool for literary agents
                        )
                        db.add(new_chat)
                        logger.info(f"✓ Created new chat thread: {chat_name}")
                    else:
                        # Update existing chat with correct tier tools and stop tool
                        existing_chat.tools = tools
                        if chat_name != "Forge" and tools:  # Only set stop tool for literary agents
                            existing_chat.stop_tool = tools[-1]
                        logger.info(f"→ Updated chat thread {chat_name} with tools and stop tool")
                    
                    db.commit()
                
        except Exception as e:
            logger.error(f"Startup failed: {str(e)}", exc_info=True)
            raise
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Running shutdown tasks...")
        # Clear the registry
        CallableRegistry()._registry.clear()
    
    # Include routers
    app.include_router(
        chat.router,
        prefix=settings.API_V1_STR
    )
    
    return app

app = create_application()

@app.get("/", response_model=dict)
async def root():
    return {
        "message": "Welcome to the Chat API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }

@app.get("/health", response_model=dict)
async def health_check():
    try:
        # Test database connection
        with Session(engine) as db:
            db.exec(select(Tool).limit(1))
        
        # Test CallableRegistry
        registry_status = "healthy" if CallableRegistry()._registry else "empty"
        
        return {
            "status": "healthy",
            "database": "connected",
            "callable_registry": registry_status,
            "registered_functions": list(CallableRegistry()._registry.keys())
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }