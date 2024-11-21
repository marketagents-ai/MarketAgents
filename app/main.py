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
from abstractions.inference.sql_models import Tool, CallableRegistry
from abstractions.hub.callable_tools import DEFAULT_CALLABLE_TOOLS
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
    
    @app.on_event("startup")
    async def startup_event():
        logger.info("Running startup tasks...")
        try:
            # Create database tables
            logger.info("Creating database tables...")
            SQLModel.metadata.create_all(engine)
            
            # Register tools
            with Session(engine) as db:
                register_default_tools(db)
                
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