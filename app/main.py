from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.base import BaseHTTPMiddleware
from app.api.v1.endpoints import chat
from app.core.config import settings
from app.db.session import engine
from sqlmodel import SQLModel
from typing import Union

class SQLAlchemyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except SQLAlchemyError as e:
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": f"Database error: {str(e)}"}
            )

def create_application() -> FastAPI:
    app = FastAPI(
        title="Chat API",
        description="API for managing AI chat interactions",
        # Add version info
        version="1.0.0",
        # Add response model by default
        default_response_class=JSONResponse
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",  # React dev server
            "http://127.0.0.1:3000",  # Alternative React dev server URL
        ],
        allow_credentials=True,
        allow_methods=["*"],  # Allows all methods
        allow_headers=["*"],  # Allows all headers
        expose_headers=["*"]  # Expose all headers to the browser
    )
    
    # Add database middleware
    app.add_middleware(SQLAlchemyMiddleware)
    
    # Create database tables
    SQLModel.metadata.create_all(engine)
    
    # Include routers
    app.include_router(
        chat.router,
        prefix=settings.API_V1_STR
    )
    
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        error_msg = f"Unhandled error: {str(exc)}"
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": error_msg}
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
    return {
        "status": "healthy",
        "database": "connected"  # You might want to add actual DB health check
    }