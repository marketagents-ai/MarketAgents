import logging
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from contextlib import asynccontextmanager
import httpx
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn.error")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    logger.info("Starting up the server...")
    yield
    # Shutdown code (if any)
    logger.info("Shutting down the server...")

app = FastAPI(lifespan=lifespan)

# Define your model configurations
model_configs = {
    "Hermes-3": {
        "api_url": "https://hermes-70b.legendarywou.com/v1/chat/completions",
        "api_key": "MarketAgents",
        "model": "/models/Hermes-3-Llama-3.1-70B-FP8",
    },
    "llama-3.1-70b": {
        "api_url": "https://llama3-70b.legendarywou.com/v1/chat/completions",
        "api_key": "MarketAgents",
        "model": "/models/Meta-Llama-3.1-70B-Instruct-FP8",
    },
}

# Local API key for authentication
API_KEY = "MarketAgents"

security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        logger.warning("Invalid API Key provided")
        raise HTTPException(status_code=403, detail="Invalid API Key")

# Add middleware to log requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Received request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status code: {response.status_code}")
    return response

@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(verify_api_key)
):
    request_body = await request.json()
    model_name = request_body.get("model")
    
    logger.info(f"Received request for model: {model_name}")
    
    if not model_name:
        logger.error("Model name is missing in the request.")
        raise HTTPException(status_code=400, detail="Model name is required.")

    # Check if the model is in our configs
    config = model_configs.get(model_name)
    if not config:
        logger.error(f"Model '{model_name}' is not available.")
        raise HTTPException(status_code=400, detail=f"Model '{model_name}' is not available.")
    
    # Prepare the request for the backend API
    backend_request_body = request_body.copy()
    # Replace the model name with the actual model identifier expected by the backend
    backend_request_body["model"] = config["model"]
    
    # Extract headers
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}",
    }
    
    # Send the request to the backend API
    async with httpx.AsyncClient() as client:
        try:
            backend_response = await client.post(
                config["api_url"],
                json=backend_request_body,
                headers=headers,
                timeout=60.0  # Adjust timeout as needed
            )
            backend_response.raise_for_status()
            logger.info(f"Successfully got response from backend model '{model_name}'.")
        except httpx.HTTPError as exc:
            logger.error(f"Error while requesting backend model '{model_name}': {exc}")
            raise HTTPException(status_code=500, detail=str(exc))
    
    # Return the response from the backend API
    return backend_response.json()

# If running directly, start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_router:app", host="127.0.0.1", port=8000, reload=True)
