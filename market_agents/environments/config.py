
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings

class EnvironmentConfig(BaseSettings):
    name: str = Field(
        ...,
        description="Name of the group chat environment"
    )
    api_url: Optional[str] = Field(
        default=None,
        description="API endpoint for the environment"
    )
    model_config = {
        "extra": "allow"
    }