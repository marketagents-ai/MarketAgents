from typing import Dict, Any
from pydantic import BaseModel, Field
from minference.lite.inference import RequestLimits

class OrchestratorRequestsLimits(BaseModel):
    """A simple container for multiple provider-based RequestLimits."""

    openai: RequestLimits = Field(default_factory=lambda: RequestLimits(provider="openai"))
    anthropic: RequestLimits = Field(default_factory=lambda: RequestLimits(provider="anthropic"))
    vllm: RequestLimits = Field(default_factory=lambda: RequestLimits(provider="vllm"))
    litellm: RequestLimits = Field(default_factory=lambda: RequestLimits(provider="litellm"))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OrchestratorRequestsLimits":
        """Build OrchestratorRequestLimits from a raw dictionary with keys like 'openai', 'anthropic', etc."""
        openai_data = data.get("openai", {})
        anthropic_data = data.get("anthropic", {})
        vllm_data = data.get("vllm", {})
        litellm_data = data.get("litellm", {})

        openai_req = RequestLimits(provider="openai", **openai_data)
        anthro_req = RequestLimits(provider="anthropic", **anthropic_data)
        vllm_req   = RequestLimits(provider="vllm", **vllm_data)
        litellm_req= RequestLimits(provider="litellm", **litellm_data)

        return cls(
            openai=openai_req,
            anthropic=anthro_req,
            vllm=vllm_req,
            litellm=litellm_req,
        )

    def to_provider_map(self) -> Dict[str, RequestLimits]:
        """Return a dictionary keyed by provider name """
        return {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "vllm": self.vllm,
            "litellm": self.litellm
        }