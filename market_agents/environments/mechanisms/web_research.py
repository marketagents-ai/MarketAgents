from typing import Dict, List, Any, Optional, Type, Union
from market_agents.orchestrators.config import WebResearchConfig
from market_agents.orchestrators.parallel_cognitive_steps import ParallelCognitiveProcessor
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from market_agents.environments.mechanisms.research import ResearchAction
from market_agents.web_search.url_processor import URLFetcher
from market_agents.web_search.web_search_config import WebSearchConfig
from market_agents.environments.environment import (
    EnvironmentHistory,
    MultiAgentEnvironment, 
    Mechanism,
    LocalAction,
    LocalObservation,
    GlobalAction,
    GlobalObservation,
    LocalEnvironmentStep,
    EnvironmentStep,
    ActionSpace,
    ObservationSpace,
    StrAction
)
from minference.lite.models import CallableTool, StructuredTool
from minference.caregistry import CallableRegistry

from market_agents.web_search.web_search_manager import SearchManager
from market_agents.web_search.content_extractor import ContentExtractor

logger = logging.getLogger(__name__)
CallableRegistry._logger = logger

class WebSearchResult(BaseModel):
    """Structure for a single search result"""
    url: str
    title: str
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class WebSearchLocalObservation(LocalObservation):
    """Local observation for a specific agent"""
    agent_id: str
    observation: Dict[str, Any]
    status: str = "pending"
    search_results: Optional[List[Dict[str, str]]] = None

    def dict(self, *args, **kwargs):
        """Custom dict method to handle nested observation"""
        d = super().dict(*args, **kwargs)
        if self.observation:
            d['observation'] = self.observation
        return d

class WebSearchGlobalObservation(GlobalObservation):
    """Global observation containing all agent observations"""
    observations: Dict[str, WebSearchLocalObservation]

class WebSearchMechanism(Mechanism):
    """Mechanism that manages web search workflow"""
    search_manager: Optional[SearchManager] = Field(default=None, exclude=True)
    search_config: Optional[WebSearchConfig] = Field(default=None, exclude=True)
    content_extractor: Optional[ContentExtractor] = Field(default=None, exclude=True)
    url_fetcher: Optional[URLFetcher] = Field(default=None, exclude=True)
    current_round: int = Field(default=0, description="Current search round")
    max_rounds: int = Field(default=3, description="Maximum search rounds")
    cohorts: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Mapping of cohort IDs to lists of agents"
    )
    search_history: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=lambda: {"default": []},
        description="History of search results by cohort"
    )
    current_query: str = ""

    model_config = {
        "arbitrary_types_allowed": True,
        "extra": "allow"
    }

    def __init__(self, **data):
        super().__init__(**data)
        
        # Handle summary model
        summary_model = data.get('summary_model')
        if summary_model:
            if isinstance(summary_model, type) and issubclass(summary_model, BaseModel):
                self.summary_model = summary_model
            else:
                raise ValueError("summary_model must be a Pydantic model class")
        
        self.search_config = WebSearchConfig(
            urls_per_query=data.get('urls_per_query', 3)
        )
            
        self.search_manager = SearchManager(config=self.search_config)
        self.content_extractor = ContentExtractor(config=self.search_config)
        self.url_fetcher = URLFetcher(config=self.search_config, prompts={})

    def step(
        self,
        action: Union[GlobalAction, str],
        cohort_id: Optional[str] = None
    ) -> Union[LocalEnvironmentStep, EnvironmentStep]:
        """Process agent actions in workflow sequence"""
        # Use provided cohort_id or default
        effective_cohort = cohort_id if cohort_id else "default"
        
        # Initialize cohort's search history if needed
        if effective_cohort not in self.search_history:
            self.search_history[effective_cohort] = []

        self.current_round += 1
        done = (self.current_round >= self.max_rounds)

        if isinstance(action, GlobalAction):
            observations = {}
            
            for agent_id, agent_action in action.actions.items():
                # Ensure action data is serializable
                if hasattr(agent_action, 'model_dump'):
                    action_data = agent_action.model_dump()
                    # Remove any ModelMetaclass references if present
                    if 'summary_model' in action_data:
                        action_data['summary_model'] = action_data['summary_model'].__name__ if hasattr(action_data['summary_model'], '__name__') else str(action_data['summary_model'])
                else:
                    action_data = str(agent_action)
                
                obs_data = {
                    "action": action_data,
                    "round": self.current_round,
                    "status": "success"
                }
                
                obs = WebSearchLocalObservation(
                    agent_id=agent_id,
                    observation=obs_data,
                    status="success",
                    search_results=[]
                )
                observations[agent_id] = obs

            step_result = EnvironmentStep(
                global_observation=GlobalObservation(observations=observations),
                reward=1.0,
                done=done,
                info={
                    "round": self.current_round,
                    "actions": {k: str(v) for k, v in action.actions.items()},
                    "cohort_id": effective_cohort
                }
            )
            self.last_step = step_result
            return step_result

    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current global state, filtered by agent's cohort if specified."""
        state = {
            "current_round": self.current_round,
            "max_rounds": self.max_rounds,
            "current_query": self.current_query,
            "summary_model": self.summary_model.__name__ if hasattr(self, 'summary_model') and self.summary_model else None

        }

        if self.cohorts:
            if agent_id:
                # Find agent's cohort
                cohort_id = next(
                    (cid for cid, agents in self.cohorts.items() 
                    if any(a.id == agent_id for a in agents)),
                    None
                )
                if cohort_id:
                    state.update({
                        "search_history": self.search_history.get(cohort_id, []),
                        "cohort_id": cohort_id,
                        "cohort_agents": [a.id for a in self.cohorts[cohort_id]]
                    })
            else:
                # Return all cohorts' information
                state.update({
                    "cohorts": {cid: [a.id for a in agents] for cid, agents in self.cohorts.items()},
                    "search_history": self.search_history
                })
        else:
            # Not using cohorts, return default history
            state["search_history"] = self.search_history.get("default", [])

        return state

    def reset(self) -> None:
        """Reset mechanism state"""
        self.current_round = 0
        self.search_history.clear()
        self.search_history["default"] = []
        self.current_query = ""

class WebResearchActionSpace(ActionSpace):
    """Action space that handles both web search and research summary actions"""
    summary_model: Optional[Type[BaseModel]] = None
    mechanism: WebSearchMechanism = Field(
        ..., 
        description="Mechanism that handles web search operations"
    )
    workflow: bool = Field(
        default=True,
        description="Whether tools should be executed sequentially as a workflow"
    )

    def __init__(self, mechanism: WebSearchMechanism, summary_model: Type[BaseModel] = None, workflow:bool = True, **data):
        data.update({
            "mechanism": mechanism,
            "workflow": workflow
        })
        super().__init__(**data)

        print(f"WebResearchActionSpace initialized with workflow={workflow}")
        
        self.summary_model = summary_model
        
        # Create web search tool
        web_search_tool = CallableTool.from_callable(
            func=self.execute_web_search,
            name="web_search",
            docstring="Execute web search and return results",
            strict_schema=True
        )
        
        # Create summary tool based on model or string action
        if summary_model:
            summary_tool = StructuredTool(
                json_schema=summary_model.model_json_schema(),
                name="research_summary",
                description="Generate research summary from search results"
            )
        else:
            summary_tool = StructuredTool(
                json_schema=StrAction.model_json_schema(),
                name="text_summary",
                description="Generate text summary from search results"
            )

        # Both tools defined in action space
        self.allowed_actions = [
            web_search_tool,
            summary_tool
        ]

    async def execute_web_search(
        self,
        query: str,
        num_results: Optional[int] = None
    ) -> List[WebSearchResult]:
        """Execute web search and return results"""
        try:
            self.mechanism.current_query = query
            
            urls = await self.mechanism.search_manager.get_urls_for_query(
                query,
                self.mechanism.search_config.urls_per_query)
            
            for url in urls:
                self.mechanism.search_manager.query_url_mapping[url] = query
            
            fetched_results = await self.mechanism.url_fetcher.process_urls(urls, self.mechanism.search_manager.query_url_mapping)
            
            search_results = [
                {
                    "url": fr.url,
                    "title": fr.title,
                    "content": fr.content.get('text', ''),
                    "timestamp": datetime.now().isoformat()
                }
                for fr in fetched_results if fr is not None
            ]

            print(search_results)
            
            return search_results
                
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            return []
        
    def get_action_schema(self) -> Dict[str, Any]:
        """Return JSON schema for both tools"""
        return {
            tool.name: tool.json_schema() 
            for tool in self.allowed_actions
        }
    
class WebSearchEnvironment(MultiAgentEnvironment):
    """Environment that manages web search operations"""
    name: str = Field(
        default="Web Search Environment",
        description="Name of the environment"
    )
    mechanism: WebSearchMechanism = Field(
        ...,
        description="Mechanism that handles web search operations"
    )
    action_space: WebResearchActionSpace = None

    internal_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Internal storage for global state"
    )
    summary_model: Optional[Type[BaseModel]] = Field(
        default=None,
        description="Optional Pydantic model for structuring research summaries"
    )
    initial_query: str = Field(
        ...,
        description="Initial search query to start the research with"
    )

    def __init__(self, **config):
        """Initialize environment with config parameters."""
        try:
            # Parse and validate config
            env_config = WebResearchConfig(**config)
            
            # Initialize mechanism with only the fields we need
            mechanism = WebSearchMechanism(
                urls_per_query=env_config.urls_per_query,
                summary_model=env_config.summary_model,
                max_rounds=env_config.max_rounds
            )

            # Initialize parent class with required fields
            super().__init__(
                name=env_config.name,
                initial_query=env_config.initial_query,
                mechanism=mechanism,
                action_space=WebResearchActionSpace(
                    mechanism=mechanism,
                    summary_model=env_config.summary_model
                ),
                observation_space=ObservationSpace()
            )

            # Initialize cognitive processor if ai_utils provided
            if 'ai_utils' in config and 'storage_service' in config:
                self.mechanism._cognitive_processor = ParallelCognitiveProcessor(
                    ai_utils=config['ai_utils'],
                    storage_service=config['storage_service'],
                    logger=logging.getLogger(__name__),
                    tool_mode=config.get('tool_mode', False)
                )

        except Exception as e:
            raise ValueError(f"Failed to initialize WebSearchEnvironment: {e}")
        
    def get_global_state(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Return the environment's global state with filtered mechanism state."""
        # Get mechanism state with agent_id
        mechanism_state = self.mechanism.get_global_state(agent_id) if agent_id else self.mechanism.get_global_state()
        
        return {
            **mechanism_state,
            "current_step": self.mechanism.current_round,
            "max_steps": self.mechanism.max_rounds
        }

    def reset(self) -> GlobalObservation:
        """Reset environment state and restore initial query"""
        self.internal_state = {}
        if hasattr(self.mechanism, 'current_query'):
            self.mechanism.current_query = self.initial_query
        self.mechanism.reset()
        return GlobalObservation(observations={})