# groupchat_api.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uvicorn
import random
import logging

from market_agents.memory.knowledge_base import KnowledgeObject
from market_agents.memory.memory import EpisodicMemoryObject, MemoryObject

app = FastAPI()
logger = logging.getLogger("groupchat_api")
logging.basicConfig(level=logging.INFO)

# In-memory storage (can be replaced with a database)
cohorts: Dict[str, List[str]] = {}
topics: Dict[str, str] = {}
messages: Dict[str, List[Dict]] = {}
agents: Dict[str, Dict] = {}
proposers: Dict[str, str] = {}
cognitive_memory: Dict[str, List[Dict]] = {}
episodic_memory: Dict[str, List[Dict]] = {}
knowledge_base: Dict[str, Dict] = {}

# Pydantic models
class Agent(BaseModel):
    id: str
    index: int

class Message(BaseModel):
    agent_id: str
    content: str
    cohort_id: str
    round_num: Optional[int] = 1
    sub_round_num: Optional[int] = 1

class TopicProposal(BaseModel):
    agent_id: str
    topic: str
    cohort_id: str
    round_num: Optional[int] = 1

class CohortFormationRequest(BaseModel):
    agent_ids: List[str]
    cohort_size: int

class CohortResponse(BaseModel):
    cohort_id: str
    agent_ids: List[str]

class ProposerSelectionRequest(BaseModel):
    cohort_id: str
    agent_ids: List[str]

class ProposerResponse(BaseModel):
    cohort_id: str
    proposer_id: str

class GetMessagesResponse(BaseModel):
    cohort_id: str
    messages: List[Dict]

class GetTopicResponse(BaseModel):
    cohort_id: str
    topic: str

class CognitiveMemoryParams(BaseModel):
    limit: Optional[int] = 10
    cognitive_step: Optional[Union[str, List[str]]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

class EpisodicMemoryParams(BaseModel):
    agent_id: str
    query: str
    top_k: Optional[int] = 5

class KnowledgeQueryParams(BaseModel):
    query: str
    top_k: Optional[int] = 5
    table_prefix: Optional[str] = None

# Endpoint to register agents (optional)
@app.post("/register_agent")
def register_agent(agent: Agent):
    agents[agent.id] = agent.dict()
    logger.info(f"Agent registered: {agent.id}")
    return {"message": "Agent registered"}

# Endpoint to form cohorts
@app.post("/form_cohorts", response_model=List[CohortResponse])
def form_cohorts(request: CohortFormationRequest):
    global cohorts
    agent_ids = request.agent_ids
    cohort_size = request.cohort_size
    random.shuffle(agent_ids)
    cohorts = {}
    cohort_responses = []
    for i in range(0, len(agent_ids), cohort_size):
        cohort_agent_ids = agent_ids[i:i + cohort_size]
        cohort_id = f"cohort_{i // cohort_size}"
        cohorts[cohort_id] = cohort_agent_ids
        cohort_responses.append(CohortResponse(cohort_id=cohort_id, agent_ids=cohort_agent_ids))
        # Initialize messages and topics for the cohort
        messages[cohort_id] = []
        topics[cohort_id] = ""
        proposers[cohort_id] = ""  # Initialize proposer
        logger.info(f"Cohort formed: {cohort_id} with agents {cohort_agent_ids}")
    return cohort_responses

# Endpoint to select a topic proposer for a cohort
@app.post("/select_proposer", response_model=ProposerResponse)
def select_proposer(request: ProposerSelectionRequest):
    cohort_id = request.cohort_id
    agent_ids = request.agent_ids
    if cohort_id not in cohorts:
        raise HTTPException(status_code=404, detail="Cohort not found")
    # Rotate proposers within the cohort
    current_proposer = proposers.get(cohort_id)
    if current_proposer in agent_ids:
        current_index = agent_ids.index(current_proposer)
        next_index = (current_index + 1) % len(agent_ids)
        proposer_id = agent_ids[next_index]
    else:
        proposer_id = random.choice(agent_ids)
    proposers[cohort_id] = proposer_id
    logger.info(f"Proposer selected for {cohort_id}: {proposer_id}")
    return ProposerResponse(cohort_id=cohort_id, proposer_id=proposer_id)

# Endpoint for agents to propose topics
@app.post("/propose_topic")
def propose_topic(proposal: TopicProposal):
    cohort_id = proposal.cohort_id
    if cohort_id not in cohorts:
        raise HTTPException(status_code=404, detail="Cohort not found")
    if proposal.agent_id != proposers.get(cohort_id):
        raise HTTPException(status_code=403, detail="Agent is not the proposer for this cohort")
    topics[cohort_id] = proposal.topic
    logger.debug(f"Topic proposed for {cohort_id} by {proposal.agent_id}: {proposal.topic}")
    return {"message": "Topic accepted"}

# Endpoint to get the topic for a cohort
@app.get("/get_topic/{cohort_id}", response_model=GetTopicResponse)
def get_topic(cohort_id: str):
    topic = topics.get(cohort_id)
    if topic == "":
        raise HTTPException(status_code=404, detail="Topic not set for this cohort")
    return GetTopicResponse(cohort_id=cohort_id, topic=topic)

# Endpoint for agents to post messages
@app.post("/post_message")
def post_message(message: Message):
    cohort_id = message.cohort_id
    if cohort_id not in cohorts:
        raise HTTPException(status_code=404, detail="Cohort not found")
    if message.agent_id not in cohorts[cohort_id]:
        raise HTTPException(status_code=403, detail="Agent not part of this cohort")
    message_entry = {
        "agent_id": message.agent_id,
        "content": message.content,
        "round_num": message.round_num,
        "sub_round_num": message.sub_round_num,
    }
    messages[cohort_id].append(message_entry)
    logger.debug(f"Message from {message.agent_id} in {cohort_id}: {message.content}")
    return {"message": "Message posted"}

# Endpoint to get messages for a cohort
@app.get("/get_messages/{cohort_id}", response_model=GetMessagesResponse)
def get_messages(cohort_id: str):
    if cohort_id not in messages:
        raise HTTPException(status_code=404, detail="No messages for this cohort")
    return GetMessagesResponse(cohort_id=cohort_id, messages=messages[cohort_id])

# Endpoint to get agents in a cohort
@app.get("/get_cohort_agents/{cohort_id}")
def get_cohort_agents(cohort_id: str):
    if cohort_id not in cohorts:
        raise HTTPException(status_code=404, detail="Cohort not found")
    return {"cohort_id": cohort_id, "agent_ids": cohorts[cohort_id]}

# Endpoint to get the current proposer for a cohort
@app.get("/get_proposer/{cohort_id}")
def get_proposer(cohort_id: str):
    proposer_id = proposers.get(cohort_id)
    if not proposer_id:
        raise HTTPException(status_code=404, detail="Proposer not set for this cohort")
    return {"cohort_id": cohort_id, "proposer_id": proposer_id}

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/memory/cognitive")
def add_cognitive_memory(memory_object: MemoryObject):
    """Adds a cognitive memory entry for an agent."""
    agent_id = memory_object.agent_id
    if agent_id not in cognitive_memory:
        cognitive_memory[agent_id] = []
    cognitive_memory[agent_id].append(memory_object.dict())
    logger.debug(f"Added cognitive memory for agent {agent_id}")
    return {"message": "Cognitive memory added"}


@app.post("/memory/episodic")
def add_episodic_memory(memory_object: EpisodicMemoryObject):
    """Adds an episodic memory entry for an agent."""
    agent_id = memory_object.agent_id
    if agent_id not in episodic_memory:
        episodic_memory[agent_id] = []
    episodic_memory[agent_id].append(memory_object.dict())
    logger.debug(f"Added episodic memory for agent {agent_id}")
    return {"message": "Episodic memory added"}


@app.get("/memory/cognitive/{agent_id}")
def get_cognitive_memory(agent_id: str, params: CognitiveMemoryParams):
    """Retrieves cognitive memory for an agent based on optional query parameters."""
    if agent_id not in cognitive_memory:
        raise HTTPException(status_code=404, detail="No cognitive memory found for this agent")

    memories = cognitive_memory.get(agent_id, [])
    filtered_memories = []

    for mem in memories:
        if params.cognitive_step:
            if isinstance(params.cognitive_step, str):
                if mem.get('cognitive_step') != params.cognitive_step:
                    continue
            elif isinstance(params.cognitive_step, list):
                if mem.get('cognitive_step') not in params.cognitive_step:
                    continue
        if params.metadata_filters:
            for key, value in params.metadata_filters.items():
                if mem.get('metadata', {}).get(key) != value:
                    continue

        filtered_memories.append(mem)
    if params.start_time or params.end_time:

        filtered_memories_time = []
        for mem in filtered_memories:

            try:
                memory_time = datetime.fromisoformat(mem.get('metadata', {}).get('timestamp'))

                if params.start_time and memory_time < params.start_time:
                    continue

                if params.end_time and memory_time > params.end_time:
                    continue

                filtered_memories_time.append(mem)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid date time or no timestamp, error: {e}")
                continue

        filtered_memories = filtered_memories_time

    return filtered_memories[:params.limit]


@app.get("/memory/episodic/{agent_id}")
def get_episodic_memory(agent_id: str, query: str, top_k: int = 5):
    """Retrieves episodic memory episodes for an agent (PLACEHOLDER)."""
    if agent_id not in episodic_memory:
        raise HTTPException(status_code=404, detail="No episodic memory found for this agent")
    # In a real implementation, you would perform a similarity search
    # Here, we just return the first 'top_k' memories.
    logger.debug(f"Retrieved episodic memory for agent {agent_id} with query {query} top k {top_k}")
    return episodic_memory.get(agent_id, [])[:top_k]


@app.delete("/memory/{agent_id}")
def delete_memory(agent_id: str):
    """Deletes all memory for a specific agent."""
    if agent_id in cognitive_memory:
        del cognitive_memory[agent_id]
    if agent_id in episodic_memory:
        del episodic_memory[agent_id]

    logger.debug(f"Deleted memory for agent {agent_id}")
    return {"message": f"Memory for agent {agent_id} deleted"}


# Knowledge Base Endpoints
@app.post("/knowledge/ingest")
def ingest_knowledge(knowledge_object: KnowledgeObject):
    """Ingests a knowledge entry into the knowledge base."""
    knowledge_id = knowledge_object.knowledge_id
    knowledge_base[knowledge_id] = knowledge_object.dict()
    logger.debug(f"Ingested knowledge with id {knowledge_id}")
    return {"message": "Knowledge ingested"}


@app.get("/knowledge/search")
def search_knowledge(query: str, top_k: int = 5, table_prefix: Optional[str] = None):
    """Searches the knowledge base (PLACEHOLDER)."""
    # In a real implementation, you'd use the table_prefix to target a specific kb.
    # You would perform similarity search logic here, but we are returning the first k items for this placeholder implementation
    results = [value for value in knowledge_base.values()][:top_k]
    logger.debug(f"Searched knowledge base with query: {query}, prefix {table_prefix}")

    return results


@app.delete("/knowledge/{knowledge_id}")
def delete_knowledge(knowledge_id: str):
    """Deletes a knowledge entry from the knowledge base."""
    if knowledge_id in knowledge_base:
        del knowledge_base[knowledge_id]
        logger.debug(f"Deleted knowledge with id {knowledge_id}")
        return {"message": f"Knowledge with id {knowledge_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="knowledge not found")

# Run the FastAPI application
if __name__ == "__main__":
    uvicorn.run("groupchat_api:app", host="0.0.0.0", port=8001, reload=True)
