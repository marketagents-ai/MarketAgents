from typing import List
from pydantic import BaseModel, Field
from market_agents.memory.config import MarketMemoryConfig
from market_agents.memory.memory import MemoryObject


class RetrievedMemory(BaseModel):
    text: str
    similarity: float
    context: str = ""

class MemoryRetriever:
    """
    MemoryQuery provides methods to search stored documents or agent memories based on embedding similarity.
    """
    def __init__(self, config, db_conn, embedding_service):
        self.config = config
        self.db = db_conn
        self.embedding_service = embedding_service
        self.full_text = ""

    def search_knowledge_base(self, query: str, top_k: int = None) -> List[RetrievedMemory]:
        self.db.connect()
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k

        self.db.cursor.execute("""
            WITH ranked_chunks AS (
                SELECT DISTINCT ON (c.text)
                    c.id, c.text, c.start_pos, c.end_pos, k.content,
                    (1 - (c.embedding <=> %s::vector)) AS similarity
                FROM knowledge_chunks c
                JOIN knowledge_objects k ON c.knowledge_id = k.knowledge_id
                WHERE (1 - (c.embedding <=> %s::vector)) >= %s
                ORDER BY c.text, similarity DESC
            )
            SELECT * FROM ranked_chunks
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, query_embedding, self.config.similarity_threshold, top_k))

        results = []
        rows = self.db.cursor.fetchall()
        for row in rows:
            _, text, start_pos, end_pos, full_content, sim = row
            self.full_text = full_content
            context = self._get_context(start_pos, end_pos, full_content)
            results.append(RetrievedMemory(text=text, similarity=sim, context=context))

        return results

    def search_agent_memory(self, agent_id: str, query: str, top_k: int = None) -> List[RetrievedMemory]:
        self.db.connect()
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k

        self.db.cursor.execute("""
            SELECT agent_id, content,
                   (1 - (embedding <=> %s::vector)) AS similarity
            FROM agent_memory
            WHERE agent_id = %s
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, agent_id, top_k))

        results = []
        rows = self.db.cursor.fetchall()
        for row in rows:
            _, content, sim = row
            results.append(RetrievedMemory(text=content, similarity=sim))

        return results

    def _get_context(self, start: int, end: int, full_text: str) -> str:
        start_idx = max(0, start - self.config.context_window)
        end_idx = min(len(full_text), end + self.config.context_window)
        context = full_text[start_idx:end_idx].strip()
        if start_idx > 0:
            context = "..." + context
        if end_idx < len(full_text):
            context = context + "..."
        return context


class LongTermMemory(BaseModel):
    memories: List[RetrievedMemory] = Field(default_factory=list)
    memory_retriever: MemoryRetriever

    def __init__(self, memory_config: MarketMemoryConfig, db_conn):
        super().__init__()
        embedder = MemoryEmbedder(memory_config)
        self.memory_retriever = MemoryRetriever(config=memory_config, db_conn=db_conn, embedding_service=embedder)
        self.memory_store = MarketMemory(memory_config, db_conn, embedder)

    def store_memory(self, memory_object: MemoryObject):
        self.memory_store.store_memory(memory_object)

    def retrieve_recent_memories(self, agent_id: str, query: str = None, top_k: int = None) -> List[RetrievedMemory]:
        return self.memory_retriever.search_agent_memory(agent_id=agent_id, query=query, top_k=top_k)

if __name__ == "__main__":
    import os
    from config import load_config_from_yaml, MarketMemoryConfig
    from setup_db import DatabaseConnection
    from embedding import MemoryEmbedder
    from knowledge_base import MarketKnowledgeBase
    from memory import MarketMemory, MemoryObject

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "memory_config.yaml")

    config = load_config_from_yaml(config_path)
    db_conn = DatabaseConnection(config)
    embedder = MemoryEmbedder(config)
    
    # Store some test documents first
    knowledge_base = MarketKnowledgeBase(config, db_conn, embedder)
    test_doc = """
    Q4 2023 Quarterly Earnings Report
    
    Revenue increased 15% year-over-year to $2.3B. 
    Operating margin expanded to 28%.
    Strong performance in cloud services division.
    Earnings per share of $1.45 exceeded analyst estimates.
    """
    knowledge_base.ingest_knowledge(test_doc)

    # Store some test agent memories
    memory_store = MarketMemory(config, db_conn, embedder)
    test_memory = MemoryObject(
        agent_id="crypto_agent_123",
        cognitive_step="reflection",
        content="Given recent market volatility, I'm shifting my strategy to focus more on stablecoins and established cryptocurrencies. The risk-reward ratio for altcoins seems unfavorable in current conditions."
    )
    memory_store.store_memory(test_memory)

    # Now perform the searches
    retriever = MemoryRetriever(config, db_conn, embedder)

    doc_results = retriever.search_knowledge_base("quarterly earnings")
    print("\nDocument Search Results:")
    for r in doc_results:
        print(f"Text: {r.text}\nSimilarity: {r.similarity}\nContext: {r.context}\n")

    agent_mem_results = retriever.search_agent_memory("crypto_agent_123", "strategy shift")
    print("\nAgent Memory Search Results:")
    for r in agent_mem_results:
        print(f"Text: {r.text}\nSimilarity: {r.similarity}\n")