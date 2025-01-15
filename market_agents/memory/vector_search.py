import json
from typing import List
from pydantic import BaseModel

class RetrievedMemory(BaseModel):
    text: str
    similarity: float
    context: str = ""


class MemoryRetriever:
    """
    MemoryRetriever provides methods to search stored documents or agent memories
    based on embedding similarity, dynamically referencing tables.
    """
    def __init__(self, config, db_conn, embedding_service):
        self.config = config
        self.db = db_conn
        self.embedding_service = embedding_service
        self.full_text = ""

    def _sanitize_id(self, agent_id: str) -> str:
        """Sanitize agent ID for table names"""
        return agent_id.replace('-', '_')

    def search_knowledge_base(self, table_prefix: str, query: str, top_k: int = None) -> List[RetrievedMemory]:
        """
        Search a specific knowledge base for relevant content based on semantic similarity.
        """
        self.db.connect()
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k

        knowledge_chunks_table = f"{table_prefix}_knowledge_chunks"
        knowledge_objects_table = f"{table_prefix}_knowledge_objects"

        self.db.cursor.execute(f"""
            WITH ranked_chunks AS (
                SELECT DISTINCT ON (c.text)
                    c.id, c.text, c.start_pos, c.end_pos, k.content,
                    (1 - (c.embedding <=> %s::vector)) AS similarity
                FROM {knowledge_chunks_table} c
                JOIN {knowledge_objects_table} k ON c.knowledge_id = k.knowledge_id
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

    def search_agent_cognitive_memory(self, agent_id: str, query: str, top_k: int = None) -> List[RetrievedMemory]:
        """Search a specific agent's cognitive memory"""
        self.db.connect()
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k
        
        safe_id = self._sanitize_id(agent_id)
        agent_cognitive_table = f"agent_{safe_id}_cognitive"

        self.db.cursor.execute(f"""
            SELECT content,
                   (1 - (embedding <=> %s::vector)) AS similarity
            FROM {agent_cognitive_table}
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, top_k))

        results = []
        rows = self.db.cursor.fetchall()
        for row in rows:
            content, sim = row
            results.append(RetrievedMemory(text=content, similarity=sim))
        return results

    def search_agent_episodic_memory(self, agent_id: str, query: str, top_k: int = None) -> List[RetrievedMemory]:
        """Search an agent's episodic memory"""
        self.db.connect()
        query_embedding = self.embedding_service.get_embeddings(query)
        top_k = top_k or self.config.top_k

        safe_id = self._sanitize_id(agent_id)
        agent_episodic_table = f"agent_{safe_id}_episodic"

        self.db.cursor.execute(f"""
            SELECT 
                memory_id, 
                task_query, 
                cognitive_steps,
                total_reward, 
                strategy_update, 
                metadata,
                created_at,
                (1 - (embedding <=> %s::vector)) AS similarity
            FROM {agent_episodic_table}
            ORDER BY similarity DESC
            LIMIT %s;
        """, (query_embedding, top_k))

        rows = self.db.cursor.fetchall()
        results = []
        for (mem_id, task_query, steps_json, total_reward, strategy_update, meta, created_at, sim) in rows:
            content_dict = {
                "memory_id": str(mem_id),
                "task_query": task_query,
                "cognitive_steps": steps_json,
                "total_reward": total_reward,
                "strategy_update": strategy_update,
                "metadata": meta,
                "created_at": created_at.isoformat()
            }
            content_str = json.dumps(content_dict)
            results.append(RetrievedMemory(text=content_str, similarity=sim, context=""))
        return results

    def _get_context(self, start: int, end: int, full_text: str) -> str:
        """
        Extracts context around a specific text chunk within a full document.
        """
        start_idx = max(0, start - self.config.context_window)
        end_idx = min(len(full_text), end + self.config.context_window)
        context = full_text[start_idx:end_idx].strip()
        if start_idx > 0:
            context = "..." + context
        if end_idx < len(full_text):
            context = context + "..."
        return context