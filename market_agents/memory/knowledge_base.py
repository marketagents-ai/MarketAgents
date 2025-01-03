from abc import ABC, abstractmethod
import json
import uuid
import re
from uuid import UUID
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from memory.embedding import MemoryEmbedder
from memory.setup_db import DatabaseConnection

class KnowledgeObject(BaseModel):
    knowledge_id: UUID = Field(default_factory=uuid.uuid4)
    content: str
    created_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

class KnowledgeChunk(BaseModel):
    text: str
    start: int
    end: int
    embedding: Optional[List[float]] = None
    knowledge_id: Optional[UUID] = None

class KnowledgeChunker(ABC):
    @abstractmethod
    def chunk(self, text: str) -> List[KnowledgeChunk]:
        pass

class MarketKnowledgeBase:
    """
    A base class for agent knowledge bases built with embeddings for semantic search.
    - method to splits text into chunks, embeds them, and stores them.
    - method to remove all stored knowledge.
    - methods to handle chunking and saving to the database.
    """

    def __init__(self, config, db_conn: DatabaseConnection, embedding_service: MemoryEmbedder, chunking_method:Optional[KnowledgeChunker] = None):
        self.config = config
        self.db = db_conn
        self.embedding_service = embedding_service
        self.chunking_method = chunking_method
        self.full_text = ""

    def ingest_knowledge(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> UUID:
        """Process a document: chunk, embed, and store as a KnowledgeObject."""
        chunks = self._chunk(text)
        embeddings = self.embedding_service.get_embeddings([c.text for c in chunks])

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        knowledge_id = self._save_knowledge_and_chunks(text, chunks, metadata)
        return knowledge_id
    
    def _chunk(self, text: str) -> List[KnowledgeChunk]:
        if self.chunking_method:
            return self.chunking_method.chunk(text)
        else:
            default_chunker = SemanticChunker(
                min_size=self.config.min_chunk_size,
                max_size=self.config.max_chunk_size
            )
            return default_chunker.chunk(text)

    def _save_knowledge_and_chunks(self, document_text: str, chunks: List[KnowledgeChunk], metadata: Optional[Dict[str, Any]]) -> UUID:
        """Save knowledge object and chunks to database."""
        knowledge_id = uuid.uuid4()
        self.db.connect()
        try:
            self.db.cursor.execute("""
                INSERT INTO knowledge_objects (knowledge_id, content, metadata)
                VALUES (%s, %s, %s)
                RETURNING created_at;
            """, (str(knowledge_id), document_text, json.dumps(metadata) if metadata else json.dumps({})))
            created_at = self.db.cursor.fetchone()[0]

            for chunk in chunks:
                self.db.cursor.execute("""
                    INSERT INTO knowledge_chunks (knowledge_id, text, start_pos, end_pos, embedding)
                    VALUES (%s, %s, %s, %s, %s);
                """, (str(knowledge_id), chunk.text, chunk.start, chunk.end, chunk.embedding))

            self.db.conn.commit()
            return knowledge_id
        except Exception as e:
            self.db.conn.rollback()
            raise e

    def clear_knowledge_base(self):
        """Clear all knowledge entries."""
        self.db.connect()
        try:
            self.db.cursor.execute("DELETE FROM knowledge_chunks;")
            self.db.cursor.execute("DELETE FROM knowledge_objects;")
            self.db.conn.commit()
        except Exception as e:
            self.db.conn.rollback()
            raise e
    
class SemanticChunker(KnowledgeChunker):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def chunk(self, text: str) -> List[KnowledgeChunk]:
        text = re.sub(r'\n{3,}', '\n\n', text)

        splits = [
            r'(?<=\.)\s*\n\n(?=[A-Z])',
            r'(?<=\n)\s*#{1,6}\s+[A-Z]',
            r'(?<=\n)[A-Z][^a-z]*?:\s*\n',
            r'(?<=\n\n)\s*(?:[-•\*]|\d+\.)\s+',
            r'(?<=\.)\s{2,}(?=[A-Z])',
            r'(?<=[\.\?\!])\s+(?=[A-Z])'
        ]

        pattern = '|'.join(splits)
        segments = re.split(pattern, text.strip())

        chunks = []
        current_chunk = []
        current_length = 0
        current_pos = 0
        chunk_start = 0

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            if current_length + len(segment) > self.max_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(KnowledgeChunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

            current_chunk.append(segment)
            current_length += len(segment)
            current_pos += len(segment) + 1

            if current_length >= self.min_size:
                chunk_text = ' '.join(current_chunk)
                chunks.append(KnowledgeChunk(
                    text=chunk_text,
                    start=chunk_start,
                    end=chunk_start + len(chunk_text)
                ))
                current_chunk = []
                current_length = 0
                chunk_start = current_pos

        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(KnowledgeChunk(
                text=chunk_text,
                start=chunk_start,
                end=chunk_start + len(chunk_text)
            ))

        return chunks

if __name__ == "__main__":
    import os
    from memory.config import load_config_from_yaml
    # Load configuration and initialize services
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, "memory_config.yaml")

    config = load_config_from_yaml(config_path)
    db_conn = DatabaseConnection(config)
    embedder = MemoryEmbedder(config)
    
    # Initialize MarketKnowledgeBase
    knowledge_base = MarketKnowledgeBase(config, db_conn, embedder)
    
    # Test document for ingestion
    test_doc = """
    Market Analysis Report - Q4 2023
    
    The technology sector showed strong performance in Q4 2023. Cloud computing companies reported significant growth, with major players expanding their market share. AI-driven solutions saw increased adoption across industries.
    
    Key Highlights:
    • Cloud revenue grew by 25% YoY
    • Enterprise AI adoption increased 40%
    • Cybersecurity spending up 15%
    
    Market leaders continued to invest heavily in R&D, focusing on next-generation technologies. The semiconductor shortage showed signs of easing, though supply chain challenges persist in some areas.
    """

    try:
        # Ingest the test document: chunk, embed, and store in knowledge base
        metadata = {"source": "test_document", "category": "financial_report"}
        knowledge_id = knowledge_base.ingest_knowledge(test_doc, metadata=metadata)
        print(f"Successfully ingested knowledge object with ID: {knowledge_id}")

        # Verify that the document (knowledge object) was stored
        db_conn.cursor.execute("SELECT content, metadata FROM knowledge_objects WHERE knowledge_id = %s", (str(knowledge_id),))
        stored_doc = db_conn.cursor.fetchone()
        print("\nStored knowledge object content and metadata:")
        if stored_doc:
            doc_content, doc_metadata = stored_doc
            print("Content:\n", doc_content)
            print("Metadata:", doc_metadata)
        else:
            print("Knowledge object not found in database.")
        
        # Retrieve and display chunks
        db_conn.cursor.execute("SELECT text, embedding FROM knowledge_chunks WHERE knowledge_id = %s", (str(knowledge_id),))
        chunks = db_conn.cursor.fetchall()
        print(f"\nRetrieved {len(chunks)} chunks from the database:")
        for i, (chunk_text, chunk_embedding) in enumerate(chunks, 1):
            print(f"\nChunk {i}:")
            print("Text:", chunk_text)
            print("Embedding present:", "Yes" if chunk_embedding is not None else "No")
            
    except Exception as e:
        print(f"Error during document ingestion or retrieval: {e}")
    finally:
        db_conn.conn.close()