import sqlite3
import json
from typing import List, Dict, Any
from datetime import datetime
from .search import BM25Searcher
from .config import Config

class LongTermMemory:
    def __init__(self, config: Config):
        self.config = config
        self.conn = sqlite3.connect(config.db_path)
        self.create_table()
        self.searcher = BM25Searcher(self)

    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            access_count INTEGER DEFAULT 1
        )
        ''')
        self.conn.commit()

    def add_memory(self, content: str, metadata: Dict[str, Any] = None) -> int:
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO memories (content, metadata) VALUES (?, ?)",
            (content, json.dumps(metadata) if metadata else None)
        )
        self.conn.commit()
        return cursor.lastrowid

    def get_memories(self, limit: int = 1000) -> List[Dict[str, Any]]:
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT id, content, metadata, created_at, last_accessed, access_count FROM memories ORDER BY last_accessed DESC LIMIT ?",
            (limit,)
        )
        return [
            {
                "id": row[0],
                "content": row[1],
                "metadata": json.loads(row[2]) if row[2] else None,
                "created_at": row[3],
                "last_accessed": row[4],
                "access_count": row[5]
            }
            for row in cursor.fetchall()
        ]

    def update_access(self, memory_id: int):
        cursor = self.conn.cursor()
        cursor.execute(
            "UPDATE memories SET last_accessed = ?, access_count = access_count + 1 WHERE id = ?",
            (datetime.now().isoformat(), memory_id)
        )
        self.conn.commit()

    def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        memories = self.get_memories(limit=self.config.max_search_limit)
        results = self.searcher.search(query, memories, top_k)
        
        for result in results:
            self.update_access(result["id"])
        
        return results

    def forget_memories(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM memories 
            WHERE julianday('now') - julianday(created_at) > ? 
            AND access_count < ?
            """,
            (self.config.forget_threshold_days, self.config.forget_access_threshold)
        )
        self.conn.commit()

    def close(self):
        self.conn.close()