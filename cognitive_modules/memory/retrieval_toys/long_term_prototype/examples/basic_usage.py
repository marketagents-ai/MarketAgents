from long_term_memory import LongTermMemory, Config
from long_term_memory.utils import format_memories_for_prompt

def main():
    config = Config(db_path="example_memory.db")
    ltm = LongTermMemory(config)

    # Add some memories
    ltm.add_memory("The capital of France is Paris.", {"category": "geography"})
    ltm.add_memory("Python is a high-level programming language.", {"category": "technology"})
    ltm.add_memory("The Earth revolves around the Sun.", {"category": "science"})

    # Search memories
    query = "France capital"
    results = ltm.search_memories(query, top_k=2)

    print(f"Search results for query: '{query}'")
    print(format_memories_for_prompt(results))

    # Forget old memories (this won't do anything in this example, but demonstrates the usage)
    ltm.forget_memories()

    ltm.close()

if __name__ == "__main__":
    main()