import argparse
import asyncio
from data_loader import load_data
from indexer import Indexer
from searcher import Searcher
from config import Config

async def main():
    parser = argparse.ArgumentParser(description="Advanced Distributed Search System")
    parser.add_argument("--config", required=True, help="Path to configuration file")
    parser.add_argument("--action", choices=["index", "search"], required=True, help="Action to perform")
    parser.add_argument("--query", help="Search query (required for search action)")
    args = parser.parse_args()

    config = Config.load(args.config)

    if args.action == "index":
        data = await load_data(config.data_sources)
        indexer = Indexer(config)
        await indexer.build_index(data)
        indexer.save_index(config.index_path)
        print(f"Index built and saved to {config.index_path}")
    elif args.action == "search":
        if not args.query:
            raise ValueError("Query is required for search action")
        searcher = Searcher(config)
        searcher.load_index(config.index_path)
        results = await searcher.search(args.query)
        for rank, (item, score) in enumerate(results, 1):
            print(f"{rank}. Score: {score:.4f}")
            print(f"   Content: {item.content[:100]}...")
            print(f"   Metadata: {item.metadata}")
            print()

if __name__ == "__main__":
    asyncio.run(main())