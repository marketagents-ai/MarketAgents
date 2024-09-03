from reptarINDEX import Graph, build_graph, save_graph, load_graph
import asyncio

async def create_index(input_folder, output_file, chunk_size=200, overlap=50, embedding_model="all-minilm", max_depth=3, min_cluster_size=5, max_clusters=10):
    args = type('Args', (), {
        'input_folder': input_folder,
        'output_file': output_file,
        'chunk_size': chunk_size,
        'overlap': overlap,
        'embedding_model': embedding_model,
        'max_depth': max_depth,
        'min_cluster_size': min_cluster_size,
        'max_clusters': max_clusters
    })()
    
    graph, embeddings = await build_graph(args)
    save_graph(graph, embeddings, args)
    print(f"Graph built and saved to {output_file}")

async def search_index(input_file, query):
    graph, embeddings = load_graph(type('Args', (), {'input_file': input_file})())
    results = await graph.search(query, embeddings)
    return results

async def main():
    # Build the index
    await create_index(
        input_folder="/path/to/your/documents",
        output_file="index.reptar",
        chunk_size=200,
        overlap=50,
        embedding_model="all-minilm",
        max_depth=3,
        min_cluster_size=5,
        max_clusters=10
    )
    
    # Search the index
    query = "Your search query here"
    results = await search_index("index.reptar", query)
    
    # Process and display results
    for node, score, depth, parent_cluster in results:
        print(f"{'Cluster: ' if node.is_cluster else 'Document: '}{node.name}")
        print(f"Text: {node.text[:200]}...")
        print(f"Score: {score}")
        print(f"Depth: {depth}")
        print("---")

if __name__ == "__main__":
    asyncio.run(main())