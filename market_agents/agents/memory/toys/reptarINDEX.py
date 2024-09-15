import argparse
import pickle
import json
from collections import defaultdict
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import os
import asyncio
from tqdm import tqdm
from colorama import init, Fore, Style
import tiktoken
import re
from nltk.tokenize import sent_tokenize
import yaml

# Initialize colorama
init(autoreset=True)

class GraphNode:
    def __init__(self, name, text="", is_folder=False, is_cluster=False, path="", start_pos=0, end_pos=0, chunk_id=None):
        self.name = name
        self.text = text
        self.is_folder = is_folder
        self.is_cluster = is_cluster
        self.children = []
        self.embedding = None
        self.path = path
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.chunk_id = chunk_id

class Graph:
    def __init__(self):
        self.root = GraphNode("root", is_folder=True)
        self.inverted_index = defaultdict(set)
        self.nodes = []
        self.clusters = []
        self.embedding_model = None
        self.config = None
        self.summary_model = None

    def add_node(self, node, parent=None):
        if parent is None:
            parent = self.root
        parent.children.append(node)
        self.nodes.append(node)
        
        # Update the path
        node.path = os.path.join(parent.path, node.name) if parent.path else node.name
        
        if not node.is_folder and not node.is_cluster:
            for word in node.text.lower().split():
                self.inverted_index[word].add(node)
        
        return node

    async def create_cluster(self, nodes):
        cluster = GraphNode("cluster", is_cluster=True)
        for node in nodes:
            cluster.children.append(node)
        
        cluster_text = " ".join([node.text for node in nodes])
        cluster.text = generate_summary(cluster_text, self.config, self.summary_model)
        cluster.embedding = np.mean([node.embedding for node in nodes], axis=0)
        
        # Generate and save summary embedding
        cluster.summary_embedding = (await get_embeddings([cluster.text], self.embedding_model))[0]
        
        self.nodes.append(cluster)
        self.clusters.append(cluster)
        return cluster

    async def search(self, query, embeddings, config=None, summary_model=None):
        query_words = query.lower().split()
        query_embedding = (await get_embeddings([query], embeddings['model']))[0]
        
        def recursive_search(node, depth=0, parent_cluster=None):
            results = []
            if node.is_cluster:
                cluster_sim = cosine_similarity([query_embedding], [node.embedding])[0][0] if node.embedding is not None else 0
                summary_sim = cosine_similarity([query_embedding], [node.summary_embedding])[0][0] if node.summary_embedding is not None else 0
                score = 0.6 * cluster_sim + 0.4 * summary_sim
                results.append((node, round(score, 2), depth, parent_cluster))
                for child in node.children:
                    results.extend(recursive_search(child, depth + 1, node))
            elif not node.is_folder:
                if any(word in node.text.lower() for word in query_words):
                    score = cosine_similarity([query_embedding], [node.embedding])[0][0] if node.embedding is not None else 0
                    results.append((node, round(score, 2), depth, parent_cluster))
            else:
                for child in node.children:
                    results.extend(recursive_search(child, depth + 1, parent_cluster))
            return results

        all_results = recursive_search(self.root)
        ranked_results = sorted(all_results, key=lambda x: x[1], reverse=True)
        
        return ranked_results[:10]  # Increase the number of results to include more context

async def recursive_clustering(graph, nodes, depth=0, max_depth=3, min_cluster_size=5, max_clusters=10):
    if depth >= max_depth or len(nodes) <= min_cluster_size:
        return nodes

    chunk_vectors = [node.embedding for node in nodes]
    distance_matrix = pdist(chunk_vectors, 'euclidean')
    Z = linkage(distance_matrix, 'ward')
    
    n_clusters = min(max(2, len(nodes) // min_cluster_size), max_clusters)
    clusters = fcluster(Z, t=n_clusters, criterion='maxclust')

    new_clusters = []
    for cluster_id in set(clusters):
        cluster_nodes = [node for node, c in zip(nodes, clusters) if c == cluster_id]
        if len(cluster_nodes) > 1:
            cluster = await graph.create_cluster(cluster_nodes)
            sub_clusters = await recursive_clustering(graph, cluster_nodes, depth + 1, max_depth, min_cluster_size, max_clusters)
            cluster.children = sub_clusters
            new_clusters.append(cluster)
        else:
            new_clusters.extend(cluster_nodes)

    return new_clusters

async def get_embeddings(texts, model):
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    try:
        embeddings = []
        for text in tqdm(texts, desc=f"{Fore.CYAN}Generating embeddings", unit="chunk"):
            response = await asyncio.to_thread(client.embeddings.create, model=model, input=[text])
            embeddings.append(response.data[0].embedding)
        return embeddings
    except Exception as e:
        print(f"{Fore.RED}Error fetching embeddings: {str(e)}")
        return [None] * len(texts)

def chunk_text(text, target_chunk_size, overlap=50):
    encoding = tiktoken.get_encoding("cl100k_base")
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_chunk_tokens = 0

    def safe_encode(text):
        try:
            return encoding.encode(text)
        except ValueError:
            return encoding.encode(text, allowed_special="all")

    for sentence in sentences:
        sentence_tokens = len(safe_encode(sentence))
        
        if current_chunk_tokens + sentence_tokens > target_chunk_size:
            if not current_chunk:
                # If the sentence is longer than target_chunk_size, split it
                words = sentence.split()
                while words:
                    chunk = []
                    chunk_tokens = 0
                    while words and chunk_tokens < target_chunk_size:
                        word = words.pop(0)
                        word_tokens = len(safe_encode(word))
                        if chunk_tokens + word_tokens <= target_chunk_size:
                            chunk.append(word)
                            chunk_tokens += word_tokens
                        else:
                            words.insert(0, word)
                            break
                    chunks.append(' '.join(chunk))
                    
                    # Handle overlap for long sentences
                    if words:
                        overlap_words = chunk[-overlap:]
                        overlap_tokens = len(safe_encode(' '.join(overlap_words)))
                        while overlap_tokens > target_chunk_size // 2:
                            overlap_words.pop(0)
                            overlap_tokens = len(safe_encode(' '.join(overlap_words)))
                        words = overlap_words + words
            else:
                chunks.append(' '.join(current_chunk))
                overlap_text = ' '.join(current_chunk[-overlap:])
                overlap_tokens = len(safe_encode(overlap_text))
                current_chunk = current_chunk[-overlap:] + [sentence]
                current_chunk_tokens = overlap_tokens + sentence_tokens
        else:
            current_chunk.append(sentence)
            current_chunk_tokens += sentence_tokens

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def load_yaml_config(yaml_file):
    if yaml_file and os.path.exists(yaml_file):
        with open(yaml_file, 'r') as file:
            return yaml.safe_load(file)
    else:
        # Default configuration
        return {
            "summary_system_prompt": "You are a helpful assistant that generates concise summaries.",
            "summary_user_prompt": "Summarize the following text in a single paragraph:\n\n{text}"
        }

def generate_summary(text, config, summary_model):
    client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
    response = client.chat.completions.create(
        model=summary_model,
        messages=[
            {"role": "system", "content": config['summary_system_prompt']},
            {"role": "user", "content": config['summary_user_prompt'].format(text=text)}
        ]
    )
    summary = response.choices[0].message.content.strip()
    print(f"{Fore.GREEN}Generated summary: {summary}")
    return summary

def process_folder(folder_path, parent_node, graph, args):
    encoding = tiktoken.get_encoding("cl100k_base")
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_node = GraphNode(item, is_folder=True, path=item_path)
            graph.add_node(folder_node, parent_node)
            print(f"{Fore.YELLOW}Processing folder: {item_path}")
            process_folder(item_path, folder_node, graph, args)
        elif item.endswith(('.md', '.txt')):
            with open(item_path, 'r', encoding='utf-8') as f:
                text = f.read()
            chunks = chunk_text(text, args.chunk_size, args.overlap)
            print(f"{Fore.BLUE}Processing file: {item_path}")
            print(f"{Fore.BLUE}Number of chunks: {len(chunks)}")
            start_pos = 0
            for i, chunk in enumerate(chunks):
                chunk_tokens = len(encoding.encode(chunk))
                end_pos = start_pos + chunk_tokens
                chunk_node = GraphNode(item, text=chunk, path=item_path, start_pos=start_pos, end_pos=end_pos, chunk_id=i)
                graph.add_node(chunk_node, parent_node)
                print(f"{Fore.CYAN}Added chunk {i+1}/{len(chunks)} to graph (tokens {start_pos}-{end_pos})")
                start_pos = end_pos - len(encoding.encode(' '.join(chunk.split()[-args.overlap:])))  # Account for overlap

async def build_graph(args, config):
    graph = Graph()
    graph.embedding_model = args.embedding_model
    graph.config = config
    graph.summary_model = args.summary_model
    process_folder(args.input_folder, graph.root, graph, args)

    text_nodes = [node for node in graph.nodes if not node.is_folder and not node.is_cluster]
    all_chunks = [node.text for node in text_nodes]

    print(f"{Fore.CYAN}Generating embeddings for all text chunks...")
    embeddings = {
        'model': args.embedding_model,
        'data': await get_embeddings(all_chunks, args.embedding_model)
    }

    for node, embedding in zip(text_nodes, embeddings['data']):
        node.embedding = embedding

    print(f"{Fore.GREEN}Creating hierarchical clusters...")
    graph.root.children = await recursive_clustering(graph, text_nodes, max_depth=args.max_depth, 
                                                     min_cluster_size=args.min_cluster_size, 
                                                     max_clusters=args.max_clusters)

    return graph, embeddings

def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
    try:
        return len(encoding.encode(text))
    except ValueError:
        # If the error is due to special tokens, encode with them allowed
        return len(encoding.encode(text, allowed_special="all"))

def save_graph(graph, embeddings, args):
    with open(args.output_file, 'wb') as f:
        pickle.dump((graph, embeddings), f)
    
    # Save metadata
    metadata = {
        "total_nodes": len(graph.nodes),
        "total_clusters": len(graph.clusters),
        "total_documents": sum(1 for node in graph.nodes if not node.is_folder and not node.is_cluster),
        "total_folders": sum(1 for node in graph.nodes if node.is_folder),
        "total_tokens": sum(count_tokens(node.text) for node in graph.nodes if not node.is_folder),
        "embedding_model": embeddings['model'],
    }
    
    metadata_file = args.output_file.rsplit('.', 1)[0] + '_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"{Fore.GREEN}Graph metadata saved to {metadata_file}")

def load_graph(args):
    with open(args.input_file, 'rb') as f:
        return pickle.load(f)

async def main():
    parser = argparse.ArgumentParser(description="Graph-based document processing and retrieval system")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Build command
    build_parser = subparsers.add_parser("build", help="Build and save the graph")
    build_parser.add_argument("--input_folder", required=True, help="Input folder containing text files")
    build_parser.add_argument("--output_file", required=True, help="Output file to save the graph")
    build_parser.add_argument("--chunk_size", type=int, default=512, help="Size of text chunks")
    build_parser.add_argument("--overlap", type=int, default=50, help="Number of words to overlap between chunks")
    build_parser.add_argument("--embedding_model", default="all-minilm", help="Embedding model to use")
    build_parser.add_argument("--max_depth", type=int, default=1, help="Maximum depth of the cluster tree")
    build_parser.add_argument("--min_cluster_size", type=int, default=5, help="Minimum number of nodes in a cluster")
    build_parser.add_argument("--max_clusters", type=int, default=10, help="Maximum number of clusters at each level")
    build_parser.add_argument("--config_file", help="Path to the YAML configuration file")
    build_parser.add_argument("--summary_model", default="qwen2:0.5b-instruct-fp16", help="Model to use for generating summaries")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search the graph")
    search_parser.add_argument("--input_file", required=True, help="Input file containing the saved graph")
    search_parser.add_argument("--query", required=True, help="Search query")
    search_parser.add_argument("--output_file", help="Output file to save search results")
    search_parser.add_argument("--config_file", help="Path to the YAML configuration file")
    search_parser.add_argument("--summary_model", default="qwen2:0.5b-instruct-fp16", help="Model to use for generating summaries")

    args = parser.parse_args()

    if args.command == "build":
        config = load_yaml_config(args.config_file)
        print(f"{Fore.CYAN}Starting graph build process...")
        graph, embeddings = await build_graph(args, config)
        save_graph(graph, embeddings, args)
        print(f"{Fore.GREEN}Graph built and saved to {args.output_file}")

    elif args.command == "search":
        graph, embeddings = load_graph(args)
        config = load_yaml_config(args.config_file)
        results = await graph.search(args.query, embeddings, config, args.summary_model)
        
        processed_results = []
        included_clusters = set()
        
        for node, score, depth, parent_cluster in results:
            if node.is_cluster:
                if node not in included_clusters:
                    result = {
                        'type': 'cluster',
                        'name': node.name,
                        'summary': node.text,
                        'score': score,
                        'depth': depth,
                        'children': []
                    }
                    included_clusters.add(node)
                    processed_results.append(result)
            else:
                result = {
                    'type': 'document',
                    'name': node.name,
                    'text': node.text,
                    'path': node.path,
                    'score': score,
                    'depth': depth,
                    'start_pos': node.start_pos,
                    'end_pos': node.end_pos,
                    'chunk_id': node.chunk_id
                }
                if parent_cluster:
                    parent_result = next((r for r in processed_results if r['type'] == 'cluster' and r['name'] == parent_cluster.name), None)
                    if parent_result:
                        parent_result['children'].append(result)
                    else:
                        processed_results.append({
                            'type': 'cluster',
                            'name': parent_cluster.name,
                            'summary': parent_cluster.text,
                            'score': parent_cluster.score if hasattr(parent_cluster, 'score') else 0,
                            'depth': parent_cluster.depth if hasattr(parent_cluster, 'depth') else 0,
                            'children': [result]
                        })
                else:
                    processed_results.append(result)
        
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(processed_results, f, indent=2)
            print(f"Search results saved to {args.output_file}")
        else:
            for result in processed_results:
                if result['type'] == 'cluster':
                    print(f"Cluster: {result['name']}")
                    print(f"Summary: {result['summary'][:200]}...")
                    print(f"Score: {result['score']}")
                    print(f"Depth: {result['depth']}")
                    print("Children:")
                    for child in result['children']:
                        print(f"  Document: {child['name']}")
                        print(f"  Text: {child['text'][:200]}...")
                        print(f"  Path: {child['path']}")
                        print(f"  Score: {child['score']}")
                        print(f"  Depth: {child['depth']}")
                        print(f"  Position: {child['start_pos']}-{child['end_pos']}")
                        print(f"  Chunk ID: {child['chunk_id']}")
                        print("  ---")
                else:
                    print(f"Document: {result['name']}")
                    print(f"Text: {result['text'][:200]}...")
                    print(f"Path: {result['path']}")
                    print(f"Score: {result['score']}")
                    print(f"Depth: {result['depth']}")
                    print(f"Position: {result['start_pos']}-{result['end_pos']}")
                    print(f"Chunk ID: {result['chunk_id']}")
                print("---")

if __name__ == "__main__":
    asyncio.run(main())
