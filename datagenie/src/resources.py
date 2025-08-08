import os
import logging
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from src import utils
from src.tools import WebSearch
from src.vectordb import VectorDB

from src.utils import agent_logger

class Resource(BaseModel):
    search_results: List[Dict[str, Any]] = None
    documents: List[Dict[str, Any]] = None
    examples: List[Dict[str, Any]] = None

class ResourceManager:
    def __init__(self, web_search_client: WebSearch = WebSearch(), local_embeddings=False):
        self.web_search_client = web_search_client
        self.vector_db = VectorDB(local_embeddings=local_embeddings)
        self.local_embeddings = local_embeddings

    def retrieve_websearch_results(self, query: str, num_results: int, folder_path: str) -> str:
        # Check if the folder already exists
        if os.path.exists(folder_path) and os.listdir(folder_path):
            # Read from existing JSON files
            search_results = utils.read_documents_from_folder(folder_path, num_results)
        else:
            # Fetch new search results
            google_results = self.web_search_client.google_search(query, num_results)
            combined_results = [url for url in google_results]
            
            #try:
            #    bing_results = self.web_search_client.bing_web_search(query, num_results)
            #    for url in bing_results:
            #        if url not in combined_results:
            #            combined_results.append(url)
            #except Exception as e:
            #    agent_logger.info(f"Could not complete bing search: {e}")
           
            search_results = self.web_search_client._scrape_results_parallel(combined_results)
            utils.save_search_results(folder_path, search_results)
            agent_logger.info(f"Search results saved successfully at {folder_path}")
        
        return search_results

        #try:
        #    combined_text = utils.combine_search_result_documents(search_results, char_limit)
        #    return combined_text
        #except Exception as e:
        #    return f"Exception in the loop: {e}"

    def retrieve_vectordb_documents(self, query: str, num_docs: int = 5) -> str:
        try:
            retrieved_docs = self.vector_db.perform_similarity_search(query, num_docs)
            #combined_examples = utils.combine_examples(retrieved_docs, type=None)
        except Exception as e:
            agent_logger.error(f"Error combining documents: {e}")
            retrieved_docs = []
        return retrieved_docs
    
    def initialize_vector_db(self, documents_path=None):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            documents_path = os.path.join(script_dir, '../documents')
            schema_path = os.path.join(script_dir, '../redis_schema.yaml')
            self.vector_db = VectorDB(local_embeddings=self.local_embeddings)
            try:
                self.vector_db.load_vector_store(schema_path)
                agent_logger.info("Existing VectorDB loaded successfully.")
            except Exception as load_error:
                agent_logger.error(f"Loading existing VectorDB failed: {load_error}. Initializing...")
                try:
                    self.vector_db.initialize_vector_store(documents_path, schema_path)
                    agent_logger.info("VectorDB initialized successfully.")
                except Exception as init_error:
                    print("Initialization failed:", init_error)
                    agent_logger.error(f"VectorDB initialization failed: {init_error}")