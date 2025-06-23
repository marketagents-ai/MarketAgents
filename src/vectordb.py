import os
from src import utils
import csv
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores.redis import Redis

class VectorDB:
    def __init__(self, local_embeddings=False):
        self.rds = None
        self.redis_url = os.getenv("REDIS_URL")
        self.index_name = os.getenv("INDEX_NAME")
        self.embedding_model = os.getenv("OLLAMA_EMBED_MODEL")
        if local_embeddings:
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
        else:
            self.embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_KEY"))
    
    def load_vector_store(self, schema_path):
        schema = utils.load_yaml(schema_path)
        self.rds = Redis.from_existing_index(
            self.embeddings,  
            index_name=self.index_name,  
            redis_url=self.redis_url,
            schema=schema
        )
        return self.rds

    def initialize_vector_store(self, document_path, schema_path):
        documents = self.load_documents_from_folder(document_path)
        self.rds = Redis.from_documents(
            documents,
            self.embeddings,
            redis_url=self.redis_url,
            index_name=self.index_name
        )
        # write the schema to a yaml file
        self.rds.write_schema(schema_path)
        return self.rds

    def load_documents_from_folder(self, folder_path):
        documents = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    loader = JSONLoader(
                        file_path=file_path,
                        jq_schema='.',
                        text_content=False
                    )
                    documents.extend(loader.load())
                elif file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    loader = PyPDFLoader(file_path)
                    documents.extend(loader.load())
                else:
                    continue
                
        return documents
    
    def load_document_from_file(self, file_path):
        loader = JSONLoader(
            file_path=file_path,
            jq_schema='.',
            text_content=False
        )
        document = loader.load()
        return document

    def load_tasks_from_csv(self, csv_path):
        with open(csv_path, 'r') as csv_file:
            reader = csv.DictReader(csv_file)
            tasks = [(row['Task'], row['Category'], row['SubCategory']) for row in reader]
        return tasks

    def perform_similarity_search(self, query, k=2):
        return self.rds.similarity_search(query, k)

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Create an instance of the VectorDB class
    vector_db = VectorDB(local_embeddings=True)

     # Load configuration from YAML file
    config_path = "./config.yaml"
    config = utils.load_yaml(config_path)

    # Load documents from a folder
    examples_path = config["paths"]["examples_path"]
    schema_path = config["paths"]["redis_schema"]
    #documents = vector_db.load_documents_from_folder(folder_path)

    # Initialize the vector store
    vector_db.initialize_vector_store(examples_path, schema_path)

    # Load tasks from CSV
    curriculum_csv_path = config["paths"]["curriculum_csv"]
    tasks = vector_db.load_tasks_from_csv(curriculum_csv_path)
    task = tasks[12]

    # Define a query
    query = f"{task[0]}, {task[1]}, {task[2]}, functions, APIs, documentation"

    # Perform similarity search
    docs = vector_db.perform_similarity_search(query, k=2)

    # Print the content of the most similar document
    print(docs[0].page_content)
