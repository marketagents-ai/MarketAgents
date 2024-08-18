import logging
from dotenv import load_dotenv
import os
import argparse
from bot import setup_bot
from repo_processor import start_background_processing_thread
from github_utils import setup_github_repo
from inverted_index import InvertedIndexSearch
from logging_config import setup_logging
from api_client import initialize_api_client
from cache_manager import CacheManager

from config import *

def main():
    # Set up logging
    setup_logging()

    # Load environment variables
    load_dotenv(override=True)

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Discord bot with selected API and model')
    parser.add_argument('--api', choices=['azure', 'ollama', 'openrouter'], default='ollama', help='Choose the API to use (default: ollama)')
    parser.add_argument('--model', type=str, help='Specify the model to use. If not provided, defaults will be used based on the API.')
    args = parser.parse_args()

    # Initialize API client
    initialize_api_client(args)

    # Bot configuration
    TOKEN = os.getenv('DISCORD_TOKEN')
    GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
    REPO_NAME = os.getenv('GITHUB_REPO')

    # Setup GitHub repo
    repo = setup_github_repo(GITHUB_TOKEN, REPO_NAME)

    # Initialize CacheManager
    cache_manager = CacheManager(repo.name)

    # Initialize InvertedIndexSearch
    inverted_index_search = InvertedIndexSearch(repo.name, chunk_percentage=10)

    # Setup the bot
    bot = setup_bot(args, repo, inverted_index_search, args.api)

    # Log startup information
    logging.info(f"Starting bot with {args.api.upper()} API")
    if args.api == 'azure':
        logging.info(f"Azure OpenAI Model: {os.getenv('AZURE_OPENAI_DEPLOYMENT', 'Not specified')}")
    elif args.api == 'ollama':
        logging.info(f"Ollama Model: {os.getenv('OLLAMA_MODEL', 'Not specified')}")
    elif args.api == 'openrouter':
        logging.info(f"OpenRouter Model: {args.model or 'openai/gpt-3.5-turbo'}")
        logging.info(f"OpenRouter Site URL: {os.getenv('YOUR_SITE_URL', 'Not specified')}")
        logging.info(f"OpenRouter App Name: {os.getenv('YOUR_APP_NAME', 'Not specified')}")

    # Start background processing
    start_background_processing_thread(repo, inverted_index_search, max_depth=2)  # Limit to 3 levels deep

    # Run the bot
    bot.run(TOKEN)

if __name__ == "__main__":
    main()