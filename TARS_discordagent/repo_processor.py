import asyncio
import logging
import os
import threading
from inverted_index import InvertedIndexSearch

ALLOWED_EXTENSIONS = {'.md', '.py', '.txt'}

# Create an event for signaling when repository processing is complete
repo_processing_event = asyncio.Event()

async def count_files(repo, path="", max_depth=None, current_depth=0):
    if max_depth is not None and current_depth > max_depth:
        return 0

    total_files = 0
    contents = await asyncio.to_thread(repo.get_contents, path)
    
    for content in contents:
        if content.type == "dir":
            if max_depth is None or current_depth < max_depth:
                total_files += await count_files(repo, content.path, max_depth, current_depth + 1)
        elif content.type == "file":
            _, file_extension = os.path.splitext(content.path)
            if file_extension.lower() in ALLOWED_EXTENSIONS:
                total_files += 1
    
    return total_files

async def fetch_and_chunk_repo_contents(repo, inverted_index_search, max_depth=None):
    contents = await asyncio.to_thread(repo.get_contents, "")
    if contents is None:
        logging.error("Failed to fetch repository contents.")
        return

    logging.info(f"Starting to fetch contents for repo: {repo.full_name}")

    if inverted_index_search.load_cache():
        logging.info("Loaded cached index.")
        return

    async def process_contents(contents, current_depth=0):
        tasks = []
        for content in contents:
            if content.type == "dir":
                if max_depth is None or current_depth < max_depth:
                    dir_contents = await asyncio.to_thread(repo.get_contents, content.path)
                    await process_contents(dir_contents, current_depth + 1)
            elif content.type == "file":
                tasks.append(asyncio.create_task(process_file(content)))

            if len(tasks) >= 10:
                await asyncio.gather(*tasks)
                tasks = []

        if tasks:
            await asyncio.gather(*tasks)

    async def process_file(file_content):
        try:
            _, file_extension = os.path.splitext(file_content.path)
            if file_extension.lower() in ALLOWED_EXTENSIONS:
                logging.debug(f"Processing file: {file_content.path}")
                try:
                    file_data = await asyncio.to_thread(lambda: file_content.decoded_content.decode('utf-8'))
                except UnicodeDecodeError:
                    logging.warning(f"UTF-8 decoding failed for {file_content.path}, trying latin-1")
                    try:
                        file_data = await asyncio.to_thread(lambda: file_content.decoded_content.decode('latin-1'))
                    except Exception as e:
                        logging.error(f"Failed to decode {file_content.path}: {str(e)}")
                        return
                inverted_index_search.process_and_index_content(file_data, file_content.path)
                logging.info(f"Successfully processed file: {file_content.path}")
            else:
                logging.debug(f"Skipping file with unsupported extension: {file_content.path}")
        except Exception as e:
            logging.error(f"Unexpected error processing {file_content.path}: {str(e)}")

    await process_contents(contents)

    inverted_index_search.save_cache()
    logging.info(f"Finished processing repo: {repo.full_name}. Total chunks: {len(inverted_index_search.chunks)}")

async def start_background_processing(repo, inverted_index_search, max_depth=None):
    global repo_processing_event
    repo_processing_event.clear()
    
    try:
        await fetch_and_chunk_repo_contents(repo, inverted_index_search, max_depth)
    except Exception as e:
        logging.error(f"Error in background processing: {str(e)}")
    finally:
        repo_processing_event.set()

def run_background_processing(repo, inverted_index_search, max_depth=None):
    asyncio.run(start_background_processing(repo, inverted_index_search, max_depth))

def start_background_processing_thread(repo, inverted_index_search, max_depth=None):
    thread = threading.Thread(target=run_background_processing, args=(repo, inverted_index_search, max_depth))
    thread.start()
    logging.info(f"Started background processing of repository contents in a separate thread (Max Depth: {max_depth if max_depth is not None else 'Unlimited'})")