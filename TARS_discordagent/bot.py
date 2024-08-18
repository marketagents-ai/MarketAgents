import discord
from discord.ext import commands
import logging
import asyncio
import os
import concurrent.futures
import mimetypes
import tiktoken
import tempfile
from discord import TextChannel

from config import *
from api_client import call_api
from repo_processor import count_files, start_background_processing, repo_processing_event
from github_utils import (
    get_file_content, 
    get_dir,
    generate_prompt, 
    get_local_file_content,  # Add this import
    extract_principles  # Add this import
)
from inverted_index import InvertedIndexSearch
from cache_manager import CacheManager
from repo_processor import start_background_processing, repo_processing_event, count_files

def setup_bot(args, repo, inverted_index_search, api):
    intents = discord.Intents.default()
    intents.message_content = True
    intents.members = True
    bot = commands.Bot(command_prefix='!', intents=intents)

    cache_manager = CacheManager(repo.name, MAX_CONVERSATION_HISTORY)

    @bot.event
    async def on_error(event, *args, **kwargs):
        logging.error(f"Unhandled error in event {event}", exc_info=True)

    @bot.event
    async def on_command_error(ctx, error):
        if isinstance(error, commands.CommandNotFound):
            await ctx.send("Command not found. Use !help to see available commands.")
        else:
            await ctx.send(f"An error occurred: {str(error)}")

    @bot.event
    async def on_message(message):
        if message.author == bot.user:
            return

        if isinstance(message.channel, discord.DMChannel):
            user_id = str(message.author.id)
            user_name = message.author.name
            logging.info(f"Received DM from {user_name} (ID: {user_id}): {message.content}")

            try:
                # Get conversation history
                conversation_history = cache_manager.get_conversation_history(user_id)

                # Prepare conversation context
                if conversation_history:
                    context = f"""
    Previous conversation history with {user_name} (User ID: {user_id}):
    (Most recent interactions first)

    {"-"*50}
    """
                    for i, msg in enumerate(reversed(conversation_history), 1):
                        context += f"""
    Interaction {i}:
    {msg['user_name']}: {msg[user_id]}
    AI: {msg['ai']}

    {"-"*50}
    """
                    context += f"""
    Current interaction:
    {user_name}: {message.content}

    Based on this conversation history, please provide a relevant and contextual response to the user's latest message.
    """
                else:
                    context = f"""
    This is the first interaction with {user_name} (User ID: {user_id}).

    Current interaction:
    {user_name}: {message.content}

    Please provide a relevant response to the user's message.
    """
                
                # Call API with context
                response_content = await call_api(message.content, context=context)

                # Send response to user
                await message.channel.send(response_content)
                logging.info(f"Sent DM response to {user_name} (ID: {user_id}): {response_content[:100]}...")

                # Update conversation history
                cache_manager.append_to_conversation(user_id, {
                    'user_name': user_name,
                    user_id: message.content,
                    'ai': response_content
                })

            except Exception as e:
                error_message = f"Looks like our AI got lost in a wormhole. Error: {str(e)}"
                await message.channel.send(error_message)
                logging.error(f"Error in DM with {user_name} (ID: {user_id}): {str(e)}")

        await bot.process_commands(message)

    @bot.command(name='ai_chat')
    async def ai_chat(ctx, *, question):
        """Get assistance from the AI model."""
        logging.info(f"Received ai_chat command: {question}")
        try:
            user_id = str(ctx.author.id)
            user_name = ctx.author.name

            # Get conversation history
            conversation_history = cache_manager.get_conversation_history(is_ai_chat=True)

            # Prepare conversation context
            if conversation_history:
                context = f"""
    Previous conversation history:
    (Most recent interactions first)

    {"-"*50}
    """
                for i, msg in enumerate(reversed(conversation_history[-5:]), 1):  # Show last 5 interactions
                    context += f"""
    Interaction {i}:
    {msg['user_name']} (ID: {msg['user_id']}): {msg['user_message']}
    AI: {msg['ai_response']}

    {"-"*50}
    """
                context += f"""
    Current interaction:
    {user_name}: {question}

    Based on this conversation history, please provide a relevant and contextual response to the user's latest message.
    """
            else:
                context = f"""
    This is the first interaction in the AI chat.

    Current interaction:
    {user_name}: {question}

    Please provide a relevant response to the user's message.
    """

            # Call API with context
            response_content = await call_api(question, context=context)

            # Send response to user
            await send_long_message(ctx, response_content)
            logging.info(f"Sent ai_chat response: {response_content[:100]}...")

            # Update conversation history
            cache_manager.append_to_conversation(user_id, {
                'user_name': user_name,
                'user_id': user_id,
                'user_message': question,
                'ai_response': response_content
            }, is_ai_chat=True)

        except Exception as e:
            error_message = f"Looks like our AI got lost in a wormhole. Error: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    @bot.command(name='analyze_code')
    async def analyze_code(ctx, *, code=None):
        """Analyze code using the AI model."""
        logging.info(f"Received analyze_code command: {code[:100] if code else 'No code provided'}")
        try:
            if ctx.message.attachments:
                if len(ctx.message.attachments) > 1:
                    await ctx.send("Please upload only one file at a time for analysis.")
                    return

                attachment = ctx.message.attachments[0]
                if not attachment.filename.endswith(('.py', '.js', '.txt', '.java', '.cpp')):
                    await ctx.send("Unsupported file type. Please upload a text-based code file.")
                    return

                try:
                    file_content = (await attachment.read()).decode('utf-8')
                    user_content = f"Analyze this code from the file '{attachment.filename}':\n\n{file_content}"
                except UnicodeDecodeError:
                    await ctx.send("Failed to read the file content. Please ensure it is a text file.")
                    return
            elif code:
                user_content = f"Analyze this code:\n\n{code}"
            else:
                await ctx.send("Please provide code text or upload a file to analyze.")
                return

            response_content = await call_api(user_content, context="", system_prompt_key="analyze_code")
            await ctx.send("Code Analysis Report:")
            await send_long_message(ctx, response_content)
            logging.info(f"Sent code analysis response: {response_content[:100]}...")
        except Exception as e:
            error_message = f"An unexpected error occurred while analyzing the code: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)


    @bot.command(name='repo_chat')
    async def repo_chat(ctx, *, question):
        """Chat with the Repo, uses a simple inverted index"""
        logging.info(f"Received repo_chat command: {question}")
        try:
            # Check if chunks are loaded
            logging.info("Checking if chunks are loaded...")
            if not inverted_index_search.chunks:
                logging.info("Chunks not loaded. Starting background processing...")
                total_files = await count_files(repo)
                progress_message = await ctx.send(f"Fetching and processing repository contents. This might take a moment... (0/{total_files} files processed)")
                await start_background_processing(repo, inverted_index_search)
                await ctx.send(f"Repository processing has started in the background for {total_files} files. You can continue to use other commands while this is happening.")
                
                while not repo_processing_event.is_set():
                    await asyncio.sleep(5)  # Check every 5 seconds
                    processed_files = len(inverted_index_search.chunks)
                    await progress_message.edit(content=f"Fetching and processing repository contents. This might take a moment... ({processed_files}/{total_files} files processed)")
                
                await progress_message.edit(content=f"Repository contents processed. {len(inverted_index_search.chunks)}/{total_files} files indexed. Generating response...")

            # Search for relevant chunks
            logging.info("Searching for relevant chunks...")
            try:
                relevant_chunks = inverted_index_search.search(question, k=8)
                logging.info(f"Found {len(relevant_chunks)} relevant chunk groups")
            except Exception as e:
                logging.error(f"Error in inverted_index_search.search: {str(e)}", exc_info=True)
                await ctx.send("An error occurred while searching the repository. Please try again later.")
                return

            # Process relevant chunks
            if not relevant_chunks:
                await ctx.send("No relevant information found in the repository. I'll try to answer based on my general knowledge.")
                context = ""
                file_links = []
            else:
                context = "Here are some relevant excerpts from the repository:\n\n"
                excerpt_number = 1
                file_links = set()  # Use a set to avoid duplicate links
                for chunk_group in relevant_chunks:
                    for item in chunk_group:
                        if isinstance(item, tuple) and len(item) == 2:
                            chunk, file_path = item
                            if isinstance(chunk, str) and isinstance(file_path, str):
                                context += f"Excerpt {excerpt_number} (File: {file_path}):\n```\n{chunk}\n```\n\n"
                                excerpt_number += 1
                                # Add file link to the set, using the full path
                                file_links.add(f"[{file_path}](https://github.com/{REPO_NAME}/blob/main/{file_path})")
                            else:
                                logging.warning(f"Unexpected data types in chunk_group: {type(chunk)}, {type(file_path)}")
                        else:
                            logging.warning(f"Unexpected item in chunk_group: {item}")

            # Prepare and send prompt to API
            logging.info("Preparing full prompt...")
            full_prompt = f"""Context from the repository:

    {context}

    User Question: {question}

    Based on the context provided above and your general knowledge, please answer the user's question. If the context doesn't contain relevant information, you can rely on your general knowledge to provide the best possible answer."""

            logging.info("Calling API...")
            try:
                response_content = await call_api(full_prompt)
            except Exception as e:
                logging.error(f"Error in call_api: {str(e)}", exc_info=True)
                await ctx.send("An error occurred while generating the response. Please try again later.")
                return

            # Append file links to the response
            if file_links:
                response_content += "\n\nRelevant files:\n" + "\n".join(sorted(file_links))

            # Send response to user
            logging.info("Sending response to user...")
            try:
                await send_long_message(ctx, response_content)
            except Exception as e:
                logging.error(f"Error in send_long_message: {str(e)}", exc_info=True)
                await ctx.send("An error occurred while sending the response. Please try again.")
                return

            logging.info(f"Sent response to user: {response_content[:100]}...")
        except Exception as e:
            error_message = f"An unexpected error occurred: {str(e)}"
            await ctx.send(error_message)
            logging.error(f"Unexpected error in repo_chat: {str(e)}", exc_info=True)

    @bot.command(name='generate_prompt')
    async def generate_prompt_command(ctx, file_path, *, user_task_description):
        """Generate a goal-oriented prompt based on the repo code, principles, and task description, then query the LLM."""
        logging.info(f"Received generate_prompt command: {file_path}, {user_task_description}")
        try:
            repo_code = get_file_content(repo, file_path)
            if repo_code.startswith("Error fetching file:"):
                await ctx.send(f"Error: Unable to fetch the file '{file_path}'. {repo_code}")
                return
            if repo_code.startswith("Binary file detected"):
                await ctx.send(f"The file at {file_path} appears to be a binary file and cannot be used for prompt generation.")
                return

            # Try to read from local principles.md file
            principles_markdown = get_local_file_content('principles.md')
            
            if principles_markdown.startswith("Error reading local file:"):
                # If local file read fails, fall back to GitHub principles.md
                logging.warning("Failed to read local principles.md, falling back to GitHub principles.md")
                principles_markdown = get_file_content(repo, 'principles.md')
                if principles_markdown.startswith("Error fetching file:"):
                    await ctx.send(f"Error: Unable to fetch principles. {principles_markdown}")
                    return

            principles = extract_principles(principles_markdown)

            # Determine the code type based on file extension
            _, file_extension = os.path.splitext(file_path)
            code_type = mimetypes.types_map.get(file_extension, "").split('/')[-1] or "plaintext"
            
            prompt = generate_prompt(file_path, repo_code, principles, user_task_description, code_type)
            
            # Use the generate_prompt system prompt
            response_content = await call_api(prompt, system_prompt_key="generate_prompt")
            
            # Send the AI-generated response
            await ctx.send("AI-Generated Response:")
            await send_long_message(ctx, response_content)
            
            logging.info(f"Sent AI response for generate_prompt: {response_content[:100]}...")

        except Exception as e:
            error_message = f"Error generating prompt and querying AI: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    @bot.command(name='dir')
    async def dir_command(ctx, max_depth: int = 0):
        """
        Display the repository file structure.
        Usage: !dir [depth]
        depth: Integer representing the depth of the directory structure to display (default: 0, root level only).
        """
        try:
            if max_depth < 0:
                await ctx.send("Depth must be a non-negative integer.")
                return

            with concurrent.futures.ThreadPoolExecutor() as executor:
                structure = await bot.loop.run_in_executor(executor, get_dir, repo, "", "", max_depth)
            
            formatted_structure = f"Repository Structure (Depth: {max_depth}):\n\n" + "\n".join(structure)
            
            if len(formatted_structure) > 1900:
                # Create a temporary file and write the structure to it
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as temp_file:
                    temp_file.write(formatted_structure)
                    temp_file_name = temp_file.name

                # Send the file
                await ctx.send("The directory structure is too large to display in chat. Here's a file containing the structure:", 
                            file=discord.File(temp_file_name, filename="repo_structure.txt"))
                
                # Delete the temporary file
                os.unlink(temp_file_name)
            else:
                # If it's small enough, send it as a regular message
                await ctx.send(f"```\n{formatted_structure}\n```")

        except Exception as e:
            error_message = f"Error fetching repository structure: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    @bot.command(name='clear_history')
    async def clear_history(ctx):
        """Clear your conversation history."""
        user_id = str(ctx.author.id)
        cache_manager.clear_conversation(user_id)
        await ctx.send("Your conversation history has been cleared.")

    @bot.command(name='channel_summary')
    async def channel_summary(ctx, message_count: int = 100):
        """Summarize the last n messages in the channel."""
        if not isinstance(ctx.channel, TextChannel):
            await ctx.send("This command can only be used in text channels.")
            return

        logging.info(f"Received channel_summary command in {ctx.channel.name}, messages: {message_count}")

        try:
            # Fetch messages
            messages = []
            async for message in ctx.channel.history(limit=message_count):
                messages.append(message)
            
            if not messages:
                await ctx.send("No messages found in the channel.")
                return
            
            messages.reverse()  # Oldest first

            # Process messages
            processed_messages = process_messages(messages)

            # Generate markdown content
            md_content = generate_markdown_summary(processed_messages)

            # Send a message to indicate that the bot is processing
            await ctx.send("Generating channel summary. This may take a moment...")

            # Generate summary using LLM with the channel_summary system prompt
            summary_response = await call_api(md_content, system_prompt_key="channel_summary")

            # Send the summary
            await ctx.send("Channel Summary:")
            await send_long_message(ctx, summary_response)

        except Exception as e:
            error_message = f"Error generating channel summary: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)

    def process_messages(messages):
        encoder = tiktoken.get_encoding("cl100k_base")
        processed_messages = []

        for message in messages:
            if message.content.strip():  # Only process non-empty messages
                tokens = encoder.encode(message.content)
                if len(tokens) > 128:
                    truncated_tokens = tokens[:64] + tokens[-64:]
                    content = encoder.decode(truncated_tokens)
                    content = content[:64] + " ... " + content[-64:]
                else:
                    content = message.content

                processed_messages.append({
                    'author': message.author.name,
                    'content': content
                })

        return processed_messages

    def generate_markdown_summary(processed_messages):
        md_content = "# Channel Summary\n\n"
        for msg in processed_messages:
            md_content += f"**{msg['author']}**: {msg['content']}\n\n"
        return md_content
        
    @bot.command(name='re_index')
    async def re_index(ctx):
        """Re-index the repository and update the cache."""
        logging.info(f"Received re_index command from {ctx.author.name}")
        
        try:
            # Check if indexing is already in progress
            if repo_processing_event.is_set():
                await ctx.send("Indexing is already in progress. Please wait for it to complete.")
                return

            # Clear the existing cache
            inverted_index_search.clear_cache()

            # Start the re-indexing process
            total_files = await count_files(repo)
            progress_message = await ctx.send(f"Starting re-indexing process. This might take a moment... (0/{total_files} files processed)")
            
            # Start background processing
            asyncio.create_task(start_background_processing(repo, inverted_index_search))

            # Monitor progress
            while not repo_processing_event.is_set():
                await asyncio.sleep(5)  # Check every 5 seconds
                processed_files = len(inverted_index_search.chunks)
                await progress_message.edit(content=f"Re-indexing in progress... ({processed_files}/{total_files} files processed)")

            # Indexing complete
            await progress_message.edit(content=f"Re-indexing complete. {len(inverted_index_search.chunks)}/{total_files} files indexed.")

            logging.info("Re-indexing process completed successfully")

        except Exception as e:
            error_message = f"An error occurred during re-indexing: {str(e)}"
            await ctx.send(error_message)
            logging.error(error_message)


    async def on_ready():
        logging.info(f'Logged in as {bot.user.name} (ID: {bot.user.id})')
        logging.info('------')
        # Start processing the repository contents in the background
        await start_background_processing(repo, inverted_index_search)

    return bot

async def send_long_message(ctx, message):
    chunks = [message[i:i+1900] for i in range(0, len(message), 1900)]
    if len(chunks) == 1:
        # If there's only one chunk, send it without the "Part" prefix
        await ctx.send(f"```{chunks[0]}```")
    else:
        # If there are multiple chunks, include the "Part" prefix
        for i, chunk in enumerate(chunks, 1):
            await ctx.send(f"```Part {i}/{len(chunks)}:\n{chunk}```")