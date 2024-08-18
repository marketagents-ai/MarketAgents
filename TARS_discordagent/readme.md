DISCORD AGENT FOR DISCUSSING CODE, PROJECT TASKS AND REPO CONTENTS

Overall, this script combines Discord bot functionality with AI model integration to provide an interactive experience for users to ask questions, analyze code, and receive assistance based on the repository contents.

1. InvertedIndexSearch:
   - IPO:
     - Input: Cache directory, max tokens, context chunks, chunk percentage
     - Process: Indexes and searches through chunks of text from repository files
     - Output: Relevant chunks based on a search query
   - Description: This class handles the indexing and searching of text chunks from the repository files, allowing for efficient information retrieval.

2. fetch_and_chunk_repo_contents:
   - IPO:
     - Input: GitHub repository object
     - Process: Fetches and processes the contents of the repository, chunking the text and adding it to the inverted index
     - Output: Populated inverted index with chunks of text from the repository
   - Description: This function fetches the contents of the repository, processes each file, chunks the text, and adds the chunks to the inverted index for searching.

3. call_api:
   - IPO:
     - Input: Prompt, context, system prompt key
     - Process: Calls the specified AI model API (Azure OpenAI or Ollama) with the given prompt and context
     - Output: Response from the AI model
   - Description: This function handles the interaction with the AI model API, sending the prompt and context, and returning the generated response.

4. send_long_message:
   - IPO:
     - Input: Discord context, message
     - Process: Splits the message into smaller chunks and sends them as separate messages in Discord
     - Output: Sent messages in Discord
   - Description: This function is used to send long messages that exceed the character limit in Discord by splitting them into smaller chunks.

5. on_message:
   - IPO:
     - Input: Message event
     - Process: Handles incoming messages, processing commands and direct messages
     - Output: Appropriate responses or actions based on the message
   - Description: This event handler processes incoming messages, routing them to the appropriate command functions or handling direct messages.

6. ai_chat:
   - IPO:
     - Input: Question from the user
     - Process: Calls the AI model API to generate a response
     - Output: Response from the AI model sent as a message
   - Description: This command allows users to ask questions and receive assistance from the AI model.

7. analyze_code:
   - IPO:
     - Input: Code snippet
     - Process: Analyzes the provided code using the AI model
     - Output: Code analysis report sent as a message
   - Description: This command analyzes the given code snippet and provides a detailed report using the AI model.

8. repo_chat:
   - IPO:
     - Input: Question related to the repository
     - Process: Searches for relevant information in the repository, generates a prompt, and queries the AI model
     - Output: Response from the AI model with relevant excerpts from the repository
   - Description: This command allows users to ask questions related to the repository, and the bot searches for relevant information, generates a prompt, and provides a response using the AI model.

9. generate_prompt:
   - IPO:
     - Input: File path, user task description
     - Process: Generates a goal-oriented prompt based on the repository code, principles.md, and task description
     - Output: Generated prompt and AI model's response
   - Description: This command generates a prompt based on the provided file path and task description, queries the AI model, and returns the generated prompt and the AI model's response.

10. dir:
    - IPO:
      - Input: None
      - Process: Retrieves the repository file structure
      - Output: Formatted repository file structure sent as a message
    - Description: This command displays the repository file structure in a formatted way.

Overall, this script combines Discord bot functionality with AI model integration to provide an interactive experience for users to ask questions, analyze code, and receive assistance based on the repository contents.
