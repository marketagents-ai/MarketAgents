import logging
import markdown2
import os
import re
from github import Github
import base64

def setup_github_repo(github_token, repo_name):
    g = Github(github_token)
    return g.get_repo(repo_name)

def get_file_content(repo, file_path):
    # Existing function to fetch content from GitHub
    try:
        logging.info(f"Fetching content for file: {file_path}")
        file_content = repo.get_contents(file_path)
        
        if file_content.size > 1000000:  # 1MB limit
            logging.warning(f"File {file_path} exceeds size limit (Size: {file_content.size} bytes)")
            return "File is too large to fetch content directly."

        try:
            content = file_content.decoded_content.decode('utf-8')
            logging.debug(f"Successfully decoded {file_path} as UTF-8")
            return content
        except UnicodeDecodeError:
            logging.warning(f"UTF-8 decoding failed for {file_path}, trying latin-1")
            try:
                content = file_content.decoded_content.decode('latin-1')
                logging.debug(f"Successfully decoded {file_path} as latin-1")
                return content
            except UnicodeDecodeError:
                logging.warning(f"Failed to decode {file_path} as text, treating as binary")
                encoded_content = base64.b64encode(file_content.decoded_content).decode('ascii')
                return f"Binary file detected. Base64 encoded content: {encoded_content[:100]}..."

    except Exception as e:
        logging.error(f"Error fetching {file_path}: {str(e)}")
        return f"Error fetching file: {str(e)}"

def get_dir(repo, path="", prefix="", max_depth=0, current_depth=0):
    if current_depth > max_depth:
        return []

    if not path:  # This is the root call
        structure = [f"{repo.name}/"]
        prefix = "  "
    else:
        structure = []

    contents = repo.get_contents(path)
    for i, content in enumerate(contents):
        is_last = (i == len(contents) - 1)
        if content.type == "dir":
            structure.append(f"{prefix}{'└── ' if is_last else '├── '}{content.name}/")
            if current_depth < max_depth:
                structure.extend(get_dir(repo, content.path, prefix + ('    ' if is_last else '│   '), max_depth, current_depth + 1))
        else:
            structure.append(f"{prefix}{'└── ' if is_last else '├── '}{content.name}")
    
    return structure

def get_local_file_content(file_path):
    # Function to read content from a local file
    try:
        logging.info(f"Reading content from local file: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logging.debug(f"Successfully read local file {file_path}")
        return content
    except Exception as e:
        logging.error(f"Error reading local file {file_path}: {str(e)}")
        return f"Error reading local file: {str(e)}"

import re
import markdown2

def extract_principles(markdown_content):
    # Convert markdown to HTML
    html_content = markdown2.markdown(markdown_content)
    
    principles = []
    current_levels = [0, 0, 0, 0, 0, 0]  # Track up to 6 levels of headers
    
    # Split the content into lines
    lines = html_content.split('\n')
    
    for line in lines:
        # Check for headers
        header_match = re.match(r'<h(\d)>(.*?)</h\d>', line)
        if header_match:
            level = int(header_match.group(1))
            header_text = header_match.group(2)
            
            # Reset lower levels
            current_levels[level-1] += 1
            for i in range(level, len(current_levels)):
                current_levels[i] = 0
            
            # Create header with proper indentation
            header = ("#" * level) + " " + ".".join(map(str, current_levels[:level])) + " " + header_text
            principles.append(header)
        
        # Check for list items
        list_match = re.match(r'<li>(.*?)</li>', line)
        if list_match:
            item_text = list_match.group(1)
            # Add list item with proper indentation
            principles.append("  " * (level-1) + "- " + item_text)
    
    return "\n".join(principles)

def generate_prompt(file_path, repo_code, principles, task_description, code_type):
    max_code_length = 8000
    if len(repo_code) > max_code_length:
        repo_code = repo_code[:max_code_length] + "... (truncated)"

    prompt = f"""# Context

# {os.path.basename(file_path)}
```{code_type}
{repo_code}
```

# Core Principles:

{principles}

# Task

{task_description}

# Prompt

Based on the provided context, core principles, and task, please provide a detailed response that addresses the following:

1. Summarize the key points of the code and core principles.
2. Provide a step-by-step approach to accomplish the given task.
3. Highlight any potential challenges or considerations.
4. Suggest any additional resources or information that might be helpful.

Please structure your response clearly and concisely.
"""
    return prompt