import os
import re
import shutil
import argparse
from pathlib import Path

def split_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Extract name
    name_match = re.search(r'<name>(.*?)</name>', content, re.DOTALL)
    if not name_match:
        raise ValueError("No <name> tag found in the file.")
    name = name_match.group(1).strip()

    # Create folder
    folder_path = Path(name)
    folder_path.mkdir(exist_ok=True)

    # Copy original file to the new folder
    shutil.copy2(file_path, folder_path / Path(file_path).name)

    # Extract and save persona
    persona_match = re.search(r'<persona>(.*?)</persona>', content, re.DOTALL)
    if persona_match:
        with open(folder_path / f"{name}_persona.md", 'w', encoding='utf-8') as f:
            f.write(persona_match.group(1).strip())

    # Extract and save urls
    urls_match = re.search(r'<urls>(.*?)</urls>', content, re.DOTALL)
    if urls_match:
        with open(folder_path / f"{name}_wikiurls.txt", 'w', encoding='utf-8') as f:
            f.write(urls_match.group(1).strip())

    # Extract and save queries
    queries_match = re.search(r'<queries>(.*?)</queries>', content, re.DOTALL)
    if queries_match:
        queries = queries_match.group(1).strip()
        # Remove leading numbers and extra spaces
        queries = re.sub(r'^\s*\d+\.\s*', '', queries, flags=re.MULTILINE)
        # Split queries and join with double newlines
        queries = '\n\n'.join(query.strip() for query in queries.split('\n') if query.strip())
        with open(folder_path / f"{name}_queries.txt", 'w', encoding='utf-8') as f:
            f.write(queries)

    print(f"Files have been created in the '{name}' folder.")

def main():
    parser = argparse.ArgumentParser(description="Split a .md file into separate text files based on XML-like tags.")
    parser.add_argument("file_path", help="Path to the .md file to be processed")
    args = parser.parse_args()

    try:
        split_md_file(args.file_path)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()