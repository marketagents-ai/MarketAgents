import os
import argparse
import sys
import re
from xml.etree.ElementTree import fromstring, ParseError
from dotenv import load_dotenv
from tqdm import tqdm

# Import all potential LLM clients
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic

# Load environment variables
load_dotenv()

# System prompts (unchanged)
initial_system_prompt = """
You are tasked with creating a diverse list of persona descriptions based on the ideas provided by the user. Each persona should be described in a single sentence that includes age, name, and a brief description of their perspective based on lifestyle, hobbies, or career.

When creating these personas, consider the following guidelines:
1. Ensure a diverse range of ages, from teenagers to seniors
2. Include a variety of lifestyles, careers, and hobbies
3. Represent different generations (e.g., Gen Z, Millennials, Gen X, Baby Boomers)
4. Consider various cultural backgrounds and geographic locations
5. Include both common and unique perspectives

Format each persona description as a single sentence on a new line, without numbering. The sentence should follow this general structure:
[Name], [Age]: [Brief description including lifestyle, hobby, or career, and their perspective]

Generate the requested number of persona descriptions, ensuring a diverse and representative sample based on the user's ideas. Write your answer inside <answer> tags.
"""

detailed_system_prompt = """
You are tasked with creating a detailed persona, a list of questions, and a set of relevant Wikipedia URLs based on a given description of a person. The description will include information about their career, lifestyle, domain knowledge, and hobbies. Formatted in .md. Here's how to proceed:

<person_description>
{{PERSON_DESCRIPTION}}
</person_description>

1. Creating the Persona:
Analyze the provided description and create a detailed persona. Consider the following aspects:
- Demographics (age, gender, location)
- Personality traits
- Career details and work environment
- Lifestyle choices and daily routines
- Areas of expertise and domain knowledge
- Hobbies and interests
- Challenges and aspirations
- Social circle and relationships

Use modern psychometric reasoning and draw upon psychological theories to develop a well-rounded character. Be creative and consider unique aspects or less common characteristics that might apply to this individual.

2. Generating Questions:
Create a list of 32 questions that this person might ask, related to their circumstances, work, personal life, and hobbies. Ensure the questions are diverse and cover various aspects of their life. Consider:
- Work-related queries
- Personal development questions
- Hobby-specific inquiries
- Lifestyle and health-related questions
- Questions about their field of expertise
- Queries related to their challenges or aspirations

3. Compiling Wikipedia URLs:
Generate a list of 50 Wikipedia page URLs that would be relevant to this person's interests, career, and knowledge domains. Include a mix of:
- Pages related to their career field
- Articles about their hobbies and interests
- Historical or cultural pages relevant to their background
- Scientific or technical pages related to their expertise
- Pages about lifestyle, health, or personal development topics they might be interested in

Output your results in the following format using XML wrappers:

<name>
[insert name here in plain text]
</name>

<persona>
[Detailed persona description]
</persona>

<queries>
[Question 1]
[Question 2]
[Question 3]
...
[Question 32]
</queries>

<urls>
[Populate with domain relevent or subjects of potential interest to the persona]
https://en.wikipedia.org/wiki/[Relevant_Page_1]
https://en.wikipedia.org/wiki/[Relevant_Page_2]
...
https://en.wikipedia.org/wiki/[Relevant_Page_50]
</urls>

Ensure that your persona is thorough and creative, the questions are diverse and in-character, and the Wikipedia URLs cover a broad range of relevant topics.
"""

def initialize_client(api_choice):
    if api_choice == 'ollama':
        return OpenAI(
            base_url=os.getenv('OLLAMA_BASE_URL'),
            api_key=os.getenv('OLLAMA_API_KEY')
        )
    elif api_choice == 'groq':
        return Groq(api_key=os.getenv('GROQ_API_KEY'))
    elif api_choice == 'anthropic':
        return Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
    else:
        raise ValueError(f"Unsupported API choice: {api_choice}")

def api_call(client, api_choice, system_prompt, user_prompt, model):
    try:
        if api_choice == 'ollama':
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response.choices[0].message.content
        elif api_choice == 'groq':
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response.choices[0].message.content
        elif api_choice == 'anthropic':
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )
            return response.content
    except Exception as e:
        print(f"Error during API call: {str(e)}")
        return None

def generate_initial_personas(client, api_choice, user_ideas, num_personas, model):
    user_prompt = f"Generate {num_personas} personas based on these ideas: {user_ideas}"
    content = api_call(client, api_choice, initial_system_prompt, user_prompt, model)
    
    if content is None:
        return []

    # Extract content from <answer> tags if present
    match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
    if match:
        personas = match.group(1).strip().split('\n')
    else:
        personas = content.strip().split('\n')
    
    return [persona.strip() for persona in personas if persona.strip()]

def extract_name(persona):
    # Extract name from the persona description
    match = re.match(r'^([^,]+),', persona)
    return match.group(1).strip() if match else None

def filter_unique_personas(personas):
    unique_personas = {}
    for persona in personas:
        name = extract_name(persona)
        if name and name not in unique_personas:
            unique_personas[name] = persona
    return list(unique_personas.values())

def generate_detailed_persona(client, api_choice, description, model):
    return api_call(client, api_choice, detailed_system_prompt, description, model)

def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|\n\t]', '', name)
    name = name.replace(' ', '_')
    return name.lower()

def ensure_section_tags(content, section):
    open_tag = f"<{section}>"
    close_tag = f"</{section}>"
    
    if open_tag not in content and close_tag not in content:
        return f"{open_tag}\n{content}\n{close_tag}"
    elif open_tag not in content:
        return f"{open_tag}\n{content}"
    elif close_tag not in content:
        return f"{content}\n{close_tag}"
    return content

def structure_content(content):
    sections = ['name', 'persona', 'queries', 'urls']
    structured_content = content

    for section in sections:
        structured_content = ensure_section_tags(structured_content, section)

    return structured_content

def create_safe_filename(content, max_length=200):
    # Extract the first line of content
    first_line = content.split('\n', 1)[0]
    
    # Remove any XML tags
    first_line = re.sub(r'<[^>]+>', '', first_line)
    
    # Keep only alphanumeric characters and underscores
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '', first_line)
    
    # Truncate if longer than max_length
    safe_name = safe_name[:max_length]
    
    # Ensure the filename is not empty
    if not safe_name:
        safe_name = "unnamed_persona"
    
    return safe_name

def process_persona(client, api_choice, persona_description, output_dir, model):
    try:
        generated_content = generate_detailed_persona(client, api_choice, persona_description, model)
        if generated_content is None:
            print(f"Failed to generate detailed persona for: {persona_description}")
            return

        structured_content = structure_content(generated_content)
        
        name_match = re.search(r'<name>(.*?)</name>', structured_content, re.DOTALL)
        if name_match:
            name = name_match.group(1).strip()
        else:
            name = "unnamed_persona"
        
        sanitized_name = sanitize_filename(name)
        
        # Create a safe filename
        safe_filename = create_safe_filename(structured_content)
        
        # Try with the sanitized name first
        filename = f"{sanitized_name}_persona.md"
        full_path = os.path.join(output_dir, filename)
        
        # If the path is too long, use the safe filename
        if len(full_path) >= 260:  # Windows MAX_PATH limit
            filename = f"{safe_filename}_persona.md"
        
        save_md(structured_content, output_dir, filename)
        print(f"Persona data has been saved as {filename}")
    except Exception as e:
        print(f"Error processing persona: {e}")
        print("Skipping this persona and moving to the next one.")

def save_md(content, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Persona data has been saved to {filepath}")
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")
        print("Please check if you have the necessary permissions to write to this directory.")

def main():
    parser = argparse.ArgumentParser(description="Generate personas and save as Markdown files with XML-like wrappers.")
    parser.add_argument("persona_ideas", help="A string of ideas for persona types, separated by commas")
    parser.add_argument("-o", "--output", default=os.getenv('DEFAULT_OUTPUT_DIR'), help="Output directory for generated Markdown files")
    parser.add_argument("-n", "--number", type=int, default=int(os.getenv('DEFAULT_NUM_PERSONAS')), help="Number of initial personas to generate")
    parser.add_argument("-a", "--api", choices=['ollama', 'groq', 'anthropic'], default='ollama', help="Choose the LLM API to use")
    parser.add_argument("-m", "--model", help="Specify the model to use for the chosen API")
    args = parser.parse_args()

    try:
        client = initialize_client(args.api)
        
        # Set default models if not specified
        if not args.model:
            args.model = os.getenv(f'{args.api.upper()}_DEFAULT_MODEL')

        print(f"Using API: {args.api}")
        print(f"Using model: {args.model}")

        initial_personas = []
        while len(initial_personas) < args.number:
            new_personas = generate_initial_personas(client, args.api, args.persona_ideas, args.number - len(initial_personas), args.model)
            initial_personas.extend(new_personas)
            initial_personas = filter_unique_personas(initial_personas)

        print(f"Generated {len(initial_personas)} unique initial personas.")

        for i, persona in enumerate(tqdm(initial_personas), 1):
            print(f"Processing persona {i} of {len(initial_personas)}...")
            process_persona(client, args.api, persona, args.output, args.model)
        
        print(f"Finished processing all {len(initial_personas)} unique personas in the '{args.output}' directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()