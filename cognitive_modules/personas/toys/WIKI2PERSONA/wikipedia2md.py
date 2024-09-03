import requests
from bs4 import BeautifulSoup
import re
import argparse
import os
from urllib.parse import urljoin

def clean_header(header_text):
    header_text = re.sub(r'\[edit\]$', '', header_text).strip()
    header_text = re.sub(r'\b(v|t|e)\b', '', header_text).strip()
    header_text = re.sub(r'view talk edit', '', header_text, flags=re.I).strip()
    return header_text

def is_placeholder_page(soup, markdown):
    if len(markdown) < 500:
        return True
    title = soup.find(id="firstHeading")
    if title and any(keyword in title.text.lower() for keyword in ["placeholder", "stub", "does not exist"]):
        return True
    if not soup.find(id="mw-content-text") or not soup.find("p"):
        return True
    content = soup.find(id="mw-content-text")
    if content and any(keyword in content.text.lower() for keyword in ["this page is a placeholder", "no content has been added"]):
        return True
    return False

def is_nonexistent_page(soup):
    if soup.find('div', class_='noarticletext'):
        return True
    if soup.find('div', id='mw-content-text', class_='mw-content-ltr'):
        content = soup.find('div', id='mw-content-text', class_='mw-content-ltr')
        if "There is currently no text in this page." in content.text:
            return True
    return False

def remove_initial_lists(content):
    first_paragraph_found = False
    elements_to_remove = []
    
    for element in content.find_all(['p', 'ul', 'ol'], recursive=False):
        if element.name == 'p':
            first_paragraph_found = True
        if not first_paragraph_found and element.name in ['ul', 'ol']:
            elements_to_remove.append(element)
        elif first_paragraph_found:
            break
    
    for element in elements_to_remove:
        element.decompose()

def remove_sidebar_elements(content):
    sidebars = content.find_all(['div', 'table'], class_=['sidebar', 'nomobile', 'nowraplinks', 'vertical-navbox', 'navbox', 'vertical-navbox', 'infobox', 'mw-collapsible'])
    for sidebar in sidebars:
        sidebar.decompose()

def wikipedia_to_markdown(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    if is_nonexistent_page(soup):
        return None, "This page does not exist on Wikipedia.", []

    title = soup.find(id="firstHeading")
    if not title:
        return None, "Unable to find the main heading of the page.", []
    title = title.text.strip()

    content = soup.find(id="mw-content-text")
    if not content:
        return None, "Unable to find the main content of the page.", []

    # Remove the table of contents and GUI elements
    toc = content.find('div', class_='toc')
    if toc:
        toc.decompose()
    for element in content.find_all(['div', 'span', 'table'], class_=['toc', 'mw-editsection', 'navbox', 'vertical-navbox', 'infobox']):
        element.decompose()

    # Remove initial lists before the first paragraph
    remove_initial_lists(content)

    # Remove sidebar elements
    remove_sidebar_elements(content)

    markdown = f"# {title}\n\n"

    # Process content elements and remove unwanted lists
    in_external_links = False
    external_links = []
    references_section = False
    references = []

    for element in content.find_all(['h2', 'h3', 'h4', 'h5', 'h6', 'p', 'ul', 'ol']):
        if element.name.startswith('h'):
            level = int(element.name[1])
            header_text = clean_header(element.text)
            if "external links" in header_text.lower():
                in_external_links = True
                continue
            if "references" in header_text.lower():
                references_section = True
                markdown += f'\n\n{"#" * level} {header_text}\n\n'
                continue
            markdown += f'\n\n{"#" * level} {header_text}\n\n'
            references_section = False
        elif element.name == 'p':
            markdown += process_paragraph(element) + '\n\n'
        elif element.name in ['ul', 'ol']:
            if references_section:
                references.extend(process_references_list(element))
            elif in_external_links:
                _, links = process_list(element, in_external_links, url)
                external_links.extend(links)
            else:
                list_content, _ = process_list(element, in_external_links, url)
                markdown += list_content + '\n'

    if references:
        markdown += "\n\n## References\n\n"
        for i, reference in enumerate(references, 1):
            markdown += f"[{i}] {reference}\n"

    # Remove multiple newlines
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    markdown = re.sub(r'^\n+', '', markdown)
    markdown = re.sub(r'\n+$', '', markdown)

    # Remove duplicate ## References header if there are no references
    markdown = re.sub(r'\n+## References\n+\n*', '\n\n', markdown)

    if is_placeholder_page(soup, markdown):
        return None, "This appears to be a placeholder page with minimal content.", []

    return title, markdown.strip(), external_links

def process_paragraph(element):
    text = element.text.strip()
    for link in element.find_all('a'):
        if link.has_attr('href') and not link['href'].startswith('#'):
            text = text.replace(f"[{link.text}]({link['href']})", link.text)
    return text

def process_list(element, in_external_links, base_url, level=0):
    markdown = ""
    external_links = []

    for li in element.find_all('li', recursive=False):
        if in_external_links:
            link = li.find('a')
            if link and link.has_attr('href'):
                href = urljoin(base_url, link['href'])
                external_links.append(f"* {link.text} ({href})")
        else:
            prefix = '  ' * level + ('* ' if element.name == 'ul' else f"{level+1}. ")
            markdown += prefix + process_list_item(li, base_url) + '\n'
        
        nested_list = li.find(['ul', 'ol'])
        if nested_list:
            nested_content, nested_links = process_list(nested_list, in_external_links, base_url, level + 1)
            markdown += nested_content + '\n'
            external_links.extend(nested_links)
    
    return markdown.rstrip(), external_links

def process_list_item(li, base_url):
    text = li.text.strip()
    for link in li.find_all('a'):
        if link.has_attr('href') and not link['href'].startswith('#'):
            text = text.replace(f"[{link.text}]({link['href']})", link.text)
    return text

def process_references_list(element):
    references = []
    for li in element.find_all('li', recursive=False):
        ref_text = li.get_text(separator=" ", strip=True)
        references.append(ref_text)
    return references

def save_markdown(filepath, content):
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def process_url(url, output_path):
    title, result, external_links = wikipedia_to_markdown(url)
    if title is None:
        print(f"Error processing {url}: {result}")
        return
    
    filename = re.sub(r'[^\w\-_\. ]', '_', title)
    markdown_filepath = os.path.join(output_path, filename + '.md')
    save_markdown(markdown_filepath, result)
    
    if external_links:
        links_filepath = os.path.join(output_path, filename + '_linkz.md')
        save_markdown(links_filepath, '\n'.join(external_links))
    
    print(f"Processed: {url} -> {markdown_filepath}")
    if external_links:
        print(f"External links saved to: {links_filepath}")

def process_file(file_path, output_path):
    with open(file_path, 'r') as f:
        urls = f.read().splitlines()
    for url in urls:
        process_url(url.strip(), output_path)

def main():
    parser = argparse.ArgumentParser(description="Convert Wikipedia pages to Markdown")
    parser.add_argument("input", help="URL of Wikipedia page or path to file containing URLs")
    parser.add_argument("output", help="Output directory for Markdown files")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    if args.input.startswith('http'):
        process_url(args.input, args.output)
    else:
        process_file(args.input, args.output)

if __name__ == "__main__":
    main()
