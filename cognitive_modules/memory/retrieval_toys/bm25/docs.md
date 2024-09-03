# BM25 Search Module

## Overview

This search module combines BM25 and TF-IDF algorithms to provide powerful and flexible search capabilities across various data sources. It's designed to handle multiple input formats, including plain text, JSON, and document folders, making it versatile for different use cases.

## Features

- Supports multiple data input formats (text, JSON, documents)
- Combines BM25 and cosine similarity for enhanced search results
- Asynchronous data loading and processing
- Parallel computation for improved performance
- Configurable via YAML files
- Separate indexing and search functionalities

## Installation


1. Install the required dependencies:
   ```
   pip install aiofiles pyyaml scikit-learn numpy
   ```

## Usage

### Configuration

Create a `config.yaml` file with the following structure:

```yaml
data_sources:
  - type: text
    path: /path/to/text/file.txt
    format: plain
  - type: json
    path: /path/to/json/file.json
    format: json
  - type: documents
    path: /path/to/document/folder
    format: markdown
index_path: /path/to/save/index.pkl
chunk_size: 1000
overlap: 200
ensemble_weight: 0.7
num_workers: 4
```

### Indexing

To build the search index:

```
python main.py --config config.yaml --action index
```

This will process the data sources specified in the config file and create an index at the specified `index_path`.

### Searching

To perform a search:

```
python main.py --config config.yaml --action search --query "your search query"
```

This will return the top matching results based on the combined BM25 and cosine similarity scores.

## Module Structure

- `main.py`: Entry point of the application
- `config.py`: Configuration handling
- `data_loader.py`: Asynchronous data loading from various sources
- `indexer.py`: Index building using BM25 and TF-IDF
- `searcher.py`: Search functionality using the built index

## Customization

You can extend the functionality of this module by:

1. Adding new data source types in `data_loader.py`
2. Modifying the scoring algorithm in `indexer.py` and `searcher.py`
3. Adjusting the configuration options in `config.py`

## Performance Considerations

- The module uses multiprocessing for BM25 score calculation, which can be CPU-intensive. Adjust the `num_workers` in the config file based on your system's capabilities.
- For very large datasets, consider implementing batch processing or streaming to manage memory usage.
