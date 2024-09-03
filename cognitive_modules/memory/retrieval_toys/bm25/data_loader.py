import os
import json
from typing import List, Dict, Any
import aiofiles
import asyncio
from config import DataSource

class DataItem:
    def __init__(self, content: str, metadata: Dict[str, Any]):
        self.content = content
        self.metadata = metadata

async def load_text(path: str) -> List[DataItem]:
    async with aiofiles.open(path, 'r') as f:
        content = await f.read()
    return [DataItem(content, {"path": path})]

async def load_json(path: str) -> List[DataItem]:
    async with aiofiles.open(path, 'r') as f:
        data = json.loads(await f.read())
    return [DataItem(str(item), {"path": path, "index": i}) for i, item in enumerate(data)]

async def load_documents(path: str) -> List[DataItem]:
    items = []
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
            items.append(DataItem(content, {"path": file_path}))
    return items

LOADERS = {
    "text": load_text,
    "json": load_json,
    "documents": load_documents
}

async def load_data(sources: List[DataSource]) -> List[DataItem]:
    tasks = [LOADERS[source.type](source.path) for source in sources]
    results = await asyncio.gather(*tasks)
    return [item for sublist in results for item in sublist]