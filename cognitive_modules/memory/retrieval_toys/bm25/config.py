import yaml
from typing import List, Dict, Any

class DataSource:
    def __init__(self, type: str, path: str, format: str):
        self.type = type
        self.path = path
        self.format = format

class Config:
    def __init__(self, data: Dict[str, Any]):
        self.data_sources = [DataSource(**source) for source in data['data_sources']]
        self.index_path = data['index_path']
        self.chunk_size = data['chunk_size']
        self.overlap = data['overlap']
        self.ensemble_weight = data['ensemble_weight']
        self.num_workers = data['num_workers']

    @classmethod
    def load(cls, config_path: str) -> 'Config':
        with open(config_path, 'r') as f:
            return cls(yaml.safe_load(f))