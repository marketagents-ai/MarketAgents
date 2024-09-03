from dataclasses import dataclass

@dataclass
class Config:
    db_path: str
    max_search_limit: int = 1000
    forget_threshold_days: int = 30
    forget_access_threshold: int = 5