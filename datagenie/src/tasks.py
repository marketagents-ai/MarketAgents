import os
import csv
from itertools import islice
from typing import Optional
from pydantic import BaseModel

from src.utils import get_source_dir

class Task(BaseModel):
    Category: str
    SubCategory: str
    Task: str
    Meta_Data: Optional[dict] = {}

def load_tasks(generation_type: str, num_tasks: int) -> list[Task]:
    # Construct the path to the curriculum CSV file
    data_genie_agents_path = get_source_dir()
    curriculum_csv_path = os.path.join("configs/curriculum", f"{generation_type}.csv")
    total_path = os.path.join(data_genie_agents_path, curriculum_csv_path)

    tasks = []
    with open(total_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file, skipinitialspace=True)
        for row in islice(reader, num_tasks):
            print(row)
            task = Task(
                Category=row['Category'],
                SubCategory=row['SubCategory'],
                Task=row['Task']
            )
            tasks.append(task)

    return tasks