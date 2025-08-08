#!/bin/bash
# activate datagen environment
source venv/bin/activate

# initialize redis vectordb on docker
#./run_vectordb.sh

# run data correction pipeline
python datagen.py --generation_type financial_rag --num_tasks 1 --agent_config finance_agents_groq.json
