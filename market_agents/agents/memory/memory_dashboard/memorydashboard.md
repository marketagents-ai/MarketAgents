# Agent Psychometrics Dashboard

A web-based tool for analyzing agent interactions in multi-agent systems, focusing on a initial skeletal framework for searching agent memories from json files. With the goal of including our existing remote sql database and presenting a table of agents with columns for each field defined in the pydantic model schema.

## Current Features
- Natural language memory search
- Agent-specific filtering
- Semantic relationship visualization

## Architecture
- Backend: FastAPI, Dash, MemoryManager
- Frontend: HTML, JavaScript, Plotly.js

## Quick Start
1. Run Python server script
2. Open HTML file in browser
3. Select agents, enter queries, explore results

## Main Dependencies
FastAPI, Dash, NumPy, scikit-learn, Plotly.js, Marked.js

## Work Yet To Be Done
- ingest raw json files to populate a dynamictable with the fields defined in the pydantic model schema
- add a search bar to the table to search for all fields in the table
- toggle between local db and remote sql database



