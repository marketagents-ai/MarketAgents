# Cognition Modules

Landing page for a all our coginition module experiments.

## Objective
Create a framework for AI cognition using keyed modules for context enhancement with LLMs.

## Core Components
1. Persona Object {keyed}
2. Memory Object {keyed}
3. Reasoning Object {keyed}
4. LLM

## Framework Flow
(Persona + {keyed}Memory + {keyed}Reasoning) → LLM Synthesizer → Response

## Key Concepts
- Memory and Reasoning objects contain {keyed} verbs for internet object correlation
- LLM uses keys to add relevant context to input text
- Synthesis occurs by combining Persona with keyed context from Memory and Reasoning

## Process
1. Retrieve relevant {keyed} information from Memory and Reasoning
2. LLM expands keys with internet-correlated context
3. Combine expanded context with Persona
4. Synthesize final response

## Implementation Focus
- Efficient key-based retrieval system
- Flexible integration with various LLM providers
- Streamlined context expansion process
- Really good Prompting to injest context