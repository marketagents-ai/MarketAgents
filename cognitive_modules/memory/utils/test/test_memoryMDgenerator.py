"""
pytest -v -s cognitive_modules/memory/utils/test/test_memoryMDgenerator.py
"""

import pytest
import sys
import os
import random
from nltk.corpus import wordnet as wn

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
sys.path.append(project_root)

from cognitive_modules.memory.utils.memoryMDgenerator import MemoryModule, generate_mock_corpus

@pytest.fixture
def mock_corpus():
    return generate_mock_corpus(num_conversations=20, num_memories=40)

@pytest.fixture
def memory_module(mock_corpus):
    return MemoryModule(max_conversation_history=5, corpus=mock_corpus)

def generate_random_sentence():
    nouns = list(wn.all_synsets(pos=wn.NOUN))
    verbs = list(wn.all_synsets(pos=wn.VERB))
    adjectives = list(wn.all_synsets(pos=wn.ADJ))

    noun = random.choice(nouns).lemmas()[0].name()
    verb = random.choice(verbs).lemmas()[0].name()
    adj = random.choice(adjectives).lemmas()[0].name()

    return f"The {adj} {noun} {verb}s."

def test_add_conversation(memory_module):
    question = generate_random_sentence()
    answer = generate_random_sentence()
    memory_module.add_conversation(question, answer)
    assert len(memory_module.conversation_history) == 6  # 5 from mock corpus + 1 new
    assert memory_module.conversation_history[-1] == {"q": question, "a": answer}
    print(f"\nAdded conversation:\nQ: {question}\nA: {answer}")

def test_add_short_term_memory(memory_module):
    memory = generate_random_sentence()
    initial_count = len(memory_module.short_term_memory)
    memory_module.add_short_term_memory(memory)
    assert len(memory_module.short_term_memory) == initial_count + 1
    assert memory_module.short_term_memory[-1] == memory
    print(f"\nAdded short-term memory: {memory}")

def test_add_long_term_memory(memory_module):
    memory = generate_random_sentence()
    initial_count = len(memory_module.long_term_memory)
    memory_module.add_long_term_memory(memory)
    assert len(memory_module.long_term_memory) == initial_count + 1
    assert memory_module.long_term_memory[-1] == memory
    print(f"\nAdded long-term memory: {memory}")

def test_add_external_reference(memory_module):
    title = f"Reference to {random.choice(list(wn.all_synsets(pos=wn.NOUN))).lemmas()[0].name()}"
    url = f"https://example.com/{title.lower().replace(' ', '-')}"
    memory_module.add_external_reference(title, url)
    assert len(memory_module.external_references) == 1
    assert memory_module.external_references[0] == {"title": title, "url": url}
    print(f"\nAdded external reference:\nTitle: {title}\nURL: {url}")

def test_get_memory_artifact(memory_module):
    # Calculate relevance score before generating the artifact
    query = generate_random_sentence()
    memory_module.calculate_relevance_score(query)
    
    artifact = memory_module.get_memory_artifact()
    assert isinstance(artifact, str)
    assert "# Memory Module for Agent Context" in artifact
    assert "## Conversation History" in artifact
    assert "## Short-term Memory" in artifact
    assert "## Long-term Memory" in artifact
    assert "## External References" in artifact
    assert "## Context Metadata" in artifact
    
    print("\nGenerated Memory Artifact:")
    print(artifact)

def test_max_conversation_history(memory_module):
    for i in range(10):
        memory_module.add_conversation(f"Question {i}", f"Answer {i}")
    assert len(memory_module.conversation_history) == 5
    assert memory_module.conversation_history[0]["q"] == "Question 5"
    print("\nConversation history after exceeding max_conversation_history:")
    for conv in memory_module.conversation_history:
        print(f"Q: {conv['q']}\nA: {conv['a']}")

def test_calculate_relevance_score(memory_module, mock_corpus):
    all_texts = (
        [f"{conv['q']} {conv['a']}" for conv in mock_corpus["conversations"]] +
        mock_corpus["short_term_memory"] +
        mock_corpus["long_term_memory"]
    )
    query = random.choice(all_texts)
    score = memory_module.calculate_relevance_score(query)
    assert isinstance(score, float)
    assert 0 <= score <= 1
    print(f"\nCalculated relevance score for query '{query}': {score}")

def test_ingest_corpus(memory_module, mock_corpus):
    assert len(memory_module.search_collection.items) == len(mock_corpus["conversations"]) + len(mock_corpus["short_term_memory"]) + len(mock_corpus["long_term_memory"])
    assert len(memory_module.conversation_history) <= memory_module.max_conversation_history
    assert len(memory_module.short_term_memory) == len(mock_corpus["short_term_memory"])
    assert len(memory_module.long_term_memory) == len(mock_corpus["long_term_memory"])
    print(f"\nIngested corpus with {len(mock_corpus['conversations'])} conversations, {len(mock_corpus['short_term_memory'])} short-term memories, and {len(mock_corpus['long_term_memory'])} long-term memories")

def test_search_functionality(memory_module, mock_corpus):
    all_texts = (
        [f"{conv['q']} {conv['a']}" for conv in mock_corpus["conversations"]] +
        mock_corpus["short_term_memory"] +
        mock_corpus["long_term_memory"]
    )
    query = random.choice(all_texts)
    results = memory_module.search(query, top_k=5)
    assert len(results) == 5
    print(f"\nSearch results for query '{query}':")
    for item, score in results:
        print(f"- {item} (score: {score})")

def test_apply_forgetting_algorithm(memory_module):
    # This test is a placeholder since the method is not implemented yet
    memory_module.apply_forgetting_algorithm()
    print("\nApplied forgetting algorithm (placeholder)")

if __name__ == "__main__":
    pytest.main([__file__])