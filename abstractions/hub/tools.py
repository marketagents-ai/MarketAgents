"""Default tools and schemas for the chat system."""

from typing import Dict, Any

CHAIN_OF_THOUGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "thought_process": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "step": {"type": "integer"},
                    "thought": {"type": "string"},
                    "reasoning": {"type": "string"}
                },
                "required": ["step", "thought", "reasoning"]
            }
        },
        "final_answer": {"type": "string"}
    },
    "required": ["thought_process", "final_answer"]
}

JUNGIAN_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "inner_thoughts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "archetype": {"type": "string", "description": "The universal symbol or theme present in the thought."},
                    "symbolism": {"type": "string", "description": "The symbolic meaning or imagery associated with the thought."},
                    "conscious_interaction": {"type": "string", "description": "How the thought interacts with conscious awareness."},
                    "unconscious_influence": {"type": "string", "description": "The influence of the unconscious mind on the thought."},
                    "emotional_tone": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "The emotional tone or feeling associated with the thought."
                    }
                },
                "required": ["archetype", "symbolism", "conscious_interaction", "unconscious_influence", "emotional_tone"]
            }
        },
        "holistic_summary": {"type": "string", "description": "A summary that integrates the various elements of the thought process into a cohesive understanding."}
    },
    "required": ["inner_thoughts", "holistic_summary"]
}

QUANTUM_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "superposition_states": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "state_vector": {"type": "string", "description": "A possible decision state or outcome"},
                    "amplitude": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "The probability amplitude of this state"
                    },
                    "interference_patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Other states this decision interacts with"
                    }
                },
                "required": ["state_vector", "amplitude", "interference_patterns"]
            }
        },
        "entangled_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "factor": {"type": "string"},
                    "influence": {"type": "string"},
                    "correlation_strength": {
                        "type": "number",
                        "minimum": -1,
                        "maximum": 1
                    }
                }
            }
        },
        "collapsed_state": {
            "type": "object",
            "properties": {
                "decision": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "uncertainty_principle": {"type": "string", "description": "What remains fundamentally uncertain about this decision"}
            },
            "required": ["decision", "confidence", "uncertainty_principle"]
        }
    },
    "required": ["superposition_states", "entangled_factors", "collapsed_state"]
}

MYTHOLOGICAL_PATTERN_SCHEMA = {
    "type": "object",
    "properties": {
        "archetypal_journey": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string", "description": "Current stage in the mythological cycle"},
                    "challenges": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Obstacles and trials faced"
                    },
                    "allies_and_mentors": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "archetype": {"type": "string"},
                                "role": {"type": "string"},
                                "gift_or_wisdom": {"type": "string"}
                            }
                        }
                    },
                    "transformation": {"type": "string", "description": "The change or growth occurring at this stage"}
                },
                "required": ["stage", "challenges", "allies_and_mentors", "transformation"]
            }
        },
        "symbolic_elements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "cultural_context": {"type": "string"},
                    "universal_meaning": {"type": "string"},
                    "personal_resonance": {"type": "string"}
                }
            }
        },
        "prophecy": {
            "type": "object",
            "properties": {
                "vision": {"type": "string", "description": "The revealed pattern or insight"},
                "omens": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "divine_message": {"type": "string"}
            },
            "required": ["vision", "omens", "divine_message"]
        }
    },
    "required": ["archetypal_journey", "symbolic_elements", "prophecy"]
}

# Default tools configuration
DEFAULT_TOOLS: Dict[str, Dict[str, Any]] = {
    "reasoning_steps": {
        "schema": CHAIN_OF_THOUGHT_SCHEMA,
        "description": "Break down the reasoning process step by step",
        "instruction": "Please follow this step-by-step reasoning format:"
    },
    "jungian_analysis": {
        "schema": JUNGIAN_ANALYSIS_SCHEMA,
        "description": "A psychological analysis tool based on Jungian principles",
        "instruction": "Please analyze using Jungian psychological concepts:"
    },
    "quantum_decision": {
        "schema": QUANTUM_DECISION_SCHEMA,
        "description": "A decision-making tool using quantum metaphors",
        "instruction": "Please analyze using quantum decision-making concepts:"
    },
    "mythological_pattern": {
        "schema": MYTHOLOGICAL_PATTERN_SCHEMA,
        "description": "An archetypal analysis tool using mythological patterns",
        "instruction": "Please interpret using mythological archetypes:"
    }
}



