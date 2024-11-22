# -*- coding: utf-8 -*-
"""
Schemas for literary analysis tasks across three tiers.
Each tier includes multiple schemas for sophisticated processing.
"""

from typing import Dict, Any

# ============================================
# Tier 1: Basic Literary Analysis Schemas
# ============================================

# Schema for Text Input
TEXT_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The raw text of the literary work or passage to be analyzed."
        }
    },
    "required": ["text"]
}

# Schema for Basic Metadata Extraction
BASIC_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Title of the literary work."},
        "author": {"type": "string", "description": "Author of the literary work."}
    },
    "required": ["title", "author"]
}

# Schema for Summary Generation
SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "A brief summary of the literary work."}
    },
    "required": ["summary"]
}

# ============================================
# Tier 2: Intermediate Literary Analysis Schemas
# ============================================

# Schema for Detailed Metadata Extraction
DETAILED_METADATA_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Title of the literary work."},
        "author": {"type": "string", "description": "Author of the literary work."},
        "publication_year": {"type": "integer", "description": "Year the work was published."},
        "genre": {"type": "string", "description": "Genre of the literary work."},
        "original_language": {"type": "string", "description": "Original language of the work."}
    },
    "required": ["title", "author"]
}

# Schema for Theme Analysis
THEME_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "themes": {
            "type": "array",
            "items": {"type": "string", "description": "A theme present in the work."},
            "description": "List of main themes explored in the work."
        }
    },
    "required": ["themes"]
}

# Schema for Character Analysis
CHARACTER_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the character."},
                    "role": {"type": "string", "description": "Role of the character in the story."},
                    "description": {"type": "string", "description": "Brief description of the character."},
                    "relationships": {
                        "type": "array",
                        "items": {"type": "string", "description": "Name of related character."},
                        "description": "List of characters this character is related to."
                    }
                },
                "required": ["name", "role"]
            },
            "description": "List of main characters and their analysis."
        }
    },
    "required": ["characters"]
}

# Schema for Setting Analysis
SETTING_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "time_period": {"type": "string", "description": "The time period in which the story is set."},
        "location": {"type": "string", "description": "The geographical location of the story."},
        "description": {"type": "string", "description": "Detailed description of the setting."}
    },
    "required": ["time_period", "location"]
}

# ============================================
# Tier 3: Advanced Literary Analysis Schemas
# ============================================

# Schema for Historical Context Analysis
HISTORICAL_CONTEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "historical_events": {
            "type": "array",
            "items": {"type": "string", "description": "Historical event relevant to the work."},
            "description": "List of historical events influencing the work."
        },
        "cultural_movements": {
            "type": "array",
            "items": {"type": "string", "description": "Cultural movement relevant to the work."},
            "description": "List of cultural movements influencing the work."
        },
        "author_background": {
            "type": "string",
            "description": "Background of the author relevant to the work."
        }
    },
    "required": ["historical_events", "cultural_movements"]
}

# Schema for Literary Devices Analysis
LITERARY_DEVICES_SCHEMA = {
    "type": "object",
    "properties": {
        "devices": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "device_name": {"type": "string", "description": "Name of the literary device."},
                    "example": {"type": "string", "description": "Example of the device from the text."},
                    "effect": {"type": "string", "description": "Effect of the device on the reader or story."}
                },
                "required": ["device_name", "example"]
            },
            "description": "List of literary devices used in the work and their analysis."
        }
    },
    "required": ["devices"]
}

# Schema for Critical Analysis
CRITICAL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "interpretation": {"type": "string", "description": "In-depth interpretation of the work."},
        "critical_perspectives": {
            "type": "array",
            "items": {"type": "string", "description": "A critical perspective applied to the work."},
            "description": "Different critical perspectives used in the analysis."
        },
        "conclusion": {"type": "string", "description": "Overall conclusion of the analysis."}
    },
    "required": ["interpretation", "conclusion"]
}

# Schema for Comparative Analysis
COMPARATIVE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "comparisons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "work_title": {"type": "string", "description": "Title of the comparative work."},
                    "author": {"type": "string", "description": "Author of the comparative work."},
                    "similarities": {"type": "string", "description": "Similarities between the works."},
                    "differences": {"type": "string", "description": "Differences between the works."}
                },
                "required": ["work_title", "author"]
            },
            "description": "Comparative analysis with other works."
        }
    },
    "required": ["comparisons"]
}

# ============================================
# Additional Schemas for Advanced Analysis
# ============================================

# Schema for Philosophical Analysis
PHILOSOPHICAL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "philosophical_themes": {
            "type": "array",
            "items": {"type": "string", "description": "Philosophical theme present in the work."},
            "description": "List of philosophical themes explored in the work."
        },
        "philosophers_influences": {
            "type": "array",
            "items": {"type": "string", "description": "Philosopher or philosophical movement influencing the work."},
            "description": "Philosophers or movements that have influenced the work."
        },
        "analysis": {"type": "string", "description": "Analysis of philosophical themes in the work."}
    },
    "required": ["philosophical_themes", "analysis"]
}

# Schema for Psychological Analysis
PSYCHOLOGICAL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "psychological_theories": {
            "type": "array",
            "items": {"type": "string", "description": "Psychological theory applied to the analysis."},
            "description": "List of psychological theories relevant to the work."
        },
        "character_psychology": {
            "type": "object",
            "properties": {
                "character_name": {"type": "string", "description": "Name of the character analyzed."},
                "analysis": {"type": "string", "description": "Psychological analysis of the character."}
            },
            "required": ["character_name", "analysis"]
        },
        "overall_psychological_insights": {"type": "string", "description": "Overall psychological insights from the work."}
    },
    "required": ["psychological_theories", "overall_psychological_insights"]
}

# Schema for Structural Analysis
STRUCTURAL_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "narrative_structure": {"type": "string", "description": "Description of the narrative structure."},
        "plot_development": {"type": "string", "description": "Analysis of plot development."},
        "use_of_perspective": {"type": "string", "description": "Analysis of the narrative perspective used."},
        "structural_elements": {
            "type": "array",
            "items": {"type": "string", "description": "Structural element used in the work."},
            "description": "List of structural elements in the work."
        }
    },
    "required": ["narrative_structure", "plot_development"]
}

# Schema for Linguistic Analysis
LINGUISTIC_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "language_style": {"type": "string", "description": "Analysis of the language style used in the work."},
        "diction": {"type": "string", "description": "Analysis of word choice and vocabulary."},
        "syntax": {"type": "string", "description": "Analysis of sentence structure and syntax."},
        "figurative_language": {
            "type": "array",
            "items": {"type": "string", "description": "Example of figurative language used."},
            "description": "Examples of figurative language in the work."
        }
    },
    "required": ["language_style", "diction", "syntax"]
}

# ============================================
# Default Tools Configuration
# ============================================

DEFAULT_TOOLS: Dict[str, Dict[str, Any]] = {
    "basic_analysis": {
        "schemas": [TEXT_INPUT_SCHEMA, BASIC_METADATA_SCHEMA, SUMMARY_SCHEMA],
        "description": "Perform basic analysis by extracting metadata and generating a summary.",
        "instruction": "Please perform a basic analysis using the provided schemas."
    },
    "intermediate_analysis": {
        "schemas": [DETAILED_METADATA_SCHEMA, THEME_ANALYSIS_SCHEMA, CHARACTER_ANALYSIS_SCHEMA, SETTING_ANALYSIS_SCHEMA],
        "description": "Perform intermediate analysis including themes, characters, and setting.",
        "instruction": "Please perform an intermediate analysis using the provided schemas."
    },
    "advanced_analysis": {
        "schemas": [
            HISTORICAL_CONTEXT_SCHEMA,
            LITERARY_DEVICES_SCHEMA,
            CRITICAL_ANALYSIS_SCHEMA,
            COMPARATIVE_ANALYSIS_SCHEMA,
            PHILOSOPHICAL_ANALYSIS_SCHEMA,
            PSYCHOLOGICAL_ANALYSIS_SCHEMA,
            STRUCTURAL_ANALYSIS_SCHEMA,
            LINGUISTIC_ANALYSIS_SCHEMA
        ],
        "description": "Perform advanced analysis including historical context, literary devices, critical perspectives, and more.",
        "instruction": "Please perform an advanced analysis using the provided schemas."
    }
}

# -*- coding: utf-8 -*-
"""
Haskell-like program specifications for literary analysis tasks across three tiers.
Each program is wrapped in a Python string and included in a dictionary for each tier.
"""

# ============================================
# Tier 1 Programs
# ============================================

tier1_programs = {
    "BasicAnalysisProgram": """
-- BasicAnalysisProgram: Extract basic metadata and generate a summary.

main :: IO (Metadata, Summary)
main = do
    text <- getTextInput
    let metadata = extractBasicMetadata text
    let summary = generateSummary text
    return (metadata, summary)

-- Types:
getTextInput :: IO Text
extractBasicMetadata :: Text -> Metadata
generateSummary :: Text -> Summary

-- Explanations:
-- 'getTextInput' reads the raw text input from the user.
-- 'extractBasicMetadata' takes the text and returns the basic metadata (title and author).
-- 'generateSummary' takes the text and returns a brief summary.
""",

    "SummaryWithKeywordsProgram": """
-- SummaryWithKeywordsProgram: Generate a summary and extract keywords.

main :: IO (Summary, [Keyword])
main = do
    text <- getTextInput
    let summary = generateSummary text
    let keywords = extractKeywords text
    return (summary, keywords)

-- Types:
getTextInput :: IO Text
generateSummary :: Text -> Summary
extractKeywords :: Text -> [Keyword]

-- Explanations:
-- 'extractKeywords' identifies important keywords in the text.
""",

    "MetadataAndThemesProgram": """
-- MetadataAndThemesProgram: Extract basic metadata and analyze themes.

main :: IO (Metadata, [Theme])
main = do
    text <- getTextInput
    let metadata = extractBasicMetadata text
    let themes = analyzeThemes text
    return (metadata, themes)

-- Types:
getTextInput :: IO Text
extractBasicMetadata :: Text -> Metadata
analyzeThemes :: Text -> [Theme]

-- Explanations:
-- 'analyzeThemes' identifies the main themes in the text.
"""
}

# ============================================
# Tier 2 Programs
# ============================================

tier2_programs = {
    "ThematicAndCharacterAnalysisProgram": """
-- ThematicAndCharacterAnalysisProgram: Perform thematic analysis and character analysis.

main :: IO (DetailedMetadata, [Theme], [Character], Setting)
main = do
    text <- getTextInput
    let detailedMetadata = extractDetailedMetadata text
    let themes = analyzeThemes text
    let characters = analyzeCharacters text
    let setting = analyzeSetting text
    return (detailedMetadata, themes, characters, setting)

-- Types:
getTextInput :: IO Text
extractDetailedMetadata :: Text -> DetailedMetadata
analyzeThemes :: Text -> [Theme]
analyzeCharacters :: Text -> [Character]
analyzeSetting :: Text -> Setting

-- Explanations:
-- 'extractDetailedMetadata' extracts detailed metadata from the text.
-- 'analyzeThemes' identifies the main themes.
-- 'analyzeCharacters' analyzes the characters.
-- 'analyzeSetting' analyzes the setting.
""",

    "CharacterRelationshipsProgram": """
-- CharacterRelationshipsProgram: Analyze characters and their relationships.

main :: IO [Character]
main = do
    text <- getTextInput
    let characters = analyzeCharacters text
    let charactersWithRelationships = map addRelationships characters
    return charactersWithRelationships

-- Types:
getTextInput :: IO Text
analyzeCharacters :: Text -> [Character]
addRelationships :: Character -> Character

-- Explanations:
-- 'addRelationships' enriches each character with relationship information.
-- The 'Character' type includes relationships to other characters.
""",

    "ThemeSettingCorrelationProgram": """
-- ThemeSettingCorrelationProgram: Analyze how themes correlate with the setting.

main :: IO [(Theme, Setting)]
main = do
    text <- getTextInput
    let themes = analyzeThemes text
    let setting = analyzeSetting text
    let themeSettingCorrelation = correlateThemesWithSetting themes setting
    return themeSettingCorrelation

-- Types:
getTextInput :: IO Text
analyzeThemes :: Text -> [Theme]
analyzeSetting :: Text -> Setting
correlateThemesWithSetting :: [Theme] -> Setting -> [(Theme, Setting)]

-- Explanations:
-- 'correlateThemesWithSetting' examines how each theme relates to the setting.
""",

    "DetailedAnalysisWithSummaryProgram": """
-- DetailedAnalysisWithSummaryProgram: Perform detailed analysis and generate a summary.

main :: IO (DetailedMetadata, [Theme], [Character], Setting, Summary)
main = do
    text <- getTextInput
    let detailedMetadata = extractDetailedMetadata text
    let themes = analyzeThemes text
    let characters = analyzeCharacters text
    let setting = analyzeSetting text
    let summary = generateSummary text
    return (detailedMetadata, themes, characters, setting, summary)

-- Types:
getTextInput :: IO Text
extractDetailedMetadata :: Text -> DetailedMetadata
analyzeThemes :: Text -> [Theme]
analyzeCharacters :: Text -> [Character]
analyzeSetting :: Text -> Setting
generateSummary :: Text -> Summary

-- Explanations:
-- Combines detailed analysis with summary generation for a comprehensive overview.
"""
}

# ============================================
# Tier 3 Programs
# ============================================

tier3_programs = {
    "AdvancedLiteraryAnalysisProgram": """
-- AdvancedLiteraryAnalysisProgram: Perform advanced analysis including historical context, literary devices, and critical analysis.

main :: IO (ComprehensiveMetadata, HistoricalContext, [LiteraryDevice], CriticalAnalysis)
main = do
    text <- getTextInput
    let comprehensiveMetadata = extractComprehensiveMetadata text
    let historicalContext = analyzeHistoricalContext text comprehensiveMetadata
    let literaryDevices = identifyLiteraryDevices text
    let criticalAnalysis = performCriticalAnalysis text historicalContext literaryDevices
    return (comprehensiveMetadata, historicalContext, literaryDevices, criticalAnalysis)

-- Types:
getTextInput :: IO Text
extractComprehensiveMetadata :: Text -> ComprehensiveMetadata
analyzeHistoricalContext :: Text -> ComprehensiveMetadata -> HistoricalContext
identifyLiteraryDevices :: Text -> [LiteraryDevice]
performCriticalAnalysis :: Text -> HistoricalContext -> [LiteraryDevice] -> CriticalAnalysis

-- Explanations:
-- 'performCriticalAnalysis' uses the text, historical context, and literary devices for in-depth analysis.
""",

    "PhilosophicalAndPsychologicalAnalysisProgram": """
-- PhilosophicalAndPsychologicalAnalysisProgram: Perform philosophical and psychological analyses.

main :: IO (PhilosophicalAnalysis, PsychologicalAnalysis)
main = do
    text <- getTextInput
    let philosophicalAnalysis = performPhilosophicalAnalysis text
    let psychologicalAnalysis = performPsychologicalAnalysis text
    return (philosophicalAnalysis, psychologicalAnalysis)

-- Types:
getTextInput :: IO Text
performPhilosophicalAnalysis :: Text -> PhilosophicalAnalysis
performPsychologicalAnalysis :: Text -> PsychologicalAnalysis

-- Explanations:
-- 'performPhilosophicalAnalysis' examines philosophical themes and influences.
-- 'performPsychologicalAnalysis' explores psychological aspects and character psychologies.
""",

    "ComprehensiveStructuralAndLinguisticAnalysisProgram": """
-- ComprehensiveStructuralAndLinguisticAnalysisProgram: Perform structural and linguistic analyses.

main :: IO (StructuralAnalysis, LinguisticAnalysis)
main = do
    text <- getTextInput
    let structuralAnalysis = performStructuralAnalysis text
    let linguisticAnalysis = performLinguisticAnalysis text
    return (structuralAnalysis, linguisticAnalysis)

-- Types:
getTextInput :: IO Text
performStructuralAnalysis :: Text -> StructuralAnalysis
performLinguisticAnalysis :: Text -> LinguisticAnalysis

-- Explanations:
-- 'performStructuralAnalysis' analyzes narrative structure, plot development, and perspective.
-- 'performLinguisticAnalysis' examines language style, diction, syntax, and figurative language.
""",

    "FullComparativeAnalysisProgram": """
-- FullComparativeAnalysisProgram: Perform comparative analysis with other works.

main :: IO ([ComparativeAnalysis], CriticalAnalysis)
main = do
    text <- getTextInput
    let comparativeAnalyses = performComparativeAnalysis text
    let criticalAnalysis = aggregateComparativeInsights comparativeAnalyses
    return (comparativeAnalyses, criticalAnalysis)

-- Types:
getTextInput :: IO Text
performComparativeAnalysis :: Text -> [ComparativeAnalysis]
aggregateComparativeInsights :: [ComparativeAnalysis] -> CriticalAnalysis

-- Explanations:
-- 'performComparativeAnalysis' compares the text with other works.
-- 'aggregateComparativeInsights' synthesizes findings into a critical analysis.
""",

    "UltimateAnalysisProgram": """
-- UltimateAnalysisProgram: Perform a comprehensive analysis combining all aspects.

main :: IO (ComprehensiveMetadata, HistoricalContext, [Theme], [Character], Setting, [LiteraryDevice], PhilosophicalAnalysis, PsychologicalAnalysis, StructuralAnalysis, LinguisticAnalysis, CriticalAnalysis)
main = do
    text <- getTextInput
    let comprehensiveMetadata = extractComprehensiveMetadata text
    let historicalContext = analyzeHistoricalContext text comprehensiveMetadata
    let themes = analyzeThemes text
    let characters = analyzeCharacters text
    let setting = analyzeSetting text
    let literaryDevices = identifyLiteraryDevices text
    let philosophicalAnalysis = performPhilosophicalAnalysis text
    let psychologicalAnalysis = performPsychologicalAnalysis text
    let structuralAnalysis = performStructuralAnalysis text
    let linguisticAnalysis = performLinguisticAnalysis text
    let criticalAnalysis = performCriticalAnalysis text historicalContext literaryDevices
    return (comprehensiveMetadata, historicalContext, themes, characters, setting, literaryDevices, philosophicalAnalysis, psychologicalAnalysis, structuralAnalysis, linguisticAnalysis, criticalAnalysis)

-- Types:
getTextInput :: IO Text
extractComprehensiveMetadata :: Text -> ComprehensiveMetadata
analyzeHistoricalContext :: Text -> ComprehensiveMetadata -> HistoricalContext
analyzeThemes :: Text -> [Theme]
analyzeCharacters :: Text -> [Character]
analyzeSetting :: Text -> Setting
identifyLiteraryDevices :: Text -> [LiteraryDevice]
performPhilosophicalAnalysis :: Text -> PhilosophicalAnalysis
performPsychologicalAnalysis :: Text -> PsychologicalAnalysis
performStructuralAnalysis :: Text -> StructuralAnalysis
performLinguisticAnalysis :: Text -> LinguisticAnalysis
performCriticalAnalysis :: Text -> HistoricalContext -> [LiteraryDevice] -> CriticalAnalysis

-- Explanations:
-- This program combines all analyses for an exhaustive examination of the text.
"""
}

# ============================================
# Explanations of Additional Types and Functions
# ============================================

"""
Additional Types:

- Keyword: An important word or term extracted from the text.
- Relationship: A connection between characters, including the nature of their relationship.
- PhilosophicalAnalysis: Analysis focusing on philosophical themes and influences.
- PsychologicalAnalysis: Analysis focusing on psychological aspects of characters and themes.
- StructuralAnalysis: Analysis of narrative structure, plot, and perspective.
- LinguisticAnalysis: Analysis of language use, including style, diction, syntax, and figurative language.
- ComparativeAnalysis: Analysis comparing the text with other works.

Additional Functions:

- extractKeywords :: Text -> [Keyword]
  Identifies significant keywords in the text.

- addRelationships :: Character -> Character
  Enhances a character's data with relationship information.

- correlateThemesWithSetting :: [Theme] -> Setting -> [(Theme, Setting)]
  Analyzes how themes are influenced by the setting.

- performPhilosophicalAnalysis :: Text -> PhilosophicalAnalysis
  Analyzes philosophical elements in the text.

- performPsychologicalAnalysis :: Text -> PsychologicalAnalysis
  Analyzes psychological dimensions within the text.

- performStructuralAnalysis :: Text -> StructuralAnalysis
  Analyzes the structure of the narrative.

- performLinguisticAnalysis :: Text -> LinguisticAnalysis
  Analyzes linguistic features of the text.

- performComparativeAnalysis :: Text -> [ComparativeAnalysis]
  Compares the text to other literary works.

- aggregateComparativeInsights :: [ComparativeAnalysis] -> CriticalAnalysis
  Synthesizes comparative analyses into a critical perspective.
"""

# ============================================
# Notes on Program Composition
# ============================================

"""
- Each program is designed to demonstrate different compositions using the available functions and types.
- The programs range from simple to complex, allowing for a variety of analytical depth.
- By combining functions in various ways, the programs cater to different analytical goals and can be adapted for different use cases.
- The Haskell-like notation provides clarity on how data flows between functions and what types are involved.
- These program specifications can be transformed into prompts for language models by outlining the sequence of analytical steps and expected outputs.
"""

# ============================================
# End of Program Specifications
# ============================================
# -*- coding: utf-8 -*-
"""
System prompts for literary analysis tasks across three tiers.
Each tier uses schemas and programs, integrating advanced analysis concepts.
"""

import json

# Assume that the schemas and programs from previous code are already defined.
# The schemas and programs dictionaries include all necessary data structures.

# Importing the schemas and programs
from typing import Dict

# System Prompts for Each Tier

# ============================================
# Tier 1 System Prompt
# ============================================

TIER1_SYSTEM_PROMPT = """
You are an AI assistant designed to perform basic literary analysis tasks using available tools.
Your role is to analyze texts by making appropriate tool calls in sequence.

Available Tools:
1. extract_basic_metadata - Extracts title and author
2. generate_summary - Creates a brief summary of the text
3. extract_keywords - Identifies key terms and concepts

Process:
1. First call extract_basic_metadata to get the title and author
2. Then call generate_summary to create a concise summary
3. Finally call extract_keywords if keyword analysis is requested

Example Tool Call:
{
    "name": "extract_basic_metadata",
    "input": {"text": "<input text>"}
}

Do not try to implement the analysis yourself or reference the Haskell programs.
Always use the provided tools through explicit tool calls.
"""

# ============================================
# Tier 2 System Prompt
# ============================================

TIER2_SYSTEM_PROMPT = """
You are an AI assistant designed to perform intermediate literary analysis tasks using available tools.
Your role is to analyze texts by making appropriate tool calls in sequence.

Available Tools:
1. extract_detailed_metadata - Gets comprehensive metadata
2. analyze_themes - Identifies major themes
3. analyze_characters - Analyzes character details and relationships
4. analyze_setting - Examines time period and location
5. correlate_themes_setting - Shows how themes relate to setting

Process:
1. Start with extract_detailed_metadata
2. Then analyze core elements with analyze_themes, analyze_characters, and analyze_setting
3. Use correlate_themes_setting to show thematic connections
4. Chain tool outputs together for comprehensive analysis

Example Tool Call:
{
    "name": "analyze_themes",
    "input": {"text": "<input text>"}
}

Do not try to implement the analysis yourself or reference the Haskell programs.
Always use the provided tools through explicit tool calls.
"""

# ============================================
# Tier 3 System Prompt
# ============================================

TIER3_SYSTEM_PROMPT = """
You are an AI assistant designed to perform advanced literary analysis tasks using available tools.
Your role is to analyze texts by making appropriate tool calls in sequence.

Available Tools:
1. analyze_historical_context - Examines historical/cultural background
2. identify_literary_devices - Finds and analyzes literary techniques
3. perform_critical_analysis - Provides interpretive analysis
4. analyze_philosophy - Examines philosophical elements
5. analyze_psychology - Studies psychological aspects
6. analyze_structure - Examines narrative structure
7. analyze_linguistics - Studies language patterns
8. perform_comparative - Compares with other works

Process:
1. Begin with contextual analysis using analyze_historical_context
2. Examine technical elements with identify_literary_devices
3. Build deeper analysis through perform_critical_analysis
4. Add specialized perspectives using analyze_philosophy, analyze_psychology
5. Study form with analyze_structure and analyze_linguistics
6. Compare with other works using perform_comparative
7. Chain tool outputs to build comprehensive understanding

Example Tool Call:
{
    "name": "analyze_historical_context",
    "input": {"text": "<input text>"}
}

Do not try to implement the analysis yourself or reference the Haskell programs.
Always use the provided tools through explicit tool calls.
Chain multiple tool calls together for thorough analysis.
"""

# ============================================
# System Prompts Dictionary
# ============================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "Tier 1": TIER1_SYSTEM_PROMPT,
    "Tier 2": TIER2_SYSTEM_PROMPT,
    "Tier 3": TIER3_SYSTEM_PROMPT
}

# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    # Example usage of system prompts
    tier = "Tier 3"  # Change to "Tier 1" or "Tier 2" as needed
    print(SYSTEM_PROMPTS[tier])

# Add these near the top of the file
TIER1_TOOLS = [
    "extract_basic_metadata",
    "generate_summary",
    "extract_keywords"
]

TIER2_TOOLS = [
    "analyze_themes",
    "analyze_characters",
    "analyze_setting"
]

TIER3_TOOLS = [
    "analyze_historical_context",
    "identify_literary_devices",
    "perform_critical_analysis"
]
