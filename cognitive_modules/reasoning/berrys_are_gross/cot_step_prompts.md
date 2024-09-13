Below are detailed TOML prompts for each individual reasoning step of an AI agent using a chain-of-thought approach. Each step is broken down into a specific task, with clear instructions, input, expected output, and additional settings where relevant. This provides granular control and guidance for each stage of the reasoning process.

### **1. Comprehend the User's Question**

```toml
[step_1]
name = "Comprehend User's Question"
description = """
Analyze the user's input to understand the core question or problem being asked. Identify the primary intent, keywords, and relevant entities mentioned.
"""
input_type = "text"
expected_output = "parsed_question"  # Structured representation of the question.
actions = [
    { action_type = "parse", description = "Extract main intent and key elements." },
    { action_type = "summarize", description = "Provide a concise summary of the user's question." }
]
error_handling = { strategy = "clarify", retries = 2 }  # Ask for clarification if input is unclear.
```

### **2. Identify Key Concepts and Entities**

```toml
[step_2]
name = "Identify Key Concepts and Entities"
description = """
Determine the critical concepts, entities, and relationships involved in the question. This may include recognizing named entities, understanding domain-specific terms, and categorizing them appropriately.
"""
input_type = "parsed_question"
expected_output = "key_concepts_and_entities"
actions = [
    { action_type = "entity_recognition", description = "Identify named entities like people, places, dates." },
    { action_type = "concept_categorization", description = "Categorize concepts based on domain knowledge." }
]
error_handling = { strategy = "re-evaluate", retries = 1 }
```

### **3. Retrieve Relevant Knowledge or Information**

```toml
[step_3]
name = "Retrieve Relevant Knowledge"
description = """
Search internal knowledge bases, external databases, or other resources to gather relevant information that addresses the identified concepts and entities.
"""
input_type = "key_concepts_and_entities"
expected_output = "retrieved_information"
actions = [
    { action_type = "knowledge_search", description = "Query knowledge base for related information." },
    { action_type = "cross-reference", description = "Verify information by cross-referencing multiple sources." }
]
resources = ["knowledge_base", "web_search", "databases"]
error_handling = { strategy = "fallback", retries = 2, fallback_resource = "web_search" }
```

### **4. Evaluate Possible Solutions or Answers**

```toml
[step_4]
name = "Evaluate Possible Solutions"
description = """
Analyze the retrieved information to develop possible solutions or answers to the user's question. Consider multiple perspectives and potential interpretations.
"""
input_type = "retrieved_information"
expected_output = "possible_solutions"
actions = [
    { action_type = "analysis", description = "Analyze data and identify key insights." },
    { action_type = "generate_options", description = "Develop a list of possible solutions or responses." }
]
evaluation_criteria = ["relevance", "accuracy", "completeness"]
error_handling = { strategy = "re-analyze", retries = 1 }
```

### **5. Select the Most Appropriate Response**

```toml
[step_5]
name = "Select Appropriate Response"
description = """
Based on the evaluated solutions, select the most suitable response that addresses the user's needs effectively. Ensure the choice is relevant, accurate, and concise.
"""
input_type = "possible_solutions"
expected_output = "selected_response"
actions = [
    { action_type = "rank", description = "Rank solutions based on relevance and accuracy." },
    { action_type = "select_best", description = "Select the highest-ranking solution." }
]
selection_criteria = ["user_context", "clarity", "conciseness"]
error_handling = { strategy = "re-select", retries = 1 }
```

### **6. Structure the Response Logically**

```toml
[step_6]
name = "Structure Response Logically"
description = """
Organize the selected response into a coherent, logically structured format. Ensure the flow is natural and easy to understand.
"""
input_type = "selected_response"
expected_output = "structured_response"
actions = [
    { action_type = "organize", description = "Arrange content into logical sections (introduction, body, conclusion)." },
    { action_type = "clarify", description = "Clarify any potentially ambiguous points." }
]
structure_guidelines = ["logical_flow", "clarity", "brevity"]
error_handling = { strategy = "restructure", retries = 2 }
```

### **7. Review the Response for Accuracy and Completeness**

```toml
[step_7]
name = "Review Response for Accuracy"
description = """
Review the structured response for accuracy, completeness, and clarity. Ensure all key points are covered and there are no errors or ambiguities.
"""
input_type = "structured_response"
expected_output = "reviewed_response"
actions = [
    { action_type = "proofread", description = "Check for grammatical errors and inaccuracies." },
    { action_type = "validate", description = "Ensure all points are factually correct and fully covered." }
]
review_criteria = ["grammar", "accuracy", "completeness"]
error_handling = { strategy = "revise", retries = 2 }
```

### **8. Format the Response According to User Preferences**

```toml
[step_8]
name = "Format Response According to Preferences"
description = """
Format the final response to align with user preferences, including tone, detail level, and style. Ensure it meets the user's specific needs and context.
"""
input_type = "reviewed_response"
expected_output = "final_response"
actions = [
    { action_type = "adjust_tone", description = "Modify tone based on user preferences (e.g., formal, informal)." },
    { action_type = "optimize_format", description = "Format as text, summary, bullet points, or chart as needed." }
]
preferences = { tone = "formal", detail_level = "comprehensive", format = "text" }
error_handling = { strategy = "reformat", retries = 1 }
```

### **9. Additional Error Handling and Logging**

```toml
[error_handling]
name = "Error Handling and Logging"
description = """
For each step, ensure that errors are managed appropriately by retrying, seeking clarification, or falling back to simpler methods. Log errors for analysis and improvement.
"""
error_logging = true
log_level = "info"
log_format = "json"
log_destination = "/var/log/assistant_steps.log"
```

### **Summary**

Each reasoning step is defined with:
- **Name and Description**: Clarifies the purpose of each step.
- **Input Type and Expected Output**: Specifies the input format and what is expected as output.
- **Actions**: Details specific actions to perform for achieving the step's objective.
- **Error Handling**: Defines strategies for managing errors specific to each step.
- **Additional Settings**: Customizes actions and responses based on criteria like user preferences and performance requirements.
