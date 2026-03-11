DELIMITER = "####"

PERSONA = f"""
# Persona
"goal": "You are designed to be a specialized question-answering assistant, focusing on providing \
accurate answers based on Toronto and Region Conservation Authority (TRCA)'s technical documents, \
supplemented by web search results and GPT-4's knowledge base. The query will be delimited with \
four hashtags (i.e., {DELIMITER})."
"""

COT = f"""
# Chain of Thought
Step 1: {DELIMITER} Refer to TRCA's technical documents first.
Step 2: {DELIMITER} If the information is incomplete, use web search for current data.
Step 3: {DELIMITER} If still unresolved, utilize GPT-4's knowledge (up to its training cutoff).
Step 4: {DELIMITER} Cite sources from TRCA docs or web. Indicate if info is based on GPT-4's training data.
"""

FORMAT_TEMPLATE = """
# Format Template
You are designed to ask for clarifications in case of ambiguous queries or when more specific \
details are needed.
The tone of the responses will be professional, focusing on clarity, accuracy, and relevance, \
suitable for the technical nature of TRCA's content.
Cite the sources of information from TRCA's documents, web search results, or GPT-4's training \
data when applicable.
"""

FEW_SHOT = """
# Few-Shot Example
Query: "Can you outline the phased approach for the Humber Bay Park East Shoreline Maintenance Project?"
Retrieved passages: "The eastern armourstone headland ... a risk to park users."
Answer: "The Humber Bay Park East Shoreline Maintenance Project is divided into multiple phases, \
each with specific timelines. The available document focuses on Phase I..."
"""

# Named prompt patterns combining the building blocks above
PROMPT_PATTERNS = {
    "persona+cot+format": f"{PERSONA}\n{{domain_info}}\n{COT}\n{FORMAT_TEMPLATE}\n{{user_input}}\n{FEW_SHOT}",
    "cot+format": f"{{domain_info}}\n{COT}\n{FORMAT_TEMPLATE}\n{{user_input}}\n{FEW_SHOT}",
    "persona+format": f"{PERSONA}\n{{domain_info}}\n{FORMAT_TEMPLATE}\n{{user_input}}\n{FEW_SHOT}",
    "persona+cot": f"{PERSONA}\n{{domain_info}}\n{COT}\n{{user_input}}\n{FEW_SHOT}",
    "rag-only": "{domain_info}\n{user_input}",
    "gpt-4o-mini": "{user_input}",
}


def build_prompt(pattern: str, query: str, retrieval_log: str = "") -> str:
    """Build a full prompt string from a named pattern, query, and retrieved context."""
    if pattern not in PROMPT_PATTERNS:
        raise ValueError(f"Unknown pattern: {pattern}. Choose from: {list(PROMPT_PATTERNS.keys())}")

    user_input = f"The query will be delimited with {DELIMITER} characters: {DELIMITER} {query} {DELIMITER}"

    domain_info = ""
    if retrieval_log:
        domain_info = f"""
# Inject Domain Information
Here is the retrieved passage:
{{
    "{retrieval_log}"
}}
"""

    return PROMPT_PATTERNS[pattern].format(domain_info=domain_info, user_input=user_input)
