delimiter = '####'
text_chunk = ''
user_query = ''
user_query_input = f"The query will be delimited with {delimiter} characters: {delimiter} {user_query} {delimiter}"

# Define each pattern
persona_pattern = """
# Persona
"goal": "You are designed to be a specialized question-answering assistant, focusing on providing accurate answers based on Toronto and Region Conservation Authority (TRCA)'s technical documents, supplemented by web search results and GPT-4's knowledge base. The query will be delimited with four hashtags (i.e., {delimiter})."
"""

cot_pattern = """
# Chain of Thought
Step 1: {delimiter} Refer to TRCA's technical documents first.
Step 2: {delimiter} If the information is incomplete, use web search for current data.
Step 3: {delimiter} If still unresolved, utilize GPT-4's knowledge (up to its training cutoff).
Step 4: {delimiter} Cite sources from TRCA docs or web. Indicate if info is based on GPT-4's training data.
"""

domain_info_pattern = f"""
# Inject Domain Information
Here is the retrieved passage:
{
  "file_name": "Humber Bay Park East Project: Phase I.pdf",
  "page":  4,
  "section": "2.3.1 Phase I – Eastern Armourstone Headland",
  "text": {text_chunk}
}
"""

format_template_pattern = """
# Format Template
You are designed to ask for clarifications in case of ambiguous queries or when more specific details are needed.
The tone of the responses will be professional, focusing on clarity, accuracy, and relevance, suitable for the technical nature of TRCA's content.
Cite the sources of information from TRCA's documents, web search results, or GPT-4's training data when applicable.
"""

few_shot_pattern = """
# Few-Shot Example
Query: "Can you outline the phased approach for the Humber Bay Park East Shoreline Maintenance Project?"
Retrieved passages: "The eastern armourstone headland ... a risk to park users."
Answer: "The Humber Bay Park East Shoreline Maintenance Project is divided into multiple phases, each with specific timelines. The available document focuses on Phase I..."
"""

# Four different prompt templates by combining the above prompt patterns
persona_cot_format_template = f"""
{persona_pattern}
{domain_info_pattern}
{cot_pattern}
{format_template_pattern}
{user_query_input}
{few_shot_pattern}
"""

cot_format_template = f"""
{domain_info_pattern}
{cot_pattern}
{format_template_pattern}
{user_query_input}
{few_shot_pattern}
"""

persona_format_template = f"""
{persona_pattern}
{domain_info_pattern}
{format_template_pattern}
{user_query_input}
{few_shot_pattern}
"""

persona_cot = f"""
{persona_pattern}
{domain_info_pattern}
{cot_pattern}
{user_query_input}
{few_shot_pattern}
"""

