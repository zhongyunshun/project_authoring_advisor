import re

def parse_question_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract query between #### markers
    query_match = re.search(r"The query will be delimited with #### characters: ####\s*(.*?)\s*####", content, re.DOTALL)
    query = query_match.group(1).strip() if query_match else None

    # Extract the block starting from the injected domain info
    inject_start = re.search(r"# Inject Domain Information\s+Here is the retrieved passage:\s+\{", content)
    if not inject_start:
        raise ValueError("Injected domain information block not found.")

    context_block = content[inject_start.end():].strip()

    # Extract each document starting with --- Document x ---
    context_list = re.findall(r"--- Document \d+ ---.*?(?=(?:--- Document \d+ ---|$))", context_block, re.DOTALL)

    return query, context_list

def parse_response_file(response_path):
    with open(response_path, 'r', encoding='utf-8') as f:
        response = f.read().strip()
    return response

# Usage
query, context = parse_question_file("prompt_engineer_ragas/prompts/COT+Format Template/gm_question1.txt")
response = parse_response_file("prompt_engineer_ragas/prompting_results/COT+Format Template/gm_question1.txt")


print("Query:\n", query)
print("\nFirst Context Snippet:\n", context[0])

print("\nResponse:\n", response[:500], "...")  # print first 500 chars
