import os
import sys

# TODO: could be resolved by init.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prompt_engineer_ragas import questions
from configs.keys import OPENAI_API_KEY, GOOGLE_API_KEY
from pipelines.rag_pipeline import PromptingRAG
from embeddings.embeddings import load_embeddings_from_file

# Set API Keys
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

prompt_types = [
    'persona+cot+format',
]

output_dir_map = {
    'persona+cot+format': "Persona+COT+Format Template",
}

# Question groups and counts
question_counts = {
    "gm": 10,
    "hbpe": 10,
    "pc": 10,
}

def main():
    # Setup RAG database embedding
    embedding_file = "vector_db/sbert_chunk_700_embedding"
    print(f"\n📂 Loading existing embeddings from {embedding_file}...")
    vector_db = load_embeddings_from_file(embedding_file, embed_model="sbert")
    print("✅ Embeddings loaded from file.")

    model = "llama"

    # Pre-load RAG models once per pattern type
    rag_models = {
        p_type: PromptingRAG(vector_db=vector_db, top_k=22, search_type="similarity", pattern=p_type, temperature=0.7, model=model)
        for p_type in prompt_types
    }

    # Prepare all prompt generation runs
    prompt_runs = []
    step = 0
    total_count = sum(question_counts.values()) * len(prompt_types)

    for prefix, count in question_counts.items():
        for i in range(1, count + 1):
            var_name = f"{prefix}_question{i}"
            question = getattr(questions, var_name)

            for p_type in prompt_types:
                rag_model = rag_models[p_type]
                answer, prompt = rag_model.invoke(question)

                prompt_runs.append({
                    "filename": f"{prefix}_question{i}.txt",
                    "prompt_type": p_type,
                    "prompt": prompt,
                    "answer": answer
                })

                step += 1
                print(f"Progress Done {step}/{total_count}")

    print("Writing to txt files...")

    # Create folders and write prompt and answer files
    for run in prompt_runs:
        folder_name = output_dir_map[run["prompt_type"]]

        # # Comment out because it is already saved
        # prompt_folder_path = os.path.join("prompt_engineer_ragas/prompts", folder_name)
        # os.makedirs(prompt_folder_path, exist_ok=True)
        # prompt_filepath = os.path.join(prompt_folder_path, run["filename"])
        # with open(prompt_filepath, "w", encoding="utf-8") as f:
        #     f.write(run["prompt"])

        answer_folder_path = os.path.join("prompt_engineer_ragas/prompting_results_sbert", model, folder_name) # Changed to sbert embed
        os.makedirs(answer_folder_path, exist_ok=True)
        answer_filepath = os.path.join(answer_folder_path, run["filename"])
        with open(answer_filepath, "w", encoding="utf-8") as f:
            f.write(run["answer"])

    print(f"Generated {len(prompt_runs)} prompt files in {len(output_dir_map)} folders.")


if __name__ == "__main__":
    main()
    