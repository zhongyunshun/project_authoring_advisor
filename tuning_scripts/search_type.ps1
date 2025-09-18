# Tune search_type while keeping chunk_length and top_k constant
$chunk_length = 200
$top_k = 5
$search_types = "similarity", "mmr" # , "similarity_score_threshold"

foreach ($search_type in $search_types) {
    Write-Output "Running main.py with chunk_length=$chunk_length, top_k=$top_k, search_type=$search_type..."
    python main.py --chunk_length $chunk_length --top_k $top_k --search_type $search_type --mode csv
    Write-Output "Output saved.\n"
}

Write-Output "All search_type tuning runs completed!"
