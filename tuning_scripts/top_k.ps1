# Tune top_k while keeping chunk_length and search_type constant
$chunk_length = 700
$top_ks = 22
$search_type = "similarity"

foreach ($top_k in $top_ks) {
    Write-Output "Running main.py with chunk_length=$chunk_length, top_k=$top_k, search_type=$search_type..."
    python main.py --chunk_length $chunk_length --top_k $top_k --search_type $search_type --mode csv
    Write-Output "Output saved."
}

Write-Output "All top_k tuning runs completed!"
