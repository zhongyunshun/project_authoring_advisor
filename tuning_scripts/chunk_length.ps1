# Tune chunk_length while keeping top_k and search_type constant
$chunk_lengths = 600,700,800,900
$top_k = 5
$search_type = "mmr"

foreach ($chunk_length in $chunk_lengths) {
    Write-Output "Running main.py with chunk_length=$chunk_length, top_k=$top_k, search_type=$search_type..."
    python main.py --chunk_length $chunk_length --top_k $top_k --search_type $search_type
    Write-Output "Output saved."
}

Write-Output "All chunk_length tuning runs completed!"
