# Tune chunk_length while keeping top_k and search_type constant
$chunk_lengths = 620,640,660,680,720,740,760,780
$top_k = 5
$search_type = "similarity"

foreach ($chunk_length in $chunk_lengths) {
    Write-Output "Running main.py with chunk_length=$chunk_length, top_k=$top_k, search_type=$search_type..."
    python main.py --chunk_length $chunk_length --top_k $top_k --search_type $search_type --mode csv
    Write-Output "Output saved."
}

Write-Output "All chunk_length tuning runs completed!"
