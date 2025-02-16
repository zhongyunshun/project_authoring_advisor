# $chunk_length = 200
$search_type = "similarity"
$top_k = 5

foreach ($chunk_length in 100,200,300,400,500) {
    Write-Output "Running main.py with chunk_length=$chunk_length..."
    python main.py --chunk_length $chunk_length --top_k $top_k --search_type $search_type
    Write-Output "Output saved."
}

Write-Output "All runs completed!"
