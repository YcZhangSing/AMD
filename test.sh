#!/bin/bash

# Set the GPU index to use 
GPU_NU=0

# Path to the trained model
MODEL_ID="/AMD/models"

# Batch size for evaluation
BATCH_SIZE=1

# List of validation JSON files (datasets to evaluate on)
VALS=(
    path/to/bbc.json
    path/to/NYT.json
)

# Build --vals argument
VALS_ARGS=("${VALS[@]}")

# Generate output log file path using timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${MODEL_ID}/domain_test_out_${TIMESTAMP}.txt"

# Run the Python evaluation script
# if you want to evaluate on DGM4 benchmark, please use scripts/test_on_DGM4.py and given the tokennizer pth, bert-base-uncased is recommanned
python scripts/test_on_DGM4.py \
    --GPU_nu "$GPU_NU" \
    --model_id "$MODEL_ID" \
    --batch_size "$BATCH_SIZE" \
    --vals "${VALS_ARGS[@]}"\
    --output_file "$OUTPUT_FILE" \
    # --tokenizer ""

# Final message
echo "Evaluation completed. Logs saved at: $OUTPUT_FILE"
