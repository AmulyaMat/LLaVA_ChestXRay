#!/bin/bash

CHUNKS=4   # 8

# Create a logs directory if it doesn't already exist
mkdir -p logs

# Loop through each chunk index
for IDX in {0..3}; do
    # Assign each chunk to a specific GPU and log outputs/errors
    CUDA_VISIBLE_DEVICES=$IDX python -m llava.eval.model_vqa_science \
        --model-path liuhaotian/llava-v1.5-7b \
        --question-file /home/amulyam/Projects/LLaVA/ScienceQA/data/scienceqa/llava_test_QCM-LEA.json \
        --image-folder /home/amulyam/Projects/LLaVA/ScienceQA/test \
        --answers-file ./test_llava-13b-chunk$CHUNKS_$IDX.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --conv-mode llava_v1 \
        > logs/chunk_${IDX}.out 2> logs/chunk_${IDX}.err &
done

# Wait for all background processes to finish
wait

