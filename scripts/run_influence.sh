#!/bin/bash

base_dir=$PWD
cd influence

python compute_influences.py \
    --train_batch_size 2 \
    --test_batch_size 2 \
    --model_name EleutherAI/pythia-1b-deduped \
    --test_dataset cjiao/nuggets-kmeans-100 \
    --train_file "$base_dir/data/prompts/prompts-all.txt" \
    --outfile "$base_dir/test/infl.pt" \
    --use_conditional