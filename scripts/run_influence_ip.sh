#!/bin/bash

base_dir=$PWD
cd influence

python compute_influences_ip.py \
    --train_batch_size 2 \
    --test_batch_size 2 \
    --model_name EleutherAI/pythia-1b-deduped \
    --checkpoints EleutherAI/pythia-1b-deduped \
    --lrs 1 \
    --test_dataset cjiao/nuggets-kmeans-100 \
    --train_file "$base_dir/data/prompts/prompts-all.txt" \
    --outfile "$base_dir/test/infl_ip.pt" \
    --use_conditional

: '
Note: the above run will take a very long time. It will be easier to get the scores from different data partitions

python compute_influences_ip.py \
    --train_batch_size 2 \
    --test_batch_size 2 \
    --model_name EleutherAI/pythia-1b-deduped \
    --checkpoints EleutherAI/pythia-1b-deduped \
    --lrs 1 \
    --test_dataset cjiao/nuggets-kmeans-100 \
    --train_file "$base_dir/data/prompts/prompts-all-1-of-9.txt" \
    --outfile "$base_dir/test/infl_ip-1-of-9.pt" \
    --use_conditional
'