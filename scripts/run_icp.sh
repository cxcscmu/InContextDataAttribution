#!/bin/bash

base_dir=$PWD
cd icp

rand=$(shuf -i 20000-65530 -n 1)
accelerate launch --main_process_port ${rand} run_eval.py \
    --model hf \
    --tasks kmeans100 \
    --verbosity DEBUG \
    --model_args pretrained=EleutherAI/pythia-1b-deduped \
    --device cuda \
    --batch_size 32 \
    --split TEST \
    --incontext_type iterate \
    --incontext_file "$base_dir/data/prompts/kmeans100.txt" \
    --output_path "$base_dir/outputs/model_probs.json"