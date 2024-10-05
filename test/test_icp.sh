#!/bin/bash
#SBATCH --job-name=icp-test
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/icp-test.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/icp-test.err
#SBATCH --gres=gpu:8000:4
#SBATCH --time=1-00:00:00
#SBATCH --mem=64GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu
#SBATCH --exclude=babel-3-19

set -x

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate icdata

cd /home/cljiao/InContextDataValuation/icp

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
    --incontext_file "/data/user_data/cljiao/data-calibration/minipile-train-pythia-1b-256-n25000.tsv" \
    --output_path "/data/user_data/cljiao/paper_outputs/ll_scores/kmeans_minipile.json"