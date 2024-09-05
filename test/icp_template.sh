#!/bin/bash
#SBATCH --mem=128GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu

set -x

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate icdata

cd /home/cljiao/InContextDataValuation/icp

rand=$(shuf -i 20000-65530 -n 1)
accelerate launch --main_process_port ${rand} run_eval.py \
    --model hf \
    --tasks $1 \
    --verbosity DEBUG \
    --model_args pretrained=$2 \
    --device cuda \
    --batch_size 8 \
    --split TEST \
    --incontext_type iterate \
    --incontext_file $3 \
    --output_path $4