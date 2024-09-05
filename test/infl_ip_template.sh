#!/bin/bash
#SBATCH --job-name=infl-minipile
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/infl-minipile.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/infl-minipile.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=3:00:00
#SBATCH --mem=16GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu

set -x

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate icdata

cd /home/cljiao/InContextDataValuation/influence
python compute_influences_ip.py \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --model_name EleutherAI/pythia-1b-deduped \
    --checkpoints EleutherAI/pythia-1b-deduped \
    --lrs 1 \
    --test_dataset $1 \
    --train_file $2 \
    --outfile $3 \
    --max_length 1024 \
    --grad_approx sign_log \
    --grad_clip \
    --model_dtype float32 \
    --use_conditional