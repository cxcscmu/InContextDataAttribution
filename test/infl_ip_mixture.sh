#!/bin/bash
#SBATCH --job-name=infl-mixture
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/infl-mixture.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/infl-mixture.err
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

base_dir=$PWD
cd influence

python compute_influences_ip.py \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --model_name EleutherAI/pythia-1b-deduped \
    --checkpoints EleutherAI/pythia-1b-deduped \
    --lrs 1 \
    --test_dataset cjiao/task-mixture-no-pythia \
    --train_file "/data/user_data/cljiao/data-calibration/minipile-valid-pythia-1b-256-n100.tsv" \
    --outfile "/data/user_data/cljiao/one_step/infl_scores/mixture_minipile_cond_optimal.pt" \
    --max_length 1024 \
    --use_conditional \
    --grad_approx sign_log \
    --model_dtype float32 \
    --grad_clip

# --test_dataset cjiao/nuggets-kmeans-100 \
# --train_file "$base_dir/data/prompts/kmeans100.txt" \
# --outfile "/data/user_data/cljiao/pretrain_test/infl_scores/self_kmeans_sign.pt" \