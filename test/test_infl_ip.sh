#!/bin/bash
#SBATCH --job-name=infl-test
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/infl-test.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/infl-test.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=5:00:00
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
for ((i = 0 ; i < 50 ; i++ ));
do
python compute_influences_ip.py \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --model_name EleutherAI/pythia-1b-deduped \
    --checkpoints EleutherAI/pythia-1b-deduped \
    --lrs 1 \
    --test_dataset /home/cljiao/heuristic-data/data/$i.json \
    --train_file "/home/cljiao/heuristic-data/data/${i}_different_tasks.tsv" \
    --outfile "/home/cljiao/InContextDataValuation/outputs/infl_scores/${i}_different_tasks.pt" \
    --grad_approx sign_log \
    --grad_clip \
    --max_length 1024 \
    --model_dtype float32 \
    --use_conditional \
    --load_local
done

# --test_dataset cjiao/nuggets-kmeans-100 \
# --train_file "$base_dir/data/prompts/kmeans100.txt" \
# --outfile "/data/user_data/cljiao/pretrain_test/infl_scores/self_kmeans_sign.pt" \