#!/bin/bash
#SBATCH --mem=16GB
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

cd /home/cljiao/InContextDataValuation/influence

python compute_influences_ip.py \
    --train_batch_size 2 \
    --test_batch_size 2 \
    --model_name $1 \
    --checkpoints $2 \
    --lrs 1 \
    --test_dataset /home/cljiao/heuristic-data/test_data/nuggets-kmeans-100.json \
    --train_file $3 \
    --outfile $4 \
    --grad_approx sign_log \
    --grad_clip \
    --max_length 256 \
    --use_conditional \
    --model_dtype float16 \
    --load_local

: '
for ((i = 0 ; i < 50 ; i++ ));
do
python compute_influences_ip.py \
    --train_batch_size 1 \
    --test_batch_size 1 \
    --model_name $1 \
    --checkpoints $2 \
    --lrs 1 \
    --test_dataset /home/cljiao/heuristic-data/data/$i.json \
    --train_file $3 \
    --outfile ${4}/${i}_${5}.pt \
    --grad_approx sign_log \
    --grad_clip \
    --max_length 1024 \
    --model_dtype float32 \
    --use_conditional \
    --load_local
done
'

# --test_dataset cjiao/nuggets-kmeans-100 \
# --train_file "$base_dir/data/prompts/kmeans100.txt" \
# --outfile "/data/user_data/cljiao/pretrain_test/infl_scores/self_kmeans_sign.pt" \