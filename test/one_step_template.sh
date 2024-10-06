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

cd /home/cljiao/InContextDataValuation/one_step_train

: '
for ((i = 0 ; i < 50 ; i++ ));
do
python one_step_trainer.py \
       --devices 1 \
       --model_name $1 \
       --model_ckpt $2 \
       --train_data_path $3 \
       --val_data_path $4/${i}.json \
       --out_dir ${5}/${i}_${6} \
       --max_seq_length 1024 \
       --eval_conditional true \
       --fsdp false \
       --load_local_data true
done
'

# --eval_conditional true \
# --max_seq_length 1024 \

python one_step_trainer.py \
       --devices 1 \
       --model_name $1 \
       --model_ckpt $2 \
       --train_data_path $3 \
       --val_data_path $4 \
       --out_dir $5 \
       --max_seq_length 256 \
       --fsdp false \
       --eval_conditional true \
       --load_local_data true \
       --train_start_idx $6 \
       --n_train_samples $7

#results_file="/data/user_data/cljiao/paper_data/one-finetune-2e-5/$i.pt"
#if ! test -f $results_file; then
#fi
# cjiao/task-mixture-no-pythia
# cjiao/nuggets-kmeans-100
# cjiao/minipile-valid-pythia-1b-256-n100