#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/ft-test.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/ft-test.err
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=16GB
#SBATCH --partition=general
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-type=fail
#SBATCH --mail-user=cljiao@andrew.cmu.edu

set -x

run_path="/home/cljiao/Lightning-Pretrain"
cd ${run_path}

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate li-pretrain-py39

for ((i = 0 ; i < 10 ; i++ ));
do
python pretrain/one_step_trainer.py \
       --devices 1 \
       --model_name pythia-1b \
       --model_ckpt /data/user_data/cljiao/cont-pretrain/pythia-1b-deduped \
       --train_data_path /home/cljiao/heuristic/data/${i}.json \
       --val_data_path /home/cljiao/heuristic/data/${i}_same_tasks.json \
       --out_dir /home/cljiao/InContextDataValuation/outputs/ft_scores/${i}_same_tasks \
       --max_seq_length 1024 \
       --eval_conditional true \
       --fsdp False
done

#results_file="/data/user_data/cljiao/paper_data/one-finetune-2e-5/$i.pt"
#if ! test -f $results_file; then
#fi
# cjiao/task-mixture-no-pythia
# cjiao/nuggets-kmeans-100
# cjiao/minipile-valid-pythia-1b-256-n100