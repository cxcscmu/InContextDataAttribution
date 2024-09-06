#!/bin/bash
#SBATCH --job-name=icp-test
#SBATCH --output=/home/cljiao/InContextDataValuation/logs/icp-test.out
#SBATCH --error=/home/cljiao/InContextDataValuation/logs/icp-test.err
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

base_dir=${PWD}
cd icp

rand=$(shuf -i 20000-65530 -n 1)
for ((i = 0 ; i < 50 ; i++ ));
do

task_config_path=${base_dir}/icp/lm_eval/tasks/instruction/instruction.yaml
rm ${task_config_path}
touch ${task_conftig_path}

# Update the instruction to evaluate
echo "task: instruction
dataset_path: json
dataset_kwargs:
  data_files: {\"test\": /home/cljiao/heuristic-data/data/$i.json}
dataset_name: default
test_split: test
output_type: loglikelihood
doc_to_text: \"{{text}}\"
doc_to_target: \"{{target}}\"
metric_list:
  - metric: perplexity
    aggregation: perplexity
    higher_is_better: false
metadata:
  version: 2.0" >> ${task_config_path}

accelerate launch --main_process_port ${rand} run_eval.py \
    --model hf \
    --tasks instruction \
    --verbosity DEBUG \
    --model_args pretrained=EleutherAI/pythia-1b-deduped \
    --device cuda \
    --batch_size 32 \
    --split TEST \
    --incontext_type iterate \
    --incontext_file "/home/cljiao/heuristic-data/data/${i}_same_tasks.tsv" \
    --output_path "/home/cljiao/InContextDataValuation/outputs/ll_scores/${i}_same_tasks.json"
done