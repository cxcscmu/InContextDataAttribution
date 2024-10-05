set -x

__conda_setup="$('/home/cljiao/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
conda activate icdata

cd /home/cljiao/InContextDataValuation/icp

rand=$(shuf -i 20000-65530 -n 1)
for ((i = 0 ; i < 50 ; i++ ));
do

task_config_path=/home/cljiao/InContextDataValuation/icp/lm_eval/tasks/instruction/instruction.yaml
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
    --model_args ${1} \
    --device cuda \
    --batch_size 32 \
    --split TEST \
    --incontext_type iterate \
    --incontext_file ${2} \
    --output_path ${3}/${i}_${4}.json
done