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
    --model_args ${1} \
    --device cuda \
    --batch_size 32 \
    --split TEST \
    --incontext_type iterate \
    --incontext_file ${2} \
    --output_path ${3}
done