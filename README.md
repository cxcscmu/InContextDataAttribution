# In-Context Data Valuation

This is the repository for the paper "In-Context Probing Approximates Influence Function for Data Valuation"

# Table of Contents
- [1 Setup](#setup)
- [2 Data Scoring](#data-scoring)
    - [2.1 In-Context Probing Scores](#in-context-probing-scores)
    - [2.2 Influence Scores](#in-context-probe-scores)
- [3 Training](#training)
- [4 Evaluation](#evaluation)
- [5 Analysis](#analysis)
- [6 Contact](#contact)

## Setup
This code has been tested on Python 3.9. Run the following to start:

```
conda create --name icdata python=3.9
pip install -r requirements.txt
```

## Data Scoring
This section details how to obtain in-context probing scores and influence scores for our data valuation experiments.

### In-Context Probing Scores
To obtain the in-context probing (ICP) scores, first run 

```
bash scripts/run_icp.sh
```

The script above uses the [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) to obtain model likelihood scores. The benefit of this is that the harness supports additional tasks, so you can customize the script above to include the tasks you wish to evaluate.

Then, to get the ICP scores, run

```
python utils/icp_scores.py --infile path_to_model_outputs --outfile path_to_save_file
```

### Influence Scores

To obtain the influence scores, run 

```
bash scripts/run_influence_ip.sh
```

However, it is not recommended to score all 52K instructions at once since takes a very long time. For convenience, the 52k instruction have been partitioned into smaller subsets (5778 examples each) in ```data/prompts```. Feel free to adjust the bash script above in order to run it on different partitions. Note: it takes about ~9hrs on a single A6000 48GB gpu to score a single partition. 

We provide a simple function for combining the influence scores obtained from the different data partitions in ```utils/combine_infl_scores.py```.

## Training

We recommend using the Stanford Alpaca Repo for training. To start, run

```
git clone https://github.com/tatsu-lab/stanford_alpaca.git
cd stanford_alpaca
```

Below is an example of a training script. You will need to set ```--output_dir```, ```--data_path``` and ```--output_dir```.

```
rand=$(shuf -i 20000-65530 -n 1)
torchrun --nproc_per_node=1 --master_port=${rand} train.py \
      --model_name_or_path "EleutherAI/pythia-1b-deduped" \
      --data_path "data/finetune/icp>0.9.json" \
      --bf16 False \
      --output_dir "outputs" \
      --num_train_epochs 3 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 4 \
      --save_strategy "epoch" \
      --evaluation_strategy "epoch" \
      --save_total_limit 3 \
      --learning_rate 2e-7 \
      --adam_epsilon 1e-8 \
      --warmup_ratio 0.03 \
      --weight_decay 0. \
      --lr_scheduler_type "cosine" \
      --logging_steps 1 \
      --fsdp "full_shard auto_wrap" \
      --fsdp_transformer_layer_cls_to_wrap 'GPTNeoXLayer' \
      --tf32 False \
```

We provide the data from different rankings to finetune on in ```data/finetune```.

## Evaluation
To evaluate the finetuned models, check out the [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval) repository:

```
git clone https://github.com/tatsu-lab/alpaca_eval.git
cd alpaca_eval
```

To get started, go to ```/src/alpaca_eval/models_configs``` and run

```
mkdir my_model
cd my_model

touch configs.yaml
echo "pythia-1b-deduped:
  prompt_template: "prompt.txt"
  fn_completions: "huggingface_local_completions"
  completions_kwargs:
    model_name: path_to_finetuned_model
    model_kwargs:
      torch_dtype: 'bfloat16'
      device_map: cuda
    max_new_tokens: 512
    temperature: 0.7
    top_p: 1.0
    do_sample: True
    batch_size: 8
  pretty_name: "my model"" >> configs.yaml

touch prompts.txt

echo "{instruction}" >> prompts.txt
```

This creates a model configuration to evaluate your trained model from the previous step.
Now, you can evaluate the model (i.e., get winrates). Below is an example of how you can evaluate your model for the vicuna dataset using GPT-4 as the evaluator:

```
export OPENAI_API_KEY=your_api_key

python -m alpaca_eval.main \
  evaluate_from_model \
  --model_configs my_model \
  --annotators_config  "alpaca_eval_gpt4_turbo_fn"  \
  --evaluation_dataset data/eval/vicuna.json \
  --output_path your_output_dir \
```

## Analysis
We provide additional functions for analysis which can be found in ```utils/analyze.py```. For instance, obtaining spearman correlation between icp and influence scores, plotting score distributions etc. 

## Contact
For questions, please email:

Cathy Jiao (cljiao@cs.cmu.edu)