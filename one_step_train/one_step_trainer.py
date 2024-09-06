from datasets import Dataset, Features, Sequence, Value, load_dataset

from transformers.trainer_pt_utils import IterableDatasetShard
from transformers import default_data_collator, AutoTokenizer

from lightning.fabric.plugins import BitsandbytesPrecision
from lightning.fabric.strategies import FSDPStrategy
from torch.nn.utils.rnn import pad_sequence
from typing import Optional, Tuple, Union
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import lightning as L

import torch
import math
import time
import sys
import os

import numpy as np
import json

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model import GPT, Block
from lit_gpt.utils import (
    get_default_supported_precision,
    chunked_cross_entropy,
    num_parameters,
    lazy_load
)
#from lit_gpt.parallelism import parallelize

# Hyperparameters
learning_rate = None
batch_size = 1
micro_batch_size = 1
gradient_accumulation_steps = batch_size // micro_batch_size
assert gradient_accumulation_steps > 0
weight_decay = 1e-1
beta1 = 0 # For one step finetune, don't keep track of moments
beta2 = 0
grad_clip = 1.0
decay_lr = False
stable_iters = 400000
lr_decay_iters = 400000
#warmup_iters = lr_decay_iters * 0.04
warmup_iters = 0
min_lr = 1e-4

hparams = {
    k: v
    for k, v in locals().items()
    if isinstance(v, (int, float, str)) and not k.startswith("_")
}
logger = None

# data parameters
train_batch_size = 1
val_batch_size = 8
ignore_token = -1


def setup(
    train_data_path: str = None,
    val_data_path: str = None,
    model_name: str = None,
    model_ckpt: Path = None,
    devices: int = 1,
    method: str = "random",
    rank: int = 0,
    precision: Optional[str] = None,
    out_dir: Path = None,
    eval_conditional: bool = False,
    max_seq_length: int = 512,
    train_start_idx: int = None,
    n_train_samples: int = None,
    fsdp: bool = False,
    lr: float = 2e-5,
    load_local_data: bool = False,
) -> None:
    global learning_rate
    learning_rate = lr
    
    precision = precision or get_default_supported_precision(training=True)
    #precision = BitsandbytesPrecision(mode="int8-training", dtype=torch.float16)
    if fsdp:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(
        devices=devices,
        num_nodes=1,
        strategy=strategy,
        precision=precision,
        loggers=logger
    )
    fabric.print(hparams)
    fabric.launch(
        main,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        rank=rank,
        model_name=model_name,
        model_ckpt=model_ckpt,
        out_dir=out_dir,
        eval_conditional=eval_conditional,
        max_seq_length=max_seq_length,
        train_start_idx=train_start_idx,
        n_train_samples=n_train_samples,
        fsdp=fsdp,
        load_local_data=load_local_data,
    )


def get_tokenizer(model_ckpt):
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    
    if tokenizer.pad_token:
        pad_token = tokenizer.pad_token
    else:
        pad_token = "<|pad|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
    
    return tokenizer


# tokenizes each row in dataset
def tokenize_function(examples, **kwargs):
    tokenizer = kwargs['tokenizer']
    eval_conditional = kwargs['eval_conditional']
    max_seq_length = kwargs['max_seq_length']

    if 'target' in examples:
        queries, targets = examples["text"], examples["target"]
        inputs = [f"{queries[i]}\n{targets[i]}" for i in range(len(examples['text']))]
    else:
        inputs = examples["text"]

    # Tokenize the inputs
    tokenized_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_seq_length)
    input_ids = tokenized_inputs['input_ids']
    attention_mask = tokenized_inputs['attention_mask']
    input_lengths = np.sum(tokenized_inputs['attention_mask'], axis=1)

    # Get labels
    labels = np.copy(input_ids)

    # Get lengths of query only
    # Adjust labels to include ignore token (default -100) over context
    query_tokenized = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=max_seq_length)
    query_lengths = np.sum(query_tokenized['attention_mask'], axis=1)

    for i in range(len(labels)):
        ignore_idxs = list(range(input_lengths[i], max_seq_length))
        if eval_conditional:
            query_end_idx = query_lengths[i] + 1 # + 1 for the \n between query and response
            ignore_idxs.extend(list(range(query_end_idx)))
          
        np.put(labels[i], ignore_idxs, ignore_token)

    return {'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels}


def main(
    fabric: L.Fabric,
    train_data_path: str,
    val_data_path: str,
    rank: int,
    model_name: str,
    model_ckpt: Path,
    out_dir: Path,
    eval_conditional: bool,
    max_seq_length: int,
    train_start_idx: int = None,
    n_train_samples: int = None,
    fsdp: bool = False,
    load_local_data: bool = False,
) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = os.path.join(out_dir, str(fabric.global_rank))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if fsdp:
        fabric.seed_everything(
            1337, workers=True
        )  # same seed for every process to init model (FSDP)
    else:
        fabric.seed_everything(workers=True)  # each process gets a different seed (DDP)

    config = Config.from_name(model_name)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()

    with fabric.init_module(empty_init=False):
        model = GPT(config)
        # model.apply(model._init_weights)

    fabric.load_raw(f"{model_ckpt}/lit_model.pth", model)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    # Load model and optimizer
    model = fabric.setup(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        foreach=False,
    )
    optimizer = fabric.setup_optimizers(optimizer)

    #with fabric.rank_zero_first():
    train_data = load_datasets(train_data_path, load_local_data, train_start_idx, n_train_samples)
    val_data = load_datasets(val_data_path, load_local_data)

    tokenizer = get_tokenizer(model_ckpt)

    tokenized_train_data = train_data.map(
        tokenize_function, 
        batched=True,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, 'eval_conditional': False, 'max_seq_length': max_seq_length},
    )

    tokenized_val_data = val_data.map(
        tokenize_function, 
        batched=True,
        desc="Running tokenizer on dataset",
        fn_kwargs={"tokenizer": tokenizer, 'eval_conditional': eval_conditional, 'max_seq_length': max_seq_length},
    )

    tokenized_train_data = IterableDatasetShard(
        tokenized_train_data,
        batch_size=micro_batch_size,
        num_processes=fabric.world_size,
        process_index=fabric.global_rank,
    )

    train_dataloader = DataLoader(tokenized_train_data, batch_size=train_batch_size, collate_fn=default_data_collator)
    val_dataloader = DataLoader(tokenized_val_data, batch_size=val_batch_size, collate_fn=default_data_collator)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader,
        val_dataloader,
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "hparams": hparams,
        "iter_num": 0,
        "step_count": 0,
    }

    baseline_scores = evaluate(fabric, model, val_dataloader, eval_conditional)
    print('baseline scores')
    print(baseline_scores)

    results_file = f"{out_dir}/{fabric.global_rank}/baseline_scores.pt"
    torch.save(baseline_scores, results_file)

    indices = [batch['index'].tolist() for batch in val_dataloader]
    indices = [x for xs in indices for x in xs]
    print(indices)
    
    train_iter = iter(train_dataloader)
    data = []

    if train_start_idx is not None and n_train_samples is not None:
        start = train_start_idx
        end = train_start_idx + n_train_samples
    else:
        start = 0
        end = len(train_dataloader)

    for i in tqdm(range(start, end)):
        batch = next(train_iter)
        index = batch['index'][0]
        results_file = f"{results_dir}/{index}.pt"
        if os.path.exists(results_file):
            continue
        
        fabric.load_raw(f"{model_ckpt}/lit_model.pth", state['model'])

        input_ids, labels = batch['input_ids'], batch['labels']
        scores = train(fabric, state, input_ids, labels, val_dataloader, eval_conditional)
        print(scores)
        torch.save(scores, results_file)

    
def train(fabric, state, input_ids, labels, val_dataloader, eval_conditional):
    model = state["model"]
    optimizer = state["optimizer"]

    lr = get_wsd_lr(state["iter_num"]) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    logits = model(input_ids)
    loss = chunked_cross_entropy(
        logits[:, :-1, :].contiguous(),
        labels[:, 1:].contiguous(),
        chunk_size=0,
    )
    fabric.backward(loss)
    fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
    optimizer.step()
    optimizer.zero_grad()

    return evaluate(fabric, model, val_dataloader, eval_conditional)


@torch.no_grad()
def evaluate(fabric, model, val_dataloader, eval_conditional):
    model.eval()

    logprobs_norm = []
    logprobs = []
    for batch in val_dataloader:
        input_ids, labels = batch['input_ids'], batch['labels']

        logits = model(input_ids)

        # Do softmax on logits
        outputs = torch.nn.functional.log_softmax(logits, dim=-1)

        batch_size = outputs.shape[0]
        for i in range(batch_size):
            label, output = labels[i], outputs[i]

            start_idx = ((label != ignore_token).nonzero()[0][0])
            end_idx = ((label != ignore_token).nonzero()[-1][0]) + 1

            if eval_conditional:
                choice_tokens = label[start_idx:end_idx].unsqueeze(1)
                
                """
                print_tokens = choice_tokens.clone()
                print_tokens[print_tokens==ignore_token] = tokenizer.pad_token_id
                print(print_tokens)
                print(tokenizer.decode(print_tokens, skip_special_tokens=True))
                sys.exit()
                """

                lm_log_p = output[start_idx-1:end_idx-1]
            else:
                choice_tokens = label[:end_idx][1:].unsqueeze(1)
                lm_log_p = output[:end_idx][:-1]

            lm_log_p = torch.gather(lm_log_p, -1, choice_tokens).squeeze(-1) # gather logits for target tokens
            lm_log_p_n = torch.mean(lm_log_p).item()
            lm_log_p_un = torch.sum(lm_log_p).item()
            
            logprobs_norm.append(lm_log_p_n)
            logprobs.append(lm_log_p_un)

    model.train()
    return logprobs_norm

def load_datasets(path: str, load_local_data: bool = False, train_start_idx: int = None, n_train_samples: int = None):
    if load_local_data:
        data = load_dataset("json", data_files={'test': path})
        data = data['test']
    else:
        data = load_dataset(path, split="test")
    
    if train_start_idx is not None and n_train_samples is not None:
        start, end = train_start_idx, train_start_idx + n_train_samples
        data = data[start:end]
        data = data.add_column("index", list(range(start, end)))
    else:
        data = data.add_column("index", list(range(len(data))))

    return data


# learning rate decay scheduler (wsd with warmup)
def get_wsd_lr(it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it < stable_iters:
        return learning_rate
    return learning_rate * math.pow(0.5, (it - stable_iters) / 400)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)

"""
@torch.no_grad()
def evaluate(fabric, model, val_dataloader):
    model.eval()
    losses = []
    for batch in val_dataloader:
        input_ids, labels = batch['input_ids'], batch['labels']

        logits = model(input_ids)
        loss = chunked_cross_entropy(
            logits[:, :-1, :].contiguous(),
            labels[:, 1:].contiguous(),
            chunk_size=0,
        )
        losses.append(loss.item())
    model.train()
    print(losses)
    return losses

def get_loss(logits, labels):
    logits = logits[:, :-1, :].contiguous()
    labels = labels[:, 1:].contiguous()
    print(logits.shape)
    print(labels.shape)

    logits = logits.view(-1, logits.size(-1))
    labels = labels.view(-1)
    print(logits.shape)
    print(labels.shape)

    loss = torch.nn.functional.cross_entropy(logits, labels, ignore_index=-100)
    print(loss)

    return loss
"""