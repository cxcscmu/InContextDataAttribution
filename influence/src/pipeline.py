from itertools import chain
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    PreTrainedTokenizer
)

from transformers.pytorch_utils import Conv1D
from datasets import load_dataset, Dataset

import numpy as np
import random
import sys
import re


random.seed(1)


class LanguageModel(nn.Module):
    def __init__(self, model_name) -> None:

        super().__init__()

        self.config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                from_tf=False,
                config=self.config,
                torch_dtype="auto",
                trust_remote_code=True,
        )
            

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).logits


def construct_model(model_name) -> nn.Module:
    return LanguageModel(model_name)


def get_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    
    if tokenizer.pad_token:
        pad_token = tokenizer.pad_token
    else:
        pad_token = "<|pad|>"
        tokenizer.add_special_tokens({"pad_token": pad_token})
    
    return tokenizer
    
def get_loaders(
    test_dataset: str = None,
    train_file: str = None,
    model: str = None,
    tokenizer: PreTrainedTokenizer = None,
    train_batch_size: int = 8,
    eval_batch_size: int = 16,
    n_train_samples: int = None,
    n_test_samples: int = None,
    use_conditional: bool = False,
    max_length: int = 512, 
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    train_dataset = load_train_dataset(train_file, n_train_samples=n_train_samples)
    test_dataset = load_test_dataset(test_dataset, n_test_samples=n_test_samples)

    pretrain_loader = get_dataloader(
        dataset=train_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=train_batch_size,
        split='train',
        max_length=max_length
    )

    test_loader = get_dataloader(
        dataset=test_dataset,
        model=model,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        split='test',
        use_conditional=use_conditional,
        max_length=max_length
    )

    return pretrain_loader, test_loader

def sample_dataset(dataset, n):
    indices = random.sample(list(range(len(dataset))), n)
    dataset = dataset.select(indices)
    return dataset

def load_test_dataset(dataset, n_test_samples=None):
    dataset = load_dataset(dataset, split="test")

    if n_test_samples is not None:
        dataset = sample_dataset(dataset, n_test_samples)
    
    return dataset


def load_train_dataset(train_file, n_train_samples=None):
    with open(train_file) as f:
        data = {'text': [l.strip() for l in f.readlines()][1:]}

    dataset = Dataset.from_dict(data)

    if n_train_samples is not None:
        dataset = sample_dataset(dataset, n_train_samples)
    
    return dataset


def get_dataloader(
    dataset: str = None,
    model: str = None,
    tokenizer: PreTrainedTokenizer = None,
    batch_size: int = 8,
    split: str = None,
    use_conditional: bool = False,
    max_length: int = 512, 
    ignore_token: int = -100
) -> torch.utils.data.DataLoader:

    # tokenizes each row in dataset
    def tokenize_function(examples):
        if split == 'test':
            inputs = [examples["text"][i] + '\n' + examples["target"][i] for i in range(len(examples['text']))]
        else:
            inputs = examples["text"]

        tokenize_output = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length)
        input_ids = tokenize_output['input_ids']
        attention_mask = tokenize_output['attention_mask']
        labels = np.copy(input_ids)

        # Adjust labels to include ignore token (default -100) over context
        ctx_tokenize_output = tokenizer(examples["text"], padding='max_length', truncation=True, max_length=max_length)
        ctx_lengths = np.sum(ctx_tokenize_output['attention_mask'], axis=1)

        input_lengths = np.sum(tokenize_output['attention_mask'], axis=1)

        for i in range(len(labels)):
            input_length = input_lengths[i]
            labels[i] = [tok if tok != tokenizer.pad_token_id else ignore_token for tok in labels[i]]

            if split == 'test' and use_conditional:
                ctx_mask_idx = ctx_lengths[i] + 1 # + 1 for the space between instruction and response
                labels[i][: ctx_mask_idx] = [ignore_token] * ctx_mask_idx

        return {'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}

    # call tokenizer on all rows
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=None,
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    dataloader = torch.utils.data.DataLoader(
        tokenized_datasets,
        batch_size=batch_size,
        collate_fn=default_data_collator,
    )
    return dataloader
