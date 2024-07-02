import argparse
import os
import re
from os.path import join
from typing import List
import sys

import torch

from src.pipeline import construct_model, get_loaders, get_tokenizer
from src.task import LanguageModelTask
from src.influence_function import InfluenceFunctionComputer

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument("--model_name", type=str, help="name of model or path to model directory")
parser.add_argument('--checkpoint', type=str, default=None, help="path to a check point of the model to use")

# data arguments
parser.add_argument("--test_dataset", type=str, help="path to test data")
parser.add_argument("--train_file", type=str, help="path to file with all the training prompts")
parser.add_argument("--n_train_samples", type=int, default=None, help="maximum number of samples from train dataset")
parser.add_argument("--n_test_samples", type=int, default=None, help="maximum number of samples from test dataset")
parser.add_argument("--train_batch_size", type=int, help="train batch size")
parser.add_argument("--test_batch_size", type=int, help="test batch size")

# other arguments
parser.add_argument('--use_conditional', action='store_true')
parser.add_argument('--outfile', type=str, default=None)
args = parser.parse_args()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_everything():
    tokenizer = get_tokenizer(args.model_name)
    
    train_loader, test_loader = get_loaders(
        test_dataset=args.test_dataset,
        model=args.model_name,
        tokenizer=tokenizer,
        train_file=args.train_file,
        eval_batch_size=args.test_batch_size,
        train_batch_size=args.train_batch_size,
        n_train_samples=args.n_train_samples,
        n_test_samples=args.n_test_samples,
        use_conditional=args.use_conditional,
        max_length=args.max_length,
    )

    model = construct_model(model_name=args.model_name)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu",))
            
    model.eval()

    task = LanguageModelTask(device=DEVICE, layers=None)

    return model.to(DEVICE), train_loader, test_loader, task


def compute_if() -> None:
    model, eval_train_loader, valid_loader, task = prepare_everything()

    ekfac = InfluenceFunctionComputer(
        model=model,
        task=task,
        n_epoch=1,
    )
    ekfac.build_curvature_blocks(eval_train_loader)
    scores = ekfac.compute_scores_with_loader(
        test_loader=valid_loader, train_loader=eval_train_loader
    )
    torch.save(scores, args.outfile)


if __name__ == "__main__":
    compute_if()
