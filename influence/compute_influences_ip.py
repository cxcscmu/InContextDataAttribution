import argparse
import os
import re
from os.path import join
from typing import List
import sys

import torch

from src.pipeline import construct_model, get_loaders, get_tokenizer
from src.task import LanguageModelTask
from src.tracin import TracinComputer

parser = argparse.ArgumentParser()

# model arguments
parser.add_argument("--model_name", type=str, help="name of model or path to model directory")
parser.add_argument("--checkpoints", nargs="*", type=str, default=[], help="list of paths to checkpoints of model to use")
parser.add_argument("--lrs", nargs="*", type=float, default=[], help="list of learning rates for each checkpoint")
parser.add_argument('--model_dtype', type=str, default='float16', help="float16|float32")

# data arguments
parser.add_argument("--test_dataset", type=str, help="path to test data")
parser.add_argument("--train_file", type=str, help="path to file with all the training prompts")
parser.add_argument("--n_train_samples", type=int, default=None, help="maximum number of samples from train dataset")
parser.add_argument("--n_test_samples", type=int, default=None, help="maximum number of samples from test dataset")
parser.add_argument("--train_batch_size", type=int, help="train batch size")
parser.add_argument("--test_batch_size", type=int, help="test batch size")
parser.add_argument('--load_local', action='store_true', help="load local test set")

# other arguments
parser.add_argument("--metric", type=str, default="dot", help="dot or cos for computing gradient similarity")
parser.add_argument("--use_conditional", action="store_true", help="conditional cross-entropy")
parser.add_argument("--max_length", type=int, default=512, help="maximum length of ")
parser.add_argument("--outfile", type=str, default=None, help="")
parser.add_argument("--grad_approx", type=str, default="sign_log", help="log|sign_log|none")
parser.add_argument('--grad_clip', action='store_true')

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
        load_local=args.load_local,
    )

    model = construct_model(model_name=args.model_name)
    if args.model_dtype == 'float32':
        model = model.float()

    model.eval()

    task = LanguageModelTask(device=DEVICE, layers=None)

    return model.to(DEVICE), train_loader, test_loader, task


def compute_if() -> None:
    model, eval_train_loader, valid_loader, task = prepare_everything()

    tracinComputer = TracinComputer(
        model=model,
        task=task,
        metric=args.metric,
        grad_approx=args.grad_approx,
        grad_clip=args.grad_clip,
    )

    scores = tracinComputer.compute_scores_with_loader(
        test_loader=valid_loader,
        train_loader=eval_train_loader,
        checkpoints=args.checkpoints,
        lrs=args.lrs,
        model_dtype=args.model_dtype,
    )

    torch.save(scores, args.outfile)


if __name__ == "__main__":
    compute_if()
