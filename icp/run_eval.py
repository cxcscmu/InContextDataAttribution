import argparse
import json
import logging
import os
import re
import sys
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np

import evaluator
from lm_eval import utils
from lm_eval.tasks import TaskManager, include_path, initialize_tasks
from lm_eval.utils import make_table

import torch
import datetime

# torch.distributed.init_process_group(backend='nccl', init_method=None, timeout=datetime.timedelta(seconds=7200), world_size=- 1, rank=- 1, store=None, group_name='', pg_options=None)

def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _int_or_none_list_arg_type(max_len: int, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items != max_len:
        raise argparse.ArgumentTypeError(
            f"Argument requires {max_len} integers or None, separated by '{split_char}'"
        )

    return items


def parse_eval_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", "-m", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        "-t",
        default=None,
        metavar="task1,task2",
        help="To get full list of tasks, use the command lm-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        "-a",
        default="",
        help="Comma separated string arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--num_fewshot",
        "-f",
        type=int,
        default=None,
        metavar="N",
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu).",
    )
    parser.add_argument(
        "--output_path",
        "-o",
        default=None,
        type=str,
        metavar="DIR|DIR/file.json",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--limit",
        "-L",
        type=float,
        default=None,
        metavar="N|0<N<1",
        help="Limit the number of examples per task. "
        "If <1, limit is a percentage of the total number of examples.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="DIR",
        help="A path to a sqlite db file for caching model responses. `None` if not caching.",
    )
    parser.add_argument("--decontamination_ngrams_path", default=None)  # TODO: not used
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks.",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="Prints the prompt for the first few documents.",
    )
    parser.add_argument(
        "--log_samples",
        "-s",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis. Use with --output_path.",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        metavar="DIR",
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default=None,
        help=(
            "String arguments for model generation on greedy_until tasks,"
            " e.g. `temperature=0,top_k=0,top_p=0`."
        ),
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        type=str.upper,
        default="INFO",
        metavar="CRITICAL|ERROR|WARNING|INFO|DEBUG",
        help="Controls the reported logging error level. Set to DEBUG when testing + adding new task configurations for comprehensive log output.",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3),
        default="0,1234,1234",  # for backward compatibility
        help=(
            "Set seed for python's random, numpy and torch.\n"
            "Accepts a comma-separated list of 3 values for python's random, numpy, and torch seeds, respectively, "
            "or a single integer to set the same seed for all three.\n"
            "The values are either an integer or 'None' to not set the seed. Default is `0,1234,1234` (for backward compatibility).\n"
            "E.g. `--seed 0,None,8` sets `random.seed(0)` and `torch.manual_seed(8)`. Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all three seeds to 42."
        ),
    )
    # NOTE: new parameters
    parser.add_argument(
        "--use_wandb",
        action='store_true',
        help="Use wandb to log"
    )
    parser.add_argument(
        "--incontext_type",
        type=str,
        default='random',
        metavar="random|iterate",
        help="Type of incontext examples to use"
    )
    parser.add_argument(
        "--incontext_file",
        type=str,
        default=None,
        help="File containing pretrain examples to use incontext to iterate over"
    )
    # NOTE: Overrides "repeats" in task config! 
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        metavar="N|1<=N",
        help="Number of repeats per request"
    )
    parser.add_argument(
        "--split",
        type=str,
        default='TEST',
        help=(
            "Split of the dataset to run evaluation on"
            "IMPORTANT: this is different from dataset_kwargs which used used for downloading the entire dataset"
            "options=TRAIN|TEST|VALID"
        ),
    )
    parser.add_argument(
        "--split_args",
        type=str,
        default=None,
        help=(
            "Arguments for `split`. E.g., [:40%] to evaluate on the first half of the dataset"
        ),
    )
    parser.add_argument(
        "--dataset_kwargs",
        type=str,
        help=(
            "String arguments for when downloading the dataset"
            "e.g. `split=train[:10]"
            "IMPORTANT: in general, there is no need to to use this parameter."
        )
    )
    parser.add_argument(
        "--eval_gold",
        action='store_true',
        help="evaluate only on the correct answer in multiple choice settings"
    )
    return parser.parse_args()


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    if not args:
        # we allow for args to be passed externally, else we parse them ourselves
        args = parse_eval_args()

    eval_logger = utils.eval_logger
    eval_logger.setLevel(getattr(logging, f"{args.verbosity}"))
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        assert args.output_path, "Specify --output_path"

    initialize_tasks(args.verbosity)
    task_manager = TaskManager(args.verbosity, include_path=args.include_path)

    if args.limit:
        eval_logger.warning(
            " --limit SHOULD ONLY BE USED FOR TESTING."
            "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )
    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
        include_path(args.include_path)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info(
            "Available Tasks:\n - {}".format("\n - ".join(task_manager.all_tasks))
        )
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n"
                    f"{utils.SPACING}Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks list` for list of available tasks, or '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    if args.output_path:
        path = Path(args.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file() or Path(args.output_path).joinpath("results.json").is_file():
            eval_logger.warning(
                f"File already exists at {path}. Results will be overwritten."
            )
            output_path_file = path.joinpath("results.json")
            assert not path.is_file(), "File already exists"
        # if path json then get parent dir
        elif path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
            output_path_file = path.joinpath("results.json")

    eval_logger.info(f"Selected Tasks: {task_names}")
    eval_logger.info("Loading selected tasks...")

    # NOTE: Initialize wandb
    if args.use_wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="cont-pretrain",
            # Track metadata
            name='{}-{}'.format(args.model_args, args.tasks),
            config={
                "": args.tasks,
            },
        )
 
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        
        # NOTE: new parameters
        output_path=args.output_path,
        incontext_type=args.incontext_type,
        incontext_file=args.incontext_file,
        repeats=args.repeats,
        split=args.split,
        split_args=args.split_args,
        eval_gold=args.eval_gold,
        dataset_kwargs=args.dataset_kwargs,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        if args.output_path:
            output_path_file.open("w", encoding="utf-8").write(dumped)

            if args.log_samples:
                for task_name, config in results["configs"].items():
                    output_name = "{}_{}".format(
                        re.sub("/|=", "__", args.model_args), task_name
                    )
                    filename = path.joinpath(f"{output_name}.jsonl")
                    samples_dumped = json.dumps(
                        samples[task_name],
                        indent=2,
                        default=_handle_non_serializable,
                        ensure_ascii=False,
                    )
                    filename.write_text(samples_dumped, encoding="utf-8")

        print(
            f"{args.model} ({args.model_args}), gen_kwargs: ({args.gen_kwargs}), limit: {args.limit}, num_fewshot: {args.num_fewshot}, "
            f"batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()