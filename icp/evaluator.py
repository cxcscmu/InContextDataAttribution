import collections
import itertools
import logging
import random
from typing import Optional, Union

import numpy as np
import torch

# NOTE: new imports
from tqdm import tqdm
import json
import sys

import lm_eval.api.metrics
import lm_eval.api.registry
import lm_eval.models
from lm_eval.tasks import TaskManager, get_task_dict

from lm_eval.utils import (
    eval_logger,
    get_git_commit_hash,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[str] = None,
    tasks=None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    limit: Optional[Union[int, float]] = None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
    task_manager: TaskManager = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    # NOTE: new parameters
    output_path: str = None,
    incontext_type: str = None,
    incontext_file: str = None,
    repeats: int = 1,
    split: str = None,
    split_args: str = None,
    dataset_kwargs: str = None,
    eval_gold: bool = False,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.

    :return
        Dictionary of results
    """
    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        eval_logger.info(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        eval_logger.info(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        eval_logger.info(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if tasks is None:
        tasks = []
    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    eval_logger.info(
        "get_task_dict has been updated to accept an optional argument, `task_manager`"
        "Read more here:https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage"
    )

    # NOTE: set dataset args if any
    if dataset_kwargs is None:
        override_configs = None
    else:
        dataset_kwargs = dict([tuple(arg.split('=')) for arg in dataset_kwargs.split(',')])
        override_configs = {'dataset_kwargs': dataset_kwargs}

    task_dict = get_task_dict(tasks, task_manager, override_configs=override_configs)

    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.set_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

            if predict_only:
                log_samples = True
                eval_logger.info(
                    f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                )
                # we have to change the class properties post-hoc. This is pretty hacky.
                task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.set_config(key="num_fewshot", value=num_fewshot)

        # NOTE: set additional task configs
        task_obj.set_config(key="incontext_type", value=incontext_type)
        task_obj.set_config(key="incontext_file", value=incontext_file)
        task_obj.set_config(key="repeats", value=repeats)
        task_obj.set_config(key="split", value=split)
        task_obj.set_config(key="split_args", value=split_args)
        task_obj.set_config(key="eval_gold", value=eval_gold)
        task_obj.dump_config()

    if check_integrity:
        run_task_tests(task_list=tasks)

    evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
        # NOTE: new parameters
        output_path=output_path,
        split = split,
    )

    return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit: Optional[int] = None,
    bootstrap_iters: Optional[int] = 100000,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    verbosity: str = "INFO",
    # NOTE: new parameters
    output_path: str = None,
    split: str = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    # decontaminate = decontamination_ngrams_path is not None

    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            _, task = task
        if not log_samples:
            assert (
                "bypass" not in getattr(task, "_metric_fn_list", {}).keys()
            ), f"log_samples must be True for 'bypass' only tasks: {task_name}"

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)

    # get lists of each type of request
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"

        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        # Number of few-shots for printing.
        if (n_shot := configs[task_name].get("num_fewshot")) == 0:
            n_shot = configs[task_name].get("metadata", {}).get("num_fewshot", 0)
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            results[task_name]["alias"] = configs[task_name]["task_alias"]

        if (
            ("group_alias" in configs[task_name])
            and (group_name not in results)
            and (group_name is not None)
        ):
            results[group_name]["alias"] = configs[task_name]["group_alias"]

        if limit is not None:
            # NOTE: update
            if split == 'TEST':
                if task.has_test_docs():
                    task_docs = task.test_docs()
                else:
                    raise RuntimeError("Task has no test_docs")
            elif split == 'VALID':
                if task.has_validation_docs():
                    task_docs = task.validation_docs()
                else:
                    raise RuntimeError("Task has no valid_docs")
            elif split == 'TRAIN':
                if task.has_training_docs():
                    task_docs = task.training_docs()
                else:
                    raise RuntimeError("Task has no train_docs")
            else:
                raise RuntimeError("Split not specified!")

            """  
            if task.has_test_docs():
                task_docs = task.test_docs()
            elif task.has_validation_docs():
                task_docs = task.validation_docs()
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")
            """

            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        task.build_all_requests(limit=limit, rank=lm.rank, world_size=lm.world_size)

        eval_logger.debug(
            f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
        )

        insts = task.instances[:8]
        for inst in insts:
            """
            eval_logger.info(
                f"Task: {task_name}; document {inst.doc_id}; context prompt (starting on next line):\
                \n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)\n"
            )
            """
            eval_logger.info(f"Request: {str(inst)}\n")

        # aggregate Instances by LM method requested to get output.
        eval_logger.info("Grouping instances by reqtype")
        for instance in tqdm(task.instances):
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")

        """
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)
        """

        resps = getattr(lm, reqtype)(reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group, task = task
            if task is None:
                continue
        task.apply_filters()

    # Open output file
    f = open(output_path, "a+")

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group, task = task
            if task is None:
                continue
        # TODO: make it possible to use a different metric per filter
        # iterate over different filters used
        for key in task.instances[0].filtered_resps.keys():
            doc_iterator = (
                itertools.islice(
                    enumerate(task.test_docs()), lm.rank, limit, lm.world_size
                )
                if task.has_test_docs()
                else itertools.islice(
                    enumerate(task.validation_docs()), lm.rank, limit, lm.world_size
                )
            )

            incontext_doc_ids = list(range(task._n_pretrain))
            eval_logger.debug('Writing results')
            doc_iterator = list(doc_iterator)
            
            for (doc_id, doc) in tqdm(doc_iterator, total=len(doc_iterator)):
                task_requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))

                for incontext_doc_id in incontext_doc_ids:
                    requests = list(filter(lambda x: x.incontext_doc_id==incontext_doc_id, task_requests))
                    requests.sort(key=lambda x: x.idx)

                    target = task.doc_to_target(doc)

                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps":  [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[key] for req in requests],
                        "incontext_doc_id" : incontext_doc_id,
                        "incontext_doc" : requests[0].incontext_doc
                    }
                
                    f.write(json.dumps(example) + '\n')
    
    if lm.world_size > 1:
        lm.accelerator.wait_for_everyone()
        f.close()

    return None