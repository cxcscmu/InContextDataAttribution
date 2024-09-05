import copy
from typing import Any, List, Union

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM


from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.gradient_similarity import GradientSimilarityComputer

class LanguageModel(nn.Module):
    def __init__(self, model_name) -> None:

        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            ignore_mismatched_sizes=False,
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


class TracinComputer(AbstractComputer):
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        metric: str = "cos",
        grad_approx: str = "sign_log",
        grad_clip: bool = False,
    ):
        """Initializes the `TracIn` class.

        This class performs TDA using similarity between gradients (over trajectories) based on a specified metric
        (for more details, see https://arxiv.org/pdf/2002.08484.pdf). Note that when `metric = "cos"`, the method
        is called Gradients Aggregated Similarity (GAS).

        Args:
            model (nn.Module):
                The PyTorch model for which similarities are computed.
            task (AbstractTask):
                The task for the pipeline.
            metric (str, optional):
                The metric used to measure similarity. Supported metrics include "dot"
                and "cos". Defaults to "dot".
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        # Save original parameters to CPU.
        self.original_state_dict = copy.deepcopy(self.model.state_dict())
        for name, tensor in self.original_state_dict.items():
            self.original_state_dict[name] = tensor.cpu()

        self.metric = metric
        self.grad_approx = grad_approx
        self.grad_clip = grad_clip

    def _load_checkpoint(self, checkpoint: str, model_dtype: str = 'float16') -> None:
        """Given the path to the checkpoint, load the parameters and buffers."""

        self.model = LanguageModel(checkpoint)

        if model_dtype == 'float32':
            self.model = self.model.float()

        self.model = self.model.to(self.task.device)
        self.model.eval()

    def _reload_original_params(self) -> None:
        """Reload the initial parameters and buffers, given at the initialization stage."""
        self.model.load_state_dict(self.original_state_dict)
        self.model = self.model.to(self.task.device)

    def compute_scores_with_batch(
        self,
        batch1: Any,
        batch2: Any,
        checkpoints: List[str],
        lrs: Union[List[float], float],
        model_dtype: str = 'float16'
    ) -> torch.Tensor:
        """Compute pairwise influence scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
            checkpoints (list):
                A list of paths to the checkpoints.
            lrs (list, float):
                Learning rates used for the checkpoints. If a single float is provided,
                it is assumed that the learning rate was fixed for all checkpoints. Otherwise,
                provide a list of floats with the same size as the list of checkpoints.
        """
        self.model.eval()

        # Perform lazy initialization.
        score_table = 0.0

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt, model_dtype)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
                grad_approx=self.grad_approx,
                grad_clip=self.grad_clip,
            )
            score_table += lrs[i] * gsc.compute_scores_with_batch(
                batch1=batch1, batch2=batch2
            )
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        checkpoints: List[str],
        lrs: Union[int, float, List[int], List[float]] = 1.0,
        model_dtype: str = 'float16'
    ) -> torch.Tensor:
        """Compute pairwise similarity scores between `test_loader` and `train_loader`.

        Args:
            test_loader (object):
                The loader with test dataset.
            train_loader (object):
                The loader with training dataset.
            checkpoints (list):
                A list of paths to the checkpoints.
            lrs (list, float):
                Learning rates used for the checkpoints. If a single float is provided,
                it is assumed that the learning rate was fixed for all checkpoints. Otherwise,
                provide a list of floats with the same size as the list of checkpoints.
        """
        self.model.eval()

        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.score_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt, model_dtype)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
                grad_approx=self.grad_approx,
                grad_clip=self.grad_clip,
            )
            scores = gsc.compute_scores_with_loader(
                test_loader=test_loader, train_loader=train_loader
            )
            score_table += lrs[i] * scores
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table

    def compute_self_scores_with_loader(
        self,
        loader: torch.utils.data.DataLoader,
        checkpoints: List[str],
        lrs: Union[int, float, List[int], List[float]] = 1.0,
        model_dtype: str = 'float16',
    ) -> torch.Tensor:
        """Compute self-similarity scores of all data points in `loader`.

        Args:
            loader (DataLoader):
                The loader for which self-similarity scores are computed.
            checkpoints (list):
                A list of paths to the checkpoints.
            lrs (list, float):
                Learning rates used for the checkpoints. If a single float is provided,
                it is assumed that the learning rate was fixed for all checkpoints. Otherwise,
                provide a list of floats with the same size as the list of checkpoints.
        """

        self.model.eval()

        # Perform lazy initialization.
        score_table = 0.0

        if isinstance(lrs, float) or isinstance(lrs, int):
            lrs = [lrs for _ in range(len(checkpoints))]

        for i, ckpt in enumerate(checkpoints):
            self._load_checkpoint(ckpt, model_dtype)
            gsc = GradientSimilarityComputer(
                model=self.model,
                task=self.task,
                metric=self.metric,
                grad_approx=self.grad_approx,
                grad_clip=self.grad_clip,
            )

            scores = gsc.compute_self_scores_with_loader(loader=loader)

            score_table += lrs[i] * scores
            del gsc

        score_table.div_(len(checkpoints))
        self._reload_original_params()
        return score_table
