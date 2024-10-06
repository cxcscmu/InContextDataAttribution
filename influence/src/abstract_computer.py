import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import torch
import torch.nn as nn

from src.abstract_task import AbstractTask, validate_task


class AbstractComputer(ABC):
    """An abstract base class for Computers."""

    # Specifies the dtype for storing TDA scores.
    score_dtype: torch.dtype = torch.float32

    # Specifies the dtype for storing gradients.
    grads_dtype: torch.dtype = torch.float32

    # Specifies the dtype for storing statistics (only applies to influence functions).
    stats_dtype: torch.dtype = torch.float32

    # Specifies the dtype for performing eigendecompositon (only applies to influence functions).
    eig_dtype: torch.dtype = torch.float64

    @abstractmethod
    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        logger_name: str,
        logging_level: int = logging.INFO,
    ) -> None:
        """Initializes the class AbstractComputer.

        Args:
            model (nn.Module):
                PyTorch model for which scores are computed.
            task (AbstractTask):
                Specifies the task for the pipeline. For details, see `AbstractTask` in
                `src/abstract_task.py`.
            logger_name (str):
                Name of the logger.
            logging_level (int, optional):
                The logging level. Defaults to `logging.INFO`.
        """
        self.model = model
        self.task = task

        # Setup logging configurations.
        logging.basicConfig(
            format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d:%H:%M:%S',)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging_level)

        validate_task(model=self.model, task=self.task, logger=self.logger)

    def _compute_train_loss(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Computes the cumulative training loss for a given batch using specified model parameters and buffers.

        Args:
            params (dict):
                Model parameters to be used for computation.
            buffers (dict):
                Model buffers to be used for computation.
            batch (Any):
                The batch of data on which the loss will be computed.
        """
        
        loss = self.task.get_train_loss(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            sample=False,
            reduction="sum",
        )
        print(loss)
        print(loss.requires_grad)
        return loss

    def _compute_train_loss_grad(self) -> Callable:
        """Returns the function that computes gradients of loss w.r.t. parameters."""
        return torch.func.grad(self._compute_train_loss, argnums=0, has_aux=False)

    def _compute_measurement(
        self,
        params: Dict[str, torch.Tensor],
        buffers: Dict[str, torch.Tensor],
        batch: Any,
    ) -> torch.Tensor:
        """Computes the cumulative measurement for a given batch using specified model parameters and buffers.

        Args:
            params (dict):
                Model parameters to be used for computation.
            buffers (dict):
                Model buffers to be used for computation.
            batch (Any):
                The batch of data on which the loss will be computed.
        """
        return self.task.get_measurement(
            model=self.model,
            batch=batch,
            parameter_and_buffer_dicts=(params, buffers),
            reduction="sum",
        )

    def _compute_measurement_grad(self) -> Callable:
        """Returns the function that computes gradients of measurement w.r.t. parameters."""
        return torch.func.grad(self._compute_measurement, argnums=0, has_aux=False)
