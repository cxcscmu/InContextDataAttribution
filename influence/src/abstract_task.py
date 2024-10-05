import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from trak.modelout_functions import AbstractModelOutput


class InvalidTaskError(Exception):
    # Raised when the task is invalid.
    pass


class AbstractTask(ABC):
    """An abstract base class for Tasks.

    Subclasses of this abstract class should facilitate computing TDA (Training Data Attribution) methods
    according to specific pipelines (e.g., models, data loaders, training objectives). For practical implementations
    of the Task class tailored to regression, classification, and language modeling, see the `examples/` directory.
    """

    @abstractmethod
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        generator: Optional[torch.Generator] = None,
        layers: Optional[list] = None,
    ) -> None:
        """Initializes the class AbstractTask.

        Args:
            device (torch.device):
                Specifies the device for computation, defaulting to CPU.
            generator (torch.Generator, optional):
                Generator for ensuring consistent experiment behavior, such as sampling outputs for
                true Fisher computation. If not provided, defaults to non-deterministic sampling.
        """
        self.device = device
        self.generator = generator
        self.layers = layers

    @abstractmethod
    def get_train_loss(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Computes the training loss for a given model and batch. Ensure operations within this
        function are traceable by autograd.

        Args:
            model (nn.Module):
                PyTorch model for which the training loss will be computed.
            batch (Any):
                Batch of data sourced from the DataLoader.
            parameter_and_buffer_dicts (tuple, optional):
                Instead of relying on the model's inherent parameters (given by `model.parameters()`),
                specific parameters can be directly provided for loss computation.
            sample (bool):
                If set to True, labels are sampled from the outputs; otherwise, the actual label is used.
            reduction (str):
                Determines the type of output. By default, it returns the cumulative loss. To alter
                this behavior, specify either 'average' (for the mean loss) or 'none'
                (to get losses for each data point).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_measurement(
        self,
        model: nn.Module,
        batch: Any,
        parameter_and_buffer_dicts: Optional[
            Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
        ] = None,
        reduction: str = "sum",
    ) -> torch.Tensor:
        """Computes the measurement (e.g., loss, margin, conditional log probability) for a given model and batch.
        The measurement is defined as `f(\theta)` from https://arxiv.org/pdf/2308.03296.pdf. Ensure operations within
        this function are traceable by autograd.

        Args:
            model (nn.Module):
                PyTorch model for which the measurement will be computed.
            batch (Any):
                Batch of data sourced from the DataLoader.
            parameter_and_buffer_dicts (tuple, optional):
                Instead of relying on the model's inherent parameters (given by `model.parameters()`),
                specific parameters can be directly provided for measurement computation.
            reduction (str):
                Determines the type of output. By default, it returns the cumulative loss. To alter
                this behavior, specify either 'average' (for the mean loss) or 'none'
                (to get losses for each data point).
        """
        raise NotImplementedError()

    @abstractmethod
    def get_batch_size(self, batch: Any) -> int:
        """Given a batch of data, return the batch size.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.
        """
        raise NotImplementedError()

    def influence_modules(self) -> Optional[List[str]]:
        """Returns module names of layers for which TDA methods should be computed based on the specified architecture.

        If the function returns `None`, influences will be computed for all available modules such as
        Embedding, Linear, Conv2d, etc.
        """
        return None

    @abstractmethod
    def representation_module(self) -> str:
        """Returns module name of layers for which `RepresentationSimilarity` should be computed based on
        the specified architecture. Typically, this is set to the module just before the last layer.
        """
        raise NotImplementedError()

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        """Returns masks for data points that have been padded to ensure consistent length, as observed in
        architectures like Transformers. For architectures with a fixed input size, the standard behavior
        should be to return `None`.

        Note that this function is used for masking activations during the EK-FAC influence computations.

        Args:
            batch (Any):
                Batch of data sourced from the DataLoader.
        """
        del batch
        return None

    def get_model_output(self) -> Optional[AbstractModelOutput]:
        """TRAK requires implementing the AbstractModeOutput. To use TRAK, this method must
        be implemented and functioning correctly.

         AbstractModelOutput should contain two functions:
        - The model output function itself.
        - The gradient of the (training) loss w.r.t. the model output function.

        See https://trak.readthedocs.io/en/latest/modeloutput.html for details.
        """
        return None


def validate_task(
    model: nn.Module, task: AbstractTask, logger: Optional[logging.Logger] = None
) -> None:
    """Tests if the `task` is properly defined for the given `model`.

    Args:
        model (nn.Module):
            PyTorch module to test.
        task (AbstractTask):
            Task specific to the given model.
        logger (Logger, optional):
            If provided, sends error messages through logger.
    """

    influence_modules = task.influence_modules()
    influence_module_exists_dict = {name: False for name in influence_modules}
    representation_module = task.representation_module()
    if representation_module is None and logger is not None:
        logger.warning("`representation_module` is not defined.")
        representation_module_exists = True
    else:
        representation_module_exists = False

    model_output = task.get_model_output()
    if model_output is None and logger is not None:
        logger.warning(
            "`model_output` is not defined (this must be implemented for TRAK)."
        )

    for name, module in model.named_modules():
        #print(name)
        #print(module)
        if name in influence_module_exists_dict.keys():
            if not isinstance(
                module,
                (nn.Linear, nn.Conv2d, nn.Embedding, nn.LayerNorm, nn.BatchNorm2d),
            ):
                error_msg = (
                    f"The provided influence module with {name} is not supported."
                )
                if logger is not None:
                    logger.error(error_msg)
                raise InvalidTaskError(error_msg)
            influence_module_exists_dict[name] = True

        if representation_module is not None and representation_module == name:
            representation_module_exists = True

    if not all(list(influence_module_exists_dict.values())):
        error_msg = (
            f"Some provided influence modules were not found. The found mapping: "
            f"{list(influence_module_exists_dict.keys())}."
            f"{list(influence_module_exists_dict.values())}."
        )
        if logger is not None:
            logger.error(error_msg)
        raise InvalidTaskError(error_msg)

    if not representation_module_exists:
        error_msg = f"Provided representation module with name {representation_module} was not found."
        if logger is not None:
            logger.error(error_msg)
        raise InvalidTaskError(error_msg)

    return
