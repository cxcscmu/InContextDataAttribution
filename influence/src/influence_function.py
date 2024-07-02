import time
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from tqdm import tqdm

from src.abstract_computer import AbstractComputer
from src.abstract_task import AbstractTask
from src.ekfac_utils import (
    InvalidModuleError,
    extract_activations,
    extract_gradients,
    make_grads_dict_to_matrix,
)

class InfluenceFunctionComputer(AbstractComputer):
    _supported_kronecker_modules = {"Linear", "Conv2d"}
    _supported_full_modules = {"LayerNorm", "BatchNorm2d"}
    _supported_diag_modules = {"Embedding"}
    eig_dtype: torch.dtype = torch.float64

    def __init__(
        self,
        model: nn.Module,
        task: AbstractTask,
        damping: Optional[float] = None,
        n_epoch: int = 1,
        use_true_fisher: bool = True,
    ) -> None:
        """Initializes the `InfluenceFunctionComputer` class.

        This class performs TDA using influence functions. More specifically, instead of using expensive
        iterative computers such as LiSSA, the class uses EK-FAC approximation. The details can be
        found in https://arxiv.org/pdf/2308.03296.pdf.

        Args:
            model (nn.Module):
                PyTorch model for which influences are computed.
            task (AbstractTask):
                Specifies the task for the pipeline.
            damping (float, optional):
                A damping term. If not provided, the module-wise damping is set as
                0.1 times the mean of the eigenvalues.
            n_epoch (int, optional):
                Number of epochs to compute covariance and lambda statistics. Defaults to 1.
            use_true_fisher (bool, optional):
                If set to True, targets are sampled from the outputs. If set to False, the
                empirical Fisher is used, and the targets are set to the true targets. Default is True.
        """
        super().__init__(model=model, task=task, logger_name=self.__class__.__name__)

        self.func_params = dict(self.model.named_parameters())
        self.func_buffers = dict(self.model.named_buffers())

        self.damping = damping
        self.n_epoch = n_epoch
        self.use_true_fisher = use_true_fisher

        # List of attributes to navigate modules.
        self._module_to_name = {
            v: k for k, v in dict(self.model.named_modules()).items()
        }
        self.modules = []
        self.modules_name = []
        # Modules, where influences are computed using EK-FAC (e.g., Linear, Conv2d).
        self.kronecker_modules_name = []
        # Modules, where influences are computed using the full Fisher (e.g., LayerNorm2d).
        self.full_modules_name = []
        # Modules, where influence are computied using the diagonal Fisher (e.g., Embedding).
        self.diag_modules_name = []

        # List of attributes to keep track EK-FAC computation.
        self.activation_cov, self.pseudograd_cov = {}, {}
        self.activation_cov_eigvecs, self.pseudograd_cov_eigvecs = {}, {}
        self.activation_cov_eigvals, self.pseudograd_cov_eigvals = {}, {}
        self.kronecker_eigvals, self.full_factors, self.diag_factors = {}, {}, {}
        self.damping_factors = {}
        self._activation_masks = None

        # Flags to indicate which computation has already finished.
        self._covariance_done = False
        self._eigendecompositon_done = False
        self._additional_factors_done = False

        self._handles = []
        self.initialize()

    def initialize(self) -> None:
        """Initializes all hooks and save module mappings."""
        for name, module in self.model.named_modules():
            classname = module.__class__.__name__

            if name in self.task.influence_modules():
                self.logger.info(f"Found module {name}.")

                if classname in self._supported_kronecker_modules:
                    print('{} in supported_kronecker_modules'.format(classname))
                    # Register all modules.
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.kronecker_modules_name.append(name)

                    # Register forward hooks.
                    handle = module.register_forward_pre_hook(self._forward_hook)
                    self._handles.append(handle)

                    # Register backward hooks.
                    handle = module.register_full_backward_hook(self._backward_hook)
                    self._handles.append(handle)

                if classname in self._supported_full_modules:
                    # This is only supported for LayerNorm & BatchNorm parameters (with affine set to True).
                    if module.weight is not None:
                        self.modules.append(module)
                        self.modules_name.append(name)
                        self.full_modules_name.append(name)

                if classname in self._supported_diag_modules:
                    # This is supported for embedding parameters.
                    self.modules.append(module)
                    self.modules_name.append(name)
                    self.diag_modules_name.append(name)

        if len(self.modules_name) == 0:
            error_msg = f"Cannot find any modules in {self.task.influence_modules()}."
            self.logger.error(error_msg)
            raise AttributeError(error_msg)

    def _forward_hook(self, module: nn.Module, inputs: Tuple[torch.Tensor]) -> None:
        """Computes the pre-activation covariance matrices."""
        assert len(inputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            acts = extract_activations(
                inputs[0].data.to(dtype=self.stats_dtype),
                module,
                self._activation_masks,
            ).to(dtype=self.stats_dtype)
            if module_name not in self.activation_cov:
                last_dim = acts.shape[-1]
                self.activation_cov[module_name] = torch.zeros(
                    (last_dim, last_dim), dtype=self.stats_dtype, device=acts.device
                )
            self.activation_cov[module_name].addmm_(acts.t(), acts)

    def _backward_hook(
        self,
        module: nn.Module,
        grad_inputs: Tuple[torch.Tensor],
        grad_outputs: Tuple[torch.Tensor],
    ) -> None:
        """Computes the pseudo-gradient covariance matrices."""
        del grad_inputs
        assert len(grad_outputs) == 1

        with torch.no_grad():
            module_name = self._module_to_name[module]
            pseudograds = extract_gradients(
                grad_outputs[0].data.to(dtype=self.stats_dtype), module
            ).to(dtype=self.stats_dtype)
            if module_name not in self.pseudograd_cov:
                last_dim = pseudograds.shape[-1]
                self.pseudograd_cov[module_name] = torch.zeros(
                    (last_dim, last_dim),
                    dtype=self.stats_dtype,
                    device=pseudograds.device,
                )
            self.pseudograd_cov[module_name].addmm_(pseudograds.t(), pseudograds)

    def _perform_forward_and_backward_pass(
        self, batch: Any, sample: bool = False, reduction: str = "sum"
    ) -> None:
        """Perform the forward and backward pass with the given `batch`."""
        loss = self.task.get_train_loss(
            model=self.model,
            batch=batch,
            sample=sample,
            reduction=reduction,
        )
        loss.backward()

    def fit_covariances(self, loader: torch.utils.data.DataLoader) -> None:
        """Fit the pre-activation and pseudo-gradient covariance matrices given the loader.

        Args:
            loader (DataLoader):
                Dataloader in which covariances are computed for.
        """
        print('Fitting covariances')

        if self._covariance_done:
            self.logger.info("Covariance computation is already done. Skipping.")
            return

        t0 = time.time()
        self.model.eval()
        examples_seen = 0
        for _ in range(self.n_epoch):
            for batch in loader:
                self.model.zero_grad()
                self._activation_masks = self.task.get_activation_masks(batch)
                self._perform_forward_and_backward_pass(
                    batch, sample=self.use_true_fisher, reduction="sum"
                )
                examples_seen += self.task.get_batch_size(batch)

        with torch.no_grad():
            # Normalize the covariances by the number of data points.
            for name in self.kronecker_modules_name:
                self.activation_cov[name] /= examples_seen
                self.activation_cov[name] = self.activation_cov[name].to(
                    self.grads_dtype
                )
                self.pseudograd_cov[name] /= examples_seen
                self.pseudograd_cov[name] = self.pseudograd_cov[name].to(
                    self.grads_dtype
                )
        self._covariance_done = True
        self.logger.info(f"Seen {examples_seen} examples.")
        self.logger.info(f"Time for computing covariances: {time.time() - t0}")

    def fit_eigendecompositions(self, keep_cache: bool = False) -> None:
        """Compute eigendecomposition of all covariances.

        Args:
            keep_cache (bool):
                If true, delete the covariance matrices and eigenvalues from memory (just keeping the eigenbasis),
                after performing the eigendecomposition. Defaults to True to save memory.
        """
        print('Fitting Eigendecompositions')
        
        if self._eigendecompositon_done:
            self.logger.info(
                "Eigendecomposition computation is already done. Skipping."
            )
            return

        t1 = time.time()
        with torch.no_grad():
            for name in self.kronecker_modules_name:
                orig_dtype = self.activation_cov[name].dtype
                eigvals, eigvecs = torch.linalg.eigh(
                    self.activation_cov[name].to(dtype=self.eig_dtype)
                )
                self.activation_cov_eigvals[name] = eigvals.to(dtype=orig_dtype)
                self.activation_cov_eigvecs[name] = eigvecs.to(dtype=orig_dtype)

                if not keep_cache:
                    del self.activation_cov[name]
                    del self.activation_cov_eigvals[name]

                orig_dtype = self.pseudograd_cov[name].dtype
                eigvals, eigvecs = torch.linalg.eigh(
                    self.pseudograd_cov[name].to(dtype=self.eig_dtype)
                )
                self.pseudograd_cov_eigvals[name] = eigvals.to(dtype=orig_dtype)
                self.pseudograd_cov_eigvecs[name] = eigvecs.to(dtype=orig_dtype)

                if not keep_cache:
                    del self.pseudograd_cov[name]
                    del self.pseudograd_cov_eigvals[name]
        self._eigendecompositon_done = True
        self.logger.info(f"Time for eigendecomposition: {time.time() - t1}")

    def compute_kronecker_lambda(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        """Compute the Lambda term (corrected eigenvalues) for the EK-FAC.

        For details, see https://arxiv.org/pdf/2308.03296.pdf.

        Args:
            module_name (str):
                Name of the module.
            per_batch_grads (torch.Tensor)
                Individual per batch gradients.
        """
        assert len(per_batch_grads.shape) == 3

        if module_name not in self.kronecker_eigvals:
            self.kronecker_eigvals[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        grads_rot = torch.matmul(
            per_batch_grads, self.activation_cov_eigvecs[module_name]
        )

        batch_size = grads_rot.shape[0]
        for i in range(batch_size):
            weight_grad_rot = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name].t(), grads_rot[i, :, :]
            )
            self.kronecker_eigvals[module_name].add_(torch.square(weight_grad_rot))

    def compute_full_factors(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        """Compute the full Fisher given the `module_name`.

        Args:
            module_name (str):
                Name of the module.
            per_batch_grads (torch.Tensor)
                Individual per batch gradients.
        """
        assert len(per_batch_grads.shape) == 2
        if module_name not in self.full_factors:
            last_dim = per_batch_grads.shape[1]
            self.full_factors[module_name] = torch.zeros(
                (last_dim, last_dim),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        self.full_factors[module_name].addmm_(per_batch_grads.t(), per_batch_grads)

    def compute_diag_lambda(
        self, module_name: str, per_batch_grads: torch.Tensor
    ) -> None:
        """Compute the diagonal Fisher given the `module_name`.

        Args:
            module_name (str):
                Name of the module.
            per_batch_grads (torch.Tensor)
                Individual per batch gradients.
        """
        assert len(per_batch_grads.shape) == 3
        if module_name not in self.diag_factors:
            self.diag_factors[module_name] = torch.zeros(
                (per_batch_grads.shape[1], per_batch_grads.shape[2]),
                dtype=self.grads_dtype,
                device=per_batch_grads.device,
            )
        self.diag_factors[module_name].add_(torch.square(per_batch_grads).sum(dim=0))

    def fit_additional_factors(self, loader: torch.utils.data.DataLoader) -> None:
        """Fit additional factors (e.g., Lambda, full Fisher, and diagonal Fisher) given the loader.

        Args:
            loader (DataLoader):
                Dataloader in which additional factors are computed for.
        """

        print('Fitting additional factors')

        if self._additional_factors_done:
            self.logger.info(
                "Additional factors computation is already done. Skipping."
            )
            return

        t2 = time.time()

        def compute_loss(_params, _buffers, _batch):
            return self.task.get_train_loss(
                model=self.model,
                batch=_batch,
                parameter_and_buffer_dicts=(_params, _buffers),
                sample=self.use_true_fisher,
                reduction="sum",
            )

        ft_compute_grad = torch.func.grad(compute_loss, has_aux=False)

        self.model.eval()
        examples_seen = 0
        for _ in range(self.n_epoch):
            for batch in loader:
                examples_seen += self.task.get_batch_size(batch)
                grads_dict = torch.func.vmap(
                    ft_compute_grad,
                    in_dims=(None, None, 0),
                    randomness="different",
                )(self.func_params, self.func_buffers, batch)

                with torch.no_grad():
                    for name, module in zip(self.modules_name, self.modules):
                        if name in self.kronecker_modules_name:
                            # Compute the Lambda on K-FAC eigenbasis.
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_kronecker_lambda(name, per_batch_grads)
                        elif name in self.full_modules_name:
                            # Compute the full Fisher.
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_full_factors(name, per_batch_grads)
                        elif name in self.diag_modules_name:
                            # Compute the diagonal Fisher.
                            per_batch_grads = make_grads_dict_to_matrix(
                                module, name, grads_dict
                            ).to(dtype=self.grads_dtype)
                            self.compute_diag_lambda(name, per_batch_grads)
                        else:
                            raise InvalidModuleError()

        with torch.no_grad():
            # Normalize the statistics by the number of data points and set up the damping term.
            for name in self.kronecker_modules_name:
                self.kronecker_eigvals[name] /= examples_seen
                self.kronecker_eigvals[name] = self.kronecker_eigvals[name].to(
                    self.grads_dtype
                )
                if self.damping is None:
                    self.damping_factors[name] = 0.1 * torch.mean(
                        self.kronecker_eigvals[name]
                    )
                else:
                    self.damping_factors[name] = self.damping

            for name in self.full_modules_name:
                self.full_factors[name] /= examples_seen

                eigvals, eigvecs = torch.linalg.eigh(
                    self.full_factors[name].to(dtype=self.eig_dtype)
                )
                if self.damping is None:
                    self.damping_factors[name] = 0.1 * torch.mean(eigvals).to(
                        dtype=self.grads_dtype
                    )
                else:
                    self.damping_factors[name] = self.damping

                # Invert the full Fisher matrix and apply damping.
                inv_eigvals = eigvals + self.damping_factors[name]
                self.full_factors[name] = torch.matmul(
                    torch.matmul(eigvecs, torch.diag(inv_eigvals)), eigvecs.t()
                )
                self.full_factors[name] = self.full_factors[name].to(self.grads_dtype)

            for name in self.diag_modules_name:
                self.diag_factors[name] /= examples_seen
                self.diag_factors[name] = self.diag_factors[name].to(self.grads_dtype)
                if isinstance(self.damping, str) and self.damping == "heuristic":
                    self.damping_factors[name] = 0.1 * torch.mean(
                        self.diag_factors[name]
                    )
                else:
                    self.damping_factors[name] = self.damping

        self._additional_factors_done = True
        print("Time for computing Lambda:", time.time() - t2)

    def build_curvature_blocks(
        self, loader: torch.utils.data.DataLoader, keep_cache: bool = False
    ) -> None:
        """Perform EK-FAC computations.

        Args:
            loader (DataLoader):
                Dataloader in EK-FAC factors are computed for.
            keep_cache (bool, optional):
                If set to False, remove all forward and backward hooks after EK-FAC.
        """
        print('Building curvature blocks')

        start = time.time()
        self.fit_covariances(loader=loader)

        self.fit_eigendecompositions(keep_cache=keep_cache)
        for handle in self._handles:
            handle.remove()
        
        self._handles = []

        self.fit_additional_factors(loader=loader)

        end = time.time()
        print('Done building curvature blocks: {}s'.format(end - start))

    def precondition_grads(
        self,
        module_name: str,
        grads: torch.Tensor,
    ) -> torch.Tensor:
        """Given the `module_name` and `grads`, apply the preconditioning.

        Args:
            module_name (str):
                Name of the module in which gradients are computed on.
            grads (torch.Tensor):
                Reshaped gradients for the given module.
        """
        if module_name in self.kronecker_modules_name:
            grads_rot = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name].t(),
                torch.matmul(
                    grads.to(dtype=self.grads_dtype),
                    self.activation_cov_eigvecs[module_name],
                ),
            )

            precond_grads_rot = grads_rot / (
                self.kronecker_eigvals[module_name] + self.damping_factors[module_name]
            )

            precond_grads = torch.matmul(
                self.pseudograd_cov_eigvecs[module_name],
                torch.matmul(
                    precond_grads_rot,
                    self.activation_cov_eigvecs[module_name].t(),
                ),
            )

        elif module_name in self.full_modules_name:
            # Note that the Fisher is already inverted and damping is applied.
            precond_grads = torch.matmul(
                grads.to(dtype=self.grads_dtype), self.full_factors[module_name]
            )

        elif module_name in self.diag_modules_name:
            precond_grads = grads / (
                self.diag_factors[module_name] + self.damping_factors[module_name]
            )

        else:
            raise InvalidModuleError()

        return precond_grads

    def _get_grads_dict(
        self, 
        batch: Any, 
        use_measurement: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Given a batch, compute the individual gradient and reshape it into a 2D matrix."""

        grads_dict = torch.func.vmap(
            self._compute_measurement_grad()
            if use_measurement
            else self._compute_train_loss_grad(),
            in_dims=(None, None, 0),
            randomness="different",
        )(self.func_params, self.func_buffers, batch)

        return grads_dict

    def _get_precond_grads_dict(
        self,
        batch,
        use_measurement: bool = False,
        disable_precondition: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Given a batch, compute the individual gradient, reshape it into a 2D matrix, and apply preconditioning."""
        batch_size = self.task.get_batch_size(batch)
        grads_dict = self._get_grads_dict(batch=batch, use_measurement=use_measurement)

        with torch.no_grad():
            precond_grads_dict = {}
            for name, module in zip(self.modules_name, self.modules):
                grads = make_grads_dict_to_matrix(
                    module, name, grads_dict, remove_grads=True
                )
                if disable_precondition:
                    precond_grads_dict[name] = grads.reshape(batch_size, -1).to(
                        dtype=self.grads_dtype
                    )
                else:
                    precond_grads_dict[name] = (
                        self.precondition_grads(module_name=name, grads=grads)
                        .reshape(batch_size, -1)
                        .to(dtype=self.grads_dtype)
                    )
        del grads_dict
        return precond_grads_dict

    def compute_scores_with_batch(
        self, batch1: Any, batch2: Any, disable_precondition: bool = False
    ) -> torch.Tensor:
        """Compute pairwise influences scores between data points in `batch1` and `batch2`.

        Args:
            batch1 (object):
                The first set of data points from the data loader.
            batch2 (object):
                The second set of data points from the data loader.
            disable_precondition (bool, optional):
                If set to True, assume the Hessian to be identity.
        """
        self.model.eval()
        precond_grads_dict1 = self._get_precond_grads_dict(
            batch=batch1,
            use_measurement=False,
            disable_precondition=disable_precondition,
        )
        grads_dict2 = self._get_grads_dict(batch=batch2, use_measurement=False)

        with torch.no_grad():
            batch_size = self.task.get_batch_size(batch2)
            total_score = 0.0

            for name, module in zip(self.modules_name, self.modules):
                grads = (
                    make_grads_dict_to_matrix(module, name, grads_dict2)
                    .reshape(batch_size, -1)
                    .to(dtype=self.grads_dtype)
                )
                if isinstance(total_score, float):
                    total_score = torch.matmul(precond_grads_dict1[name], grads.t())
                else:
                    total_score.addmm_(precond_grads_dict1[name], grads.t())
                del precond_grads_dict1[name], grads
        return total_score

    def compute_scores_with_loader(
        self,
        test_loader: torch.utils.data.DataLoader,
        train_loader: torch.utils.data.DataLoader,
        disable_precondition: bool = False,
    ) -> torch.Tensor:
        """Compute pairwise influence scores between `test_loader` and `train_loader`.

        Args:
            test_loader (DataLoader):
                The loader with test dataset.
            train_loader (DataLoader):
                The loader with training dataset.
            disable_precondition (bool, optional):
                If set to True, assume the Hessian to be identity.
        """
        self.model.eval()

        score_table = torch.zeros(
            (len(test_loader.dataset), len(train_loader.dataset)),
            dtype=self.grads_dtype,
            device=self.task.device,
            requires_grad=False,
        )

        num_processed_test = 0

        for test_batch in tqdm(test_loader):
            start = time.time()
            test_batch_size = self.task.get_batch_size(test_batch)
            precond_test_grads_dict = self._get_precond_grads_dict(
                batch=test_batch,
                use_measurement=True,
                disable_precondition=disable_precondition,
            )

            num_processed_train = 0
            for train_batch in train_loader:
                train_batch_size = self.task.get_batch_size(train_batch)
                train_grads_dict = self._get_grads_dict(
                    batch=train_batch, use_measurement=False
                )

                with torch.no_grad():
                    for name, module in zip(self.modules_name, self.modules):
                        train_grads = (
                            make_grads_dict_to_matrix(module, name, train_grads_dict)
                            .reshape(train_batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )
                        score_table[
                            num_processed_test : num_processed_test + test_batch_size,
                            num_processed_train : num_processed_train
                            + train_batch_size,
                        ].addmm_(precond_test_grads_dict[name], train_grads.t())
                        del train_grads
                num_processed_train += train_batch_size
            del precond_test_grads_dict
            num_processed_test += test_batch_size
            end = time.time()
            print(
                f"Processed {num_processed_test} test data points (out of {len(test_loader.dataset)}). Elaspsed: {round(end-start, 2)}s"
            )

        return score_table

    def compute_self_scores_with_loader(
        self, loader: torch.utils.data.DataLoader, disable_precondition: bool = False
    ) -> torch.Tensor:
        """Compute self-influence scores of all data points in `loader`.

        Args:
            loader (DataLoader):
                The loader for which self-influence scores are computed.
            disable_precondition (bool, optional):
                If set to True, assume the Hessian to be identity.
        """
        self.model.eval()

        scores = []
        for batch in loader:
            batch_size = self.task.get_batch_size(batch)
            current_score = torch.zeros(
                (batch_size,),
                dtype=self.score_dtype,
                device=self.task.device,
                requires_grad=False,
            )
            grads_dict = self._get_grads_dict(batch=batch, use_measurement=False)

            with torch.no_grad():
                for name, module in zip(self.modules_name, self.modules):
                    grads = make_grads_dict_to_matrix(
                        module, name, grads_dict, remove_grads=True
                    )
                    if disable_precondition:
                        precond_grads = grads.reshape(batch_size, -1).to(
                            dtype=self.grads_dtype
                        )
                    else:
                        precond_grads = (
                            self.precondition_grads(module_name=name, grads=grads)
                            .reshape(batch_size, -1)
                            .to(dtype=self.grads_dtype)
                        )
                    grads = grads.reshape(batch_size, -1).to(dtype=self.grads_dtype)
                    current_score.add_(torch.sum(precond_grads * grads, dim=-1))

                del grads_dict
                scores.append(current_score)
        return torch.cat(scores)
