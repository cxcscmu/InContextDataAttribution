from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class InvalidModuleError(Exception):
    # Raised when the provided module is invalid.
    pass


def extract_patches(
    inputs: torch.Tensor,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: Tuple[int, int],
) -> torch.Tensor:
    """Extract patches for the KFC approximation.

    This method is based on the technique described in https://arxiv.org/pdf/1602.01407.pdf.

    Args:
        inputs (torch.Tensor):
            Activations before the convolutional layer.
        kernel_size (tuple):
            Dimensions of the convolutional layer's kernel.
        stride (tuple):
            Stride applied in the convolutional layer.
        padding (tuple):
            Padding dimensions applied in the convolutional layer.
    """
    if padding[0] + padding[1] > 0:
        inputs = torch.nn.functional.pad(
            inputs,
            (padding[1], padding[1], padding[0], padding[0]),
        ).data
    inputs = inputs.unfold(2, kernel_size[0], stride[0])
    inputs = inputs.unfold(3, kernel_size[1], stride[1])
    inputs = inputs.transpose_(1, 2).transpose_(2, 3).contiguous()
    inputs = inputs.view(
        inputs.size(0),
        inputs.size(1),
        inputs.size(2),
        inputs.size(3) * inputs.size(4) * inputs.size(5),
    )
    return inputs


def make_grads_dict_to_matrix(
    module: nn.Module,
    module_name: str,
    grads_dict: Dict[str, torch.Tensor],
    remove_grads: bool = True,
) -> torch.Tensor:
    """Extracts and reshapes the homogeneous matrix of gradients for the specified `module`
    from the provided dictionary of batched gradients.

    The provided module must be an instance of one of the following: `Linear`, `Conv`, `Embedding`,
     `LayerNorm`, or `BatchNorm2d`.

    Args:
        module (nn.Module):
            The module for which the matrix will be reshaped.
        module_name (str):
            The name of the module, specific to the architecture it belongs to.
        grads_dict (dict):
            A dictionary that maps parameter names to their corresponding gradients.
        remove_grads (bool):
            If set to True, remove the reference to the original gradients. Defaults to True to
            reduce memory overheads.
    """
    if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
        grads_mat = grads_dict[module_name + ".weight"]
        if remove_grads:
            del grads_dict[module_name + ".weight"]
        if module_name + ".bias" in grads_dict:
            grads_mat = torch.cat(
                (grads_mat, grads_dict[module_name + ".bias"].unsqueeze(-1)), -1
            )
            if remove_grads:
                del grads_dict[module_name + ".bias"]
    elif isinstance(module, nn.Conv2d):
        grads_mat = grads_dict[module_name + ".weight"]
        grads_mat = grads_mat.view(grads_mat.size(0), grads_mat.size(1), -1)
        if remove_grads:
            del grads_dict[module_name + ".weight"]
        if module_name + ".bias" in grads_dict:
            grads_mat = torch.cat(
                [grads_mat, grads_dict[module_name + ".bias"].unsqueeze(-1)], -1
            )
            if remove_grads:
                del grads_dict[module_name + ".bias"]
    elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.LayerNorm):
        # Concatenate weights and bias.
        grads_mat = torch.cat(
            (grads_dict[module_name + ".weight"], grads_dict[module_name + ".bias"]), -1
        )
        if remove_grads:
            del grads_dict[module_name + ".weight"], grads_dict[module_name + ".bias"]
    else:
        raise InvalidModuleError()
    return grads_mat


def extract_activations(
    activations: torch.Tensor,
    module: nn.Module,
    activations_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract and reshape activations into valid shapes for covariance computations.

    Args:
        activations (torch.Tensor):
            Raw pre-activations supplied to the module.
        module (nn.Module):
            The module where the activations are applied.
        activations_mask (torch.Tensor, optional):
             If padding with dummy inputs is applied to the batch, provide the same mask.
    """
    if isinstance(module, nn.Linear):
        if (
            activations_mask is not None
            and activations_mask.shape[:-1] == activations.shape[:-1]
        ):
            activations *= activations_mask
        reshaped_activations = activations.reshape(-1, activations.shape[-1])
        if module.bias is not None:
            shape = list(reshaped_activations.shape[:-1]) + [1]
            append_term = reshaped_activations.new_ones(shape)
            if (
                activations_mask is not None
                and activations_mask.shape[:-1] == activations.shape[:-1]
            ):
                append_term *= activations_mask.view(-1, 1)
            reshaped_activations = torch.cat(
                [reshaped_activations, append_term], dim=-1
            )
    elif isinstance(module, nn.Conv2d):
        del activations_mask
        reshaped_activations = extract_patches(
            activations, module.kernel_size, module.stride, module.padding
        )
        reshaped_activations = reshaped_activations.view(
            -1, reshaped_activations.size(-1)
        )
        if module.bias is not None:
            shape = list(reshaped_activations.shape[:-1]) + [1]
            reshaped_activations = torch.cat(
                [reshaped_activations, reshaped_activations.new_ones(shape)], dim=-1
            )
    else:
        raise InvalidModuleError()
    return reshaped_activations


def extract_gradients(gradients: torch.Tensor, module: nn.Module) -> torch.Tensor:
    """Extract and reshape gradients into valid shapes for covariance computations.

    Args:
        gradients (torch.Tensor):
            Raw gradients on the output to the module.
        module (nn.Module):
            The module where the gradients are computed.
    """
    if isinstance(module, nn.Linear):
        del module
        reshaped_grads = gradients.reshape(-1, gradients.shape[-1])
        return reshaped_grads
    elif isinstance(module, nn.Conv2d):
        del module
        reshaped_grads = gradients.permute(0, 2, 3, 1)
        reshaped_grads = reshaped_grads.reshape(-1, reshaped_grads.size(-1))
    else:
        raise InvalidModuleError()
    return reshaped_grads
