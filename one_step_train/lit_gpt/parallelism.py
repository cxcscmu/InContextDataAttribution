import sys
import torch

from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from torch.distributed._composable.fsdp.fully_shard import fully_shard
from torch.distributed._tensor import Replicate, Shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)

from lit_gpt.model import GPT

# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def parallelize(model: GPT, device_mesh: DeviceMesh) -> GPT:
    """Apply parallelisms and activation checkpointing to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    """

    dp_mesh = device_mesh["data_parallel"]
    tp_mesh = device_mesh["tensor_parallel"]

    if tp_mesh.size() > 1:
        # 1. Parallelize the first embedding and the last linear proj layer
        # 2. Parallelize the root norm layer over the sequence dim
        # 3. Shard the first transformer block's inputs

        """
        for name, transformer_block in model.transformer.items():
            print(name) # wte, h, ln_f
        """

        # Parallelize the first embedding and the last linear out projection
        plan = {
            "transformer.wte": RowwiseParallel(input_layouts=Replicate()),
            "lm_head": ColwiseParallel(
                input_layouts=Shard(1),
                # Optional: Shard the output along the class dimension to compute the loss in parallel.
                # See `loss_parallel` in `train.py`
                output_layouts=Shard(-1),
                use_local_output=False,
            ),
            "transformers.ln_f": SequenceParallel(),
            "transformers.h.0": PrepareModuleInput(
                input_layouts=(Replicate(), None),
                desired_input_layouts=(Shard(1), None),
                use_local_output=True,
            ),
        }
        model = parallelize_module(model, tp_mesh, plan)

        # Parallelize each transformer block
        for block in model.transformer.h:
            print(block)
            # Adjust attention module to use the local number of heads
            n_heads = 8
            n_query_groups = 8
            
            block.attn.config.n_head = n_heads // tp_mesh.size()
            block.attn.config.n_query_groups = n_query_groups // tp_mesh.size()

            plan = {
                "norm_1": SequenceParallel(use_local_output=True),
                #"attn.attn": ColwiseParallel(),
                #"attn.proj": ColwiseParallel(),
                "norm_2": SequenceParallel(use_local_output=True),
                "mlp.fc": ColwiseParallel(),
                "mlp.proj": ColwiseParallel(use_local_output=True),
            }

            """
            plan = {
                "norm_1": SequenceParallel(),
                "attn": PrepareModuleInput(
                    input_layouts=(Shard(1), None),
                    desired_input_layouts=(Replicate(), None),
                ),
                "attn.attn": ColwiseParallel(),
                "attn.proj": ColwiseParallel(output_layouts=Shard(1)),
                "norm_2": SequenceParallel(),
                "mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
                "mlp.fc": ColwiseParallel(),
                "mlp.proj": ColwiseParallel(),
            }
            """

            # Apply the plan for the current transformer block
            parallelize_module(block, tp_mesh, plan)

    """
    if dp_mesh.size() > 1:
        assert dp_mesh.ndim == 1  # Hybrid-sharding not supported

        # NOTE: Currently, the user is required to manually handle precision settings such as the `mp_policy` here
        # because the model parallel strategy does not respect all settings of `Fabric(precision=...)` at the moment.
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

        fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
        for layer_id, transformer_block in model.transformer.h.items():
            # Apply activation checkpointing
            transformer_block = checkpoint_wrapper(transformer_block)
            # As an optimization, do not reshard after forward for the last
            # transformer block since FSDP would prefetch it immediately
            reshard_after_forward = int(layer_id) < len(model.transformer.h) - 1
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
            model.transformer.h[layer_id] = transformer_block
        model = fully_shard(model, **fsdp_config)
    """
    
    return model
