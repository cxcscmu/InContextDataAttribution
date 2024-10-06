from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.abstract_task import AbstractTask
import sys
import numpy as np

BATCH_DTYPE = Dict[str, torch.Tensor]

class LanguageModelTask(AbstractTask):
    def __init__(
        self, 
        device: torch.device = "cpu", 
        generator: Optional[torch.Generator] = None,
        layers: Optional[list] = None,
    ) -> None:
        super().__init__(device=device, generator=generator, layers=layers)

    def get_train_loss(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:

        if parameter_and_buffer_dicts is None:
            inputs = (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
            lm_logits = model(*inputs)
        else:
            params, buffers = parameter_and_buffer_dicts
            print(params)
            lm_logits = torch.func.functional_call(
                model,
                (params, buffers),
                args=(
                    batch["input_ids"].unsqueeze(0).to(self.device),
                    batch["attention_mask"].unsqueeze(0).to(self.device),
                ),
            )
            batch["labels"] = batch["labels"].unsqueeze(0).to(self.device)

        """
        inputs = (
                batch["input_ids"].to(self.device),
                batch["attention_mask"].to(self.device),
            )
        lm_logits = model(*inputs)
         """

        # [batch_size, seq_len, vocab_size]
        batch_size = lm_logits.shape[0]

         # [batch_size, seq_len-1, vocab_size]
        # last token in input is removed
        shift_logits = lm_logits[..., :-1, :].contiguous()

        if not sample:
            #print('NOT SAMPLE!')

            labels = batch["labels"].to(self.device)

            # move labels to right
            shift_labels = labels[..., 1:].contiguous()
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))

            # [(batch size x vocab size)]
            summed_loss = F.cross_entropy(
                reshaped_shift_logits, shift_labels.view(-1), reduction="sum"
            )
            #print(summed_loss)
        else:
            #print('SAMPLE!')
            reshaped_shift_logits = shift_logits.view(-1, shift_logits.size(-1))

            with torch.no_grad():
                 # [(seq_len - 1) x batch_size, vocab_size]
                probs = torch.nn.functional.softmax(reshaped_shift_logits, dim=-1)

                # [batch, (seq_len - 1)]
                # move labels to right
                labels = batch["labels"].to(self.device)
                shift_labels = labels[..., 1:].contiguous().view(-1)

                # Find tokens to ignore
                mask = torch.where(shift_labels == -100, True, False)

                # Sample labels according to probs
                # [batch, (seq_len - 1)] -> [batch x (seq_len - 1)]
                sampled_labels = torch.multinomial(
                    probs, num_samples=1, generator=self.generator
                ).flatten()

                #print(shift_labels[510:])
                #print(sampled_labels[510:])
                #print(mask[510:])

                # Mask the ignore tokens
                sampled_labels = torch.where(mask, shift_labels, sampled_labels)

            summed_loss = F.cross_entropy(
                reshaped_shift_logits, sampled_labels.detach(), reduction="sum"
            )
        
        if reduction == "sum":
            return summed_loss
        elif reduction == "mean":
            return summed_loss / batch_size
        else:
            raise NotImplementedError("Not supported reduction provided.")


    def get_measurement(
        self,
        model: nn.Module,
        batch: BATCH_DTYPE,
        parameter_and_buffer_dicts: Optional[Union[Dict[str, torch.Tensor]]] = None,
        sample: bool = False,
        reduction: str = "sum",
    ) -> torch.Tensor:
        # Alternatively, we can provide the conditional log-likelihood (given the prompt and completion).
        return self.get_train_loss(
            model, batch, parameter_and_buffer_dicts, sample, reduction
        )

    def get_batch_size(self, batch: BATCH_DTYPE) -> int:
        return batch["labels"].shape[0]

    def influence_modules(self) -> List[str]:
        total_modules = []

        if self.layers is None:
            #layer_idxs = list(range(16))
            layer_idxs = list(range(28))
            #layer_idxs = list(range(0, 32, 4))
        else:
            layer_idxs = self.layers

        
        """
        model.model.layers.27.self_attn.q_proj
        model.model.layers.27.self_attn.k_proj
        model.model.layers.27.self_attn.v_proj
        """

        # Add attention layers:
        for i in layer_idxs:
            #total_modules.append(f"model.gpt_neox.layers.{i}.attention.query_key_value")
            #total_modules.append(f"model.gpt_neox.layers.{i}.attention.dense")
            total_modules.append(f"model.model.layers.{i}.self_attn.o_proj")

        # Add MLP layers:``
        for i in layer_idxs:
            #total_modules.append(f"model.gpt_neox.layers.{i}.mlp.dense_h_to_4h")
            #total_modules.append(f"model.gpt_neox.layers.{i}.mlp.dense_4h_to_h")
            total_modules.append(f"model.model.layers.{i}.mlp.up_proj")
            total_modules.append(f"model.model.layers.{i}.mlp.down_proj")
        
        #total_modules = ['model.embed_out']
        #total_modules = ['model.lm_head']
        return total_modules

    def representation_module(self) -> str:
        return "model.model.norm"
        #return "model.gpt_neox.final_layer_norm"

    def get_activation_masks(self, batch: Any) -> Optional[torch.Tensor]:
        return batch["attention_mask"].unsqueeze(-1).to(self.device)
