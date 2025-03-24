# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Apply monkey-patch function to models
"""
import sys
from typing import Optional

import torch
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from verl.utils.ulysses import gather_heads_scatter_seq, gather_seq_scatter_heads, get_ulysses_sequence_parallel_world_size, get_ulysses_sequence_parallel_group


def _ulysses_flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    position_ids: Optional[torch.Tensor] = None,
    **kwargs,
):
    """Insert all-to-all before and after flash attention.
    DeepSpeed-Ulysses: https://arxiv.org/pdf/2309.14509

    Args:
        query_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads, head_dim)
        key_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        value_states (torch.Tensor): (batch_size, seqlen/sp_size, nheads_k, head_dim)
        position_ids (torch.Tensor, optional): (batch_size, seqlen/sp_size)

    Returns:
        torch.Tensor: (batch_size, seqlen/sp_size, nheads, head_dim)
    """
    ulysses_sp_size = get_ulysses_sequence_parallel_world_size()

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        assert position_ids is not None, "position_ids is required for Ulysses sequence parallelism"
        # (bsz, seq_len/n, n_head, head_dim) -> (bsz, seq_len, n_head/n, head_dim)
        query_states = gather_seq_scatter_heads(query_states, seq_dim=1, head_dim=2)
        key_states = gather_seq_scatter_heads(key_states, seq_dim=1, head_dim=2)
        value_states = gather_seq_scatter_heads(value_states, seq_dim=1, head_dim=2)

        # (bsz, seq_len/n) -> (bsz, seq_len)
        position_ids_list = [torch.empty_like(position_ids) for _ in range(ulysses_sp_size)]
        torch.distributed.all_gather(position_ids_list, position_ids, group=get_ulysses_sequence_parallel_group())
        position_ids = torch.concat(position_ids_list, dim=-1)

    # (bsz, seq_len, n_head/n, head_dim)
    attn_output = _flash_attention_forward(query_states,
                                           key_states,
                                           value_states,
                                           *args,
                                           position_ids=position_ids,
                                           **kwargs)

    ########## AlltoAll for Ulysses ##########
    if ulysses_sp_size > 1:
        # (bsz, seq_len, n_head/n, head_dim) -> (bsz, seq_len/n, n_head, head_dim)
        attn_output = gather_heads_scatter_seq(attn_output, seq_dim=1, head_dim=2)

    return attn_output


def apply_monkey_patch(model: PreTrainedModel):
    """Replace _flash_attention_forward to _ulysses_flash_attention_forward"""
    module = sys.modules[model.__module__]

    # TODO: VLM models only, unify monkey patch to LLM models.
    if model.config.model_type in ("qwen2_vl", "qwen2_5_vl"):  # patch remove padding for qwen2vl mrope
        from verl.models.transformers.qwen2_vl import ulysses_flash_attn_forward
        from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2

        Qwen2VLFlashAttention2.forward = ulysses_flash_attn_forward
        Qwen2_5_VLFlashAttention2.forward = ulysses_flash_attn_forward
        print("Monkey patch FlashAttention2.forward in Qwen2VL")
        return

    # transformers<=4.47.1
    if hasattr(module, "_flash_attention_forward"):
        module._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {model.__module__}")
    else:
        # transformers>=4.48.0
        from transformers.integrations import flash_attention
        flash_attention._flash_attention_forward = _ulysses_flash_attention_forward
        print(f"Monkey patch _flash_attention_forward in {flash_attention.__name__}")


from functools import lru_cache
from packaging import version
import importlib.metadata


@lru_cache()
def is_transformers_version_in_range(min_version: str, max_version: str) -> bool:
    try:
        # Get the installed version of the transformers library
        transformers_version = importlib.metadata.version("transformers")
    except importlib.metadata.PackageNotFoundError:
        raise ModuleNotFoundError("The `transformers` package is not installed.")

    # Check if the version is within the specified range
    return version.parse(min_version) <= version.parse(transformers_version) <= version.parse(max_version)
