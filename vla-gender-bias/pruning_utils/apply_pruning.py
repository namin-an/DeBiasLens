import re
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import Tensor
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.custom_nethook import get_module
from transformers.pytorch_utils import prune_linear_layer
from transformers.pytorch_utils import find_pruneable_heads_and_indices


def prune_linear(module: nn.Linear, keep_indices: Tensor, dim: int = 0) -> nn.Linear:
    device = next(iter(module.parameters())).device
    dtype = next(iter(module.parameters())).dtype

    new_module = prune_linear_layer(module, keep_indices.to(device=device), dim=dim)
    new_module = new_module.to(device=device, dtype=dtype)
    return new_module


def prune_mlp_llama(module: nn.Module, mask: np.ndarray) -> None:
    keep_indices, *_ = np.logical_not(mask).nonzero()
    keep_indices = torch.from_numpy(keep_indices).long()

    module.gate_proj = prune_linear(module.gate_proj, keep_indices, dim=0)
    module.up_proj = prune_linear(module.up_proj, keep_indices, dim=0)
    module.down_proj = prune_linear(module.down_proj, keep_indices, dim=1)

def prune_mlp_internvl(module: nn.Module, mask: np.ndarray) -> None:
    keep_indices, *_ = np.logical_not(mask).nonzero()
    keep_indices = torch.from_numpy(keep_indices).long()

    module.w1 = prune_linear(module.w1, keep_indices, dim=0)
    module.w3 = prune_linear(module.w3, keep_indices, dim=0)
    module.w2 = prune_linear(module.w2, keep_indices, dim=1)          


def prune_attention_llama(module: nn.Module, key_value_pruning_mask: np.ndarray, attention_pruning_mask: np.ndarray) -> None:
    num_key_value_groups = module.num_key_value_groups
    key_value_pruning_mask = key_value_pruning_mask.reshape(1, -1).repeat(num_key_value_groups, 1)
    key_value_pruning_mask = key_value_pruning_mask.reshape(-1)
    attention_pruning_mask = attention_pruning_mask.reshape(1, -1).repeat(num_key_value_groups, 1)
    attention_pruning_mask = attention_pruning_mask.reshape(-1)

    key_value_head_prune_indices, *_ = key_value_pruning_mask.nonzero()
    key_value_head_prune_indices = torch.from_numpy(key_value_head_prune_indices).long()
    attention_head_prune_indices, *_ = attention_pruning_mask.nonzero()
    attention_head_prune_indices = torch.from_numpy(attention_head_prune_indices).long()

    num_pruned_key_value_heads = len(key_value_head_prune_indices)
    num_pruned_attention_heads = len(attention_head_prune_indices)

    _, key_value_head_prune_indices = find_pruneable_heads_and_indices(
        key_value_head_prune_indices, module.num_key_value_heads, module.head_dim, set()
    )
    _, attention_head_prune_indices = find_pruneable_heads_and_indices(
        attention_head_prune_indices, module.num_heads, module.head_dim, set()
    )

    # module.q_proj = prune_linear(module.q_proj, attention_head_prune_indices, dim=0)
    # module.k_proj = prune_linear(module.k_proj, key_value_head_prune_indices, dim=0)
    # module.v_proj = prune_linear(module.v_proj, key_value_head_prune_indices, dim=0)
    # module.o_proj = prune_linear(module.o_proj, attention_head_prune_indices, dim=1)
    old_data = module.o_proj.weight.data.clone()
    module.o_proj.weight.data = torch.zeros_like(module.o_proj.weight.data)
    module.o_proj.weight.data[:, attention_head_prune_indices] = old_data[:, attention_head_prune_indices]

    # module.num_key_value_heads = module.num_key_value_heads - num_pruned_key_value_heads
    # module.num_heads = module.num_heads - num_pruned_attention_heads
    # module.num_key_value_groups = module.num_heads // module.num_key_value_heads


def prune_attention_internvl(
    module: nn.Module,
    key_value_pruning_mask: np.ndarray,
    attention_pruning_mask: np.ndarray,
) -> None:

    num_key_value_groups = module.num_key_value_groups
    key_value_pruning_mask = key_value_pruning_mask.reshape(1, -1).repeat(
        num_key_value_groups, 1
    )
    key_value_pruning_mask = key_value_pruning_mask.reshape(-1)
    attention_pruning_mask = attention_pruning_mask.reshape(1, -1).repeat(
        num_key_value_groups, 1
    )
    attention_pruning_mask = attention_pruning_mask.reshape(-1)

    wqkv_pruning_mask = np.concatenate(
        [attention_pruning_mask, key_value_pruning_mask, key_value_pruning_mask]
    )

    wqkv_pruning_indices, *_ = wqkv_pruning_mask.nonzero()
    wqkv_pruning_indices = torch.from_numpy(wqkv_pruning_indices).long()
    attention_head_prune_indices, *_ = attention_pruning_mask.nonzero()
    attention_head_prune_indices = torch.from_numpy(attention_head_prune_indices).long()

    num_pruned_key_value_heads = len(key_value_pruning_mask.nonzero()[0])
    num_pruned_attention_heads = len(attention_head_prune_indices)

    _, wqkv_pruning_indices = find_pruneable_heads_and_indices(
        wqkv_pruning_indices,
        module.num_heads + 2 * module.num_key_value_heads,
        module.head_dim,
        set(),
    )
    _, attention_head_prune_indices = find_pruneable_heads_and_indices(
        attention_head_prune_indices, module.num_heads, module.head_dim, set()
    )

    # module.wqkv = prune_linear(module.wqkv, wqkv_pruning_indices, dim=0)
    # module.wo = prune_linear(module.wo, attention_head_prune_indices, dim=1)
    old_data = module.wo.weight.data.clone()
    module.wo.weight.data = torch.zeros_like(module.wo.weight.data)
    module.wo.weight.data[:, attention_head_prune_indices] = old_data[:, attention_head_prune_indices]

    # module.num_key_value_heads = module.num_key_value_heads - num_pruned_key_value_heads
    # module.num_heads = module.num_heads - num_pruned_attention_heads
    # module.num_key_value_groups = module.num_heads // module.num_key_value_heads
    # module.hidden_size = module.wo.in_features


PRUNING_FUNCTIONS = {
    "llama": (prune_mlp_llama, prune_attention_llama),
    "internvl": (prune_mlp_internvl, prune_attention_internvl),
}


def apply_pruning(
    model: nn.Module, sparsity: float, granularity: str, importance_scores: dict[str, np.ndarray], model_type: str = "llama",
) -> None:
    assert model_type in PRUNING_FUNCTIONS, f"Invalid model type: {model_type}"

    if model_type == "llama":
        mlp_module_name = "mlp"
        attention_module_name = "self_attn"
    elif model_type == "internvl":
        mlp_module_name = "feed_forward"
        attention_module_name = "attention"
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Pool importance scores based on granularity
    print("Pooling importance scores...")
    if granularity == "global":
        all_importance_scores = np.concatenate(list(importance_scores.values()))
        # Drop nan values
        all_importance_scores = all_importance_scores[~np.isnan(all_importance_scores)]
        cutoff = np.quantile(all_importance_scores, q=sparsity)
        module_to_cutoff = {module_name: cutoff for module_name in importance_scores}
    elif granularity == "type":
        module_to_cutoff = dict()
        # Pool importance scores based on module type
        # 1) MLPs
        mlp_scores = [
            module_scores
            for module_name, module_scores in importance_scores.items()
            if module_name.endswith(mlp_module_name)
        ]
        # 2) Self-Attention
        key_value_head_scores = []
        attention_head_scores = []
        for module_name, module_scores in importance_scores.items():
            if module_name.endswith(attention_module_name):
                key_value_head_scores.append(module_scores[0])
                attention_head_scores.append(module_scores[1])

        # Calculate cutoff for MLPs
        mlp_scores = np.concatenate(mlp_scores)
        mlp_scores = mlp_scores[~np.isnan(mlp_scores)]
        mlp_cutoff = np.quantile(mlp_scores, q=sparsity)

        # Calculate cutoff for Self-Attention
        key_value_head_scores = np.concatenate(key_value_head_scores)
        attention_head_scores = np.concatenate(attention_head_scores)
        key_value_head_scores = key_value_head_scores[~np.isnan(key_value_head_scores)]
        attention_head_scores = attention_head_scores[~np.isnan(attention_head_scores)]
        key_value_head_cutoff = np.quantile(key_value_head_scores, q=sparsity)
        attention_head_cutoff = np.quantile(attention_head_scores, q=sparsity)

        # Assign cutoffs to modules
        for module_name in importance_scores.keys():
            if module_name.endswith(mlp_module_name):
                module_to_cutoff[module_name] = mlp_cutoff
            elif module_name.endswith(attention_module_name):
                module_to_cutoff[module_name] = (key_value_head_cutoff, attention_head_cutoff)
            else:
                raise ValueError(f"Unknown module type: {module_name}")
    elif granularity == "local":
        module_to_cutoff = dict()
        for module_name, module_scores in importance_scores.items():
            if module_name.endswith(mlp_module_name):
                module_scores = module_scores[~np.isnan(module_scores)]
                cutoff = np.quantile(module_scores, q=sparsity)
                module_to_cutoff[module_name] = cutoff
            elif module_name.endswith(attention_module_name):
                key_value_head_scores = module_scores[0]
                attention_head_scores = np.array(module_scores[1])
                key_value_head_scores = key_value_head_scores[~np.isnan(key_value_head_scores)]
                #print(attention_head_scores)
                #print(np.isnan(attention_head_scores))
                attention_head_scores = attention_head_scores[~np.isnan(attention_head_scores)]
                key_value_head_cutoff = np.quantile(key_value_head_scores, q=sparsity)
                attention_head_cutoff = np.quantile(attention_head_scores, q=sparsity)
                module_to_cutoff[module_name] = (key_value_head_cutoff, attention_head_cutoff)
            else:
                raise ValueError(f"Unknown module type: {module_name}")

    else:
        raise ValueError(f"Invalid granularity: {granularity}")

    # Apply pruning
    print("Applying pruning...")
    prune_mlp, prune_attention = PRUNING_FUNCTIONS[model_type]
    module_to_cutoff_iterator = tqdm(
        module_to_cutoff.items(), desc="Pruning", total=len(module_to_cutoff)
    )
    for module, cutoff in module_to_cutoff_iterator:
        # Get layer index
        layer_index = re.findall(r"\d+", module).pop()
        if int(layer_index) <= 1:
            continue

        if module.endswith(attention_module_name):
            key_value_head_cutoff, attention_head_cutoff = cutoff
            key_value_head_importances, attention_head_importances = importance_scores[
                module
            ]
            key_value_pruning_mask = key_value_head_importances < key_value_head_cutoff
            attention_pruning_mask = attention_head_importances < attention_head_cutoff
            prune_attention(
                get_module(model.model, module),
                key_value_pruning_mask,
                attention_pruning_mask,
            )

        elif module.endswith(mlp_module_name):
            mlp_importances = importance_scores[module]
            pruning_mask = mlp_importances < cutoff
            prune_mlp(get_module(model.model, module), pruning_mask)
        else:
            raise ValueError(f"Unknown module type: {module}")
