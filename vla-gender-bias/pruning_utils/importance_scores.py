import re
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm
from torch import Tensor
from collections import defaultdict
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.custom_nethook import get_module


def get_attn_head_importance_llama(
    model: nn.Module,
    name: str,
    gradients: dict[str, Tensor],
    aggregation_method: str,
    o_proj_name: str = "o_proj",
) -> tuple[np.ndarray, np.ndarray]:
    # Perform the following checks to ensure that the input arguments are valid
    assert name.endswith("self_attn")
    assert aggregation_method in ["sum", "max", "last"]
    assert o_proj_name in [
        "o_proj",
        "out_proj",
    ], "o_proj_name must be either 'o_proj' or 'out_proj'"

    # Get the module corresponding to the given name
    module = get_module(model, name)
    # Get the submodules that are wrapped by the given module
    # (i.e. linear layers within the Self-Attention module)
    submodules = [
        module_name for module_name in gradients if module_name.startswith(name)
    ]
    submodules = {
        submodule.split(".")[-2]: submodule for submodule in submodules
    }

    # Calculate the importance of the key and value projection heads
    head_dim = module.head_dim
    num_key_value_groups = module.num_key_value_groups

    k_proj = submodules["k_proj"]
    v_proj = submodules["v_proj"]
    k_proj_head_importances = torch.split(gradients[k_proj], head_dim * num_key_value_groups, dim=0)
    k_proj_head_importances = [torch.sum(k_proj_head_importance).item() for k_proj_head_importance in k_proj_head_importances]
    v_proj_head_importances = torch.split(gradients[v_proj], head_dim * num_key_value_groups, dim=0)
    v_proj_head_importances = [torch.sum(v_proj_head_importance).item() for v_proj_head_importance in v_proj_head_importances]

    if aggregation_method == "sum":
        key_value_head_importances = np.sum(np.array([k_proj_head_importances, v_proj_head_importances]), axis=0)
    elif aggregation_method == "max":
        key_value_head_importances = np.maximum(np.array(k_proj_head_importances), np.array(v_proj_head_importances))
    elif aggregation_method == "last":
        key_value_head_importances = np.array(v_proj_head_importances)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    # Calculate the importance of attention heads
    q_proj = submodules["q_proj"]
    o_proj = submodules["o_proj"]
    q_proj_head_importances = torch.split(gradients[q_proj], head_dim * num_key_value_groups, dim=0)
    q_proj_head_importances = [torch.sum(q_proj_head_importance).item() for q_proj_head_importance in q_proj_head_importances]
    o_proj_head_importances = torch.split(gradients[o_proj], head_dim * num_key_value_groups, dim=1)
    o_proj_head_importances = [torch.sum(o_proj_head_importance).item() for o_proj_head_importance in o_proj_head_importances]

    if aggregation_method == "sum":
        attention_head_importances = np.sum(np.array([q_proj_head_importances, o_proj_head_importances]), axis=0)
    elif aggregation_method == "max":
        attention_head_importances = np.maximum(np.array(q_proj_head_importances), np.array(o_proj_head_importances))
    elif aggregation_method == "last":
        attention_head_importances = np.array(o_proj_head_importances)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    return key_value_head_importances, attention_head_importances


def get_attn_head_importance_internvl(
    model: nn.Module,
    name: str,
    gradients: dict[str, Tensor],
    aggregation_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    # Perform the following checks to ensure that the input arguments are valid
    assert name.endswith("attention")
    assert aggregation_method in ["sum", "max", "last"]

    # Get the module corresponding to the given name
    module = get_module(model, name)
    # Get the submodules that are wrapped by the given module
    # (i.e. linear layers within the Self-Attention module)
    submodules = [
        module_name for module_name in gradients if module_name.startswith(name)
    ]
    submodules = {submodule.split(".")[-2]: submodule for submodule in submodules}

    wqkv_gradients = gradients[submodules["wqkv"]]
    wo_gradients = gradients[submodules["wo"]]

    head_dim = module.head_dim
    num_heads = module.num_heads
    num_key_value_heads = module.num_key_value_heads

    q_gradients = wqkv_gradients[:head_dim * num_heads]
    k_gradients = wqkv_gradients[head_dim * num_heads: head_dim * num_heads + num_key_value_heads * head_dim]
    v_gradients = wqkv_gradients[head_dim * num_heads + num_key_value_heads * head_dim:]

    # Calculate the importance of the key and value projection heads
    num_key_value_groups = module.num_key_value_groups

    k_proj_head_importances = torch.split(
        k_gradients, head_dim * num_key_value_groups, dim=0
    )
    k_proj_head_importances = [
        torch.sum(k_proj_head_importance).item()
        for k_proj_head_importance in k_proj_head_importances
    ]
    v_proj_head_importances = torch.split(
        v_gradients, head_dim * num_key_value_groups, dim=0
    )
    v_proj_head_importances = [
        torch.sum(v_proj_head_importance).item()
        for v_proj_head_importance in v_proj_head_importances
    ]

    if aggregation_method == "sum":
        key_value_head_importances = np.sum(
            np.array([k_proj_head_importances, v_proj_head_importances]), axis=0
        )
    elif aggregation_method == "max":
        key_value_head_importances = np.maximum(
            np.array(k_proj_head_importances), np.array(v_proj_head_importances)
        )
    elif aggregation_method == "last":
        key_value_head_importances = np.array(v_proj_head_importances)
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    # Calculate the importance of attention heads
    q_proj_head_importances = torch.split(
        q_gradients, head_dim * num_key_value_groups, dim=0
    )
    q_proj_head_importances = [
        torch.sum(q_proj_head_importance).item()
        for q_proj_head_importance in q_proj_head_importances
    ]
    o_proj_head_importances = torch.split(
        wo_gradients, head_dim * num_key_value_groups, dim=1
    )
    o_proj_head_importances = [
        torch.sum(o_proj_head_importance).item()
        for o_proj_head_importance in o_proj_head_importances
    ]

    attention_head_importances = o_proj_head_importances
    return key_value_head_importances, attention_head_importances


def get_mlp_importance_internvl(
    model: nn.Module,
    name: str,
    gradient: dict[str, Tensor],
    aggregation_method: str,
) -> np.ndarray:
    # Perform the following checks to ensure that the input arguments are valid
    assert name.endswith("feed_forward")
    assert aggregation_method in ["sum", "max", "last"]

    # Get the submodules that are wrapped by the given module
    submodules = [
        module_name for module_name in gradient
        if module_name.startswith(name) and "weight" in module_name
    ]
    submodules = {
        submodule.replace(".weight", "").split(".")[-1]: submodule
        for submodule in submodules
    }

    w1_gradients = torch.split(gradient[submodules["w1"]], 1, dim=0)
    w2_gradients = torch.split(gradient[submodules["w2"]], 1, dim=1)
    w3_gradients = torch.split(gradient[submodules["w3"]], 1, dim=0)

    w1_gradients = [torch.sum(channel_gradient).item() for channel_gradient in w1_gradients]
    w2_gradients = [torch.sum(channel_gradient).item() for channel_gradient in w2_gradients]
    w3_gradients = [torch.sum(channel_gradient).item() for channel_gradient in w3_gradients]

    w1_gradients = np.array(w1_gradients)
    w2_gradients = np.array(w2_gradients)
    w3_gradients = np.array(w3_gradients)

    if aggregation_method == "sum":
        mlp_scores = w1_gradients + w2_gradients + w3_gradients
    elif aggregation_method == "max":
        mlp_scores = np.maximum(w1_gradients, w2_gradients, w3_gradients)
    elif aggregation_method == "last":
        mlp_scores = w2_gradients
    else:
        raise ValueError(f"Invalid aggregation method: {aggregation_method}")
    
    return mlp_scores


def get_mlp_importance_llama(
    model: nn.Module,
    name: str,
    gradient: dict[str, Tensor],
    aggregation_method: str,
    down_proj_name: str = "down_proj",
) -> np.ndarray:
    # Perform the following checks to ensure that the input arguments are valid
    assert name.endswith("mlp")
    assert aggregation_method in ["sum", "max", "last"]
    assert down_proj_name in [
        "down_proj",
        "fc2",
    ], "down_proj_name must be either 'down_proj' or 'fc2'"

    # Get the submodules that are wrapped by the given module
    submodules = [
        module_name for module_name in gradient if module_name.startswith(name)
    ]

    # Initialize a dictionary to store the gradients of the submodules
    mlp_gradients = defaultdict(dict)

    # Iterate over the submodules and store the gradients of the submodules
    for submodule in submodules:
        # Ignore the bias gradients
        if "weight" not in submodule:
            continue

        submodule_name = submodule.replace(".weight", "").split(".")[-1]
        submodule_gradient = gradient[submodule]
        # We must remember that the gradient in the down proj layer
        # is transposed compared to the up proj layer
        dim = 1 if submodule_name == down_proj_name else 0
        channel_gradients = torch.split(submodule_gradient, 1, dim=dim)
        channel_gradients = [
            torch.sum(channel_gradient).item() for channel_gradient in channel_gradients
        ]
        channel_gradients = np.array(channel_gradients)
        channel_gradients[np.isnan(channel_gradients)] = 0
        for channel_index, channel_gradient in enumerate(channel_gradients):
            mlp_gradients[channel_index][submodule_name] = channel_gradient

    # Aggregate the gradients of each channel in the MLP
    mlp_scores = np.zeros(len(mlp_gradients))
    for channel_index, channel_gradients in mlp_gradients.items():
        if aggregation_method == "sum":
            mlp_scores[channel_index] = sum(channel_gradients.values())
        elif aggregation_method == "max":
            mlp_scores[channel_index] = max(channel_gradients.values())
        elif aggregation_method == "last":
            mlp_scores[channel_index] = channel_gradients[down_proj_name]
        else:
            raise ValueError(f"Invalid aggregation method: {aggregation_method}")

    return mlp_scores


def get_importance_scores(
    model: nn.Module,
    gradients: dict[str, Tensor],
    aggregation_method: str,
    model_type: str = "llama",
) -> dict[str, np.ndarray]:
    # Perform the following checks to ensure that the input arguments are valid
    assert aggregation_method in ["sum", "max", "last"]
    assert model_type in [
        "llama",
        "internvl",
    ], "model_type must be either 'llama' or 'internvl'"

    if model_type == "internvl":
        attention_module_name = "attention"
        mlp_module_name = "feed_forward"
        get_mlp_importance = get_mlp_importance_internvl
        get_attn_head_importance = get_attn_head_importance_internvl
    else:
        attention_module_name = "self_attn"
        mlp_module_name = "mlp"
        get_mlp_importance = get_mlp_importance_llama
        get_attn_head_importance = get_attn_head_importance_llama

    # Collect all mlp modules (modules that end with "mlp")
    all_mlp_modules = set()
    for parameter_name in gradients.keys():
        all_mlp_modules.update(set(re.findall(r".*\." + mlp_module_name, parameter_name)))
    # Collect all self-attention modules (modules that end with "self_attn")
    all_attn_modules = set()
    for parameter_name in gradients.keys():
        all_attn_modules.update(
            set(re.findall(r".*\." + attention_module_name, parameter_name))
        )

    # Get importance scores for each module
    all_importance_scores = dict()
    progress = tqdm(
        total=len(all_mlp_modules) + len(all_attn_modules),
        desc="Calculating Importance Scores",
    )
    for mlp_module in all_mlp_modules:
        mlp_scores = get_mlp_importance(
            model.model,
            mlp_module,
            gradients,
            aggregation_method,
        )
        all_importance_scores[mlp_module] = mlp_scores
        progress.update(1)
    for attn_module in all_attn_modules:
        attn_scores = get_attn_head_importance(
            model.model,
            attn_module,
            gradients,
            aggregation_method,
        )
        all_importance_scores[attn_module] = attn_scores
        progress.update(1)

    progress.close()
    return all_importance_scores
