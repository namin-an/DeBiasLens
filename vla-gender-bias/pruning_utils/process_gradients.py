import os
import torch

from tqdm import tqdm
from typing import Any
from torch import Tensor
from collections import defaultdict


METRICS = ["bias", "mme",]
TASKS = ["sentiment", "skills", "occupations"]


def defaultdict_to_dict(dfltdict: Any) -> dict:
    # Recursively convert defaultdict to dict
    if isinstance(dfltdict, defaultdict):
        dfltdict = {k: defaultdict_to_dict(v) for k, v in dfltdict.items()}
    return dfltdict


def load_gradients(
    model: str,
    only_gradients: bool,
    bias_num_images: int,
    # vision_gradients: bool,
    # normalize_gradients: bool,
    # bias_task: str,
    # bias_sample: int,
    path_to_gradients: str = "./results/gradients/",
) -> dict[str, dict[str, Tensor]]:
    # Initialize a dictionary to store the gradients of the model
    all_gradient_dicts = defaultdict(dict)
    # Determine whether to load pure gradients or gradients times weights
    gradient_file_name = (
        "gradients.pt" if only_gradients else "gradients_times_weight.pt"
    )

    # Load MME gradients
    mme_gradients_path = os.path.join(path_to_gradients, model, "mme", gradient_file_name)
    mme_gradients = torch.load(mme_gradients_path, map_location="cpu")
    for parameter_name, gradient in mme_gradients.items():
        all_gradient_dicts[parameter_name]["mme"] = gradient
    
    # Load gradients for all tasks
    for task in TASKS:
        task_gradients_path = os.path.join(path_to_gradients, model, "bias", task, str(bias_num_images), gradient_file_name)
        print(task, task_gradients_path)
        if not os.path.exists(task_gradients_path):
            continue

        task_gradients = torch.load(task_gradients_path, map_location="cpu")
        for parameter_name, gradient in task_gradients.items():
            all_gradient_dicts[parameter_name][f"bias-{task}"] = gradient

    # Convert defaultdict to dict
    all_gradient_dicts = defaultdict_to_dict(all_gradient_dicts)
    return all_gradient_dicts


def merge_gradients(gradients: dict[str, dict[str, Tensor]]) -> dict[str, Tensor]:
    merged_gradients = {}
    gradient_iterator = tqdm(
        gradients.items(), desc="Merging Gradients", total=len(gradients)
    )
    print(gradients.keys())
    for parameter_name, parameter_gradients in gradient_iterator:
        mme_gradient = torch.abs(
            parameter_gradients["mme"].to(
                device="cpu",
                dtype=torch.float32,
            )
        )

        bias_gradient = None
        for task in TASKS:
            if f"bias-{task}" not in parameter_gradients:
                continue
            
            task_gradient = torch.abs(
                parameter_gradients[f"bias-{task}"].to(
                    device="cpu",
                    dtype=torch.float32,
                )
            )
            if bias_gradient is not None:
                bias_gradient += task_gradient
            else:
                bias_gradient = task_gradient
        
        bias_gradient = bias_gradient / len(TASKS)

        combined_gradient = mme_gradient - bias_gradient
        combined_gradient = combined_gradient.to(device="cpu", dtype=torch.float32)
        merged_gradients[parameter_name] = combined_gradient

    return merged_gradients
