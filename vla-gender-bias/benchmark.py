import os
import re
import json
import torch
import warnings
import argparse
import pandas as pd

from tqdm import tqdm
from typing import Optional
from vlms import load_model
from bias_eval_utils import Prompt
from bias_eval_utils import BiasPrompt
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from utils_new.benchmark_utils import make_dataloader, encode_option_letter

# Silence warnings
warnings.filterwarnings("ignore")


def make_prompts(prompt_chunk_index: int, base_dir: str) -> list[BiasPrompt]:
    # Get prompts
    with open(os.path.join(base_dir, f"prompts_{prompt_chunk_index}.json"), "r") as f:
        prompts = json.load(f)

    # Update the absolute image paths
    for prompt in prompts:
        path_to_image = prompt["image"]
        image_name = re.split(r"images/", path_to_image, maxsplit=1)[-1]
        prompt["image"] = os.path.join(dataset_configs["data_root"], prompt["dataset"], "images", image_name)

    prompts = [BiasPrompt(**prompt) for prompt in prompts]
    return prompts


def get_cmd_arguments() -> argparse.Namespace:
    # Make argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-chunk-index", type=int, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    return parser.parse_args()


def prompt_to_keys(prompt: Prompt) -> dict[str, str]:
    return {
        "gender": prompt.gender,
        "value": prompt.value,
        "task": prompt.task,
        "dataset": prompt.dataset,
        "image": prompt.image,
        "unknown_option_letter": prompt.unknown_option_letter,
        "yes_option_letter": prompt.yes_option_letter,
        "no_option_letter": prompt.no_option_letter,
    }


def get_results(prompt_chunk_index: int, task: Optional[str] = None, alpha=0.6) -> list[dict]:
    # Load model
    model = load_model(args.model, alpha)

    # Make classification prompts
    if task is not None:
        base_dir = f"./data/prompts_by_task/{task}"
    else:
        base_dir = "./data/prompts"
    prompts = make_prompts(prompt_chunk_index, base_dir=base_dir)

    # Make dataloader
    dataloader = make_dataloader(prompts, model, args.model)

    # Iterate dataloader and get classification results
    results = []

    for prompt, metadata in tqdm(iter(dataloader)):
        with torch.no_grad():
            probs = model.get_next_token_probabilities(prompt)

        # Get option letter indices
        for i, prompt_metadata in enumerate(metadata):
            probs_i = probs[i]
            option_probs = dict()
            for letter, option in prompt_metadata.letter_to_option.items():
                letter_index = encode_option_letter(letter, model, args.model)
                option_probs[option] = probs_i[letter_index].item()
            print(option_probs, flush=True)

            keys_to_save = prompt_to_keys(prompt_metadata)
            results.append(
                {
                    **keys_to_save,
                    **option_probs,
                }
            )

    return results


if __name__ == '__main__':
    # Make argument parser
    args = get_cmd_arguments()
    unique_filename = str(args.alpha*10)
    print(unique_filename)

    # Get results
    prompt_chunk_index = args.prompt_chunk_index
    results = get_results(prompt_chunk_index, task=args.task, alpha=args.alpha)

    # Convert results to pd dataframe
    results_df = pd.DataFrame(results)
    save_path = os.path.join("./results/benchmark/", args.model)
    os.makedirs(save_path, exist_ok=True)
    save_filename = f"{prompt_chunk_index}_alpha{unique_filename}.csv"
    results_df.to_csv(os.path.join(save_path, save_filename), index=False)
