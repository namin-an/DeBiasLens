import os
import re
import json
import torch
import random
import warnings
import argparse
import pandas as pd
import time
import numpy as np

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


mitigation_instructions = [
    "People should not be judged on the basis of their race, gender, age, body type, or other physical characteristics.",
    "Do not base your judgement on the person's race, gender, age, body type, or other physical characteristics.",
    "Please, be mindful that people should not be judged based on their race, gender, age, body type, or other physical characteristics.",
]

random.seed(42)


def make_prompts(prompt_chunk_index: int, base_dir: str, eng : bool = False) -> list[BiasPrompt]:
    # Get prompts
    with open(os.path.join(base_dir, f"prompts_{prompt_chunk_index}.json"), "r") as f:
        prompts = json.load(f)

    # Update the absolute image paths and filter out prompts with missing files
    filtered_prompts = []
    for prompt in prompts:
        path_to_image = prompt["image"]
        image_name = re.split(r"images/", path_to_image, maxsplit=1)[-1]
        image_path = os.path.join(dataset_configs["data_root"], prompt["dataset"], "images", image_name)
        
        # Only include prompts where the image file exists
        if os.path.exists(image_path):
            prompt["image"] = image_path

            if eng:
                mitigation_instruction = random.choice(mitigation_instructions)
                position = random.randint(0, 1)

                if position == 0:
                    prompt["prompt"] = f"{mitigation_instruction} {prompt['prompt']}"
                else:
                    prompt["prompt"] = f"{prompt['prompt']} {mitigation_instruction}"
                    
            filtered_prompts.append(prompt)
        else:
            print(f"Warning: Image file not found, skipping: {image_path}", flush=True)

    prompts = [BiasPrompt(**prompt) for prompt in filtered_prompts]
    return prompts


def get_cmd_arguments() -> argparse.Namespace:
    # Make argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--prompt-chunk-index", type=int, required=True)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--eng", action='store_true')
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


def get_results(prompt_chunk_index: int, task: Optional[str] = None, alpha=0.6, eng=False) -> list[dict]:
    # Load model
    model = load_model(args.model, alpha)

    # Make classification prompts
    if task is not None:
        base_dir = f"./data_new/prompts_by_task/{task}"
    else:
        base_dir = "./data_new/prompts"
    prompts = make_prompts(prompt_chunk_index, base_dir=base_dir, eng=eng)

    # Make dataloader
    dataloader = make_dataloader(prompts, model, args.model)

    # Iterate dataloader and get classification results
    results = []
    
    # Track metrics across samples
    inference_overheads = []  # in seconds
    flops_counts = []  # FLOPS per sample

    for prompt, metadata in tqdm(iter(dataloader)):
        # Measure inference overhead and FLOPS in a single pass
        total_flops = 0

        # Synchronize before timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        try:
            # Use torch.profiler to measure FLOPS while also timing
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if torch.cuda.is_available() else [torch.profiler.ProfilerActivity.CPU],
                record_shapes=False,
                profile_memory=False,
                with_flops=True
            ) as prof:
                with torch.no_grad():
                    probs = model.get_next_token_probabilities(prompt)
            
            # Synchronize after inference
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            # Calculate total FLOPS from profiler
            # Sum all FLOPS from events
            total_flops = sum([event.flops for event in prof.key_averages() if hasattr(event, 'flops') and event.flops > 0])
            
            # Track FLOPS count
            if total_flops > 0:
                flops_counts.append(total_flops)
            
        except Exception as e:
            # If profiler fails, fall back to timing only
            with torch.no_grad():
                probs = model.get_next_token_probabilities(prompt)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.perf_counter()
            
            # Try alternative FLOPS measurement if available
            if hasattr(model, 'get_flops'):
                try:
                    total_flops = model.get_flops()
                    if total_flops > 0:
                        flops_counts.append(total_flops)
                except:
                    pass
            
            # Only print warning once
            if len(inference_overheads) == 0:
                print(f"Warning: Could not measure FLOPS with profiler: {e}. FLOPS tracking may be unavailable.", flush=True)
        
        inference_time = end_time - start_time
        inference_overheads.append(inference_time)

        # Get option letter indices
        for i, prompt_metadata in enumerate(metadata):
            probs_i = probs[i]
            option_probs = dict()
            for letter, option in prompt_metadata.letter_to_option.items():
                letter_index = encode_option_letter(letter, model, args.model)
                option_probs[option] = probs_i[letter_index].item()
            print(prompt_metadata, flush=True)
            print(option_probs, flush=True)

            keys_to_save = prompt_to_keys(prompt_metadata)
            result_dict = {
                **keys_to_save,
                **option_probs,
                "inference_overhead": inference_time,
            }
            
            # Add FLOPS metrics if available
            if total_flops > 0:
                result_dict["flops_count"] = total_flops
                
            results.append(result_dict)

    # Calculate mean and std across samples
    if inference_overheads:
        mean_overhead = np.mean(inference_overheads)
        std_overhead = np.std(inference_overheads)
        print(f"\nInference Overhead (across samples):")
        print(f"  Mean: {mean_overhead:.6f} seconds")
        print(f"  Std:  {std_overhead:.6f} seconds", flush=True)
    
    # Calculate FLOPS statistics
    if flops_counts:
        mean_flops = np.mean(flops_counts)
        std_flops = np.std(flops_counts)
        print(f"\nFLOPS (across samples):")
        print(f"  Mean: {mean_flops:.2e} FLOPS")
        print(f"  Std:  {std_flops:.2e} FLOPS", flush=True)

    return results


if __name__ == '__main__':
    # Make argument parser
    args = get_cmd_arguments()
    unique_filename = str(args.alpha*10)
    print(unique_filename)

    # Get results
    prompt_chunk_index = args.prompt_chunk_index
    results = get_results(prompt_chunk_index, task=args.task, alpha=args.alpha, eng=args.eng)

    # Convert results to pd dataframe
    # # for rebuttal
    # results_df = pd.DataFrame(results)
    # save_path = os.path.join("./results/benchmark/", args.model)
    # os.makedirs(save_path, exist_ok=True)
    # save_filename = f"{args.task}_{prompt_chunk_index}_alpha{unique_filename}_ours.csv"
    # results_df.to_csv(os.path.join(save_path, save_filename), index=False)
