import os
import torch
import argparse
import pandas as pd
import multiprocessing as mp

from tqdm import tqdm
from vlms import load_model
from dataclasses import dataclass
from typing import Callable, Optional
from torch.utils.data import DataLoader

import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from utils_new.configs import dataset_configs
from utils_new.configs import variant_to_a100_batch_size_mapping
from vlms.base import BaseVLM, BasePreprocessor, PreprocessedPromptWithImage


@dataclass
class Prompt:
    prompt: str
    image: str
    correct_option: str
    letter_to_option: dict[str, str]


MAKE_PROMPT_FROM_ROW_TYPE = Callable[[pd.Series, str], Prompt]
MAKE_PROMPTS_TYPE = Callable[[str, Optional[MAKE_PROMPT_FROM_ROW_TYPE]], list[Prompt]]


class DataCollator:
    def __init__(self, preprocessor: BasePreprocessor) -> None:
        self.preprocessor = preprocessor

    def __call__(
        self, batch: list[Prompt]
    ) -> tuple[PreprocessedPromptWithImage, list[Prompt]]:
        # Extract prompts and images
        prompts = [prompt.prompt for prompt in batch]
        images = [prompt.image for prompt in batch]
        return self.preprocessor.preprocess(prompts=prompts, images=images), batch


def make_dataloader(prompts: list[Prompt], model: BaseVLM, model_name: str) -> DataLoader:
    # Get preprocessor
    preprocessor = model.get_preprocessor()

    # Make batch size and num workers
    if ":" in model_name:
        model_name = model_name.split(":")[1]
        
    batch_size = variant_to_a100_batch_size_mapping[model_name]
    if model_name.startswith("internvl"):
        batch_size = max(1, batch_size // 2)
    num_workers = min(8, mp.cpu_count())

    # Instantiate dataloader
    return DataLoader(
        prompts,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollator(preprocessor),
        num_workers=num_workers,
    )


def load_dataset(dataset: str, sample: bool = False) -> pd.DataFrame:
    path_to_dataset = os.path.join(dataset_configs["data_root"], dataset)
    annotations = pd.read_csv(os.path.join(path_to_dataset, "annotations.csv"))
    # Filter annotations for files that exist
    """
    annotations = annotations[
        annotations["name"].apply(
            lambda x: os.path.exists(os.path.join(path_to_dataset, "images", x))
        )
    ]
    """
    annotations = annotations[
        annotations["name"].apply(
            lambda x: os.path.exists(os.path.join(path_to_dataset, "images", x))
        )
    ]

    if sample:
        annotations = annotations.sample(200)

    return annotations


def make_classification_prompts(dataset: str, make_prompt_from_row: Optional[MAKE_PROMPT_FROM_ROW_TYPE] = None) -> list[Prompt]:
    annotations = load_dataset(dataset)
    path_to_dataset = os.path.join(dataset_configs["data_root"], dataset)

    # Iterate annotations and create prompts
    prompts = []
    for _, row in annotations.iterrows():
        prompt = make_prompt_from_row(row, path_to_dataset)
        prompts.append(prompt)

    return prompts


def classification_prompt_to_keys(prompt: Prompt) -> dict:
    return {
        "image": os.path.split(prompt.image)[1],
        "correct_option": prompt.correct_option,
    }

def encode_option_letter(letter: str, model: BaseVLM, model_name: str) -> int:
    if ":" in model_name:
        model_name = model_name.split(":")[1]
    if model_name.startswith("internvl") or "34b" in model_name:
        return model.tokenizer.convert_tokens_to_ids(letter)
    
    if model_name != "qwen":
        letter = " " + letter

    encoded_letter = model.tokenizer.encode(letter, add_special_tokens=False)
    encoded_letter = encoded_letter[-1]
    return encoded_letter


def get_cmd_arguments() -> argparse.Namespace:
    # Make argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    return parser.parse_args()


def get_results(args: argparse.Namespace, make_prompts_fn: Callable[[str,], list[Prompt]], prompt_to_keys_fn: Callable[[Prompt,], dict]) -> list[dict]:
    # Load model
    model = load_model(args.model)

    # Make classification prompts
    prompts = make_prompts_fn(args.dataset)

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
            
            keys_to_save = prompt_to_keys_fn(prompt_metadata)
            results.append(
                {
                    **keys_to_save,
                    **option_probs,
                }
            )
    
    return results
