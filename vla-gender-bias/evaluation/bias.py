import os
import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from torch.utils.data import DataLoader, Dataset
from bias_eval_utils import BiasPromptIterator, Prompt


class BiasDataset(Dataset):
    def __init__(self, dataset: list[Prompt], model) -> None:
        self.dataset = dataset
        self.model = model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        prompt = self.dataset[idx]
        image_name = prompt.image
        prompt_text = prompt.prompt
        gender = prompt.gender
        dataset = prompt.dataset
        yes_option_letter = prompt.yes_option_letter

        path_to_image = os.path.join(dataset_configs["data_root"], dataset, "images", image_name)
        if os.path.exists(path_to_image):
            generation_info = self.model.preprocess(
                prompt=prompt_text, image=path_to_image
            )
        else:
            generation_info = None

        return {
            "generation_info": generation_info,
            "gender": gender,
            "dataset": dataset,
            "yes_option_letter": yes_option_letter,
            "name": image_name
        }


def get_bias_performance(model, task: int, num_images: int) -> float:
    results = []

    # Get prompts
    prompts = BiasPromptIterator(
        task=task,
        # split="test",
        datasets=dataset_configs["benchmark_datasets"],
        num_images_per_dataset=num_images,
        include_unknown=False,
        options_num_permutations=1,
        sample_value=True,
        sample_question=True,
        sample_instructions=True,
        sample_unknown=False,
        num_values_per_image=1,
    ).get_prompts()

    # Make dataloader
    dataset = BiasDataset(prompts, model)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8, collate_fn=lambda x: x[0])

    # Get responses for all image, prompt pairs in dataset
    for dp in tqdm(dataloader, leave=False):
        gender = dp["gender"]
        yes_option_letter = dp["yes_option_letter"]
        dataset = dp["dataset"]
        generation_info = dp["generation_info"]
        image_name = dp["name"]

        if generation_info is None:
            continue

        with torch.no_grad():
            response = model.generate(**generation_info, max_new_tokens=1)

        yes_option_index = model.tokenizer.encode(yes_option_letter)[-1]
        yes_prob = np.exp(response["scores"][0, yes_option_index]).item()

        results.append(
            {
                "dataset": dataset,
                "name": image_name,
                "gender": gender,
                "yes_prob": yes_prob,
            }
        )

    results_df = pd.DataFrame(results)
    return results_df


class BiasScorer:
    def __init__(self, task: str, num_images: int) -> None:
        self.num_images = num_images
        self.task = task

    def score(self, model) -> pd.DataFrame:
        bias_performance = get_bias_performance(
            model, task=self.task, num_images=self.num_images
        )
        return bias_performance
