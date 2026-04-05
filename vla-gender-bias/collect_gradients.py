from gpu_utils import set_gpu
set_gpu()

import os
import torch
import argparse
import numpy as np
import multiprocessing as mp

from tqdm import tqdm
from typing import Any
from vlms import load_model
from typing import Optional
from bias_eval_utils import BiasPrompt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from bias_eval_utils import BiasPromptIterator
from datasets import load_dataset as load_hf_dataset
from utils_new.benchmark_utils import encode_option_letter

from tuning import DatasetArguments, DataCollator
from vlms.base import BaseVLM, BasePreprocessor, PreprocessedPromptWithImage


class MMEDataset(Dataset):
    def __init__(self, model_name: str, preprocessor: BasePreprocessor) -> None:
        self.model_name = model_name
        self.preprocessor = preprocessor
        self.dataset = load_hf_dataset("lmms-lab/MME")["test"]
        # self.dataset = load_hf_dataset("./data/MME/")["test"]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dp = self.dataset[idx]
        image = dp["image"].convert("RGB")
        # Save image
        if not os.path.exists(f"./data/MME_images/{idx}.png"):
            os.makedirs("./data/MME_images", exist_ok=True)
            image.save(f"./data/MME_images/{idx}.png", format="PNG")
        image = f"./data/MME_images/{idx}.png"

        generation_info = self.preprocessor.preprocess(prompts=dp["question"], images=image)
        answer = dp["answer"]
        if self.model_name == "qwen":
            answer_index = self.preprocessor.tokenizer.encode(answer, add_special_tokens=False)[-1]
        elif self.model_name.startswith("internvl"):
            answer_index = self.preprocessor.tokenizer.convert_tokens_to_ids(answer)
        else:
            answer_index = self.preprocessor.tokenizer.encode(" " + answer, add_special_tokens=False)[-1]

        return {
            "prompt": generation_info,
            "answer_index": answer_index,
        }


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=False, choices=["sentiment", "skills", "occupations"])
    parser.add_argument("--num-images-per-dataset", type=int, default=1000) #required=False, default=1000)
    return parser


def make_dataset(args: argparse.Namespace, model: BaseVLM, model_name: str) -> tuple[DataLoader, DataLoader]:
    num_workers = min(8, mp.cpu_count())
    preprocessor = model.get_preprocessor()
    # Make Bias data
    dataset_args = DatasetArguments(
        task=args.task, datasets=dataset_configs["benchmark_datasets"], num_images_per_dataset=args.num_images_per_dataset,
        include_unknown=True, sample_unknown=True,
    )
    prompts = BiasPromptIterator(**dataset_args.__dict__).get_prompts()
    collator = DataCollator(preprocessor)
    bias_dataloader = DataLoader(prompts, batch_size=1, shuffle=False, num_workers=num_workers, collate_fn=collator)
    
    # Make MME data
    mme_dataset = MMEDataset(model_name, preprocessor)
    # Make dataloader
    mme_dataloader = DataLoader(
        mme_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda x: x[0],
    )
    
    # Return dataloaders
    return bias_dataloader, mme_dataloader


def get_gradients_for_mme(
        model,
        num_dps: int,
        prompt: PreprocessedPromptWithImage,
        answer_index: int,
    ) -> None:
    probs = model.get_next_token_probabilities(prompt)
    if probs[0, answer_index] < 1e-2:
        return
    
    # probs = torch.log(probs)

    # Extract logits of target token
    probs = probs.squeeze(0)

    # Get NLL with `answer_index` as target
    nll_of_target = -probs[answer_index] / num_dps

    # Calculate gradients
    nll_of_target.backward()


def get_gradients_for_bias(
        model,
        model_name: str,
        num_dps: int,
        prompt: PreprocessedPromptWithImage,
        prompt_metadata: list[BiasPrompt],
        equalize_yes_no: bool = False,
    ) -> None:
    # Get probabilities of next token
    probs = model.get_next_token_probabilities(prompt).squeeze(0)
    
    if equalize_yes_no:
        yes_option_letter = prompt_metadata[0].yes_option_letter
        no_option_letter = prompt_metadata[0].no_option_letter

        yes_option_letter_index = encode_option_letter(yes_option_letter, model, model_name)
        no_option_letter_index = encode_option_letter(no_option_letter, model, model_name)

        # Get log-probs of yes and no options
        yes_option_prob = probs[yes_option_letter_index]
        no_option_prob = probs[no_option_letter_index]

        # Loss is deviation of log-prob of yes/no from 0.5
        loss = torch.abs(yes_option_prob - 0.5) + torch.abs(no_option_prob - 0.5)
        loss = torch.div(loss, num_dps)
    else:
        unknown_option_letter = prompt_metadata[0].unknown_option_letter
        unknown_option_letter_index = encode_option_letter(unknown_option_letter, model, model_name)
        unknown_option_prob = probs[unknown_option_letter_index]
        loss = 1 - unknown_option_prob
        loss = torch.div(loss, num_dps)
    
    if torch.isnan(loss) or torch.isinf(loss):
        return

    # Calculate gradients
    loss.backward()


metric_to_get_gradients_func = {
    "bias": get_gradients_for_bias,
    "mme": get_gradients_for_mme,
}


if __name__ == "__main__":
    # Get cmd arguments
    parser = make_argument_parser()
    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Disable gradients for model parameters
    for parameter in model.model.parameters():
        parameter.requires_grad = False

    # Get LLM layers and enable gradients
    llm_layers = model.get_llm_layers()
    for parameter in llm_layers.parameters():
        parameter.requires_grad = True

    # Make dataset
    bias_dataloader, mme_dataloader = make_dataset(args, model, args.model)
    dataloaders = {
        "bias": bias_dataloader,
        "mme": mme_dataloader,
    }

    for metric in [
        "mme",
        "bias",
    ]:
        # Zero gradients in model
        model.model.zero_grad()

        # Retrieve function for calculating gradients
        get_gradients = metric_to_get_gradients_func[metric]

        # Get dataloader
        dataloader = dataloaders[metric]

        # This loop iterates over all samples and calculates gradients
        # Gradients are accumulated in the model's parameters over all samples
        total_num_dps = len(dataloader)
        for dp in tqdm(dataloader):
            # Calculate gradients
            if metric == "mme":
                get_gradients(model, num_dps=total_num_dps, **dp)
            else:
                get_gradients(model, args.model, num_dps=total_num_dps, prompt=dp[0], prompt_metadata=dp[1])

        # Extract gradients from model
        gradients = dict()
        gradients_times_weight = dict()
        for name, parameter in model.model.named_parameters():
            if parameter.grad is None:
                continue

            gradient = parameter.grad.to(device="cpu", dtype=torch.float32)
            weight = parameter.data.to(device="cpu", dtype=torch.float32)
            gradients[name] = gradient
            gradients_times_weight[name] = gradient * weight

        # Save gradients
        save_dir = os.path.join(".", "results", "gradients", args.model, metric)
        if metric == "bias":
            save_dir = os.path.join(save_dir, args.task, str(args.num_images_per_dataset))

        os.makedirs(save_dir, exist_ok=True)
        torch.save(gradients, os.path.join(save_dir, "gradients.pt"))
        torch.save(gradients_times_weight, os.path.join(save_dir, "gradients_times_weight.pt"))
