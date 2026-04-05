from gpu_utils import set_gpu
set_gpu()

import os
import sys
import torch
import argparse
import time

from tqdm import tqdm
from torch import Tensor
from typing import Optional
from vlms import load_model
from dataclasses import dataclass
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from torch.utils.data import DataLoader, Dataset
from bias_eval_utils import BiasPromptIterator, BiasPrompt
from peft import LoraConfig, get_peft_model

from vlms.base import PreprocessedPromptWithImage, BaseVLM
from utils_new.benchmark_utils import encode_option_letter
from utils_new.benchmark_utils import DataCollator


@dataclass
class DatasetArguments:
    task: str
    datasets: list[str]
    num_images_per_dataset: int
    include_unknown: bool = True
    options_num_permutations: int = 1
    sample_value: bool = True
    sample_question: bool = True
    sample_instructions: bool = True
    sample_unknown: bool = True
    num_values_per_image: Optional[int] = 1
    image_split: Optional[str] = "train"
    value_split: Optional[str] = "train"
    prompt_split: Optional[str] = "train"
    seed: int = 42


@dataclass
class TrainingArguments:
    max_steps: int = 100
    lr: float = 1e-3
    gradient_accumulation_steps: int = 1
    threshold: float = 0.0
    max_steps_below_threshold: int = 10
    use_lora: int = 0


# Create argument parser from DatasetArguments and TrainingArguments
def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--num-images-per-dataset", type=int, default=500)
    for field in TrainingArguments.__dataclass_fields__.values():
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)
    
    # Add model argument
    parser.add_argument("--model", type=str, required=True)
    return parser


def forward_and_get_loss(model: BaseVLM, model_name: str, prompt: PreprocessedPromptWithImage, prompt_metadata: list[BiasPrompt], equalize_yes_no: bool = False) -> Tensor:
    probs = model.get_next_token_probabilities(prompt)

    # Make sure batch size is 1
    assert len(prompt_metadata) == 1

    # Get the prompt metadata
    prompt_metadata: BiasPrompt = prompt_metadata[0]
    probs = probs.squeeze(0)

    # Get log-probs of yes and no options
    option_to_letter = {option: letter for letter, option in prompt_metadata.letter_to_option.items()}
    if equalize_yes_no:
        yes_option_letter = option_to_letter["Yes"]
        no_option_letter = option_to_letter["No"]
        yes_option_index = encode_option_letter(yes_option_letter, model, model_name)
        no_option_index = encode_option_letter(no_option_letter, model, model_name)

        yes_option_prob = probs[yes_option_index]
        no_option_prob = probs[no_option_index]

        # Loss is deviation of log-prob of yes/no from 0.5
        loss = torch.abs(yes_option_prob - 0.5) + torch.abs(no_option_prob - 0.5)
    
    else:
        unsure_option_letter = option_to_letter["Unknown"]
        unsure_option_index = encode_option_letter(unsure_option_letter, model, model_name)
        unsure_option_prob = probs[unsure_option_index]
        loss = 1 - unsure_option_prob

    # Shoots nan & inf
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return torch.tensor(1.0, requires_grad=True)
    return loss


def _find_all_linear_names(module: torch.nn.Module) -> set[str]:
    all_linear_layer_names = set()
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            all_linear_layer_names.add(name)
        else:
            all_linear_layer_names.update(_find_all_linear_names(child))
    return all_linear_layer_names


def count_trainable_parameters(model: BaseVLM, peft_model=None) -> float:
    """Count trainable parameters in millions."""
    if peft_model is not None:
        # For LoRA models, count trainable parameters from PEFT model
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    else:
        # For full fine-tuning, count trainable parameters
        trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    
    # Convert to millions
    trainable_params_m = trainable_params / 1_000_000
    return trainable_params_m


def prepare_model_for_training(
    model: BaseVLM,
    use_lora: bool = False,
) -> tuple:
    # Activate training mode
    model.model.train()

    # Freeze all parameters
    for parameter in model.model.parameters():
        parameter.requires_grad = False

    # Get LLM layers
    llm_layers = model.get_llm_layers()

    # Prepare parameters of LLM for training
    if use_lora:
        config = LoraConfig(
            r=128,
            lora_alpha=128,
            target_modules=_find_all_linear_names(llm_layers),
            lora_dropout=0.0,
            bias="none",
            modules_to_save=[],
        )
        peft_model = get_peft_model(llm_layers, config)
        return peft_model, None
    else:
        # Unfreeze parameters in llm layers
        for parameter in llm_layers.parameters():
            parameter.requires_grad = True
        return None, None


def train(
    model: BaseVLM,
    model_name: str,
    dataset: Dataset,
    max_steps: int = 1,
    lr: float = 1e-3,
    gradient_accumulation_steps: int = 1,
    threshold: float = 0.08,
    max_steps_below_threshold: int = 10,
):
    trainable_parameters = [parameter for parameter in model.model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable_parameters, lr=lr)
    progress_bar = tqdm(total=max_steps, desc="Training")
    loss_running_mean = None
    optimizer.zero_grad()
    steps_below_threshold = 0

    # Start training time measurement
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    train_start_time = time.perf_counter()

    for prompt, prompt_metadata in dataset:
        loss = forward_and_get_loss(model, model_name, prompt, prompt_metadata) / gradient_accumulation_steps
        loss.backward()

        # Implement gradient accumulation
        if progress_bar.n % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad.clip_grad_value_(model.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar
        loss_item = loss.item()
        if loss_running_mean is None:
            loss_running_mean = loss_item
        else:
            loss_running_mean = 0.9 * loss_running_mean + 0.1 * loss_item

        if progress_bar.n % gradient_accumulation_steps == 0 and loss_running_mean * gradient_accumulation_steps < threshold:
            steps_below_threshold += 1
            if steps_below_threshold > max_steps_below_threshold:
                break
        if loss_running_mean * gradient_accumulation_steps >= threshold:
            steps_below_threshold = 0

        progress_bar.set_postfix(
            {"Loss": loss_running_mean * gradient_accumulation_steps}
        )
        progress_bar.update(1)
        if progress_bar.n >= max_steps:
            break

    # End training time measurement
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_mem_bytes = torch.cuda.max_memory_allocated()
        peak_mem_mb = peak_mem_bytes / (1024 ** 2)
    else:
        peak_mem_mb = 0.0
    train_end_time = time.perf_counter()
    train_time_seconds = train_end_time - train_start_time
    
    # Convert to GPU hours (assuming single GPU)
    train_time_gpu_hours = train_time_seconds / 3600.0
    
    progress_bar.close()
    return model, train_time_gpu_hours, peak_mem_mb


if __name__ == '__main__':
    # Make argument parser
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.task is None:
        task_str = "all"
    else:
        task_str = args.task
    save_path = os.path.join("./results/tuned_models", args.model, task_str, "lora" if args.use_lora else "full")
    # if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
    #     print(f"Model {args.model} already tuned for task {args.task} with LORA={args.use_lora}")
    #     sys.exit(0)

    # Load Model
    model = load_model(args.model)
    preprocessor = model.get_preprocessor()

    # Prepare model for training
    peft_model, _ = prepare_model_for_training(model, use_lora=args.use_lora)
    
    # Count trainable parameters
    trainable_params_m = count_trainable_parameters(model, peft_model)
    print(f"\nTrainable Parameters: {trainable_params_m:.2f}M", flush=True)

    # Make prompts
    if args.task is not None:
        dataset_args = DatasetArguments(
            task=args.task, datasets=dataset_configs["benchmark_datasets"], num_images_per_dataset=args.num_images_per_dataset
        )
        # Make dataset and data loader
        prompts = BiasPromptIterator(**dataset_args.__dict__).get_prompts()
    else:
        prompts = []
        for task in ["sentiment", "skills", "occupations"]:
            dataset_args = DatasetArguments(
                task=task, datasets=dataset_configs["benchmark_datasets"], num_images_per_dataset=args.num_images_per_dataset
            )
            # Make dataset and data loader
            task_prompts = BiasPromptIterator(**dataset_args.__dict__).get_prompts()
            prompts.extend(task_prompts)
    
    collator = DataCollator(preprocessor)
    dataloader = DataLoader(prompts, batch_size=1, shuffle=True, num_workers=8, collate_fn=collator)

    # Train model
    model, train_time_gpu_hours, peak_mem_mb = train(
        model,
        args.model,
        dataloader,
        max_steps=args.max_steps,
        lr=args.lr,
        threshold=args.threshold,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps_below_threshold=args.max_steps_below_threshold,
    )
    
    # Print training statistics
    print(f"\nTraining Statistics:")
    print(f"  Trainable Params: {trainable_params_m:.2f}M")
    print(f"  Train Time: {train_time_gpu_hours:.4f} GPU hrs")
    print(f"  Peak GPU Memory: {peak_mem_mb:.2f} MB")

    # Save model
    # os.makedirs(save_path, exist_ok=True)
    # if args.use_lora:
    #     peft_model.save_pretrained(save_path)
    # else:
    #     torch.save(model.model.state_dict(), os.path.join(save_path, "model.pt"))
