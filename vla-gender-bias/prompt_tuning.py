from gpu_utils import set_gpu
set_gpu()

import os
import sys
import torch
import argparse
import time

from tqdm import tqdm
from torch import Tensor
from vlms import load_model
from typing import Optional
from vlms.base import BaseVLM
from dataclasses import dataclass
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from bias_eval_utils import BiasPromptIterator, BiasPrompt
from vlms.base import PreprocessedPromptWithImage, BaseVLM
from utils_new.benchmark_utils import encode_option_letter
from utils_new.benchmark_utils import DataCollator
from torch.utils.data import DataLoader, Dataset


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


def patch_mobilevlm(model: BaseVLM, prompt_prefix: torch.nn.Parameter) -> None:
    original_prepare_multimodal = model.model.prepare_inputs_labels_for_multimodal

    def wrap_prepare_multimodal(self, *args, **kwargs):
        position_ids, attention_mask, past_key_values, new_input_embeds, new_labels = (
            original_prepare_multimodal(
                *args,
            )
        )

        prefix = (
            prompt_prefix.unsqueeze(0)
            .repeat(new_input_embeds.shape[0], 1, 1)
            .to(new_input_embeds.device, dtype=new_input_embeds.dtype)
        )
        new_input_embeds = torch.cat([new_input_embeds[:, :6, :], prefix, new_input_embeds[:, 6:, :]], dim=1)
        return (
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    model.model.prepare_inputs_labels_for_multimodal = wrap_prepare_multimodal.__get__(
        model.model, type(model.model)
    )


def patch_llava(model: BaseVLM, prompt_prefix: torch.nn.Parameter) -> None:
    original_prepare_multimodal = model.model.prepare_inputs_labels_for_multimodal

    def wrap_prepare_multimodal(self, *args, **kwargs):
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        ) = original_prepare_multimodal(
            *args,
        )

        prefix = (
            prompt_prefix.unsqueeze(0)
            .repeat(new_input_embeds.shape[0], 1, 1)
            .to(new_input_embeds.device, dtype=new_input_embeds.dtype)
        )
        new_input_embeds = torch.cat([new_input_embeds[:, :5, :], prefix, new_input_embeds[:, 5:, :]], dim=1)
        return (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    model.model.prepare_inputs_labels_for_multimodal = wrap_prepare_multimodal.__get__(
        model.model, type(model.model)
    )

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


def patch_internvl2(model: BaseVLM, prompt_prefix: torch.nn.Parameter, MAX_LEN = 4096) -> None:
    from vlms.internvl2 import InternVLPreprocessedPromptWithImage, IMG_CONTEXT_TOKEN

    def patched_get_next_token_probabilities(
        self, prompt: InternVLPreprocessedPromptWithImage
    ) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device) # (1, 7505)
        images = prompt.images.to(self.model.device, dtype=torch.bfloat16)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Pad input_ids and attention_mask
        input_ids_padding = torch.full(
            (input_ids.shape[0], prompt_prefix.shape[1]),
            fill_value=self.tokenizer.pad_token_id,
        )
        input_ids_padding = input_ids_padding.to(
            self.model.device, dtype=input_ids.dtype
        )
        input_ids = torch.cat([input_ids_padding, input_ids], dim=1) # (7505, 4096)

        attention_mask_padding = torch.zeros(
            (attention_mask.shape[0], prompt_prefix.shape[1]),
        )
        attention_mask_padding = attention_mask_padding.to(
            self.model.device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([attention_mask_padding, attention_mask], dim=1)

        # Find the input lengths
        input_lengths = attention_mask.sum(dim=1)
        # Find the start indices for the prompt prefix
        max_length = input_ids.shape[1]
        start_indices = max_length - input_lengths - prompt_prefix.shape[1]
        insert_indices = start_indices.unsqueeze(1).repeat(
            1, prompt_prefix.shape[0]
        ) + torch.arange(prompt_prefix.shape[0]).unsqueeze(0).to(start_indices.device)

        # Set img_context_token_id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # Forward pass
        images = images.to(self.model.device, dtype=torch.bfloat16) # inserted
        vit_embeds = self.model.extract_feature(images)
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = torch.eq(input_ids, self.model.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device) # (7505, 4096)

        input_embeds = input_embeds.reshape(B, N, C)
        # Insert the prompt prefix using start_indices along the time dimension
        input_embeds[torch.arange(B).unsqueeze(1), insert_indices] = prompt_prefix.to(
            input_embeds.device, dtype=input_embeds.dtype
        )
        attention_mask[torch.arange(B).unsqueeze(1), insert_indices] = 1
        
        input_embeds = input_embeds[:, :MAX_LEN] # torch.Size([1, max_len, 4096])
        attention_mask = attention_mask[:, :MAX_LEN]
        

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model.language_model.forward(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
            ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    model.get_next_token_probabilities = patched_get_next_token_probabilities.__get__(
        model, type(model)
    )


def forward_and_get_loss(
    model: BaseVLM,
    model_name: str,
    prompt: PreprocessedPromptWithImage,
    prompt_metadata: list[BiasPrompt],
    equalize_yes_no: bool = False,
) -> Tensor:
    probs = model.get_next_token_probabilities(prompt)

    # Make sure batch size is 1
    assert len(prompt_metadata) == 1

    # Get the prompt metadata
    prompt_metadata: BiasPrompt = prompt_metadata[0]
    probs = probs.squeeze(0)

    option_to_letter = {
            option: letter for letter, option in prompt_metadata.letter_to_option.items()
        }
    
    if equalize_yes_no:
        # Get probs of yes and no options
        yes_option_letter = option_to_letter["Yes"]
        no_option_letter = option_to_letter["No"]
        yes_option_index = encode_option_letter(yes_option_letter, model, model_name)
        no_option_index = encode_option_letter(no_option_letter, model, model_name)

        yes_option_prob = probs[yes_option_index]
        no_option_prob = probs[no_option_index]

        # Loss is deviation of log-prob of yes/no from 0.5
        loss = torch.abs(yes_option_prob - 0.5) + torch.abs(no_option_prob - 0.5)
    else:
        # Get prob of unsure option
        unsure_option_letter = option_to_letter["Unknown"]
        unsure_option_index = encode_option_letter(unsure_option_letter, model, model_name)
        unsure_option_prob = probs[unsure_option_index]
        loss = 1 - unsure_option_prob

    # Shoots nan & inf
    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return torch.tensor(1.0, requires_grad=True)
    return loss


def train(
    model: BaseVLM,
    model_name: str,
    prefix: torch.nn.Parameter,
    dataset: Dataset,
    max_steps: int = 1,
    lr: float = 1e-3,
    gradient_accumulation_steps: int = 1,
    threshold: float = 0.08,
    max_steps_below_threshold: int = 10,
):
    trainable_parameters = [
        prefix,
    ]
    optimizer = torch.optim.Adam(trainable_parameters, lr=lr, weight_decay=0.0001)
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
        loss = (
            forward_and_get_loss(model, model_name, prompt, prompt_metadata)
            / gradient_accumulation_steps
        )
        loss.backward()

        # Implement gradient accumulation
        if progress_bar.n % gradient_accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad.clip_grad_value_([prefix], 1.0)
            optimizer.step()
            optimizer.zero_grad()

        # Update progress bar
        loss_item = loss.item()
        if loss_running_mean is None:
            loss_running_mean = loss_item
        else:
            loss_running_mean = 0.9 * loss_running_mean + 0.1 * loss_item

        if (
            progress_bar.n % gradient_accumulation_steps == 0
            and loss_running_mean * gradient_accumulation_steps < threshold
        ):
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


def prepare_model_for_training(model: BaseVLM, model_name: str, num_tunable_tokens: int,) -> torch.nn.Parameter:
    for parameter in model.model.parameters():
        parameter.requires_grad = False

    # Get the prompt prefix
    embedding_dim = model.get_embedding_size()
    prompt_prefix = torch.nn.Parameter(
        torch.empty(num_tunable_tokens, embedding_dim),
        requires_grad=True,
    )
    torch.nn.init.normal_(prompt_prefix, mean=0.0, std=0.02)

    if model_name.startswith("mobilevlm"):
        patch_mobilevlm(model, prompt_prefix)
    elif model_name.startswith("llava"):
        patch_llava(model, prompt_prefix)
    elif model_name.startswith("internvl2"):
        patch_internvl2(model, prompt_prefix)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return prompt_prefix


MODELS = ["llava-7b", "llava-13b", "mobilevlm-7b", "llava-1.6-vicuna-7b", "llava-1.6-mistral-7b", "internvl2-8b"]


def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-images-per-dataset", type=int, default=500)
    for field in TrainingArguments.__dataclass_fields__.values():
        parser.add_argument(f"--{field.name.replace('_', '-')}", type=field.type, default=field.default)

    # Add model argument
    parser.add_argument("--model", type=str, required=True, choices=MODELS)
    parser.add_argument("--trainable-tokens", type=int, default=20) #required=True)
    return parser


if __name__ == "__main__":
    # Make argument parser
    parser = make_argument_parser()
    args = parser.parse_args()

    save_path = os.path.join(
        "./results/prompt_tuning/", args.model,
    )
    # if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
    #     print(
    #         f"Model {args.model} already tuned"
    #     )
    #     sys.exit(0)

    # Load Model
    model = load_model(args.model)
    preprocessor = model.get_preprocessor()

    model.model.language_model.gradient_checkpointing_enable()
    model.model.language_model.config.use_cache = False

    # Count trainable parameters
    trainable_params_m = count_trainable_parameters(model)
    print(f"\nTrainable Parameters: {trainable_params_m:.2f}M")

    # Prepare model for training
    tuning_prefix = prepare_model_for_training(model, args.model, args.trainable_tokens)

    # Count trainable parameters
    trainable_params_m = count_trainable_parameters(model)
    print(f"\nTrainable Parameters: {trainable_params_m:.2f}M")

    prefix_params_m = tuning_prefix.numel() / 1_000_000
    print(f"Prompt Prefix Parameters: {prefix_params_m:.4f}M")

    # Make prompts
    prompts = []
    for task in ["sentiment", "skills", "occupations"]:
        dataset_args = DatasetArguments(
            task=task,
            datasets=dataset_configs["benchmark_datasets"],
            num_images_per_dataset=args.num_images_per_dataset,
        )
        # Make dataset and data loader
        task_prompts = BiasPromptIterator(**dataset_args.__dict__).get_prompts()
        prompts.extend(task_prompts)

    collator = DataCollator(preprocessor)
    dataloader = DataLoader(
        prompts, batch_size=1, shuffle=True, num_workers=8, collate_fn=collator
    )

    # Train model
    model, train_time_gpu_hours, peak_mem_mb = train(
        model,
        args.model,
        tuning_prefix,
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
    # torch.save(tuning_prefix, os.path.join(save_path, "tuning_prefix.pt"))
