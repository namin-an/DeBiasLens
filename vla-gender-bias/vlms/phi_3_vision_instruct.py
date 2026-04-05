import torch

from PIL import Image
from torch import Tensor
from typing import Union
from dataclasses import dataclass
from .base import BasePreprocessor
from utils.configs import model_configs
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoProcessor
from .base import BaseVLM, PreprocessedPrompt, PreprocessedPromptWithImage


@dataclass
class PhiVPreprocessedPrompt(PreprocessedPrompt):
    attention_mask: Tensor


@dataclass
class PhiVPreprocessedPromptWithImage(PreprocessedPromptWithImage):
    attention_mask: Tensor
    image_sizes: Tensor


class Phi3VPreprocessor(BasePreprocessor):
    def __init__(self, processor,) -> None:
        super().__init__()
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PhiVPreprocessedPromptWithImage:
        # Make sure prompts and images are lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images

        # Load Images
        images = [Image.open(image).convert("RGB") for image in images]

        # Set tokenizer padding side to left
        self.processor.tokenizer.padding_side = "left"

        # Process prompts and images
        all_input_ids = []
        all_pixel_values = []
        all_attention_masks = []
        all_image_sizes = []

        for prompt, image in zip(prompts, images):
            # Add image placeholder to prompt
            prompt = "<|image_1|>\n" + prompt
            # Wrap prompts in dicts
            prompt = [{"role": "user", "content": prompt}]
            # Apply chat template
            prompt = self.processor.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                prompt, [image], return_tensors="pt", padding="longest"
            )
            all_input_ids.append(inputs["input_ids"].squeeze(0).flip(dims=(0,)))
            all_pixel_values.append(inputs["pixel_values"])
            all_attention_masks.append(
                inputs["attention_mask"].squeeze(0).flip(dims=(0,))
            )
            all_image_sizes.append(inputs["image_sizes"])

        # Stack inputs
        all_input_ids = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        ).flip(dims=(-1,))
        all_pixel_values = torch.cat(all_pixel_values, dim=0)
        all_attention_masks = pad_sequence(
            all_attention_masks, batch_first=True, padding_value=0
        ).flip(dims=(-1,))
        all_image_sizes = torch.cat(all_image_sizes, dim=0)

        # Calculate input lengths
        input_lengths = all_attention_masks.sum(dim=1)

        return PhiVPreprocessedPromptWithImage(
            input_ids=all_input_ids,
            input_lengths=input_lengths,
            images=all_pixel_values,
            attention_mask=all_attention_masks,
            image_sizes=all_image_sizes,
        )

    def preprocess_for_lm(
        self,
        prompts: Union[str, list[str]],
    ) -> PhiVPreprocessedPrompt:
        # Make sure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Process prompts and images
        all_input_ids = []
        all_attention_masks = []

        # Set tokenizer padding side to left
        self.processor.tokenizer.padding_side = "left"

        for prompt in prompts:
            prompt = [{"role": "user", "content": prompt}]
            # Apply chat template
            prompt = self.processor.tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(
                prompt, None, return_tensors="pt", padding="longest"
            )
            all_input_ids.append(inputs["input_ids"].squeeze(0).flip(dims=(0,)))
            all_attention_masks.append(
                inputs["attention_mask"].squeeze(0).flip(dims=(0,))
            )

        # Stack inputs
        all_input_ids = pad_sequence(
            all_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id,
        ).flip(dims=(-1,))
        all_attention_masks = pad_sequence(
            all_attention_masks, batch_first=True, padding_value=0
        ).flip(dims=(-1,))
        # Calculate input lengths
        input_lengths = all_attention_masks.sum(dim=1)

        return PhiVPreprocessedPrompt(
            input_ids=all_input_ids,
            input_lengths=input_lengths,
            attention_mask=all_attention_masks,
        )


class Phi3VModel(BaseVLM):
    def __init__(self, variant: str) -> None:
        super().__init__()
        model_path = model_configs["phi3v"][variant]["model_path"]

        # Load Model
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                _attn_implementation="flash_attention_2",
            )
        # In case FlashAttention 2 is not available, fall back to standard implementation
        except ImportError:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                trust_remote_code=True,
                torch_dtype="auto",
                 _attn_implementation="eager",
            )

        self.model = model

        # Load Processor
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, num_crops=16
        )
        self.tokenizer = self.processor.tokenizer

    def get_next_token_probabilities(self, prompt: PhiVPreprocessedPromptWithImage) -> Tensor:
        # Run model
        output = self.model.forward(
            prompt.input_ids.to(self.model.device),
            attention_mask=prompt.attention_mask.to(self.model.device),
            pixel_values=prompt.images.to(self.model.device),
            image_sizes=prompt.image_sizes.to(self.model.device),
        )

        # Get logits
        logits = output.logits
        # Get next-token prediction logits
        logits = logits[torch.arange(logits.size(0)), - 1]
        # Apply softmax
        probabilities = torch.softmax(logits, dim=-1)
        # Return
        return probabilities

    def get_next_token_probabilities_from_lm(self, prompt: PhiVPreprocessedPrompt) -> Tensor:
        # Run model
        output = self.model.forward(
            prompt.input_ids.to(self.model.device),
            attention_mask=prompt.attention_mask.to(self.model.device),
        )

        # Get logits
        logits = output.logits
        # Get next-token prediction logits
        logits = logits[torch.arange(logits.size(0)), - 1]
        # Apply softmax
        probabilities = torch.softmax(logits, dim=-1)
        # Return
        return probabilities

    def get_preprocessor(self) -> Phi3VPreprocessor:
        return Phi3VPreprocessor(self.processor)

    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_embedding_size(self) -> int:
        return self.model.model.embed_tokens.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.model.embed_tokens.weight.mean(dim=0)
