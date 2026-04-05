import torch

from PIL import Image
from torch import Tensor
from typing import Union
from torch.nn import ModuleList
from dataclasses import dataclass
from utils.configs import model_configs
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers.models.paligemma.processing_paligemma import PaliGemmaProcessor
from .base import BaseVLM, BasePreprocessor, PreprocessedPrompt, PreprocessedPromptWithImage


@dataclass
class PaliGemmaPreprocessedPrompt(PreprocessedPrompt):
    attention_mask: Tensor


@dataclass
class PaliGemmaPreprocessedPromptWithImage(PreprocessedPromptWithImage):
    attention_mask: Tensor


class PaliGemmaPreprocessor(BasePreprocessor):
    def __init__(self, processor: PaliGemmaProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PaliGemmaPreprocessedPromptWithImage:
        # Make sure prompts and images are lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images

        # Use PIL to load images
        images = [
            Image.open(image) if isinstance(image, str) else image for image in images
        ]

        # Encode
        model_inputs = self.processor(text=prompts, images=images, return_tensors="pt", padding=True)
        input_lengths = (model_inputs["input_ids"] != self.tokenizer.pad_token_id).sum(
            dim=1
        )

        return PaliGemmaPreprocessedPromptWithImage(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            input_lengths=input_lengths,
            images=model_inputs["pixel_values"],
        )

    def preprocess_for_lm(
        self, prompts: Union[str, list[str]]
    ) -> PaliGemmaPreprocessedPrompt:
        raise RuntimeError("PaliGemma does not support language-only")


class PaliGemma(BaseVLM):
    def __init__(self, variant: str) -> None:
        self.variant = variant
        variant_config = model_configs["paligemma"][variant]
        model_path = variant_config["model_path"]
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            revision="bfloat16",
        )
        self.model = self.model.eval()

        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = self.processor.tokenizer

    def get_next_token_probabilities(
        self, prompt: PaliGemmaPreprocessedPromptWithImage
    ) -> Tensor:
        response = self.model.forward(
            input_ids=prompt.input_ids.to(self.model.device),
            attention_mask=prompt.attention_mask.to(self.model.device),
            pixel_values=prompt.images.to(self.model.device),
        )
        return torch.softmax(response.logits[:, -1, :], dim=-1)

    def get_next_token_probabilities_from_lm(
        self, prompt: PreprocessedPrompt
    ) -> Tensor:
        raise RuntimeError("PaliGemma does not support language-only")

    def get_preprocessor(self) -> BasePreprocessor:
        return PaliGemmaPreprocessor(self.processor)

    def get_llm_layers(self) -> ModuleList:
        return self.model.language_model.model.layers
    
    def get_embedding_size(self) -> int:
        return self.model.language_model.model.embed_tokens.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.language_model.model.embed_tokens.weight.mean(dim=0)
