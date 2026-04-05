import torch

from torch import Tensor
from typing import Union
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class PreprocessedPrompt:
    input_ids: Tensor
    input_lengths: Tensor


@dataclass
class PreprocessedPromptWithImage(PreprocessedPrompt):
    images: Tensor


class BaseVLM(ABC):
    @abstractmethod
    def get_next_token_probabilities(self, prompt: PreprocessedPrompt) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_next_token_probabilities_from_lm(
        self, prompt: PreprocessedPrompt
    ) -> Tensor:
        raise NotImplementedError
    
    @abstractmethod
    def get_preprocessor(self) -> "BasePreprocessor":
        raise NotImplementedError
    
    @abstractmethod
    def get_llm_layers(self) -> torch.nn.ModuleList:
        raise NotImplementedError
    
    @abstractmethod
    def get_embedding_size(self) -> int:
        raise NotImplementedError
    
    @abstractmethod
    def get_avg_embedding(self) -> Tensor:
        raise NotImplementedError

class BasePreprocessor(ABC):
    @abstractmethod
    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPrompt:
        raise NotImplementedError

    @abstractmethod
    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> PreprocessedPrompt:
        raise NotImplementedError


class LLaVATypeVLM(BaseVLM):
    def get_next_token_probabilities(
        self, prompt: PreprocessedPromptWithImage
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)
        image = prompt.images.to(device=self.model.device, dtype=torch.float16)

        # Run inferece
        outputs = self.model(input_ids, images=image)
        outputs = outputs.logits


        # Extract next token probabilities
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)
        return logits

    def get_next_token_probabilities_from_lm(
        self, prompt: PreprocessedPrompt
    ) -> Tensor:
        # Extract input_ids and image from prompt
        input_ids = prompt.input_ids.to(device=self.model.device)

        # Run inferece
        outputs = self.model.model.forward(input_ids).last_hidden_state
        outputs = self.model.lm_head(outputs)

        # Extract next token probabilities
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)
        return logits
    
    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.model.layers
    
    def get_embedding_size(self) -> int:
        return self.model.model.embed_tokens.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.model.embed_tokens.weight.mean(dim=0)
