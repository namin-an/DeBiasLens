import torch

from torch import Tensor
from typing import Union
from .base import BasePreprocessor
from transformers import AutoTokenizer
import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from utils_new.configs import model_configs
from .base import BaseVLM, PreprocessedPrompt
from .backbones.qwen.modeling_qwen import QWenLMHeadModel
from .backbones.qwen.qwen_generation_utils import make_context


class QwenPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, generation_config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPrompt:
        # Make sure prompts and images are lists and have the same length
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images
        assert len(prompts) == len(
            images
        ), "The number of prompts and images must be the same."

        # Process prompt-image pairs
        all_input_ids = []
        input_lengths = []

        for prompt, image in zip(prompts, images):
            query = self.tokenizer.from_list_format(
                [
                    {"image": image},
                    {"text": prompt},
                ]
            )

            generation_config = self.generation_config
            history = []

            max_window_size = generation_config.max_window_size
            _, context_tokens = make_context(
                tokenizer=self.tokenizer,
                query=query,
                assistant_prefix=None,
                history=history,
                system="",
                max_window_size=max_window_size,
                chat_format=generation_config.chat_format,
            )

            input_ids = torch.tensor(context_tokens, dtype=torch.long)
            all_input_ids.append(input_ids)
            input_lengths.append(len(context_tokens))

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=0
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        # Return
        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)

    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> Tensor:
        # Make sure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Get tokenizer
        tokenizer = self.tokenizer
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            role_encoded = tokenizer.encode(
                role, add_special_tokens=set(tokenizer.IMAGE_ST)
            )
            content_encoded = tokenizer.encode(
                content, add_special_tokens=set(tokenizer.IMAGE_ST)
            )
            return role_encoded + nl_tokens + content_encoded

        # Process prompts
        all_input_ids = []
        input_lengths = []
        for prompt in prompts:
            system_tokens = (
                im_start_tokens
                + _tokenize_str("system", "")
                + im_end_tokens
                + nl_tokens
            )
            query_tokens = (
                im_start_tokens + _tokenize_str("user", prompt) + im_end_tokens
            )
            response_tokens_part = _tokenize_str("assistant", "")
            response_tokens = im_start_tokens + response_tokens_part
            context_tokens = system_tokens + query_tokens + nl_tokens + response_tokens
            input_ids = torch.tensor(context_tokens, dtype=torch.long)
            all_input_ids.append(input_ids)
            input_lengths.append(len(context_tokens))

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=0
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)


class Qwen(BaseVLM):
    def __init__(self, variant: str) -> None:
        # Select configuaration
        variant_config = model_configs["qwen"][variant]

        self.model = QWenLMHeadModel.from_pretrained(
            **variant_config, device_map="cuda", trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            **variant_config, trust_remote_code=True,
        )
        self.model.generation_config.top_k = 50

    def get_next_token_probabilities(self, prompt: PreprocessedPrompt) -> Tensor:
        # Get input ids and lengths
        input_ids = prompt.input_ids.to(self.model.device)
        input_lengths = prompt.input_lengths

        # Forward pass
        output = self.model.forward(input_ids).logits

        # Select the last token probabilities
        output = output[torch.arange(output.size(0)), input_lengths - 1, :]
        # Apply softmax
        output = torch.softmax(output, dim=-1)

        return output

    def get_next_token_probabilities_from_lm(self, prompt: PreprocessedPrompt) -> Tensor:
        return self.get_next_token_probabilities(prompt)

    def get_preprocessor(self) -> QwenPreprocessor:
        return QwenPreprocessor(self.tokenizer, self.model.generation_config)

    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.transformer.h
    
    def get_embedding_size(self) -> int:
        return self.model.transformer.wte.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.transformer.wte.weight.mean(dim=0)
