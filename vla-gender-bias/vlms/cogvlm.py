import torch

from PIL import Image
from math import floor
from torch import Tensor
from typing import Union
from .base import BaseVLM
from typing import Optional
from dataclasses import dataclass
from torchvision import transforms
from .base import BasePreprocessor
from utils.configs import model_configs
from vlms.base import PreprocessedPrompt
from accelerate import init_empty_weights
from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM, AutoTokenizer


text_only_template = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: {} ASSISTANT:"

LANGUAGE_TOKEN_TYPE = 0
VISION_TOKEN_TYPE = 1


@dataclass
class CogVLMPreprocessedPrompt(PreprocessedPrompt):
    attention_mask: Tensor
    token_type_ids: Tensor
    images: Optional[Tensor] = None


class CogVLMPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer: AutoTokenizer, config) -> None:
        self.tokenizer = tokenizer
        self.config = config
        self.tokenizer.pad_token_id = 128002

    def preprocess(
        self,
        prompts: Union[str, list[str]],
        images: Optional[Union[str, list[str]]] = None,
    ) -> CogVLMPreprocessedPrompt:
        # Make sure prompts and images are lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if images is not None:
            images = [images] if isinstance(images, str) else images

        # Process Images
        if images is not None:
            images = [Image.open(image).convert("RGB") for image in images]

            # Build Image Transform
            image_size = self.config.vision_config["image_size"]
            patch_size: int = self.config.vision_config["patch_size"]
            vision_token_num = (image_size // patch_size // 2) * (
                image_size // patch_size // 2
            ) + 2

            transform = transforms.Compose(
                [
                    transforms.Resize(
                        (image_size, image_size),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.48145466, 0.4578275, 0.40821073),
                        (0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

            images = [transform(image) for image in images]

        # Process prompts
        self.tokenizer.pad_token_id = 128002  # llama3 adapt for cogvlm

        all_input_ids = []
        attention_masks = []
        all_token_type_ids = []
        input_lengths = []

        for prompt in prompts:
            if images is not None:
                text = "Question: {} {}".format(prompt, "Short answer:")
            else:
                text = text_only_template.format(prompt)

            input_ids = [self.tokenizer.bos_token_id]
            token_type_ids = [LANGUAGE_TOKEN_TYPE]

            if images is not None:
                input_ids += [self.tokenizer.pad_token_id] * vision_token_num
                token_type_ids += [VISION_TOKEN_TYPE] * vision_token_num

            text_ids = self.tokenizer.encode(text, add_special_tokens=False)
            input_ids += text_ids
            token_type_ids += [LANGUAGE_TOKEN_TYPE] * len(text_ids)
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            attention_masks.append(torch.tensor(attention_mask, dtype=torch.bool))
            all_token_type_ids.append(torch.tensor(token_type_ids, dtype=torch.long))
            input_lengths.append(len(input_ids))

        # Stack inputs
        all_input_ids = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_masks = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=False
        )
        all_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            all_token_type_ids, batch_first=True, padding_value=LANGUAGE_TOKEN_TYPE
        )
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        return CogVLMPreprocessedPrompt(
            input_ids=all_input_ids,
            input_lengths=input_lengths,
            attention_mask=attention_masks,
            token_type_ids=all_token_type_ids,
            images=images,
        )

    def preprocess_for_lm(
        self, prompts: Union[str, list[str]]
    ) -> CogVLMPreprocessedPrompt:
        return self.preprocess(prompts, images=None)


class CogVLM(BaseVLM):
    def __init__(self, variant: str) -> None:
        # Get path to model
        model_path = model_configs["cogvlm"][variant]["model_path"]
        # Load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

            # Determine how much memory to use on each device
            num_gpus = torch.cuda.device_count()
            required_memory = 40
            memory_per_gpu = [torch.cuda.get_device_properties(i).total_memory for i in range(num_gpus)]
            memory_per_gpu = [floor(memory / 1024 ** 3) for memory in memory_per_gpu]
            total_available_memory = sum(memory_per_gpu)
            memory_per_gpu = [memory / total_available_memory * required_memory for memory in memory_per_gpu]
            max_memory = {i: f"{memory}GiB" for i, memory in enumerate(memory_per_gpu)}
            max_memory["cpu"] = "0GiB"

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=["CogVLMDecoderLayer", "TransformerLayer"],
            )

        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
        )
        self.model = model.eval()
    
    def get_next_token_probabilities(self, prompt: CogVLMPreprocessedPrompt) -> Tensor:
        # Extract inputs
        device = self.model.device
        input_ids = prompt.input_ids.to(device)
        attention_mask = prompt.attention_mask.to(device)
        token_type_ids = prompt.token_type_ids.to(device)
        if prompt.images is not None:
            images = [[image.to(device, dtype=torch.bfloat16)] for image in prompt.images]
        else:
            images = None

        # Encode images
        outputs = self.model.forward(
            input_ids = input_ids,
            images = images,
            token_type_ids = token_type_ids,
            attention_mask = attention_mask,
        ).logits

        # Extract next token probabilities
        batch_size = outputs.shape[0]
        last_token_positions = (
            outputs.shape[1] - input_ids.shape[1] + prompt.input_lengths - 1
        )
        logits = outputs[torch.arange(batch_size), last_token_positions]
        # Apply softmax
        logits = torch.softmax(logits, dim=-1)

        return logits
    
    def get_next_token_probabilities_from_lm(self, prompt: CogVLMPreprocessedPrompt) -> Tensor:
        return self.get_next_token_probabilities(prompt)
    
    def get_preprocessor(self) -> CogVLMPreprocessor:
        return CogVLMPreprocessor(self.tokenizer, self.model.config)
