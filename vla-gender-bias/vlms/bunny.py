import torch

from PIL import Image
from typing import Union
from .base import BasePreprocessor
from utils.configs import model_configs
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import LLaVATypeVLM, PreprocessedPrompt, PreprocessedPromptWithImage


class BunnyProcessor(BasePreprocessor):
    def __init__(self, tokenizer, image_processor, config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

    def _expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def _process_images(self, images, model_cfg):
        image_processor = self.image_processor
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self._expand2square(
                    image, tuple(int(x * 255) for x in image_processor.image_mean)
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPromptWithImage:
        # Make prompt and images into lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images
        # Process prompt
        processed_prompts = []
        for prompt in prompts:
            text = f"USER: <image>\n{prompt} ASSISTANT:"
            text_chunks = [
                self.tokenizer(chunk).input_ids for chunk in text.split("<image>")
            ]

            input_ids = torch.tensor(
                text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long
            )
            processed_prompts.append(input_ids)

        # Get lengths
        input_lengths = torch.tensor(
            [len(prompt) for prompt in processed_prompts], dtype=torch.long
        )

        # Stack the processed prompts
        input_ids = torch.nn.utils.rnn.pad_sequence(
            processed_prompts,
            batch_first=True,
            padding_value=self.tokenizer.eos_token_id,
        )

        # Process Image
        images = [Image.open(image).convert("RGB") for image in images]
        images = self._process_images(images, self.config)

        return PreprocessedPromptWithImage(
            input_ids=input_ids, input_lengths=input_lengths, images=images
        )

    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> PreprocessedPrompt:
        # Make prompt into lis
        prompts = [prompts] if isinstance(prompts, str) else prompts
        # Process prompt
        processed_prompts = []
        for prompt in prompts:
            text = f"USER: {prompt} ASSISTANT:"
            processed_prompts.append(self.tokenizer(text).input_ids)

        # Get lengths
        input_lengths = torch.tensor(
            [len(prompt) for prompt in processed_prompts], dtype=torch.long
        )

        # Stack the processed prompts
        input_ids = [
            torch.tensor(prompt, dtype=torch.long) for prompt in processed_prompts
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)


class Bunny(LLaVATypeVLM):
    def __init__(self, variant: str) -> None:
        self.variant = variant
        variant_config = model_configs["bunny"][variant]
        variant_path = variant_config["model_path"]

        # Load Model
        self.model = AutoModelForCausalLM.from_pretrained(
            variant_path,
            torch_dtype=torch.float16, # float32 for cpu
            device_map='auto',
            trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            variant_path,
            trust_remote_code=True
        )

        # Load vision tower
        self.vision_tower = self.model.get_vision_tower()
        if not self.vision_tower.is_loaded:
            self.vision_tower.load_model()
            self.vision_tower.vision_tower.to(device=self.model.device)

    def get_preprocessor(self) -> BunnyProcessor:
        return BunnyProcessor(
            self.tokenizer, self.vision_tower.image_processor, self.model.config
        )
