import re
import torch

from PIL import Image
from typing import Union
from torch import Tensor
from peft import PeftModel
from .base import BasePreprocessor
import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from utils_new.configs import model_configs
from .backbones.llava.mm_utils import process_images
from .backbones.llava.utils import disable_torch_init
from .backbones.llava.conversation import Conversation
from .backbones.llava.conversation import SeparatorStyle
from .backbones.llava.constants import IMAGE_TOKEN_INDEX
from .backbones.llava.constants import IMAGE_PLACEHOLDER
from .backbones.llava.constants import DEFAULT_IMAGE_TOKEN
from .backbones.llava.constants import DEFAULT_IM_END_TOKEN
from .backbones.llava.mm_utils import tokenizer_image_token
from .backbones.llava.constants import DEFAULT_IM_START_TOKEN
from .backbones.llava.model.builder import load_pretrained_model
from .base import LLaVATypeVLM, PreprocessedPrompt, PreprocessedPromptWithImage
from .backbones.bakllava.model.builder import load_pretrained_model as load_bakllava_model
from .backbones.llava.model.build_llava_rlhf import load_pretrained_model as load_llava_rlhf_model


conv_template = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2=" ",
)


class LLaVAMyPreprocessor():
    def __init__(self, processor) -> None:
        super().__init__()
        self.processor = processor

    def preprocess(
        self,
        prompts: Union[str, list[str]],
        images: Union[str, list[str]],
    ) -> PreprocessedPromptWithImage:
        # 0) Normalize inputs
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        images  = [images]  if isinstance(images,  str) else list(images)

        # Broadcast a single image to all prompts, or enforce 1:1 mapping
        if len(images) == 1 and len(prompts) > 1:
            images = images * len(prompts)
        elif len(images) != len(prompts):
            raise ValueError(f"#images ({len(images)}) must be 1 or match #prompts ({len(prompts)}).")

        all_prompts = []
        for text_prompt in prompts:
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text_prompt},
                ],
            }]

            formatted_prompt = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True
            )
            all_prompts.append(formatted_prompt)

        pil_images = []
        for p in images:
            with Image.open(p) as img:
                pil_images.append(img.convert("RGB").copy())  # ensure file handle closes

        return self.processor(images=pil_images, text=all_prompts,
                                padding=True, return_tensors="pt").to(torch.float16)



class LLaVAPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, image_processor, config, mm_use_im_start_end) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.mm_use_im_start_end = mm_use_im_start_end

    def _process_prompt(self, prompt: str) -> Tensor:
        prompt = prompt.strip()

        image_token_se = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        if IMAGE_PLACEHOLDER in prompt:
            if self.mm_use_im_start_end:
                prompt = re.sub(IMAGE_PLACEHOLDER, image_token_se, prompt)
            else:
                prompt = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, prompt)
        else:
            if self.mm_use_im_start_end:
                prompt = image_token_se + "\n" + prompt
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

        conv = conv_template.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt().strip()

        # Tokenize Prompt
        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids

        return input_ids

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> PreprocessedPromptWithImage:
        # Process prompt
        prompts = prompts if isinstance(prompts, list) else [prompts]
        input_ids = [self._process_prompt(prompt) for prompt in prompts]

        # Get input lengths
        input_lengths = [len(ids) for ids in input_ids]
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Process Image
        images = images if isinstance(images, list) else [images]
        images = [Image.open(image).convert("RGB") for image in images]
        images = process_images(images, self.image_processor, self.config)
        return PreprocessedPromptWithImage(input_ids, input_lengths, images)

    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> PreprocessedPrompt:
        prompts = prompts if isinstance(prompts, list) else [prompts]

        # Process prompt
        processed_prompts = []
        for prompt in prompts:
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt().strip()

            # Tokenize Prompt
            input_ids = self.tokenizer(
                prompt, return_tensors="pt", padding=True
            ).input_ids
            processed_prompts.append(input_ids.squeeze(0))

        # Get input lengths
        input_lengths = [len(ids) for ids in processed_prompts]
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            processed_prompts,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        return PreprocessedPrompt(input_ids, input_lengths)


class LLaVA(LLaVATypeVLM):
    def __init__(self, variant: str) -> None:
        super().__init__()
        disable_torch_init()
        self.image_dtype = torch.float16

        # Select configuration
        variant_config = model_configs["llava"][variant]

        # Load Model
        if "rlhf" in variant:
            tokenizer, model, image_processor, context_len = load_llava_rlhf_model(
                model_name=variant_config["model_name"],
                model_path=variant_config["sft"],
                model_base=None,
                load_bf16=True,
            )
            model = PeftModel.from_pretrained(
                model,
                variant_config["lora"],
            )
            model = model.merge_and_unload()
            model.to(torch.float16)
        elif "bakllava" in variant:
            tokenizer, model, image_processor, context_len = load_bakllava_model(
                **variant_config
            )
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        else:
            tokenizer, model, image_processor, context_len = load_pretrained_model(
                **variant_config
            )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        self.mm_use_im_start_end = model.config.mm_use_im_start_end

    def get_preprocessor(self) -> BasePreprocessor:
        return LLaVAPreprocessor(
            self.tokenizer,
            self.image_processor,
            self.model.config,
            self.mm_use_im_start_end,
        )
