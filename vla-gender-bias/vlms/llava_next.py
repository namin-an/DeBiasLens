import re
import torch

from PIL import Image
from typing import Union
from torch import Tensor
from .base import BasePreprocessor
import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from utils_new.configs import model_configs
from .backbones.llava.mm_utils import process_images
from .backbones.llava_next.conversation import Conversation
from .backbones.llava_next.conversation import SeparatorStyle
from .backbones.llava_next.constants import IMAGE_TOKEN_INDEX
from .backbones.llava_next.constants import IMAGE_PLACEHOLDER
from .backbones.llava_next.constants import DEFAULT_IMAGE_TOKEN
from .backbones.llava_next.constants import DEFAULT_IM_END_TOKEN
from .backbones.llava_next.mm_utils import tokenizer_image_token
from .backbones.llava_next.constants import DEFAULT_IM_START_TOKEN
from .backbones.llava_next.model.builder import load_pretrained_model
from .base import LLaVATypeVLM, PreprocessedPrompt, PreprocessedPromptWithImage


conv_mistral_instruct = Conversation(
    system="",
    roles=("USER", "ASSISTANT"),
    version="llama_v2",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.LLAMA_2,
    sep="",
    sep2="</s>",
)

conv_chatml_direct = Conversation(
    system="""<|im_start|>system
Answer the questions.""",
    roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
    version="mpt",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.MPT,
    sep="<|im_end|>",
)

conv_llava_v0 = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=(
    ),
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

class LLaVANextPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, image_processor, config, conv_template, mm_use_im_start_end) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config
        self.conv_template = conv_template
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

        conv = self.conv_template.copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )

        return input_ids.squeeze(0)

    def preprocess(
        self,
        prompts: Union[str, list[str]],
        images: Union[str, list[str]],
    ) -> PreprocessedPromptWithImage:
        # Process prompt
        prompts = prompts if isinstance(prompts, list) else [prompts]
        input_ids = [self._process_prompt(prompt) for prompt in prompts]

        # Get lengths
        input_lengths = [len(input_id) for input_id in input_ids]
        input_lengths = torch.tensor(input_lengths).long()

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        # Process Image
        images = images if isinstance(images, list) else [images]
        images = [Image.open(image_file).convert("RGB") for image_file in images]
        images_tensor = process_images(images, self.image_processor, self.config)

        # Return Preprocessed Prompt
        return PreprocessedPromptWithImage(
            input_ids=input_ids,
            input_lengths=input_lengths,
            images=images_tensor,
        )

    def preprocess_for_lm(self, prompts: Union[str, list[str]]) -> PreprocessedPrompt:
        # Process prompt
        prompts = prompts if isinstance(prompts, list) else [prompts]
        all_input_ids = []
        for prompt in prompts:
            conv = self.conv_template.copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt()

            # Tokenize Prompt
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            all_input_ids.append(input_ids.squeeze(0))

        input_ids = all_input_ids

        # Get lengths
        input_lengths = [len(input_id) for input_id in input_ids]
        input_lengths = torch.tensor(input_lengths).long()

        # Stack input_ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )

        return PreprocessedPrompt(input_ids=input_ids, input_lengths=input_lengths)


class LLaVANext(LLaVATypeVLM):
    def __init__(self, variant: str) -> None:
        super().__init__()
        self.variant = variant

        # Select configuration
        variant_config = model_configs["llava-next"][variant]

        tokenizer, model, image_processor, context_len = load_pretrained_model(
            **variant_config
        )

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.context_len = context_len

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.mm_use_im_start_end = model.config.mm_use_im_start_end

        # Select conversation template
        if "mistral" in variant_config["model_name"].lower():
            self.conv_template = conv_mistral_instruct
        elif "34b" in variant_config["model_name"].lower():
            self.conv_template = conv_chatml_direct
        else:
            self.conv_template = conv_llava_v0

    def get_preprocessor(self) -> LLaVANextPreprocessor:
        return LLaVANextPreprocessor(
            self.tokenizer, self.image_processor, self.model.config, self.conv_template, self.mm_use_im_start_end
        )
            
