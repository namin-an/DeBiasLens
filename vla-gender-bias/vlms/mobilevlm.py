import torch

from PIL import Image
from typing import Union
from .base import BasePreprocessor
from utils.configs import model_configs
from .backbones.mobilevlm.utils import process_images
from .backbones.mobilevlm.utils import disable_torch_init
from .backbones.mobilevlm.conversation import Conversation
from .backbones.mobilevlm.utils import tokenizer_image_token
from .backbones.mobilevlm.conversation import  SeparatorStyle
from .backbones.mobilevlm.model.mobilevlm import load_pretrained_model
from .base import LLaVATypeVLM, PreprocessedPrompt, PreprocessedPromptWithImage
from .backbones.mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


conv_template = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=tuple(),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)


class MobileVLMPreprocessor(BasePreprocessor):
    def __init__(self, tokenizer, image_processor, config) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.config = config

    def preprocess(
        self,
        prompts: Union[str, list[str]],
        images: Union[str, list[str]],
    ) -> PreprocessedPromptWithImage:
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Process the prompts
        processed_prompts = []
        for prompt in prompts:
            conv = conv_template.copy()
            conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt)
            conv.append_message(conv.roles[1], "")
            prompt = conv.get_prompt().strip()

            # Tokenize the prompt
            input_ids = tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            processed_prompts.append(input_ids)

        # Get lengths
        input_lengths = [len(input_ids) for input_ids in processed_prompts]
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)

        # Stack input ids
        input_ids = torch.nn.utils.rnn.pad_sequence(
            processed_prompts,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )

        # Process the image
        images = [images] if isinstance(images, str) else images
        images = [Image.open(image).convert("RGB") for image in images]
        images = process_images(images, self.image_processor, self.config)

        return PreprocessedPromptWithImage(
            input_ids=input_ids,
            input_lengths=input_lengths,
            images=images,
        )

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


class MobileVLM(LLaVATypeVLM):
    def __init__(self, variant: str):
        # Select configuration
        variant_config = model_configs["mobilevlm"][variant]

        disable_torch_init()
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            **variant_config, load_4bit=False, load_8bit=False
        )
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.context_len = context_len
    
    def get_preprocessor(self) -> MobileVLMPreprocessor:
        return MobileVLMPreprocessor(self.tokenizer, self.image_processor, self.model.config)
