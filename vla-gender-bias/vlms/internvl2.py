import torch

from PIL import Image
from math import floor
from torch import Tensor
from .base import BaseVLM
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Optional
from .base import BasePreprocessor
import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from utils_new.configs import model_configs
from accelerate import init_empty_weights
from accelerate import infer_auto_device_map
from accelerate import load_checkpoint_and_dispatch
from .backbones.internvl2.utils import load_image
from .backbones.internvl2.conversation import get_conv_template
from .base import PreprocessedPrompt, PreprocessedPromptWithImage
from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer as InternVL1BTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer as InternVL4BTokenizer
from transformers.models.llama.tokenization_llama import LlamaTokenizer as InternVL40BTokenizer
# from .backbones.internvl2.internvl2_1b.modeling_internvl_chat import InternVLChatModel as InternVL1B
# from .backbones.internvl2.internvl2_2b.modeling_internvl_chat import InternVLChatModel as InternVL2B
# from .backbones.internvl2.internvl2_4b.modeling_internvl_chat import InternVLChatModel as InternVL4B
from .backbones.internvl2.internvl2_8b.modeling_internvl_chat import InternVLChatModel as InternVL8B
# from .backbones.internvl2.internvl2_26b.modeling_internvl_chat import InternVLChatModel as InternVL26B
# from .backbones.internvl2.internvl2_40b.modeling_internvl_chat import InternVLChatModel as InternVL40B
# from .backbones.internvl2.internvl2_2b.tokenization_internlm2 import InternLM2Tokenizer as InternVL2BTokenizer
from .backbones.internvl2.internvl2_8b.tokenization_internlm2 import InternLM2Tokenizer as InternVL8BTokenizer
# from .backbones.internvl2.internvl2_26b.tokenization_internlm2 import InternLM2Tokenizer as InternVL26BTokenizer

import sys
# sys.path.append('/workspace/cvml_user/namin/bias_vlm/sae-for-vlm')
# from models.llava import SAEWrapper
import torch.nn as nn
from typing import Optional, Tuple
import copy


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"


variant_to_cls_mapping = {
    # "internvl2-1b": (InternVL1B, InternVL1BTokenizer),
    # "internvl2-2b": (InternVL2B, InternVL2BTokenizer),
    # "internvl2-4b": (InternVL4B, InternVL4BTokenizer),
    "internvl2-8b": (InternVL8B, InternVL8BTokenizer),
    # "internvl2-26b": (InternVL26B, InternVL26BTokenizer),
    # "internvl2-40b": (InternVL40B, InternVL40BTokenizer),
}


variant_to_memory_mapping = {
    "internvl2-1b": 3,
    "internvl2-2b": 6,
    "internvl2-4b": 9,
    "internvl2-8b": 17,
    "internvl2-26b": 55,
    "internvl2-40b": 80,
}


class SAEWrapper(nn.Module):

    def __init__(self, sae, neurons_to_fix, pre_zero):
        super().__init__()
        self.sae = sae
        self.neurons_to_fix = neurons_to_fix
        self.pre_zero = pre_zero

    def encode(self, x):
        x = self.sae.encode(x) #, use_threshold=False)
        if self.pre_zero:
            x = torch.zeros_like(x)
        
        for neuron_id, value in self.neurons_to_fix.items():
            print(x[:, :, neuron_id])
            x[:, :, neuron_id] = value
        print(hi)
        return x
    

    def decode(self, x):
        x = self.sae.decode(x)
        x = x.to(dtype=torch.float16)
        return x




@dataclass
class InternVLPreprocessedPrompt(PreprocessedPrompt):
    attention_mask: Tensor


@dataclass
class InternVLPreprocessedPromptWithImage(PreprocessedPromptWithImage):
    attention_mask: Optional[Tensor] = None
    num_patches_list: Optional[list[int]] = None


class InternVL2Preprocessor(BasePreprocessor):
    def __init__(self, tokenizer, img_context_token_id: int, template, system_message, num_image_token) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.img_context_token_id = img_context_token_id
        self.template = template
        self.system_message = system_message
        self.num_image_token = num_image_token

    def preprocess(
        self, prompts: Union[str, list[str]], images: Union[str, list[str]]
    ) -> InternVLPreprocessedPromptWithImage:
        # Make prompts and images lists
        prompts = [prompts] if isinstance(prompts, str) else prompts
        images = [images] if isinstance(images, str) else images

        # Load images
        images = [load_image(image, max_num=12) for image in images]

        # Make `num_patches_list`
        num_patches_list = [image.size(0) for image in images]

        # Concatenate images
        images = torch.cat(images, dim=0)

        # Process prompts
        processed_prompts = []
        for idx, num_patches in enumerate(num_patches_list):
            prompt = prompts[idx]
            if "<image>" not in prompt:
                prompt = "<image>\n" + prompt

            template = get_conv_template(self.template)
            template = deepcopy(template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            prompt = prompt.replace("<image>", image_tokens, 1)
            processed_prompts.append(prompt)

        # Tokenize prompts
        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(
            processed_prompts, return_tensors="pt", padding=True
        )
        input_ids = model_inputs["input_ids"]

        # Return
        return InternVLPreprocessedPromptWithImage(
            input_ids=input_ids,
            input_lengths=None,
            images=images,
            attention_mask=model_inputs.attention_mask,
            num_patches_list=num_patches_list,
        )

    def preprocess_for_lm(
        self, prompts: Union[str, list[str]]
    ) -> InternVLPreprocessedPrompt:
        # Make sure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts

        # Process prompts
        processed_prompts = []
        for prompt in prompts:
            template = get_conv_template(self.model.template)
            template.system_message = self.model.system_message
            template.append_message(template.roles[0], prompt)
            template.append_message(template.roles[1], None)
            prompt = template.get_prompt()
            processed_prompts.append(prompt)

        self.tokenizer.padding_side = "left"
        model_inputs = self.tokenizer(
            processed_prompts, return_tensors="pt", padding=True
        )
        input_ids = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        return InternVLPreprocessedPrompt(
            input_ids=input_ids,
            input_lengths=None,
            attention_mask=attention_mask,
        )

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as T

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image2(image_file, input_size=448, max_num=6, upscale=False):
    image = Image.open(image_file).convert('RGB')
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def load_image3(image, input_size=448, max_num=6, upscale=False):
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def listinstr(lst, s):
    assert isinstance(lst, list)
    for item in lst:
        if item in s:
            return True
    return False

def reorganize_prompt(message, image_num, dataset=None):
    if dataset is not None and listinstr(['MUIRBench'], dataset):
        prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        images_to_remove = ' '.join(['<image>'] * image_num)
        prompt = prompt.replace(images_to_remove, '')
        for i in range(image_num):
            prompt = prompt.replace('<image>', f'<Image-{i + 1}>', 1)
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
    elif dataset is not None and listinstr(["bmmr"], dataset.lower()):
        if image_num == 1:
            prompt = "\n".join([x["value"] for x in message if x["type"] == "text"])
        else:
            prompt, image_idx = "", 1
            for x in message:
                if x["type"] == "text":
                    prompt += x["value"]
                elif x["type"] == "image":
                    image_idx += 1
    elif image_num == 1:
        prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
    else:
        prompt, image_idx = '', 1
        for x in message:
            if x['type'] == 'text':
                prompt += x['value']
            elif x['type'] == 'image':
                prompt += f'<Image-{image_idx}>'
                image_idx += 1
        prompt = ''.join([f'Image-{i + 1}: <image>\n' for i in range(image_num)]) + prompt
        images_to_remove = ''.join([f'<Image-{i + 1}>' for i in range(image_num)])
        prompt = prompt.replace(images_to_remove, '')
    return prompt

class InternVL2(BaseVLM):
    def __init__(self, variant: str):
        # Get path to model
        model_path = model_configs["internvl2"][variant]["model_path"]

        # Get model class
        model_cls, tokenizer_cls = variant_to_cls_mapping[variant]

        # Load model
        with init_empty_weights():
            model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch.float16, #torch.bfloat16, 
            )

            # Determine how much memory to use on each device
            num_gpus = torch.cuda.device_count()
            required_memory = variant_to_memory_mapping[variant]
            memory_per_gpu = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_gpus)
            ]
            memory_per_gpu = [floor(memory / 1024**3) for memory in memory_per_gpu]
            total_available_memory = sum(memory_per_gpu)
            memory_per_gpu = [
                min(memory / total_available_memory * required_memory, memory)  # Can't use more memory than available
                for memory in memory_per_gpu
            ]
            max_memory = {i: f"{memory}GiB" for i, memory in enumerate(memory_per_gpu)}
            # max_memory["cpu"] = "0GiB"

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=[
                    "LlamaDecoderLayer",
                    "InternVisionEncoderLayer",
                    "InternLM2DecoderLayer",
                ],
            )

        model = load_checkpoint_and_dispatch(
            model,
            model_path,
            device_map=device_map,
            dtype=torch.bfloat16,   # 🔥 CHANGE THIS
            offload_folder="./offload",   # <-- ADD THIS
            offload_state_dict=True       # optional but good practice
        )
        self.model = model.eval()

        self.tokenizer = tokenizer_cls.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self.layer = 23
        self.base_InternViTEncoderLayerPostMlpResidual = copy.deepcopy(
            self.model.vision_model.encoder.layers[self.layer]
        )
        print(self.model)
        print('namin')


    def get_next_token_probabilities(self, prompt: InternVLPreprocessedPromptWithImage) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        images = prompt.images.to(self.model.device, dtype=torch.bfloat16)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Set img_context_token_id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # Forward pass
        vit_embeds = self.model.extract_feature(images.to(torch.float16))
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = torch.eq(input_ids, self.model.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)
        logits = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    def get_next_token_probabilities_from_lm(self, prompt: InternVLPreprocessedPrompt) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Forward pass
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        logits = self.model.language_model.forward(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
        ).logits

        # Extract logits of last timestep and apply softmax
        next_token_logits = logits[:, -1, :]
        next_token_probabilities = torch.softmax(next_token_logits, dim=-1)

        return next_token_probabilities

    def get_preprocessor(self) -> InternVL2Preprocessor:
        return InternVL2Preprocessor(
            tokenizer=self.tokenizer,
            img_context_token_id=self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
            template=self.model.template,
            system_message=self.model.system_message,
            num_image_token=self.model.num_image_token,
        )

    def get_llm_layers(self) -> torch.nn.ModuleList:
        return self.model.language_model.model.layers

    def get_embedding_size(self) -> int:
        return self.model.language_model.model.tok_embeddings.embedding_dim
    
    def get_avg_embedding(self) -> Tensor:
        return self.model.language_model.model.tok_embeddings.weight.mean(dim=0)

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False, alpha=0.6):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self.model.vision_model.encoder.layers[self.layer] = InternViTEncoderLayerPostMlpResidual_fix(
            self.base_InternViTEncoderLayerPostMlpResidual,
            modified_sae,
            alpha=alpha
        )
        #print(self.model)

    def generate(self, message, dataset=None):
        self.set_max_num(dataset)

        image_num = len([x for x in message if x['type'] == 'image'])
        max_num = max(1, min(self.max_num, self.total_max_num // image_num))
        prompt = reorganize_prompt(message, image_num, dataset=dataset)

        image_path = [x['value'] for x in message if x['type'] == 'image'][0]
        upscale_flag = dataset is not None and listinstr(['MMMU'], dataset)
        pixel_values = load_image2(
            image_path, max_num=max_num, upscale=upscale_flag).to(self.device).to(self.model.dtype)
        num_patches_list = [pixel_values.size(0)]

        with torch.inference_mode():
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = 0 > 0 or kwargs_default.get('do_sample', False)
            kwargs_default['temperature'] = 0.6
            kwargs_default['top_p'] = 0.95
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=kwargs_default,
                verbose=0 == 0,
            )

        return response

    def generate_2(self, pil_image, prompt, max_new_tokens=1024, temperature=0.0, do_sample=False):

        # print('Prompt: ', prompt, flush=True)
        pixel_values = load_image3(pil_image).to(self.device).to(self.model.dtype)
        num_patches_list = [pixel_values.size(0)]

        with torch.inference_mode():
            kwargs_default = self.kwargs.copy()
            kwargs_default['do_sample'] = do_sample
            kwargs_default['temperature'] = temperature
            kwargs_default['top_p'] = 0.95
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=prompt,
                generation_config=kwargs_default,
                verbose=0 == 0,
            )

        return response


class InternViTEncoderLayerPostMlpResidual_fix(nn.Module):
    def __init__(self, base, sae, alpha=0.6):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.attn
        self.layer_norm1 = base.norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.norm2

        self.sae = sae
        self.alpha = alpha

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        encoded_hidden_states = self.sae.encode(hidden_states)
  
        decoded_hidden_states = self.sae.decode(encoded_hidden_states)
        hidden_states = (1-self.alpha)*hidden_states + self.alpha*decoded_hidden_states 
        # print(self.alpha, hidden_states[0][651])

        return hidden_states