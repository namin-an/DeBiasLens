from base64 import encode
from transformers import AutoModel, CLIPImageProcessor, AutoTokenizer, AutoProcessor

from PIL import Image
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
import copy

import sys
# sys.path.append('[your_working_path]/DeBiasLens/sae-for-vlm')
# from models.llava import SAEWrapper

sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias/vlms')
from base import BaseVLM
from copy import deepcopy
from dataclasses import dataclass
from typing import Union, Optional
from base import BasePreprocessor
from backbones.internvl2.utils import load_image
from backbones.internvl2.conversation import get_conv_template
from backbones.internvl2.internvl2_8b.tokenization_internlm2 import InternLM2Tokenizer as InternVL8BTokenizer
from base import PreprocessedPrompt, PreprocessedPromptWithImage


IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"



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
            x[:, :, neuron_id] = value
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


class InternVL2:
    def __init__(self, model_path="OpenGVLab/InternVL2-8B", device='cuda', **kwargs):
        self.device = device
        self.layer = 23
        self.model = AutoModel.from_pretrained(model_path,
                                               torch_dtype=torch.float16,
                                               device_map=self.device,
                                               trust_remote_code=True)
        
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.base_InternViTEncoderLayerPostMlpResidual = copy.deepcopy(
            self.model.vision_model.encoder.layers[self.layer]
        )

        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        kwargs_default = dict(do_sample=False, max_new_tokens=4096, top_p=None)
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

        self.tokenizer = InternVL8BTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )
        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)



    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False, alpha=0.6):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self.model.vision_model.encoder.layers[self.layer] = InternViTEncoderLayerPostMlpResidual_fix(
            self.base_InternViTEncoderLayerPostMlpResidual,
            modified_sae,
            alpha=alpha
        )
    
    def get_preprocessor(self) -> InternVL2Preprocessor:
        return InternVL2Preprocessor(
            tokenizer=self.tokenizer,
            img_context_token_id=self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
            template=self.model.template,
            system_message=self.model.system_message,
            num_image_token=self.model.num_image_token,
        )

    def get_next_token_probabilities(self, prompt: InternVLPreprocessedPromptWithImage) -> Tensor:
        # Extract input_ids and image from prompts
        input_ids = prompt.input_ids.to(self.model.device)
        images = prompt.images.to(self.model.device, dtype=torch.bfloat16)
        attention_mask = prompt.attention_mask.to(self.model.device)

        # Set img_context_token_id
        img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = img_context_token_id

        # Forward pass
        images = images.to(dtype=self.model.dtype)
        vit_embeds = self.model.extract_feature(images)
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

    def set_dump_image(self, dump_image_func):
        self.dump_image_func = dump_image_func

    def dump_image(self, line, dataset):
        return self.dump_image_func(line)

    def concat_tilist(self, message):
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    def set_max_num(self, dataset):
        # The total limit on the number of images processed, set to avoid Out-of-Memory issues.
        self.total_max_num = 64
        if dataset is None:
            self.max_num = 6
            return None
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'VCR_EN', 'VCR_ZH', 'OCRVQA', 'BMMR']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST', 'DUDE', 'MMLongBench_DOC', 'SLIDEVQA']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 6

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


# OpenGVLab/InternViT-300M-448px
class InternViT:
    def __init__(self, model_name, device, layer=23, layer_txt=11):
        self.device = device
        self.dtype = torch.bfloat16
        self.model =  AutoModel.from_pretrained(f"OpenGVLab/{model_name}", trust_remote_code=True, torch_dtype=self.dtype).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(f'OpenGVLab/{model_name}')

        self.register = {}
        self.attach_methods = {
            'post_mlp_residual': self._attach_post_mlp_residual,
            'post_projection': self._attach_post_projection,
        }
        self.sae = None
        self.layer = 23

        # try:
        self.base_InternViTEncoderLayerPostMlpResidual = copy.deepcopy(
            self.model.encoder.layers[self.layer]
        )
        # except:
        #     print('SAE (image encoder) not properly initialized...')
        #     pass

    def encode(self, inputs, probe_text_enc):
        for hook in self.register.keys():
            self.register[hook] = []
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output

        # if self.sae is not None:
        #     pooled_output = self.sae.encode(pooled_output)
        #     self.register[f'post_projection_{self.layer}'].append(pooled_output.detach().cpu())
        #     pooled_output = self.sae.decode(pooled_output)
        # elif self.layer is not None:
        #     self.register[f'post_projection_{self.layer}'].append(pooled_output.detach().cpu())

        return pooled_output

    def encode_image(self, x, is_proj=True):
        hidden_states = self.model.embeddings(x)
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
            output_hidden_states=None,
            return_dict=None,
        )
        last_hidden_state = encoder_outputs.last_hidden_state # sae: (batch, 257, 1024)
        if is_proj:
            pooled_output = last_hidden_state[:, 0, :]
        else:
            pooled_output = last_hidden_state
        return pooled_output

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae)
            self.register[f'{attachment_point}_{layer}'] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.encoder.layers[layer] = InternViTEncoderLayerPostMlpResidual(
            self.model.encoder.layers[layer],
            sae,
            layer,
            self.register,
        )

    def _attach_post_projection(self, layer, sae):
        self.sae = sae
        self.layer = layer

    def attach_and_fix_saeembs(self, sae, neurons_to_fix={}, pre_zero=False, is_image=True):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        if is_image:
            self.model.encoder.layers[self.layer] = InternViTEncoderLayerPostMlpResidual_saeembs(
                self.base_InternViTEncoderLayerPostMlpResidual,
                modified_sae,
            )
        else:
            pass
    

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False, alpha=0.6, is_image=True):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        if is_image:
            self.model.encoder.layers[self.layer] = InternViTEncoderLayerPostMlpResidual_fix(
                    self.base_InternViTEncoderLayerPostMlpResidual,
                    modified_sae,
                    alpha=alpha
                )
        else:
            pass


class InternViTEncoderLayerPostMlpResidual(nn.Module):
    def __init__(self, base, sae, layer, register):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.attn
        self.layer_norm1 = base.norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.norm2

        self.sae = sae
        self.layer = layer
        self.register = register

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

        if self.sae is not None:
            hidden_states = self.sae.encode(hidden_states)
            self.register[f'post_mlp_residual_{self.layer}'].append(hidden_states.detach().cpu())
            hidden_states = self.sae.decode(hidden_states)
        else:
            self.register[f'post_mlp_residual_{self.layer}'].append(hidden_states.detach().cpu())

        return hidden_states


class InternViTEncoderLayerPostMlpResidual_saeembs(nn.Module):
    def __init__(self, base, sae):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.attn
        self.layer_norm1 = base.norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.norm2

        self.sae = sae

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
  
        return encoded_hidden_states


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

        return hidden_states