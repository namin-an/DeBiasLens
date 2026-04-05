from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, CLIPImageProcessor
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple
import copy
from PIL import Image
from abc import abstractproperty

import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from vlms.base import BasePreprocessor
from vlms.llava import LLaVAMyPreprocessor
from vlms.backbones.llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
from vlms.backbones.llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

@dataclass
class PreprocessedPrompt:
    input_ids: Tensor
    input_lengths: Tensor

@dataclass
class PreprocessedPromptWithImage(PreprocessedPrompt):
    images: Tensor

class Llava:
    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", device='cuda', **kwargs):
        self.device = device
        self.layer = 23
        self.model = LlavaForConditionalGeneration.from_pretrained(model_path,
                                                                   torch_dtype=torch.float16,
                                                                   device_map=self.device)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.base_CLIPEncoderLayerPostMlpResidual = copy.deepcopy(
            self.model.vision_tower.vision_model.encoder.layers[self.layer]
        )

        self.tokenizer = self.processor.tokenizer

        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"


    def prompt(self, text, image, max_tokens=5):
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=[image], text=[prompt],
                                padding=True, return_tensors="pt").to(self.model.device, torch.float16)

        with torch.no_grad():
            res = self.model.generate(**inputs, max_new_tokens=max_tokens, 
                                      return_dict_in_generate=True, output_hidden_states=True)

        output = self.processor.batch_decode(res.sequences, skip_special_tokens=True)
        output = [x.split('ASSISTANT: ')[-1] for x in output][0]

        emb = torch.cat([step[-1] for step in res.hidden_states], dim=1).squeeze(0).detach().cpu()  # [gen_len, 4096]
        emb = emb.mean(0)
        
        return output, emb

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False, alpha=0.6):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        self.model.vision_tower.vision_model.encoder.layers[self.layer] = CLIPEncoderLayerPostMlpResidual(
            self.base_CLIPEncoderLayerPostMlpResidual,
            modified_sae,
            alpha=alpha
        )
    
    def get_preprocessor(self) -> BasePreprocessor:
        return LLaVAMyPreprocessor(
            self.processor
        )

    def get_next_token_probabilities(
        self, inputs
        ) -> Tensor:
        
        inputs = inputs.to(device=self.model.device) # torch.Size([32, 623])
        # Run inferece
        with torch.no_grad():
            outputs = self.model(**inputs).logits # torch.Size([32, 623, 32064])
        #     res = self.model.generate(**inputs, 
        #                               return_dict_in_generate=True, output_hidden_states=True)

        # output = self.processor.batch_decode(res.sequences, skip_special_tokens=True) # torch.Size([32, 627])
        # output = [x.split('ASSISTANT: ')[-1] for x in output]

        # Extract next token probabilities
        batch_size = outputs.shape[0]
        logits = outputs[torch.arange(batch_size), -1]
        logits = torch.softmax(logits, dim=-1)
        return logits

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

    # def generate(self, message):

    #     # Support interleave text and image
    #     content, images = self.concat_tilist(message)

    #     images = [Image.open(s).convert("RGB") for s in images]
        
    #     prompt = self.system_prompt + "USER: " + content + " ASSISTANT: "

    #     inputs = self.processor(images=images, text=[prompt],
    #                             padding=True, return_tensors="pt").to(self.model.device, torch.float16)

    #     with torch.inference_mode():
    #         res = self.model.generate(**inputs, max_new_tokens=2048, 
    #                                   return_dict_in_generate=True, output_hidden_states=True)

    #     output = self.processor.batch_decode(res.sequences, skip_special_tokens=True)
    #     output = [x.split('ASSISTANT: ')[-1] for x in output][0].strip()

    #     return output

    def generate(self, inputs, max_new_tokens=1024, temperature=0.0, do_sample=False, pad_token_id=None):

        with torch.inference_mode():
            res = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      do_sample=do_sample,
                                      pad_token_id=pad_token_id)
                                    #   return_dict_in_generate=True, output_hidden_states=True)

        # output = self.processor.batch_decode(res.sequences, skip_special_tokens=True)
        # output = [x.split('ASSISTANT: ')[-1] for x in output][0].strip()

        return res #output
    

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




class CLIPEncoderLayerPostMlpResidual(nn.Module):

    def __init__(self, base, sae, alpha=0.6):
        super().__init__()

        """
        base

        0-21): 22 x CLIPEncoderLayer(
            (self_attn): CLIPSdpaAttention(
              (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
              (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
            )
            (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
            (mlp): CLIPMLP(
              (activation_fn): QuickGELUActivation()
              (fc1): Linear(in_features=1024, out_features=4096, bias=True)
              (fc2): Linear(in_features=4096, out_features=1024, bias=True)
            )
            (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        """

        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2
        
        self.sae = sae
        self.alpha = alpha

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = self.mlp(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = residual + hidden_states # (batch_size, 577, 1024)
        encoded_hidden_states = self.sae.encode(hidden_states)
        decoded_hidden_states = self.sae.decode(encoded_hidden_states)
        hidden_states = (1-self.alpha)*hidden_states + self.alpha*decoded_hidden_states 
        print(self.alpha, hidden_states[0][0])
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
    


class CLIPEncoderLayerPostMlpResidual_notfromLLaVA(nn.Module):

    def __init__(self, base, sae, attn_mask: torch.Tensor = None, alpha=0.6):
        super().__init__()
        
        """
        base

        ResidualAttentionBlock(
        (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
        )
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
            (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
        )
        (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        """
        
        self.embed_dim = base.attn.embed_dim # 1024
        self.self_attn = base.attn
        self.layer_norm1 = base.ln_1
        self.mlp = base.mlp
        self.layer_norm2 = base.ln_2
        self.attn_mask = attn_mask
        self.sae = sae
        self.alpha = alpha

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        causal_attention_mask: torch.Tensor = None,
        output_attentions: Optional[bool] = False
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = self.mlp(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = residual + hidden_states # (batch_size, 577, 1024)
        encoded_hidden_states = self.sae.encode(hidden_states) # (batch_size, 197, 1024)
        # print(encoded_hidden_states.shape) # 577, 32, 8192
        decoded_hidden_states = self.sae.decode(encoded_hidden_states)
        # # print(encoded_hidden_states.shape, decoded_hidden_states.shape) # torch.Size([197, 32, 6144]) torch.Size([197, 32, 768])
        hidden_states = (1-self.alpha)*hidden_states + self.alpha*decoded_hidden_states 
        return hidden_states


class CLIPEncoderLayerPostMlpResidual_notfromLLaVA_saeembs(nn.Module):

    def __init__(self, base, sae, attn_mask: torch.Tensor = None):
        super().__init__()
        
        """
        base

        ResidualAttentionBlock(
        (attn): MultiheadAttention(
            (out_proj): NonDynamicallyQuantizableLinear(in_features=1024, out_features=1024, bias=True)
        )
        (ln_1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (mlp): Sequential(
            (c_fc): Linear(in_features=1024, out_features=4096, bias=True)
            (gelu): QuickGELU()
            (c_proj): Linear(in_features=4096, out_features=1024, bias=True)
        )
        (ln_2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
        """
        
        self.embed_dim = base.attn.embed_dim # 1024
        self.self_attn = base.attn
        self.layer_norm1 = base.ln_1
        self.mlp = base.mlp
        self.layer_norm2 = base.ln_2
        self.attn_mask = attn_mask
        self.sae = sae

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.self_attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.FloatTensor]:

        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = self.mlp(hidden_states) # torch.Size([1, 577, 1024])
        hidden_states = residual + hidden_states # (batch_size, 577, 1024)
        encoded_hidden_states = self.sae.encode(hidden_states) # (batch_size, 197, 1024)dden_states #decoded_hidden_states # torch.Size([577, 64, 1024])
        return encoded_hidden_states 

  