from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Tuple
import copy
from PIL import Image
from abc import abstractproperty

import sys
sys.path.append('/workspace/cvml_user/namin/bias_vlm/vla-gender-bias')
from vlms.base import BasePreprocessor
from vlms.llava_next import LLaVANextPreprocessor
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

class LlavaOneVision:
    def __init__(self, model_name="llava-onevision-qwen2-7b-ov-hf", device='cuda', layer=25, layer_txt=11, is_vision=True):
        self.device = device
        
        self.dtype = torch.float32 # for saving activations after training #torch.bfloat16
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(f'llava-hf/{model_name}',
                                                                   torch_dtype=self.dtype,
                                                                   device_map=self.device)
        self.layer = layer
        self.is_vision = is_vision
        if self.is_vision:
            self.model = self.model.vision_tower.vision_model
            self.model.dtype = self.dtype
            self.processor = AutoProcessor.from_pretrained(f"google/siglip-so400m-patch14-384")

            self.register = {}
            self.attach_methods = {
                'post_mlp_residual': self._attach_post_mlp_residual,
                'post_projection': self._attach_post_projection,
            }
            self.sae = None

            self.base_SIGLIPEncoderLayerPostMlpResidual = copy.deepcopy(
                self.model.encoder.layers[self.layer]
            )
            self.full_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384").to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained("google/siglip-so400m-patch14-384")
          
        else:
            self.processor = AutoProcessor.from_pretrained(f'llava-hf/{model_name}')
            self.base_SIGLIPEncoderLayerPostMlpResidual = copy.deepcopy(
                    self.model.vision_tower.vision_model.encoder.layers[self.layer]
                )

    def encode(self, inputs, probe_text_enc):
        for hook in self.register.keys():
            self.register[hook] = []
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output

        return pooled_output

    def encode_image(self, x, is_proj=True):
        hidden_states = self.model.embeddings(x)
        encoder_outputs = self.model.encoder(
            inputs_embeds=hidden_states,
        )
        last_hidden_state = encoder_outputs.last_hidden_state # sae: (batch, 257, 1024)
        if is_proj:
            pooled_output = last_hidden_state[:, 0, :]
        else:
            pooled_output = last_hidden_state
        return pooled_output

    def encode_text(self, text):
        inputs = self.tokenizer(text=text, return_tensors="pt", padding=True)
        outputs = self.full_model.get_text_features(**inputs.to(self.device))
    
        return outputs

    def attach(self, attachment_point, layer, sae=None):
        if attachment_point in self.attach_methods:
            self.attach_methods[attachment_point](layer, sae)
            self.register[f'{attachment_point}_{layer}'] = []
        else:
            raise NotImplementedError(f"Attachment point {attachment_point} not implemented")

    def _attach_post_mlp_residual(self, layer, sae):
        self.model.encoder.layers[layer] = SiglipEncoderLayerPostMlpResidual(
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
            self.model.encoder.layers[self.layer] = SiglipEncoderLayerPostMlpResidual_saeembs(
                self.base_SIGLIPEncoderLayerPostMlpResidual,
                modified_sae,
            )
        else:
            pass

    def attach_and_fix(self, sae, neurons_to_fix={}, pre_zero=False, alpha=0.6, is_image=True):
        modified_sae = SAEWrapper(sae, neurons_to_fix, pre_zero)
        if is_image:
            if self.is_vision:
                self.model.encoder.layers[self.layer] = SiglipEncoderLayerPostMlpResidual_fix(
                    self.base_SIGLIPEncoderLayerPostMlpResidual,
                    modified_sae,
                    alpha=alpha
                )
            else:
                self.model.vision_tower.vision_model.encoder.layers[self.layer] = SiglipEncoderLayerPostMlpResidual_fix(
                        self.base_SIGLIPEncoderLayerPostMlpResidual,
                        modified_sae,
                        alpha=alpha
                    )
        else:
            pass

    def generate(self, image, text, processor, max_new_tokens=1024, temperature=0.0, do_sample=False, pad_token_id=None):
        conversation = [
            {

            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image"},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = processor(image=image, text=prompt, return_tensors='pt').to(self.model.device, torch.float16)

        with torch.inference_mode():
            res = self.model.generate(**inputs,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      do_sample=do_sample,
                                      pad_token_id=pad_token_id)

        generated_text = processor.batch_decode(res, skip_special_tokens=True)[0]
 
        return generated_text

    def get_preprocessor(self) -> LLaVANextPreprocessor:
        return LLaVANextPreprocessor(
            self.tokenizer, self.image_processor, self.model.config, self.conv_template, self.mm_use_im_start_end
        )

    def get_next_token_probabilities(
        self, inputs
        ) -> Tensor:
        
        inputs = inputs.to(device=self.model.device) # torch.Size([32, 623])
        # Run inferece
        with torch.inference_mode():
            with torch.no_grad():
                outputs = self.model(**inputs).logits # torch.Size([32, 623, 32064])
            #     res = self.model.generate(**inputs, 
            #                               return_dict_in_generate=True, output_hidden_states=True)

            # output = self.processor.batch_decode(res.sequences, skip_special_tokens=True) # torch.Size([32, 627])
            # output = [x.split('ASSISTANT: ')[-1] for x in output]
        print(output)
        # Extract next token probabilities
        batch_size = outputs.shape[0]
        logits = outputs[torch.arange(batch_size), -1]
        logits = torch.softmax(logits, dim=-1)
        return logits

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





class SiglipEncoderLayerPostMlpResidual(nn.Module):
    def __init__(self, base, sae, layer, register):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2

        self.sae = sae
        self.layer = layer
        self.register = register

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
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

        outputs = hidden_states

        if output_attentions:
            outputs += attn_weights

        return outputs



class SiglipEncoderLayerPostMlpResidual_saeembs(nn.Module):
    def __init__(self, base, sae):
        super().__init__()
        self.embed_dim = base.embed_dim
        self.self_attn = base.self_attn
        self.layer_norm1 = base.layer_norm1
        self.mlp = base.mlp
        self.layer_norm2 = base.layer_norm2

        self.sae = sae

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        encoded_hidden_states = self.sae.encode(hidden_states)

        return encoded_hidden_states


class SiglipEncoderLayerPostMlpResidual_fix(nn.Module):
    def __init__(self, base, sae, alpha=0.6):
        super().__init__()
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
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        encoded_hidden_states = self.sae.encode(hidden_states)

        decoded_hidden_states = self.sae.decode(encoded_hidden_states)
        hidden_states = (1-self.alpha)*hidden_states + self.alpha*decoded_hidden_states 
        print(self.alpha, hidden_states[0][0], flush=True)

        return hidden_states