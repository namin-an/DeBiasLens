template = """
llava:
  llava-1.5-7b:
    model_path: {path}/vlm-models/llava-v1.5-7b
    model_base: null
    model_name: llava-v1.5-7b
  llava-1.5-13b:
    model_path: {path}/vlm-models/llava-v1.5-13b
    model_base: null
    model_name: llava-v1.5-13b
  llava-rlhf-sft-7b:
    model_name: llava-rlhf-7b-v1.5-224
    sft: {path}/vlm-models/LLaVA-RLHF-7b-v1.5-224/sft_model
    lora: {path}/vlm-models/LLaVA-RLHF-7b-v1.5-224/rlhf_lora_adapter_model
  llava-rlhf-sft-13b:
    model_name: llava-rlhf-13b-v1.5-336
    sft: {path}/vlm-models/LLaVA-RLHF-13b-v1.5-336/sft_model
    lora: {path}/vlm-models/LLaVA-RLHF-13b-v1.5-336/rlhf_lora_adapter_model
  llava-phi:
    model_path: {path}/vlm-models/llava-phi
    model_base: null
    model_name: llava-phi
  bakllava:
    model_path: {path}/vlm-models/BakLLaVA-1/
    model_base: null
    model_name: bakllava

llava-next:
  llava-1.6-vicuna-7b:
    model_path: {path}/vlm-models/llava-v1.6-vicuna-7b
    model_base: null
    model_name: llava-v1.6-vicuna-7b
  llava-1.6-vicuna-13b:
    model_path: {path}/vlm-models/llava-v1.6-vicuna-13b
    model_base: null
    model_name: llava-v1.6-vicuna-13b
  llava-1.6-mistral-7b:
    model_path: {path}/vlm-models/llava-v1.6-mistral-7b
    model_base: null
    model_name: llava-v1.6-mistral-7b
  llava-1.6-hermes-34b:
    model_path: {path}/vlm-models/llava-v1.6-34b
    model_base: null
    model_name: llava-v1.6-34b

minigpt:
  minigptv2:
    ckpt: {path}/vlm-models/MiniGPTv2/checkpoint_stage3.pth
    path_to_llama: {path}/hf-llama2/llama-2-chat-hf/7B/

tinygpt:
  tinygpt:
    ckpt: {path}/vlm-models/TinyGPT-V/TinyGPT-V_for_Stage4.pth
    path_to_phi: {path}/vlm-models/phi-2
    path_to_bert: {path}/vlm-models/bert-base-uncased

qwen:
  qwen-chat-7b:
    pretrained_model_name_or_path: {path}/vlm-models/Qwen-VL-Chat/
    cache_dir: {path}/vlm-models/

mobilevlm:
  mobilevlm-v2-1.7b:
    model_path: {path}/vlm-models/MobileVLM_V2-1.7B/
  mobilevlm-v2-3b:
    model_path: {path}/vlm-models/MobileVLM_V2-3B/
  mobilevlm-v2-7b:
    model_path: {path}/vlm-models/MobileVLM_V2-7B/

bunny:
  bunny-1.1-4b:
    model_path: {path}/vlm-models/Bunny-v1_1-4B/
  bunny-1.0-3b:
    model_path: {path}/vlm-models/Bunny-v1_0-3B/
  bunny-1.1-llama3-8b:
    model_path: {path}/vlm-models/Bunny-v1_1-Llama-3-8B-V

internvl2:
  internvl2-1b:
    model_path: {path}/vlm-models/InternVL2-1B/
  internvl2-2b:
    model_path: {path}/vlm-models/InternVL2-2B/
  internvl2-4b:
    model_path: {path}/vlm-models/InternVL2-4B/
  internvl2-8b:
    model_path: {path}/vlm-models/InternVL2-8B/
  internvl2-26b:
    model_path: {path}/vlm-models/InternVL2-26B/
  internvl2-40b:
    model_path: {path}/vlm-models/InternVL2-40B/

phi3v:
  phi-3.5-vision-instruct:
    model_path: {path}/vlm-models/Phi-3.5-vision-instruct/
"""

import os
import argparse


if __name__ == '__main__':
    # Create parser
    parser = argparse.ArgumentParser(description="Update model config file")
    parser.add_argument('--path', type=str, help="Path to the models")
    args = parser.parse_args()
    path = args.path

    # Get absolute path
    path = os.path.abspath(path)
    # Remove trailing slash
    path = path.rstrip('/')

    # Update the template
    template = template.format(path=path)

    with open("./model_configs.yaml", "w") as f:
        f.write(template)
    
    print("Model config file updated successfully!")
