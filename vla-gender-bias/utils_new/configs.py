import yaml

import os

# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load model configuration file
model_configs_path = os.path.join(current_dir, '..', 'configs', 'model_configs.yaml')
with open(model_configs_path) as file:
    model_configs = yaml.load(file, Loader=yaml.FullLoader)

# Load dataset configuration file
data_config_path = os.path.join(current_dir, '..', 'configs', 'data_config.yaml')
with open(data_config_path) as file:
    dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

# Expand environment variables in dataset_configs
def expand_vars(d):
    if isinstance(d, dict):
        return {k: expand_vars(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [expand_vars(v) for v in d]
    elif isinstance(d, str):
        return os.path.expandvars(d)
    return d

dataset_configs = expand_vars(dataset_configs)


model_to_variant_mapping = {
    "qwen": "qwen-chat-7b",
    "mobilevlm-1.7b": "mobilevlm-v2-1.7b",
    "mobilevlm-3b": "mobilevlm-v2-3b",
    "mobilevlm-7b": "mobilevlm-v2-7b",
    "llava-7b": "llava-1.5-7b",
    "llava-13b": "llava-1.5-13b",
    "llava-rlhf-7b": "llava-rlhf-sft-7b",
    "llava-rlhf-13b": "llava-rlhf-sft-13b",
    "bakllava": "bakllava",
    "llava-1.6-vicuna-7b": "llava-1.6-vicuna-7b",
    "llava-1.6-vicuna-13b": "llava-1.6-vicuna-13b",
    "llava-1.6-mistral-7b": "llava-1.6-mistral-7b",
    "llava-1.6-hermes-34b": "llava-1.6-hermes-34b",
    "bunny-3b": "bunny-1.0-3b",
    "bunny-4b": "bunny-1.1-4b",
    "bunny-8b": "bunny-1.1-llama3-8b",
    "internvl2-1b": "internvl2-1b",
    "internvl2-2b": "internvl2-2b",
    "internvl2-4b": "internvl2-4b",
    "internvl2-8b": "internvl2-8b",
    "internvl2-26b": "internvl2-26b",
    "internvl2-40b": "internvl2-40b",
    "phi-3.5-vision-instruct": "phi-3.5-vision-instruct",
    "paligemma-3b": "paligemma-3b",
}

variant_to_a100_batch_size_mapping = {
    "qwen": 32,
    "mobilevlm-1.7b": 256,
    "mobilevlm-3b": 128,
    "mobilevlm-7b": 64,
    "llava-7b": 32,
    "llava-13b": 10,
    "llava-rlhf-7b": 150,
    "llava-rlhf-13b": 32,
    "bakllava": 80,
    "llava-1.6-vicuna-7b": 32,
    "llava-1.6-vicuna-13b": 10,
    "llava-1.6-mistral-7b": 64,
    "llava-1.6-hermes-34b": 1,
    "bunny-3b": 32,
    "bunny-4b": 20,
    "bunny-8b": 10,
    "internvl2-1b": 10,
    "internvl2-2b": 10,
    "internvl2-4b": 5,
    "internvl2-8b": 5,
    "internvl2-26b": 2,
    "cogvlm2": 6,
    "phi-3.5-vision-instruct": 5,
    "paligemma-3b": 30,
}


variant_to_a100_est10ksampletime = {
    "qwen": "00:30:00",
    "mobilevlm-1.7b": "00:05:00",
    "mobilevlm-3b": "00:05:00",
    "mobilevlm-7b": "00:08:00",
    "llava-7b": "00:20:00",
    "llava-13b": "00:35:00",
    "llava-rlhf-7b": "00:10:00",
    "llava-rlhf-13b": "00:35:00",
    "bakllava": "00:20:00",
    "llava-1.6-vicuna-7b": "00:20:00",
    "llava-1.6-vicuna-13b": "00:35:00",
    "llava-1.6-mistral-7b": "00:20:00",
    "llava-1.6-hermes-34b": "02:00:00",
    "bunny-3b": "00:40:00",
    "bunny-4b": "01:20:00",
    "bunny-8b": "01:20:00",
    "internvl2-1b": "00:30:00",
    "internvl2-2b": "00:30:00",
    "internvl2-4b": "01:00:00",
    "internvl2-8b": "01:30:00",
    "internvl2-26b": "04:00:00",
    "cogvlm2": "02:00:00",
    "phi-3.5-vision-instruct": "01:05:00",
    "paligemma-3b": "00:10:00",
}
