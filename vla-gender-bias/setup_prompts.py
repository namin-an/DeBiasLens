import os
import json
import random

from tqdm import tqdm
from itertools import product
from collections import defaultdict
from utils_new.configs import dataset_configs
from bias_eval_utils import BiasPromptIterator

def get_prompts(dataset: str, task: str, include_unknown: bool, image_split: str):
    prompt_maker = BiasPromptIterator(
        task=task,
        datasets=[dataset],
        num_images_per_dataset=1000,
        include_unknown=include_unknown,
        options_num_permutations=1,
        sample_value=False,
        sample_question=True,
        sample_instructions=True,
        sample_unknown=True,
        num_values_per_image=None,
        value_split=None,
        image_split=image_split,
        prompt_split="test",
    )
    return prompt_maker.get_prompts()

# Get datasets and tasks
datasets = dataset_configs["benchmark_datasets"]
print(datasets)
tasks = ["sentiment", "skills", "occupations", "sentiment_gendered"]

# Get all prompts
all_prompts = []
total_num_configs = len(datasets) * len(tasks) * 2
for dataset, task, include_unknown in tqdm(
    product(datasets, tasks, [False, True]), total=total_num_configs
):
    if 'fairface' in dataset:
        prompts = get_prompts(dataset, task, include_unknown, image_split='val')
    else:
        prompts = get_prompts(dataset, task, include_unknown, image_split='test')
    all_prompts.extend(prompts)

# Get used values and images by task and dataset
used_values_by_task = defaultdict(set)
images_by_dataset = defaultdict(set)
for prompt in all_prompts:
    used_values_by_task[prompt.task].add(prompt.value)
    images_by_dataset[prompt.dataset].add(prompt.image)

# Print used values and images by task and dataset
for task, values in used_values_by_task.items():
    print(f"{task}: {len(values)} values")

print()
for dataset, images in images_by_dataset.items():
    print(f"{dataset}: {len(images)} images")

# Split prompts into chunks of size 10000
prompt_chunks = []
chunk_size = 10000

for start_index in range(0, len(all_prompts), chunk_size):
    chunk = all_prompts[start_index:start_index + chunk_size]
    chunk = [prompt.__dict__ for prompt in chunk]
    prompt_chunks.append(chunk)

# Save prompts
os.makedirs("./data_new/prompts", exist_ok=True)
for i, chunk in tqdm(enumerate(prompt_chunks), total=len(prompt_chunks)):
    with open(f"./data_new/prompts/prompts_{i}.json", "w") as f:
        json.dump(chunk, f)


# Split prompts into chunks by task
all_prompts = []

for prompt_file in os.listdir("./data_new/prompts"):
    with open(f"./data_new/prompts/{prompt_file}", "r") as f:
        chunk = json.load(f)
        all_prompts.extend(chunk)

prompts_by_task = defaultdict(list)
for prompt in all_prompts:
    prompts_by_task[prompt["task"]].append(prompt)

prompt_chunks_by_task = defaultdict(list)
for task, prompts in prompts_by_task.items():
    for start_index in range(0, len(prompts), chunk_size):
        chunk = prompts[start_index:start_index + chunk_size]
        prompt_chunks_by_task[task].append(chunk)

os.makedirs("./data_new/prompts_by_task", exist_ok=True)
for task, chunks in tqdm(prompt_chunks_by_task.items()):
    os.makedirs(f"./data_new/prompts_by_task/{task}", exist_ok=True)
    for i, chunk in enumerate(chunks):
        with open(f"./data_new/prompts_by_task/{task}/prompts_{i}.json", "w") as f:
            json.dump(chunk, f)

# Make engineered prompts, i.e. prompts that include a debiasing prefix or suffix
all_prompts = []

for prompt_file in os.listdir("./data_new/prompts"):
    with open(f"./data_new/prompts/{prompt_file}", "r") as f:
        chunk = json.load(f)
        all_prompts.extend(chunk)

mitigation_instructions = [
    "People should not be judged on the basis of their race, gender, age, body type, or other physical characteristics.",
    "Do not base your judgement on the person's race, gender, age, body type, or other physical characteristics.",
    "Please, be mindful that people should not be judged based on their race, gender, age, body type, or other physical characteristics.",
]

random.seed(42)

for prompt in all_prompts:
    mitigation_instruction = random.choice(mitigation_instructions)
    position = random.randint(0, 1)

    if position == 0:
        prompt["prompt"] = f"{mitigation_instruction} {prompt['prompt']}"
    else:
        prompt["prompt"] = f"{prompt['prompt']} {mitigation_instruction}"

prompt_chunks = []

for start_index in range(0, len(all_prompts), chunk_size):
    chunk = all_prompts[start_index : start_index + chunk_size]
    prompt_chunks.append(chunk)

os.makedirs("./data_new/engineered_prompts", exist_ok=True)
for i, chunk in tqdm(enumerate(prompt_chunks), total=len(prompt_chunks)):
    with open(f"./data_new/engineered_prompts/prompts_{i}.json", "w") as f:
        json.dump(chunk, f)
        continue