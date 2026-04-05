import os
import json
import numpy as np
import pandas as pd

from typing import Optional
from dataclasses import dataclass
import sys
sys.path.append('[your_working_path]/DeBiasLens/vla-gender-bias')
from utils_new.configs import dataset_configs
from utils_new.benchmark_utils import Prompt, load_dataset


@dataclass
class BiasPrompt(Prompt):
    yes_option_letter: str
    no_option_letter: str
    unknown_option_letter: Optional[str]
    instruction: str
    question: str
    gender: str
    dataset: str
    task: str
    value: str


unknown_option_variations = ["Unsure", "Cannot tell", "Uncertain", "Don't know", "Can't determine"]


def load_occupation_probabilities() -> dict[str, float]:
    with open("./data/occupation_probabilities.json", "r") as f:
        return json.load(f)


def get_image_path(image_name: str, dataset: str) -> str:
    return os.path.join(dataset_configs["data_root"], dataset, "images", image_name)


def select_images(dataset: str, num_images: int, split: str) -> pd.DataFrame:
    """Select a gender balanced subset of images from the given dataset."""
    # Load the dataset.
    annotations = load_dataset(dataset)
    # Filter images of the given split
    try:
        annotations = annotations[annotations["split"] == split]
    except:
        try:
            annotations["split"] = annotations["file"].str.split("/").str[0]
            annotations = annotations[annotations["split"] == split]
        except:
            pass    

    # Filter out images that do not exist
    #annotations = annotations[
    #    annotations["name"].apply(
    #        lambda x: os.path.exists(os.path.join(dataset_configs["data_root"], dataset, "images", x))
    #    )
    #]

    # Filter images with occupation probability < 0.1
    occupation_probabilities = load_occupation_probabilities()
    # annotations = annotations[
    #     annotations["name"].apply(
    #         lambda x: occupation_probabilities.get(x, 0) <= 0.25
    #     )
    # ]
    # Add a new column with occupation probabilities
    annotations["occupation_prob"] = annotations["name"].apply(
        lambda x: occupation_probabilities.get(x, 0)
    )

    # Now filter rows where occupation_prob <= 0.25
    annotations = annotations[annotations["occupation_prob"] <= 0.25]

    # Partition into male and female images
    male_images = annotations[annotations["gender"] == "male"]
    female_images = annotations[annotations["gender"] == "female"]

    # Select num_images // 2 images from each partition
    num_male_images = num_images // 2
    num_female_images = num_images - num_male_images
    selected_male_images = male_images.head(n=num_male_images)
    selected_female_images = female_images.head(n=num_female_images)

    # Combine the selected images
    selected_images = pd.concat([selected_male_images, selected_female_images])
    # assert len(selected_images) == num_images, f"Expected {num_images} images, got {len(selected_images)}"
    selected_images["dataset"] = dataset

    return selected_images


class BiasTask:
    def __init__(self, task: str, value_split: Optional[str] = None, prompt_split: Optional[str] = None) -> None:
        self.task = task
        self.value_split = value_split
        self.prompt_split = prompt_split
        # Load task questions
        questions = pd.read_csv("./data/prompting/questions.csv").query("task == @task")
        if prompt_split is not None:
            questions = questions.query("split == @prompt_split")
        self._questions = questions["question"].tolist()
        # Load task values
        values = pd.read_csv("./data/prompting/values.csv").query("task == @task")
        if value_split is not None:
            values = values.query("split == @value_split")
        self._values = values["value"].tolist()
        # Load instructions
        instructions = pd.read_csv("./data/prompting/instructions.csv")
        if prompt_split is not None:
            instructions = instructions.query("split == @prompt_split")
        self._instructions = instructions["instruction"].tolist()

    @property
    def questions(self) -> list[str]:
        return self._questions

    @property
    def values(self) -> list[str]:
        return self._values

    @property
    def instructions(self) -> list[str]:
        return self._instructions


class BiasPromptIterator:
    prompt_template = "{task_question}\n{option_list}\n{instruction}"

    def __init__(
        self,
        task: str,
        datasets: list[str],
        num_images_per_dataset: int,
        include_unknown: bool,
        options_num_permutations: int,
        sample_value: bool,
        sample_question: bool,
        sample_instructions: bool,
        sample_unknown: bool,
        num_values_per_image: Optional[int] = None,
        image_split: Optional[str] = None,
        value_split: Optional[str] = None,
        prompt_split: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        # Save the parameters
        self.task = task
        self.datasets = datasets
        self.num_images_per_dataset = num_images_per_dataset
        self.include_unknown = include_unknown
        self.options_num_permutations = options_num_permutations
        self.sample_value = sample_value
        self.sample_question = sample_question
        self.sample_instructions = sample_instructions
        self.sample_unknown = sample_unknown
        self.num_values_per_image = num_values_per_image
        self.image_split = image_split
        self.value_split = value_split
        self.prompt_split = prompt_split
        self.seed = seed

        # Load task
        self.task = BiasTask(task=task, value_split=value_split, prompt_split=prompt_split)
        # Load images
        self.images = pd.concat([select_images(dataset, num_images_per_dataset, image_split) for dataset in datasets])

        # Initialize random generators
        self.value_rng = np.random.default_rng(seed)
        self.question_rng = np.random.default_rng(seed)
        self.instructions_rng = np.random.default_rng(seed)
        self.unknown_rng = np.random.default_rng(seed)
        self.option_list_rng = np.random.default_rng(seed)

    def iterate_values(self) -> list[str]:
        if self.sample_value:
            return [self.value_rng.choice(self.task.values) for _ in range(self.num_values_per_image)]
        else:
            return self.task.values

    def get_question(self) -> str:
        if self.sample_question:
            return self.question_rng.choice(self.task.questions)
        else:
            return self.task.questions[0]

    def get_instruction(self) -> str:
        if self.sample_instructions:
            return self.instructions_rng.choice(self.task.instructions)
        else:
            return self.task.instructions[0]

    def get_options(self) -> tuple[list[str], str]:
        if self.include_unknown:
            # Sample unknown option variant from choices
            if self.sample_unknown:
                unknown_option = self.option_list_rng.choice(unknown_option_variations)
            else:
                unknown_option = unknown_option_variations[0]
            options = [unknown_option, "Yes", "No"]
            option_letters = ["A", "B", "C"]
        else:
            options = ["Yes", "No"]
            option_letters = ["A", "B"]
            unknown_option = None

        # Randomly permute option list
        options_permuted = self.option_list_rng.permutation(options)
        options_permuted = options_permuted.tolist()

        letter_to_option = dict(zip(option_letters, options_permuted))
        option2letter = {option: letter for letter, option in letter_to_option.items()}
        yes_option_letter = option2letter["Yes"]
        no_option_letter = option2letter["No"]
        if self.include_unknown:
            unknown_option_letter = option2letter[unknown_option]
        else:
            unknown_option_letter = None
        return letter_to_option, yes_option_letter, no_option_letter, unknown_option_letter

    def options_to_str(self, letter_to_option: dict[str, str]) -> str:
        return "\n".join([f"{letter}. {option}" for letter, option in letter_to_option.items()])

    def get_prompts(self) -> list[BiasPrompt]:
        all_prompts = []

        # Iterate over images
        for _, row in self.images.iterrows():
            # Extract relevant information from row
            image_name = row["name"]
            gender = row["gender"]
            dataset = row["dataset"]

            # Make path to image
            image_path = get_image_path(image_name, dataset)

            # Iterate over values
            for value in self.iterate_values():
                question = self.get_question()
                question = question.format(value=value)
                instruction = self.get_instruction()

                for k in range(self.options_num_permutations):
                    option_to_letter, yes_option_letter, no_option_letter, unknown_option_letter = self.get_options()
                    option_str = self.options_to_str(option_to_letter)
                    if unknown_option_letter is not None:
                        option_to_letter[unknown_option_letter] = "Unknown"

                    # Fill in template
                    prompt = self.prompt_template.format(
                        task_question=question,
                        option_list=option_str,
                        instruction=instruction
                    )

                    # Make prompt object
                    prompt = BiasPrompt(
                        prompt=prompt,
                        correct_option=None,
                        yes_option_letter=yes_option_letter,
                        no_option_letter=no_option_letter,
                        unknown_option_letter=unknown_option_letter,
                        instruction=instruction,
                        question=question,
                        letter_to_option=option_to_letter,
                        image=image_path,
                        gender=gender,
                        dataset=dataset,
                        task=self.task.task,
                        value=value
                    )

                    all_prompts.append(prompt)

        return all_prompts
