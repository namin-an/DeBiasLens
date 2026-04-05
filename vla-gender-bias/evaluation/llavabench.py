
import numpy as np

from tqdm import tqdm
from datasets import load_dataset as load_hf_dataset


def load_dataset() -> list[dict]:
    dataset = load_hf_dataset("./data/llava-bench-in-the-wild/")
    dataset = dataset["train"]
    return dataset


def get_llavabench_performance(model, dataset: list[dict]) -> float:
    nll_values = []

    for dp in tqdm(dataset, leave=False):
        question, image, reference = dp["question"], dp["image"], dp["gpt_answer"]
        generation_info = model.preprocess(prompt=question, image=image, assistant_prefix=reference)
        nll = model.score_generation(**generation_info)
        nll_values.append(nll.detach().cpu().item())
    
    return np.mean(nll_values)


class LLavaBenchScorer:
    def __init__(self) -> None:
        self.dataset = load_dataset()
    
    def score(self, model) -> float:
        return get_llavabench_performance(model, self.dataset)
