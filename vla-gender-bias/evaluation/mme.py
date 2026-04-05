from tqdm import tqdm
from typing import Optional
from datasets import Dataset
from datasets import load_dataset
from datasets import load_from_disk
from sklearn.metrics import accuracy_score


def load_filtered_mme(num_questions: Optional[int] = None) -> Dataset:
    if num_questions is not None:
        dataset = load_dataset("lmms-lab/MME")["test"] #load_from_disk(f"./data/mme_subsets/{num_questions}/")["test"]
    else:
        dataset = load_dataset("./data/MME/")["test"]
    
    return dataset


def get_mme_performance(model, dataset: Dataset) -> float:
    y_pred, y_true = [], []

    for dp in tqdm(dataset, leave=False):
        image, question = dp["image"], dp["question"]
        image = image.convert("RGB")

        generation_info = model.preprocess(prompt=question, image=image.convert("RGB"))
        response = model.generate(**generation_info, max_new_tokens=1)

        y_pred.append(response["generated_text"].lower())
        y_true.append(dp["answer"].lower())
    
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


class MMEScorer:
    def __init__(self, num_questions: Optional[int] = None) -> None:
        self.dataset = load_filtered_mme(num_questions=num_questions)
    
    def score(self, model) -> float:
        return get_mme_performance(model, self.dataset)
