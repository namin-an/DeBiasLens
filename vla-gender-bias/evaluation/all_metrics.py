import torch
import pandas as pd

from typing import Optional
from evaluation.mme import MMEScorer
from evaluation.bias import BiasScorer
from evaluation.bias_proxy import BiasProxyScorer
# from evaluation.llavabench import LLavaBenchScorer


class CombinedScorer:
    def __init__(self, bias_task: str, bias_num_images: int, use_bias_proxy: bool = False, mme_num_questions: Optional[int] = None) -> None:
        if use_bias_proxy:
            self.bias_scorer = BiasProxyScorer(task=bias_task, num_images=bias_num_images)
        else:
            self.bias_scorer = BiasScorer(task=bias_task, num_images=bias_num_images)
        
        self.mme_scorer = MMEScorer(num_questions=mme_num_questions)
        # self.llavabench_scorer = LLavaBenchScorer()

        self.scorer_dict = {
            "bias": self.bias_scorer,
            "mme": self.mme_scorer
            # "llavabench": self.llavabench_scorer
        }
    
    @torch.no_grad()
    def evaluate_model(self, model) -> dict:
        result_dict = dict()
    
        for metric, scorer in self.scorer_dict.items():
            # Run evaluation with metric
            score = scorer.score(model)
            assert isinstance(score, float) or isinstance(score, pd.DataFrame), f"Score must be float or DataFrame, got {type(score)}"
            result_dict[metric] = score
        
        return result_dict
