import numpy as np

from evaluation.bias import BiasScorer


class BiasProxyScorer(BiasScorer):
    def score(self, model) -> float:
        scores = super().score(model)
        yes_prob = scores["yes_prob"]

        # Calculate mean absolute deviation of yes_probs from 0.5
        mad = np.mean(np.abs(yes_prob - 0.5))
        return mad