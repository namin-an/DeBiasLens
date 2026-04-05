from gpu_utils import set_gpu
set_gpu()

import os
import pickle
import argparse

from vlms import load_model
from pruning_utils import apply_pruning
from pruning_utils import load_gradients
from pruning_utils import merge_gradients
from pruning_utils import get_importance_scores
from evaluation.all_metrics import CombinedScorer
from pruning_utils.patched_clip_forward import clip_attn_forward
from transformers.models.clip.modeling_clip import CLIPAttention


def parse_cmd_arguments() -> argparse.Namespace:
    # Make parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Name of the model to prune")
    parser.add_argument("--sparsity", type=float, required=True, help="Sparsity level for pruning")
    parser.add_argument("--eval-num-images", type=int, default=1000)
    parser.add_argument("--only-gradients", type=int, choices=[0, 1], help="Only use gradients (vs. grad * weight)")
    parser.add_argument("--aggregation-method", type=str, default="sum", choices=["sum", "max", "last",], help="Aggregation method for importance scores")
    parser.add_argument("--location", type=str, default="local", choices=["local", "type", "global"], help="Granularity of pruning")
    parser.add_argument("--normalized-gradients", type=int, choices=[0, 1], default=0, help="Use normalized gradients")
    parser.add_argument("--bias-grad-num-images", type=int, default=1000, help="Number of images to use for bias gradient calculation")
    parser.add_argument("--bias-grad-sample", type=int, default=0, choices=[0, 1], help="Sample number for bias gradient calculation")
    parser.add_argument(
        "--bias-task",
        type=str,
        default="sentiment",
        choices=["sentiment", "skills", "occupations"],
        help="Bias task for bias metric",
    )
    parser.add_argument("--vision", type=int, choices=[0, 1], default=1, help="Prune vision encoder")
    parser.add_argument("--experiment-name", type=str, required=True, help="Name of the experiment")
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_cmd_arguments()

    # Load model
    model = load_model(args.model)
    path_to_importance_scores = os.path.join("./importance_scores", args.experiment_name)
    os.makedirs(path_to_importance_scores, exist_ok=True)
    path_to_importance_scores = os.path.join(
        path_to_importance_scores, "importance_scores.pkl"
    )

    try:
        with open(path_to_importance_scores, "rb") as ipf:
            importance_scores = pickle.load(ipf)
    except FileNotFoundError:
        # Load gradients
        gradients = load_gradients(
            args.model,
            only_gradients=bool(args.only_gradients),
            # vision_gradients=bool(args.vision),
            # normalize_gradients=bool(args.normalized_gradients),
            bias_num_images=args.bias_grad_num_images,
            # bias_task=args.bias_task,
            # bias_sample=args.bias_grad_sample
        )
        gradients = merge_gradients(gradients)

        # Calculate importance scores
        importance_scores = get_importance_scores(model, gradients, aggregation_method=args.aggregation_method)
        with open(path_to_importance_scores, "wb") as ipf:
            pickle.dump(importance_scores, ipf, fix_imports=True)

    # Prune model
    if args.vision == 1:
        # Patch CLIP forward function
        CLIPAttention.forward = clip_attn_forward
    
    print(model, 'which model')
    apply_pruning(
        model=model,
        sparsity=args.sparsity,
        granularity=args.location,
        importance_scores=importance_scores,
        model_type="internvl" #clip" if args.vision == 1 else "llama",
    )

    # Evaluate model
    scorer = CombinedScorer(bias_task=args.bias_task, bias_num_images=args.eval_num_images, use_bias_proxy=False, mme_num_questions=100)
    scores = scorer.evaluate_model(model)

    # Save scores
    save_path = os.path.join(
        ".",
        "results",
        "structured-pruning",
        args.experiment_name,
        str(args.only_gradients),
        args.model,
        str(args.sparsity),
        args.location,
        args.bias_task,
        str(args.eval_num_images),
    )
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "scores.pkl"), "wb") as f:
        pickle.dump(scores, f, fix_imports=True)
