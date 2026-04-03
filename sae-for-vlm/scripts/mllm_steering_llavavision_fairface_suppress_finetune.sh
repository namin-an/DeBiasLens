#!/bin/bash

export KAGGLEHUB_CACHE="/workspace/cvml_user/namin/bias_vlm/data"
export WANDB_API_KEY=""

export FAIRFACE_PATH="/workspace/cvml_user/namin/bias_vlm/data/sbbench-syn-crop"
DATASET_PATH="${FAIRFACE_PATH}"

# 1. Save original activations
# for SPLIT in "train" "val"; do
#   python save_activations.py \
#     --batch_size 32 \
#     --model_name "llava-onevision-qwen2-7b-ov-hf" \
#     --attachment_point "post_mlp_residual" \
#     --layer 25 \
#     --dataset_name "sbbench-syn-crop" \
#     --split "${SPLIT}" \
#     --data_path "${DATASET_PATH}" \
#     --num_workers 0 \
#     --output_dir "./activations_dir/ours/random_k_2/sbbench-syn-crop_${SPLIT}_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
#     --random_k 2 \
#     --save_every 100
# done

# # # # # # 2. Train SAE
# python sae_train.py \
#   --sae_model "matroyshka_batch_top_k" \
#   --activations_dir "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/activations_dir/ours/random_k_2/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
#   --val_activations_dir "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/activations_dir/ours/random_k_2/sbbench-syn-crop_val_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
#   --checkpoints_dir "checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/" \
#   --expansion_factor 8 \
#   --steps 110000 \
#   --save_steps 20000 \
#   --log_steps 10000 \
#   --batch_size 4096 \
#   --k 20 \
#   --auxk_alpha 0.03 \
#   --decay_start 109999 \
#   --group_fractions 0.0625 0.125 0.25 0.5625

# # # # # # 3. Save SAE activations
python save_activations.py \
  --batch_size 32 \
  --model_name "llava-onevision-qwen2-7b-ov-hf" \
  --attachment_point "post_mlp_residual" \
  --layer 25 \
  --dataset_name "sbbench-syn-crop" \
  --split "train" \
  --data_path "${DATASET_PATH}" \
  --num_workers 0 \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
  --mean_pool \
  --save_every 100 \
  --sae_model "matroyshka_batch_top_k" \
  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt"

# # # # # 4. Visualize neurons
python find_hai_indices.py \
  --activations_dir "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
  --dataset_name "sbbench-syn-crop" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --k 16 \
  --chunk_size 1000 #1000

python visualize_neurons.py \
  --output_dir "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual" \
  --top_k 16 \
  --dataset_name "sbbench-syn-crop" \
  --data_path "${DATASET_PATH}" \
  --split "train" \
  --group_fractions 0.0625 0.125 0.25 0.5625 \
  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_llava-onevision-qwen2-7b-ov-hf_25_post_mlp_residual/hai_indices_16.npy"

# # # 5. Compute steering score
# python encode_images.py \
#   --embeddings_path "embeddings_dir/sbbench-syn-crop_train_embeddings_clip-vit-large-patch14-336.pt" \
#   --model_name "llava-onevision-qwen2-7b-ov-hf" \
#   --dataset_name "sbbench-syn-crop" \
#   --split "train" \
#   --data_path "${DATASET_PATH}" \
#   --batch_size 128

# export FAIRFACE_PATH="/workspace/cvml_user/namin/bias_vlm/data/sbbench-syn-crop/imgs/train_val/train"
# DATASET_PATH="${FAIRFACE_PATH}"

# python imagenet_subset.py \
#   --imagenet_root "${DATASET_PATH}" \
#   --output_dir "./images_sbbench-syn-crop"

# # no steering (1000 images, 10 neurons) 
# python steering_score.py \
#  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual/hai_indices_16.npy" \
#  --embeddings_path "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/embeddings_dir/sbbench-syn-crop_train_embeddings_clip-vit-base-patch32.pt" \
#  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt" \
#  --images_path "./images_sbbench-syn-crop/" \
#  --no-pre_zero \
#  --model_name "clip-vit-base-patch32" \
#  --neuron_prefix 10 \
#  --no-steer \
#  --output_path "./llava_results_dir/1000/no_steering/"

# # steering (1000 images, 10 neurons)
# python steering_score.py \
#  --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual/hai_indices_16.npy" \
#  --embeddings_path "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/embeddings_dir/sbbench-syn-crop_train_embeddings_clip-vit-large-patch14-336.pt" \
#  --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt" \
#  --images_path "./images_sbbench-syn-crop/" \
#  --no-pre_zero \
#  --model_name "clip-vit-base-patch32" \
#  --neuron_prefix 10 \
#  --steer \
#  --output_path "./llava_results_dir/1000/steering/"

# # # steering (1 image, 1000 neurons)
# python steering_score.py \
#   --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual/hai_indices_16.npy" \
#   --embeddings_path "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/embeddings_dir/sbbench-syn-crop_train_embeddings_clip-vit-base-patch32.pt" \
#   --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt" \
#   --images_path "./images/" \
#   --no-pre_zero \
#   --model_name "clip-vit-base-patch32" \
#   --neuron_prefix 1000 \
#   --no-steer \
#   --output_path "./llava_results_dir/1/no_steering/"

# python steering_score.py \
#   --hai_indices_path "./activations_dir/matroyshka_batch_top_k_20_x8/mean_pool/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual/hai_indices_16.npy" \
#   --embeddings_path "/workspace/cvml_user/namin/bias_vlm/sae-for-vlm/embeddings_dir/sbbench-syn-crop_train_embeddings_clip-vit-base-patch32.pt" \
#   --sae_path "./checkpoints_dir/matroyshka_batch_top_k_20_x8/random_k_2/sbbench-syn-crop_train_activations_clip-vit-large-patch14-336_25_post_mlp_residual_matroyshka_batch_top_k_20_x8/trainer_0/checkpoints/ae_100000.pt" \
#   --images_path "./images/" \
#   --no-pre_zero \
#   --model_name "clip-vit-base-patch32" \
#   --neuron_prefix 1000 \
#   --steer \
#   --output_path "./llava_results_dir/1/steering/"

# # 6. Compute baseline scores
# python similarity_baseline.py

# # 7. Finding qualitative examples (e.g. steering pencil neuron)
# python steering_qualitative.py
