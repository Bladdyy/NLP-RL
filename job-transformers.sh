#!/bin/bash
#
#SBATCH --job-name=crl-transformer
#SBATCH --partition=a100
#SBATCH --qos=jh479001_a100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-crl-transformer.txt
#SBATCH --error=logs/slurm-crl-transformer-error.txt
#SBATCH --time=16:00:00

echo "Started"

set -eux

uv run main.py \
  --env_id "ant_u4_maze" \
  --eval_env_id "ant_u4_maze" \
  --num_epochs 100 \
  --episode_length 1000 \
  --total_env_steps 100_000_000 \
  --critic_depth 8 \
  --actor_depth 8 \
  --actor_skip_connections 4 \
  --critic_skip_connections 4 \
  --vis_length 1000  \
  --save_buffer 0  \
  --num_envs 512 \
  --batch_size 512 \
  --min_replay_size 1000 \
  --max_replay_size 10_000 \
  --use_transformer  1 \
  --transformer_embed_dim 144\
  --transformer_num_layers 2 \
  --transformer_num_heads 4 \
  --transformer_mlp_ratio 4 \
  --transformer_num_patches 8 \
  --transformer_dropout 0.0 \
  --transformer_pooling flatten \
  --tokenization per_dim

 echo "Finished"
                   