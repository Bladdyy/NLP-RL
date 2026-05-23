#!/bin/bash
#
#SBATCH --job-name=crl-transformer
#SBATCH --partition=common
#SBATCH --qos=ok479034_common
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-crl-transformer.txt
#SBATCH --error=logs/slurm-crl-transformer-error.txt
#SBATCH --time=16:00:00

echo "Started"

set -eux

uv run main.py \
  --env_id "ant_maze_u4" \
  --eval_env_id "ant_maze_u4" \
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
  --transformer_use_cls_token 1 \
  --semantic_tokens 0

 echo "Finished"
                   