#!/bin/bash
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
  --critic_network_width 256 \
  --actor_network_width 256 \
  --vis_length 1000  \
  --save_buffer 0  \
  --num_envs 512 \
  --batch_size 512 \
  --min_replay_size 1000 \
  --max_replay_size 100_000 \
  --transformer_mode StateActor \
  --transformer_embed_dim 144\
  --transformer_num_layers 2 \
  --transformer_num_heads 4 \
  --transformer_mlp_ratio 4 \
  --transformer_num_patches 8 \
  --transformer_dropout 0.0 \
  --transformer_pooling cls \
  --tokenization semantic \
  --grad_clip_max_norm 15.0 \
  --transformer_lr 1e-4 \
  --transformer_weight_decay 1e-4 \
  --loss_temperature 0.1 \
  --entropy_param 0.5

 echo "Finished"
                   