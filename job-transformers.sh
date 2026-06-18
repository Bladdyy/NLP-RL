#!/bin/bash
#
#SBATCH --job-name=crl-transformer
#SBATCH --partition=a100
#SBATCH --qos=ok479034_a100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-crl-transformer.txt
#SBATCH --error=logs/slurm-crl-transformer-error.txt
#SBATCH --time=16:00:00

export XLA_FLAGS="--xla_gpu_cuda_data_dir=/home/ok479034/NLP-RL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc ${XLA_FLAGS}"
export PATH="/home/ok479034/NLP-RL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH"

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
  --max_replay_size 100_00 \
  --transformer_mode State \
  --transformer_embed_dim 144\
  --transformer_num_layers 2 \
  --transformer_num_heads 4 \
  --transformer_mlp_ratio 4 \
  --transformer_num_patches 8 \
  --transformer_dropout 0 \
  --transformer_pooling cls \
  --tokenization semantic \
  --grad_clip_max_norm 10.0 \
  --transformer_lr 1e-4 \
  --transformer_weight_decay 1e-4 \
  --embed_norm base \
  --entropy_param 0.5 \
  --sigreg_coeff 0.1 \
  --sigreg_bandwidth 1.0 \
  --sigreg_num_t_nodes 64 \
  --sigreg_num_slices 256 \
  --sigreg_sketch_dim 64 

 echo "Finished"
                   