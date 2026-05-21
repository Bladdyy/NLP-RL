#!/bin/bash
#
#SBATCH --job-name=transformer-test
#SBATCH --partition=common
#SBATCH --qos=ok479034_common
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-transformer-test.txt
#SBATCH --error=logs/slurm-transformer-test-error.txt
#SBATCH --time=30

echo "Started"

set -eux

uv run main.py \
   --env_id "ant" \
   --eval_env_id "ant" \
   --num_epochs 10 \
   --total_env_steps 300_000 \
   --critic_depth 16 \
   --actor_depth 16 \
   --actor_skip_connections 4 \
   --critic_skip_connections 4 \
   --vis_length 1000  \
   --save_buffer 0  \
   --num_envs 16 \
   --min_replay_size 2000 \
   --unroll_length 20

 echo "Finished"
                   