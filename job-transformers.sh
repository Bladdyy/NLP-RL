#
#SBATCH --job-name=transformer-small
#SBATCH --partition=common
#SBATCH --qos=ok479034_common
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-transformer-small.txt
#SBATCH --error=logs/slurm-transformer-small-error.txt
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
  --unroll_length 20 \
  --use_transformer  1 \
  --transformer_embed_dim 144\
  --transformer_num_layers 4 \
  --transformer_num_heads 4 \
  --transformer_mlp_ratio 4 \
  --transformer_num_patches 8 \
  --transformer_dropout 0.0 \
  --transformer_use_cls_token 1

 echo "Finished"
                   