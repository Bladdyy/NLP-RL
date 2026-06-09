#!/bin/bash
#
#SBATCH --job-name=crl-transformer
#SBATCH --partition=a100
#SBATCH --qos=ok479034_a100
#SBATCH --gres=gpu:1
#SBATCH --output=logs/slurm-crl-transformer.txt
#SBATCH --error=logs/slurm-crl-transformer-error.txt
#SBATCH --time=16:00:00

bash job-transformers.sh