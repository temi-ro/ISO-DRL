#!/bin/bash
#SBATCH --partition=a100
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=trm25

export PATH=/vol/bitbucket/trm25/myvenv/bin/:$PATH

source /vol/bitbucket/trm25/pixeDRL_env/bin/activate

echo "Running on host: $(hostname)"
echo "GPU in use:"
/usr/bin/nvidia-smi

echo "Starting training..."

python -u /vol/gpudata/trm25-iso/pixelDRL/training.py --dataset /vol/gpudata/trm25-iso/braTS2023/Dataset001_BraTS2023

echo "Training finished. Uptime:"
uptime