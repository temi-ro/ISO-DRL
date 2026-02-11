#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=trm25
#SBATCH --output=14_ratio.out
export PATH=/vol/bitbucket/trm25/myvenv/bin/:$PATH

source /vol/bitbucket/trm25/pixeDRL_env/bin/activate

echo "Running on host: $(hostname)"
echo "GPU in use:"
/usr/bin/nvidia-smi

echo "Starting evaluation..."

python /vol/gpudata/trm25-iso/pixelDRL2/evaluation.py --data_dir /vol/gpudata/trm25-iso/braTS2023/Dataset001_BraTS2023 --checkpoint /vol/biomedic2/bglocker_>
echo "Evaluation finished. Uptime:"
uptime

