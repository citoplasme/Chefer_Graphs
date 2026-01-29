#!/bin/bash
#SBATCH --account=aip-frudzicz
#SBATCH --nodes=1
#SBATCH --exclude=kn122,kn117
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
#SBATCH --job-name=Chunk-and-stride
#SBATCH --output=Chunk-and-stride_%j.out

module load python/3.13.2 scipy-stack/2025a cuda12.8/toolkit/12.8.1
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs/bin/activate
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh

echo "SLURM NODE: $SLURMD_NODENAME"

python chunk_and_stride.py