#!/bin/bash
#SBATCH --account=aip-frudzicz
#SBATCH --nodes=1
#SBATCH --exclude=kn122,kn117,kn050
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=9
#SBATCH --time=72:00:00
#SBATCH --mem=72GB
#SBATCH --job-name=Sliding-windows-IMDb-1k
#SBATCH --output=Sliding-windows-IMDb-1k_%j.out

module load python/3.13.2 scipy-stack/2025a cuda12.8/toolkit/12.8.1
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs/bin/activate
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh

echo "SLURM NODE: $SLURMD_NODENAME"

python main.py --data_set IMDb-1k --use_label_smoothing 0 --use_gradient_clipping 0 --checkpoint_validation_loss 1 --use_accuracy 1 --use_balanced_loss 0 --fine_tuning_trial_number 56 --subsample_percentage 1.0