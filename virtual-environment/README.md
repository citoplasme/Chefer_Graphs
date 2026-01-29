# How to setup a virtual environment in Killarney

## 1. Create a general directory for environments if it does not exist

mkdir /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments

## 2. Create a specific directory for the new environment

mkdir /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs

## 3. Load the desired Python version

module load python/3.13.2

## 4. Load the scipy stack (Pandas, Numpy, etc.) for the selected Python version

module load scipy-stack/2025a

## 5. Create a new virtual environment

virtualenv --no-download /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs

## 6. Activate the virtual environment

source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs/bin/activate

## 7. Upgrade pip on the new environment

pip install --no-index --upgrade pip

## 8. Create cache directories for PyTorch and HuggingFace 

mkdir /home/jpimen/projects/aip-frudzicz/jpimen/cache/
mkdir /home/jpimen/projects/aip-frudzicz/jpimen/cache/PLM_graphs/
mkdir /home/jpimen/projects/aip-frudzicz/jpimen/cache/PLM_graphs/torch
mkdir /home/jpimen/projects/aip-frudzicz/jpimen/cache/PLM_graphs/huggingface

## 9. Create a directory for .sh files needed for each virtual environment

mkdir /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files

## 10. Define the cache directories for the environment (/home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh)

#!/bin/bash
export TORCH_HOME=/home/jpimen/projects/aip-frudzicz/jpimen/cache/PLM_graphs/torch
export HF_HOME=/home/jpimen/projects/aip-frudzicz/jpimen/cache/PLM_graphs/huggingface

## 11. Attribute execution permision to the .sh file

chmod +x /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh

## 12. Activate the cache directories

source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh

## 13. Install packages

### a. Already installed on the server

pip install PACKAGE --no-index

### b. Not installed on the server

pip install PACKAGE

### Installed packages on the PLM_graphs virtual environment

pip install sqlalchemy==1.4.2
pip install torch_geometric==2.6.1 networkx==3.5 optuna==4.1.0 plotnine==0.14.1 transformers==4.53.0 torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 sklearn==0.0 --no-index
pip install textstat==0.7.8
pip install evaluate==0.4.3 --no-index (requires module load gcc arrow/21.0.0)
pip install nltk==3.9.1 absl_py==2.3.1 rouge_score==0.1.2 --no-index
pip install bert-score==0.3.13
pip install librosa==0.11.0 --no-index
pip install openai-whisper==v20250625
pip install torchtext==0.15.2

## 14. Exit the virtual environment

deactivate

## 15. Load the previosuly created virtual environment on a slurm job (example of a .sh file)

#!/bin/bash
#SBATCH --account=aip-frudzicz
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --mem=1GB
#SBATCH --job-name=Test
#SBATCH --output=Test_%j.out

module load python/3.13.2 scipy-stack/2025a
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environments/PLM_graphs/bin/activate
source /home/jpimen/projects/aip-frudzicz/jpimen/virtual-environment-files/PLM_graphs.sh

python test.py

# Relevant resources

1. A basic tutoral on virtual environments: https://docs.alliancecan.ca/wiki/Python
2. Packages installed on the server: https://docs.alliancecan.ca/wiki/Available_Python_wheels

# Notes

1. /scratch periodically deletes files (every 60 days), being mostly used for caching during jobs or storing model checkpoints
2. Slurm jobs must be submitted from either /project or /scratch -- cannot be submitted from /home 
3. sklearn==0.0 is actually 1.6.1
4. Sbatch all .sh files in a directory: find . -name "*.sh" -exec sbatch {} \;