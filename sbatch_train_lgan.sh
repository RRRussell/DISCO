#!/bin/bash
#SBATCH --job-name=train_lgan       # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --cpus-per-task=32       # CPU cores/threads
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=200000M               # memory (per node)
#SBATCH --time=30-30:00            # time (DD-HH:MM)
#SBATCH --partition=zhanglab.p    # use zhanglab partition
#SBATCH --output=./train_lgan_1.log       # Standard output
#SBATCH --nodelist=galaxy

echo -e "Running train_lgan on galaxy"
python "./script_train_lgan.py" > ./train_lgan_2.log