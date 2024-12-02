#!/bin/bash
#SBATCH --job-name=train_vae       # Job name
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --cpus-per-task=32       # CPU cores/threads
#SBATCH --gres=gpu:1              # Number of GPUs (per node)
#SBATCH --mem=400000M               # memory (per node)
#SBATCH --time=30-30:00            # time (DD-HH:MM)
#SBATCH --partition=zhanglab.p    # use zhanglab partition
#SBATCH --output=./train_vae_1.log       # Standard output
#SBATCH --nodelist=galaxy

echo -e "Running train_vae on galaxy"
python "./script_train_vae.py" > ./train_vae_2.log