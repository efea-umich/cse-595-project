#!/bin/bash

#SBATCH --job-name=efea-bert
#SBATCH --account=cse595s001f24_class
#SBATCH --partition=spgpu
#SBATCH --time=00-00:50:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=32GB
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --mail-type=BEGIN,END

module load python cuda/11.7.1

# set up job
pushd /home/efea/cse-583-hw3

# run job
python bert_finetune.py