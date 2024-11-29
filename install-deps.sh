#!/bin/bash

#SBATCH --job-name=install-deps
#SBATCH --mail-user=efea@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=59:00
#SBATCH --account=cse595s001f24_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
#SBATCH --error=/home/%u/%x-%j-err.log

module load python

# set up job
pushd /home/efea/cse-595-project

# run job
uv pip install -r pyproject.toml