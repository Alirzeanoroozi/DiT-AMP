#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/comp.out
#SBATCH --error=logs/comp.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
# source ${CONDA_BASE}/bin/activate comp
source ${CONDA_BASE}/bin/activate amplify
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

# nvidia-smi

AMPlify -s genbio/ampgan_sequences.fasta
# AMPlify -s ampdiffusion_results/genbio/hydramp_sequences.fasta
AMPlify -s genbio/pepcvae_sequences.fasta
