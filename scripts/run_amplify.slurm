#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=kutem_gpu
#SBATCH --account=kutem
#SBATCH --gres=gpu:tesla_a100:1
#SBATCH --time=72:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/amplify.out
#SBATCH --error=logs/amplify.out

# Initialise environment and modules
CONDA_BASE=$(conda info --base)
source ${CONDA_BASE}/bin/activate amplify
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

echo "Running AMPlify"

config="lr2e-04_wd0p01_hlc0p001_d512_l6_h8"
mkdir -p samples/results/$config

# AMPlify -s samples/sampled_amps_$config.fasta -od samples/results/$config

echo "Evaluating AMPlify results"
python src/eval_summary.py --name $config

echo "Switching to comp env for MACREL summary"
source ${CONDA_BASE}/bin/activate comp
export LD_LIBRARY_PATH=${CONDA_BASE}/lib

python src/eval_macrel_summary.py --name $config