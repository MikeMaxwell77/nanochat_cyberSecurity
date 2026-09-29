#!/bin/bash
#SBATCH --job-name=nanochat_no_ctf
#SBATCH --output=logs/no_ctf_%j.out
#SBATCH --error=logs/no_ctf_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=rc_general
#SBATCH -p gpu-H200
#SBATCH --gres=gpu:1

set -eo pipefail
cd /work/mm401/nanochat
module load shared python3/anaconda/3.12
module load cuda/12.8
source /home/mm401/.bashrc
source /home/mm401/.cargo/env
source .venv/bin/activate
set -u
bash runs/ctf_control.sh