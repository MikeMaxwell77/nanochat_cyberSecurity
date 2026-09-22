#!/bin/bash
#SBATCH --job-name=nanochat_run
#SBATCH --output=logs/nano_%j.out
#SBATCH --error=logs/nano_%j.err
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account rc_general
#SBATCH -p gpu-A100  
#SBATCH --gres=gpu:2

# --- Rest of your script follows ---
module load shared python3/anaconda/3.12
module load cuda/12.8

source /home/mm401/.bashrc
source /home/mm401/.cargo/env

cd /work/mm401/nanochat
source .venv/bin/activate

# Build the tokenizer bridge
# uv run maturin develop --release --manifest-path Cargo.toml

# Start the training
bash runs/speedrun.sh
