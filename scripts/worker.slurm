#!/bin/bash
#SBATCH --job-name=enso-aurora
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/worker-%j.log
# Add institution-specific partition/account/GPU constraints.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
# Initialise and activate the agreed Conda environment here if required.
python scripts/server.py check
python scripts/worker.py run --job "${ENSO_JOB:-pilot}" --through "${ENSO_THROUGH:-1}"
