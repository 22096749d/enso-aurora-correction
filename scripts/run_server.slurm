#!/bin/bash
# Template only. Ask the administrator for the actual partition/account/GPU request syntax.
#SBATCH --job-name=aurora-pilot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/aurora-%j.log
set -euo pipefail
# Submit from the aurora_v2 project directory after activating aurora-server.
# Some clusters do not export activated environments; ask the administrator about activation here.
cd "${SLURM_SUBMIT_DIR:?Submit from the project directory}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8
python scripts/server.py check
python scripts/server.py run --plan transfer/pilot/plan.csv --output server_runs/pilot --steps 20 --limit 1 --save-msl
