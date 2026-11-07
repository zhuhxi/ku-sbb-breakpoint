#!/usr/bin/env bash
#SBATCH --job-name=kusbb-bystruct
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/bystruct.sbatch MENU N   (break-by-structure table on fakes, grouped by the fitted structure)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
PYTHONPATH=. ~/venvs/kusbb/bin/python probe/break_by_struct_fake.py "$1" "$2" --fit
