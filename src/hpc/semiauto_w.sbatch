#!/bin/bash
#SBATCH --job-name=kusbb-saw
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/semiauto_w.sbatch TAG  (semi-auto models trained with realism weights -> outputs/factory/semiauto_TAG_w.parquet)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export SA_JOBS=32 OMP_NUM_THREADS=32 PYTHONWARNINGS=ignore SA_WEIGHT=1
PYTHONPATH=. ~/venvs/kusbb/bin/python -u probe/semiauto_full.py "$1" "outputs/factory/semiauto_$1_w.parquet"
