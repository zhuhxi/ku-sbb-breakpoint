#!/bin/bash
#SBATCH --job-name=kusbb-saf3b
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# re-run of only the training step of hpc/semiauto_full3.sbatch (fakes + features already in outputs/aug/X_simk3.parquet),
# LightGBM threads pinned to 32 (n_jobs=-1 on the exclusive node was oversubscribed and slow)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export SA_JOBS=32 OMP_NUM_THREADS=32 PYTHONWARNINGS=ignore
PYTHONPATH=. ~/venvs/kusbb/bin/python -u probe/semiauto_full.py simk3 outputs/factory/semiauto_full3.parquet
