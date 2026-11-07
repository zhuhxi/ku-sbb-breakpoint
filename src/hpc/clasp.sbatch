#!/usr/bin/env bash
#SBATCH --job-name=kusbb-clasp
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/clasp.sbatch   (ClaSP scores for all series, then combo10g + clasp at production settings)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/clasp_all.py --jobs 64
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^clasp_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB clasp --pat "$PAT" --tag combo10gc
PYTHONPATH=. $PY probe/cmp_confirm.py combo10g combo10gc
