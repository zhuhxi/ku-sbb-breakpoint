#!/usr/bin/env bash
#SBATCH --job-name=kusbb-gen
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/gen.sbatch   (generator features for all series, then combo10 + genA + genB at production settings)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/genfeat_all.py --jobs 64
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB --pat "$PAT" --tag combo10g
PYTHONPATH=. $PY probe/cmp_confirm.py combo10 combo10g
