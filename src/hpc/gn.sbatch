#!/usr/bin/env bash
#SBATCH --job-name=kusbb-gn
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# combo10g columns with top-200 features per fold, compared with combo10g
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=~/venvs/kusbb/bin/python
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB gnoise --pat "$PAT" --tag combo10gn
PYTHONPATH=. $PY probe/cmp_confirm.py combo10g combo10gn
