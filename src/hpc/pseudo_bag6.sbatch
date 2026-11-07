#!/usr/bin/env bash
#SBATCH --job-name=kusbb-ps-bag6
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# production confirm with fold-clean pseudo-labels (top/bottom 25% of teacher test preds, hard labels), arm bag6
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CONF_JOBS=64
PY=~/venvs/kusbb/bin/python
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_'
PYTHONPATH=. $PY -W ignore probe/confirm_pseudo.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB gnoise gjlr semiauto_bag6 --pat "$PAT" --tag ps_bag6
