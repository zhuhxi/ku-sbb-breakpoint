#!/usr/bin/env bash
#SBATCH --job-name=kusbb-lof6
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# combo10gnjbag8r + LOF break features (lof6), production confirm vs combo10gnjbag8r
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CONF_JOBS=64
PY=~/venvs/kusbb/bin/python
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_|^lof_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB gnoise gjlr semiauto_bag8r lof6 --pat "$PAT" --tag combo10gnjbag8rlof
PYTHONPATH=. $PY probe/cmp_confirm.py combo10gnjbag8r combo10gnjbag8rlof
