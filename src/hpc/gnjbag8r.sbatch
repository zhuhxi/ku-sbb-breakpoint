#!/usr/bin/env bash
#SBATCH --job-name=kusbb-bag8r
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# combo10gnj + 31 semi-automatic columns averaged over 6 realism-weighted fake bags (semiauto_bag6w), production confirm, compared with combo10gnj
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export CONF_JOBS=64
PY=~/venvs/kusbb/bin/python
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^saf_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB gnoise gjlr semiauto_bag8r --pat "$PAT" --tag combo10gnjbag8r
PYTHONPATH=. $PY probe/cmp_confirm.py combo10gnjbag6w combo10gnjbag8r
