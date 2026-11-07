#!/usr/bin/env bash
#SBATCH --job-name=kusbb-gnjsa2
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# combo10gnj + 31 semi-automatic v2 (200k fakes) ABC columns (probe/semiauto.py), production confirm, compared with combo10gnj
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
PY=~/venvs/kusbb/bin/python
PAT='^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$|^g_|^sa_'
PYTHONPATH=. $PY confirm.py st_RES10 st_ARES10 fe fe2 ar2 cost genA genB gnoise gjlr semiauto2 --pat "$PAT" --tag combo10gnjsa2
PYTHONPATH=. $PY probe/cmp_confirm.py combo10gnj combo10gnjsa2
