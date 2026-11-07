#!/bin/bash
#SBATCH --job-name=kusbb-c2stnb
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/c2st_nb.sbatch N MENU   (fakes -> combo10 features -> no-break real-vs-fake C2ST, probe/c2st_full.py)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
for f in series_sim meta_sim X_sim; do [ -e outputs/aug/${f}_ok4.parquet ] || cp outputs/aug/$f.parquet outputs/aug/${f}_ok4.parquet; done
PYTHONPATH=. $PY probe/sim_series.py "$1" "$2"
PYTHONPATH=. $PY aug_features.py sim 2>&1 | grep -v -i warn
echo "== $2 with bits"; PYTHONPATH=. $PY probe/c2st_full.py 2>&1 | grep -v -i warn
echo "== $2 no bits"; PYTHONPATH=. $PY probe/c2st_full.py --nobits 2>&1 | grep -v -i warn
