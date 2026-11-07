#!/usr/bin/env bash
#SBATCH --job-name=kusbb-c2st2
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/c2st2.sbatch N MENU [MENU ...]   (hpc/shellc2st.sbatch for several menus in a row; they share outputs/aug files)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
for f in series_sim meta_sim X_sim; do [ -e outputs/aug/${f}_10k.parquet ] || cp outputs/aug/$f.parquet outputs/aug/${f}_10k.parquet; done
N=$1; shift
for M in "$@"; do
  echo "=== MENU $M"
  PYTHONPATH=. $PY probe/sim_series.py "$N" "$M"
  PYTHONPATH=. $PY aug_features.py sim 2>&1 | grep -v -i warn
  PYTHONPATH=. $PY probe/c2st_big.py --nomb
done
