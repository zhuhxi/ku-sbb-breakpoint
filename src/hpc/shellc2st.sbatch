#!/usr/bin/env bash
#SBATCH --job-name=kusbb-shellc2st
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/shellc2st.sbatch N MENU   (fakes from MENU -> combo10 features -> real-vs-fake test without mb_ columns)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
for f in series_sim meta_sim X_sim; do [ -e outputs/aug/${f}_10k.parquet ] || cp outputs/aug/$f.parquet outputs/aug/${f}_10k.parquet; done
PYTHONPATH=. $PY probe/sim_series.py "$1" "$2"
PYTHONPATH=. $PY aug_features.py sim 2>&1 | grep -v -i warn
PYTHONPATH=. $PY probe/c2st_big.py --nomb
