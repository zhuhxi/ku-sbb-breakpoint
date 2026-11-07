#!/bin/bash
#SBATCH --job-name=kusbb-sabag
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:40:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/semiauto_bag.sbatch PARAMS.json TAG SEED0   (one more bag of 30k fakes -> full features -> semi-auto ABC)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore SA_JOBS=32
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/sim_series_ku2.py 30000 "$1" "$2" "$3"
PYTHONPATH=. $PY aug_features.py "$2" 2>&1 | grep -v -i warn
OMP_NUM_THREADS=32 PYTHONPATH=. $PY -u probe/semiauto_full.py "$2" "outputs/factory/semiauto_$2.parquet"
