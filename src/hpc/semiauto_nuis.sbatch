#!/bin/bash
#SBATCH --job-name=kusbb-sanuis
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --exclusive
#SBATCH --exclude=cn-0901-02
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/semiauto_shock.sbatch N  (same seeds as simk3, generator + nuisances also in break series -> full features -> semi-auto)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/sim_series_ku2.py "$1" outputs/factory/ku_gen_nuis.json simnuis 8000000
PYTHONPATH=. $PY aug_features.py simnuis 2>&1 | grep -v -i warn
export SA_JOBS=32 OMP_NUM_THREADS=32
PYTHONPATH=. $PY -u probe/semiauto_full.py simnuis outputs/factory/semiauto_nuis.parquet 2>&1 | grep -v -i warn
