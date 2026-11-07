#!/bin/bash
#SBATCH --job-name=kusbb-saf3
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
# usage: sbatch hpc/semiauto_full.sbatch N  (retuned three-bin fakes with hidden parameters -> full features -> semi-automatic ABC on real)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
PY=~/venvs/kusbb/bin/python
PYTHONPATH=. $PY probe/sim_series_ku2.py "$1" outputs/factory/ku_gen_size_three-bin.json simk3 8000000
PYTHONPATH=. $PY aug_features.py simk3 2>&1 | grep -v -i warn
PYTHONPATH=. $PY probe/semiauto_full.py simk3 outputs/factory/semiauto_full3.parquet 2>&1 | grep -v -i warn
