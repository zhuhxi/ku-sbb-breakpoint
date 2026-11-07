#!/usr/bin/env bash
#SBATCH --job-name=kusbb-sweep
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sweep/%x_%A_%a.out
# usage: sbatch --array=1-N%40 hpc/sweep.sbatch hpc/sweep_tasks.txt   (one "FAMILY TRANSFORM" per line)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
read FAM TR < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$1")
PY=~/venvs/kusbb/bin/python
[ -f outputs/factory/${FAM}_${TR}.parquet ] || $PY factory.py $FAM $TR
$PY screen.py --base combo10 --tag sw_${FAM}_${TR} ${FAM}_${TR}
