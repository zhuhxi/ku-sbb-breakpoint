#!/usr/bin/env bash
#SBATCH --job-name=kusbb-inter
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=700G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/inter.sbatch TOP SEEDS
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
TOP=$1; SEEDS=${2:-2}
~/venvs/kusbb/bin/python -u brute/lgb_inter.py --extra outputs/factory/combo10x_train.parquet --top $TOP \
    --tag _combo10t$TOP --seeds $SEEDS
