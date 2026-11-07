#!/usr/bin/env bash
#SBATCH --job-name=kusbb-inter2
#SBATCH --partition=prod
#SBATCH --account=kuin0094
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=600G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch hpc/inter2.sbatch TOP SEEDS SEED_OFFSET LR TAG
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export NTHREADS=$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK LR=$4
~/venvs/kusbb/bin/python -u brute/lgb_inter.py --extra outputs/factory/combo10x_train.parquet --top $1 \
    --seeds $2 --seed-offset $3 --tag _combo10$5
