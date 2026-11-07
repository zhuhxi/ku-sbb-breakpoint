#!/bin/bash
#SBATCH --job-name=kusbb-saxtra
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
# semi-auto models from bags simk3..7, each also scoring the simk2 fakes (out-of-bag) -> outputs/factory/semiauto_TAG_x2.parquet
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export SA_JOBS=32 OMP_NUM_THREADS=32 PYTHONWARNINGS=ignore
for t in simk3 simk4 simk5 simk6 simk7; do
  PYTHONPATH=. ~/venvs/kusbb/bin/python -u probe/semiauto_full.py $t outputs/factory/semiauto_${t}_x2.parquet simk2 | tail -1
done
