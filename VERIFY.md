# Reproduction check (2026-11-07, local workstation)

Command (2 CPU cores pinned with taskset, OMP_NUM_THREADS=2, no network):

```bash
python predict.py --data data --feat outputs --ckpt ckpt --out submission.csv --threads 2
```

| Check | Result |
|---|---|
| Each of the 11 pair arms vs the training run's test predictions | max abs diff < 5e-16 |
| Interaction arm (top-800, 5 folds x 2 seeds) vs training run | max abs diff < 5e-16 |
| Final file vs `submission_blendD_r12.csv` | **max abs diff 0.0, identical ids/order** |
| Wall time / peak RAM | 31 s / 1.0 GB |
| md5 of the reference file `submission_blendD_r12.csv` | `a11350eb14a31ad52bc43d5d27a138ef` |

Why "r12": the blend is rank-based; arm predictions reproduce to ~1e-16, which can swap exact near-ties when ranking.
Rounding every vector to 12 decimals before ranking makes the output bit-for-bit reproducible. The first upload
(`submission_blendD.csv`, public 0.86077) was ranked without rounding and differs from the r12 file by at most 0.0005
(a few near-tied ranks; correlation 0.99999998). The checkpoints are the exact models of that submission, re-trained with
identical seeds and settings (test predictions identical to the original runs within 5e-16).

The training runs that produced the checkpoints: `src/probe/confirm_pair.py` (SAVE=ckpt, 64 threads, KU HPC) and
`src/brute/lgb_inter.py` (SAVE=ckpt, 96 threads, args `--top 162 --seeds 2 --seed-offset 0`, LR=0.02, extra pool
`combo10gnjx`), see TRAINING.md.
