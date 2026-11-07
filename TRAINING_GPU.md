# GPU route: how the designated submission blendG2 is produced

blendG2 (Kaggle public **0.86386**, OOF 0.8657) = the CPU route (blendD: 26 pair arms + interaction arm, see TRAINING.md) plus
two new ingredients. Run every command from the project root with `PYTHONPATH=.`; all splits are StratifiedKFold(5, shuffle, 42)
and every selection / augmentation happens inside the training part of each fold.

## 1. Time-reversal data (CPU, HPC, minutes per step)
Each series is reversed (`time -> n-1-time`, `period -> 1-period`, so the boundary moves to n-n0; the label is kept).
1. Reversed raw files: `src/probe/rev_feats.py` logic (also applied to test) -> `REV/data/{train,test}_series.parquet`.
2. Every feature family of TRAINING.md stages 1-3 re-run with REV as working directory (same scripts, relative paths):
   features.py, brute/magic_b.py, harvest.py, rawstats.py, `factory.py st RES10 | st ARES10 | fe fe2 | ar2 cost`,
   probe/genfeat_all.py, probe/noise_lr.py, probe/joint_lr.py.
3. Semi-automatic columns for the reversed series: `REV_ROOT=REV SA_WEIGHT=1 SA_JOBS=32 python src/probe/semiauto_full.py simk{2..7} OUT`
   (the reversed rows are scored by the same estimators; the original rows are unchanged, checked <= 1.5e-11); bag6w = mean of 6 bags.

## 2. Time-reversal pair arms (CPU, HPC 64 cores, ~3.5 min each)
`J=64 TOPSEED=T [SPLIT=S] python src/probe/confirm_pair_rev.py K TAG semiauto_bag6w REV` — train on originals + reversed copies,
predict = mean(original, reversed). Arms used: (K,T) = (60,0) (50,1) (70,2) (60,5) (65,6) and SPLIT=TOPSEED in {43,44,45} with K=60.

## 3. TabPFN arms (GPU; we used 2 x RTX 4090, ~1-5 min per 5-fold run)
`TABPFN_TOKEN=... CUDA_VISIBLE_DEVICES=g python src/probe/tabpfn_top.py K folds...` then `... K merge`
(per fold: top-K features by a quick LightGBM on the inner-train part, rank-transformed, whole fold-train as context, 8 estimators).
- 8 plain arms: K=60, 100, (80,TS=1), (70,TS=2), (65,TS=3), (90,TS=4), (75,TS=5), (60,TS=6).
- 5 reversed-context arms: `REV=REV CTXREV=1` (context = originals + reversed copies, prediction = mean of both):
  K=60, (70,TS=2), (80,TS=1), (65,TS=3), (60,TS=4).
TabPFN 3.5 (`pip install tabpfn`, license accepted, weights downloaded once). Public pretrained model, used under its license.

## 4. Blend
`python make_blendG2.py --root <project>` -> `submission_blendG2.csv` (verified identical to the uploaded file, max diff 0.0).

## Resources / audit note
TabPFN inference needs a GPU: on 2 CPU cores one fold model did not finish 100 test rows in 7 minutes. The designated GPU route
therefore needs an audit machine with one consumer GPU (>= 16 GB, < 10 min for the whole blend). The fully CPU-auditable fallback is
`submission_blendD_r12.csv` (predict.py + ckpt/, 31 s on 2 cores, bit-exact), also selected on Kaggle.
