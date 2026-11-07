# KU-SBB (KU Applied AI Challenge 2026) — team **BreakPoint**

Private technical package for qualification (tag `qualification`).

## Designated qualification submission

| | |
|---|---|
| Kaggle file | `submission_blendD.csv` |
| Submitted | 2026-11-07, ~18:15 UAE (UTC+4), team BreakPoint |
| Kaggle description | `blendD: avg of 11 pairwise-rank-interaction LGB arms x2 + inter2 x1, OOF .8606` |
| Public AUC | **0.86077** |
| Out-of-fold AUC | 0.8606 (StratifiedKFold(5, shuffle, 42), whole series) |
| Inference | `predict.py` + `ckpt/` (CPU, deterministic) |

Other submissions referred to in the report (identify by file name / time on the team's Kaggle Submissions page):

| File | Date (UAE) | Public AUC | Note |
|---|---|---|---|
| `submission_blendF_tabpfn.csv` | 2026-11-07 ~19:30 | 0.86324 | adds TabPFN arms (GPU); **not CPU-auditable, not designated** |
| `submission_blendB.csv` | 2026-11-07 ~17:00 | 0.85833 | CV-weighted blend of earlier arms |
| `submission_saf_bag6_inter.csv` | 2026-11-05 | 0.85830 | earlier best |

## Repository layout

- `report.pdf` / `report.html` — 3-page qualification report (method, validation, error analysis, limitations, sources, AI use).
- `predict.py` — CPU inference for the designated submission from the frozen checkpoints.
- `ckpt/` — frozen checkpoints: 11 pair-interaction arms × 5 folds × 3 seeds (LightGBM text models, gzip) + per-fold selected columns; interaction arm `inter_gnjt162_ck` × 5 folds × 2 seeds (selected pool columns, fill values, kept interaction list, LightGBM models).
- `src/` — all project code (feature pipeline, synthetic generator, semi-automatic estimators, models, HPC job scripts). `src/outputs/factory/ku_gen_*.json` are the calibrated generator parameters; `src/outputs/harvest_keep.json` the kept feature list.
- `TRAINING.md` — ordered recipe to regenerate every feature file and checkpoint from the raw competition data.
- `AI_DISCLOSURE.md`, `requirements.txt`.

**No competition data, labels or predictions are included** (per the rules). Put the official files in `data/` (`train_series.parquet`, `test_series.parquet`, `train_labels.csv`, `sample_submission.csv`).

## Inference (audit route: 2 CPU cores, < 30 min, no network)

```bash
pip install -r requirements.txt
# feature files in outputs/ (see TRAINING.md, stage 1-3), checkpoints in ckpt/
python predict.py --data data --feat outputs --ckpt ckpt --out submission.csv --threads 2
```

Verified on 2026-11-07: every arm reproduces the predictions of the training run to < 1e-15 (the pair arms take ~2.5 s each on 2 threads); the final file equals the submitted `submission_blendD.csv` (see `VERIFY.md`).

## Training (full pipeline)

See `TRAINING.md`. Feature generation and the 6 synthetic-data bags were run on the KU Madar HPC (Slurm, 64–96 CPU cores per job); the 11 pair arms take ~2.5 min each on 64 cores, the interaction arm ~25 min on 96 cores. All seeds are fixed in code.

## Team

Team alias: BreakPoint. Contact through the Kaggle competition host.
