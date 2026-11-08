# KU-SBB (KU Applied AI Challenge 2026) — team **BreakPoint**

Private technical package for qualification (tag `qualification`).

## Latest best version (submitted after the qualification freeze, for the organisers' information)

| Kaggle submission ID | File | Public AUC | OOF AUC | Notes |
|---|---|---|---|---|
| submitted 2026-11-08 ~22:52 UAE | **`submission_blendRFS.csv`** | **0.87550** | **0.8719** | best public and best OOF; **selected on Kaggle**; GPU route |
| 58123057 | `submission_blendNFS.csv` | 0.87416 | 0.8706 | GPU route |
| 58126032 | `submission_blendOFS.csv` | 0.87356 | 0.8710 | GPU route |

New since blendG2: CNN pretrained on synthetic series whose break class also contains transient nuisances (zero-shot AUC 0.83 without
real labels), forecast-error features from the public Chronos-2 and TimesFM-2.5 foundation models, four new TabPFN groups, full-training-set
TabPFN refits for test predictions, and a TabPFN stacker; blendRFS adds the public Chronos-2 model fine-tuned on our synthetic
series as a break classifier (zero-shot AUC 0.84 without real labels) and three TabPFN groups on its embeddings.
**How to reproduce blendRFS: `TRAINING_LATEST.md`, section 5** (sections 1-4 are its prerequisites); checkpoints of the pretrained CNNs: `ckpt/seq/`;
stacker inputs: `manifests/`; updated 3-page report: **`report_final.pdf`** (`REPORT_CHANGELOG.md`).
The designated qualification submission below is unchanged.

## Designated qualification submission

| | |
|---|---|
| Kaggle submission ID | **58101103** |
| Kaggle file | **`submission_blendG2.csv`** (GPU route) |
| Submitted | 2026-11-07, ~22:10 UAE (UTC+4) |
| Kaggle description | starts with `QUALIFICATION DESIGNATED (GPU route): blendG2 ...` |
| Public AUC | **0.86386** |
| Out-of-fold AUC | 0.8657 |
| How it is built | `TRAINING_GPU.md`, `make_blendG2.py` (needs one GPU for the TabPFN arms) |

**CPU-auditable fallback:** `submission_blendD_r12.csv` (submission ID 58097903), public 0.86076, OOF 0.8606 —
`predict.py` + `ckpt/` reproduce it bit-for-bit in 31 s on 2 CPU cores (`VERIFY.md`).

Kaggle final selections (changed 2026-11-08 ~23:30 UAE): **`submission_blendRFS.csv` + `submission_blendG2.csv`**.
`submission_blendD_r12.csv` is no longer selected but stays the CPU-auditable, bit-exact reference (`predict.py`, `VERIFY.md`).

Other submissions referred to in the report:

| File | Date (UAE) | Public AUC | Note |
|---|---|---|---|
| `submission_blendK.csv` (ID 58113676) | 2026-11-08 ~07:46 | 0.86971 | blendG2 + MT/raw/side TabPFN groups (report_final.pdf) |
| `submission_blendF_tabpfn.csv` | 2026-11-07 ~19:30 | 0.86324 | earlier GPU blend (no time reversal) |
| `submission_blendD.csv` | 2026-11-07 ~18:15 | 0.86077 | same models as r12, ranked without rounding |
| `submission_blendB.csv` | 2026-11-07 ~17:00 | 0.85833 | CV-weighted blend of earlier arms |

## Repository layout

- `report_final.pdf` / `report_final.html` — updated 3-page report (best version, ablations); `TRAINING_LATEST.md` — recipe for blendNFS / blendOFS.
- `report.pdf` / `report.html` — 3-page qualification report (frozen at the qualification tag) (method, validation, error analysis, limitations, sources, AI use).
- `predict.py` — CPU inference for the designated submission from the frozen checkpoints.
- `ckpt/` — frozen checkpoints: 11 pair-interaction arms × 5 folds × 3 seeds (LightGBM text models, gzip) + per-fold selected columns; interaction arm `inter_gnjt162_ck` × 5 folds × 2 seeds (selected pool columns, fill values, kept interaction list, LightGBM models).
- `src/` — all project code (feature pipeline, synthetic generator, semi-automatic estimators, models, HPC job scripts). `src/outputs/factory/ku_gen_*.json` are the calibrated generator parameters; `src/outputs/harvest_keep.json` the kept feature list.
- `TRAINING_GPU.md`, `make_blendG2.py` — GPU route of the designated submission (time reversal + TabPFN).
- `TRAINING.md` — ordered recipe to regenerate every feature file and checkpoint from the raw competition data.
- `AI_DISCLOSURE.md`, `requirements.txt`.

**No competition data, labels or predictions are included** (per the rules). Put the official files in `data/` (`train_series.parquet`, `test_series.parquet`, `train_labels.csv`, `sample_submission.csv`).

## Inference (audit route: 2 CPU cores, < 30 min, no network)

```bash
pip install -r requirements.txt
# feature files in outputs/ (see TRAINING.md, stage 1-3), checkpoints in ckpt/
python predict.py --data data --feat outputs --ckpt ckpt --out submission.csv --threads 2
```

Verified on 2026-11-07: every arm reproduces the predictions of the training run to < 1e-15 (the pair arms take ~2.5 s each on 2 threads); the final file equals the designated `submission_blendD_r12.csv` exactly (max abs diff 0.0; 31 s, 1 GB RAM; see `VERIFY.md`).

## Training (full pipeline)

See `TRAINING.md`. Feature generation and the 6 synthetic-data bags were run on the KU Madar HPC (Slurm, 64–96 CPU cores per job); the 11 pair arms take ~2.5 min each on 64 cores, the interaction arm ~25 min on 96 cores. All seeds are fixed in code.

## Team

Team alias: BreakPoint. Contact through the Kaggle competition host.
