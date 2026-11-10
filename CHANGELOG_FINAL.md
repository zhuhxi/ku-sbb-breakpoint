# Change log: qualification freeze (tag `qualification`, 2026-11-07) → finals freeze (tag `final`, 2026-11-10)

All times are UAE (UTC+4). Every OOF number is a 5-fold StratifiedKFold(5, shuffle, 42) on whole series with all label-dependent
steps inside the training folds; "zero-shot" = AUC on the real training set of a model trained only on synthetic series (no real label).
Full day-by-day notes with every negative result are kept by the team (LESSONS.md, not in git); the recipe is `TRAINING_LATEST.md`.

## Submissions and selections

| Date | Submission | Public AUC | OOF | Kaggle selections after this |
|---|---|---|---|---|
| 11-07 22:10 | blendG2 (designated for qualification) | 0.86386 | 0.8657 | blendG2 + blendD_r12 |
| 11-08 07:46 | blendK | 0.86971 | 0.8684 | unchanged |
| 11-08 | blendNFS / blendOFS | 0.87416 / 0.87356 | 0.8706 / 0.8710 | unchanged |
| 11-08 22:52 | blendRFS | 0.87550 | 0.8719 | blendRFS + blendG2 (11-08 ~23:30) |
| 11-09 13:06 | blendTFS (ID 58158556) | 0.87641 | 0.8724 | unchanged |
| 11-09 18:48 | blendWFS | Kaggle scoring "Error" (not selected) | 0.8732 | unchanged |
| 11-10 00:24 | **blendXFS** (ID 58174578) | **0.87791** | **0.8741** | **blendXFS + blendTFS (11-10)** |

## Method changes

1. **CNN pretrained on synthetic series with nuisances inside break series** (11-08). Zero-shot 0.8115 → 0.830; 53 label-free features.
2. **Forecast-error features from public foundation models** Chronos-2 and TimesFM-2.5 (11-08; help only together with 1).
3. **TabPFN stacker** on the logits of all arms + label-free context; TabPFN test arms refitted on the full training set (11-08, blendNFS/OFS).
4. **Chronos-2 fine-tuned as a break classifier on synthetic data** (11-08, blendRFS): zero-shot 0.8408; three TabPFN groups on its embeddings.
5. **Generator fitted for realism** (11-09): criterion changed from "zero-shot score" to a real-vs-fake classifier (C2ST, lower = more real),
   because realism fixes did not move zero-shot score but are what the method claims. genT 0.662 → genR10 0.580 in 10 steps
   (`src/outputs/factory/ku_gen_genR*.json`, new default-off options in `src/probe/ku_gen.py`).
6. **Chronos-2 classifier retrained on genR10 fakes** (c2mv_R10 / R10b, zero-shot 0.8442 / 0.8455) + R10 TabPFN arms → blendTFS.
7. **More realistic fakes in the semi-automatic estimators** (bag8r: +0.0074 in the quick LightGBM test) + rebuilt pair arms → blendWFS.
8. **Segment noise classifier** (`src/probe/seg_noise.py`, trained on genR10 fakes) after an oracle test showed the pre-segment noise
   distribution is the largest reducible error (+0.0070 of +0.0095) → 8 TabPFN arms + stacker context → blendXFS.

## Measured, did not help (kept out of the blends)

| Idea | Result |
|---|---|
| Real labels in Chronos-2 fine-tuning (real + fake mix, fold-clean) | fold-0 0.8445–0.8497 vs zero-shot 0.8483: no reliable gain |
| genR10 fakes in TabPFN context (with / without real-fake flag) | 0.8512–0.8528 vs 0.8520: no gain (fakes help LightGBM only) |
| TiRex / TOTO forecast-error features | TabPFN arms −0.0008 to +0.0004 |
| genR11 (level breaks ×1.1) | more real break statistics, classifier unchanged (0.8441 vs 0.8442) |
| TabPFN arms on a second CV split | stacker 0.8732 vs 0.8734: no |
| Linear probe on full Chronos embeddings | 0.813–0.819 < zero-shot 0.843 |
| Negation test-time augmentation for Chronos | 4 views 0.8446 vs fwd+rev 0.8436 (within chunk noise) |

## Code added or changed in `src/` since the qualification tag

- New: `probe/seg_noise.py`, `probe/oracle_nuis.py`, `probe/ceiling.py`, `probe/gen_c2st.py`, `probe/gen_disc.py`, `probe/gen_ks.py`,
  `probe/c2st_brk.py`, `probe/c2st_brkvar.py`, `probe/disc_c2.py`, `probe/c2mv_mix.py`, `probe/c2_tta.py`, `probe/make_zs.py`,
  `probe/tpf_aug.py`, `probe/lof_feat.py`, `probe/spec_rw.py`, `probe/tirex_feats.py`, `probe/toto_feats.py`;
  HPC jobs `hpc/gnjbag8r.sbatch`, `hpc/gnjsn.sbatch`, `hpc/gnjlof.sbatch`, `hpc/pair_any.sbatch`, `hpc/sweep_tasks*.txt`.
- Changed: `probe/ku_gen.py` (new generator options, all default off), `probe/stack_tpf.py` / `probe/stack_tpf_sub.py` (`CTXSN`, `CTXSNF`,
  more arms), `probe/c2mv_score.py`, `probe/feats.py` (keeps `^sn_` columns), `probe/semiauto_full.py` (classes absent from fakes → 0 column),
  `probe/weighted_aug.py`, `probe/genfeat_all.py`, `probe/joint_lr.py`, `probe/noise_lr.py`.
- Generator configs: `src/outputs/factory/ku_gen_genT.json`, `ku_gen_genR1.json` … `ku_gen_genR11.json`, variants in `src/outputs/factory/var/`.
- Checkpoints: `ckpt/seq/seqpre_mnR9_mt_c64.pt`, `ckpt/seq/seqpre_mnR10_mt_c64.pt`; manifests `manifests/blendXFS_*`.
- Slides and figure scripts: `slides/`.
- Report: `report_final.html` / `.pdf` updated to blendXFS / blendTFS (still 3 pages; entries in `REPORT_CHANGELOG.md`).

## Known limitations at the freeze

- Inference of the final blends needs one ≥ 16 GB GPU and the fine-tuned Chronos-2 checkpoints (not in git, ~486 MB each, SHA-256 in
  `TRAINING_LATEST.md`); the CPU-only bit-exact route is blendD_r12 (`predict.py`).
- Chronos fine-tuning steps vary by about ±0.004 zero-shot AUC between fake chunks; retraining gives equal quality, not identical weights.
- Stacker settings were chosen on the same OOF table, so the stacked OOF may be slightly optimistic (public-board sd ≈ 0.009).
