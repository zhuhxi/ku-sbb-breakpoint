# Latest (post-qualification-freeze) submissions: blendNFS and blendOFS

| File | Kaggle submission ID | Public AUC | OOF AUC |
|---|---|---|---|
| `submission_blendNFS.csv` | **58123057** | **0.87416** (our best public) | 0.8706 |
| `submission_blendOFS.csv` | 58126032 | 0.87356 | 0.8710 (our best OOF) |

Both extend the designated GPU route (blendG2 → blendK, see TRAINING_GPU.md and report_final.pdf) with four new ingredients.
All splits are StratifiedKFold(5, shuffle, 42) on whole series; every label-dependent step (feature selection, TabPFN context,
blend weights, stacker) is fitted inside the training part of a fold. Run from the project root with `PYTHONPATH=.`.
No competition data, labels or predictions are in this repository.

## 1. Synthetic pretraining of a sequence CNN, with nuisances also inside break series (GPU, ~2 min per step)
Generator: `src/probe/ku_gen.py` with the calibrated parameters in `src/outputs/factory/`.
Key finding: real break series also contain transient nuisances (short level bumps, variance bursts, drifts). Adding them to the
synthetic BREAK class (`ku_gen_nuis.json` = three-bin parameters + `nuis_brk: [decoy, vburst, drift]`) raised the zero-shot AUC of the
pretrained CNN on the real training set from 0.8115 to 0.828–0.830 **without using any real label**.

1. Fresh fakes (50k per file, ~3 s each on 32 cores): `python src/probe/sim_tensor.py 50000 src/outputs/factory/ku_gen_nuis.json DIR/sim_i.npz SEED`
   (`REVP=.5` makes half of the fakes time-reversed; label kept).
2. Continued pretraining (dilated CNN, multi-task heads for break kind / family / sizes / noise type / nuisances):
   `SIMDIR=DIR MT=1 INIT=prev.pt LRMAX=5e-4 OUT=new.pt python src/probe/seq_pretrain.py 1`
   Lineage of the checkpoints in `ckpt/seq/` (all trained on synthetic data only):
   - `seqpre_mtw_c64.pt`: three-bin fakes (600k, multi-epoch, realism-weighted continuation)
   - `seqpre_mtf1_mt_c64.pt` (mf1): mtw + 1 epoch on 600k fresh three-bin fakes (seeds 20,000,000+), zero-shot 0.8115
   - `seqpre_mn_mt_c64.pt` (mn): mtw + 1 epoch on 600k nuis fakes (seeds 40,000,000+), zero-shot 0.8281
   - `seqpre_mn2_mt_c64.pt` (mn2): mn + 1 epoch on 600k new nuis fakes (seeds 40,600,000+), 0.8288
   - `seqpre_mnr_mt_c64.pt` (mnr): mn2 + 1 epoch on 600k nuis fakes with REVP=.5 (seeds 42,400,000+), 0.8297 (reversed view 0.8237)
   - `seqpre_mnx_mt_c64.pt` (mnx): mn + 1 epoch on `ku_gen_nuisx2.json` fakes (seeds 41,800,000+), 0.8295
3. Label-free features for real series (original and time-reversed tensors, same PCA):
   `python src/probe/seq_prep.py` and `REVSEQ=1 python src/probe/seq_prep.py`, then
   `REVOUT=REV/outputs/factory/seqembX.parquet python src/probe/seq_emb.py ckpt/seq/<ckpt> outputs/factory/seqembX.parquet se_X_`
   (PCA32 of the pooled embedding + zero-shot logit + 20 auxiliary-head outputs = 53 columns).

## 2. Forecast-error features from public time-series foundation models (GPU, < 1 min each)
Forecast the first 64 post-boundary points from the pre-boundary context and compare with the actual values (scaled by the
80% prediction-interval width), plus the same back-test inside the pre segment for self-normalisation; also on reversed series.
- Chronos-2 (amazon/chronos-2, Apache-2.0): `python src/probe/chronos_feats.py amazon/chronos-2 all outputs/factory/chronos_2.parquet` (+ `REVSEQ=1` into REV)
- TimesFM-2.5 (google/timesfm-2.5-200m-pytorch, `pip install timesfm`): `python src/probe/timesfm_feats.py outputs/factory/timesfm.parquet` (+ `REVSEQ=1`)
They help TabPFN only together with the nuisance-pretrained CNN features (with the older CNN features Chronos hurt).

## 3. New TabPFN groups (raw values, original + reversed features side by side; `src/probe/tabpfn_top.py`)
Common settings: `REV=REV SIDE=1 RAWV=1`, 8 estimators, per-fold top-K by a quick LightGBM on the inner-train part.
| Group | EXTRA | (K, TS) bags | group OOF |
|---|---|---|---|
| mf1 | `seqembmf1` | (60,0) (70,2) (50,1) (65,3) (40,5) (80,4) (55,6) | 0.8685 |
| nuis | `seqembmnr` ×6 bags + `seqembmn` (60,0)(70,2) + `seqembmn2` (60,0) | mnr: (60,0) (70,2) (50,1) (65,3) (40,5) (80,4) | 0.8689 |
| cb (Chronos) | `seqembmnr,chronos_2` | (60,0) (70,2) (50,1) (65,3) (40,5) (80,4) (55,6) (75,7) | 0.8698 |
| tf (TimesFM) | `seqembmnr,timesfm` | (70,2) (55,6) (60,0) (50,1) (65,3) | 0.8702 |
Test predictions: every TabPFN arm is refitted with the WHOLE training set as context (`FULL=1 ... tabpfn_top.py K 0`, ~30 s per arm);
on OOF a full-context model beat the average of four 80%-context models (0.8676 vs 0.8672).

## 4. Blend and TabPFN stacker
Linear rank blend (group = mean of ranks; `src/probe/stack_tpf.py` holds the exact definition):
- blendN = 2·pairs + 1·interaction + 2·reversal-pairs + 6·rctx + 9·MT + 3·rctx-MT + 9·raw + 3·rctx-raw + 9·side-raw + 6·siderw + 12·mf1 + 24·nuis + 144·cb (OOF 0.8701)
- blendO = same with 96·cb + 96·tf (OOF 0.8704)

Stacker: TabPFN on the logits of all arm OOF ranks (lists in `manifests/blendNFS_stacker_arms.txt` / `manifests/blendOFS_stacker_arms.txt`)
+ the blend + context (series length, boundary position, semi-automatic family / no-break / noise-type estimates, rescaled flag,
zero-shot / auxiliary outputs of the mn2 CNN). Fitted on all 24k OOF rows, applied to the test arm predictions; final = rank(stacker) + rank(blend), 1:1.
Validated with an outer 5-fold CV on the OOF table (seeds 11 and 13): blendNFS 0.8706 / 0.8706, blendOFS 0.8710 / 0.8709, all 5 folds above the linear blend.
```
FULLT=1 ARMS=1 SW=1 XARMS=outputs/arms/zsnuis_oof.npy,outputs/arms/zsmnrrev_oof.npy CTXEXTRA=outputs/factory/seqembmn2.parquet \
  OUT=outputs/submission_blendOFS.csv python src/probe/stack_tpf_sub.py
```
(`zsnuis` = mean rank of the zero-shot logits of mn, mn2, mnx; `zsmnrrev` = zero-shot logit of mnr on the reversed series.)

## Hardware and honesty notes
- GPU: 2 × RTX 4090 (TabPFN, CNN pretraining, Chronos-2, TimesFM); CPU feature pipeline on the KU HPC. The whole test-time
  inference of these blends needs one ≥ 16 GB GPU; a CPU-only LightGBM route reaches OOF 0.8651.
- Group weights after the LightGBM stage were chosen on the full OOF with a CI > 0 and 5/5-fold rule, so the OOF numbers may be
  slightly optimistic; public scores (0.87416 / 0.87356) are consistent with them (public-board sd ≈ 0.009).
- These submissions were made after the qualification freeze; the designated qualification submission is unchanged (README).
