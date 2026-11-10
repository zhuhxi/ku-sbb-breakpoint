# Latest submissions: blendXFS (best, public 0.87791) back to blendNFS

**Final Kaggle selections (2026-11-10): blendXFS (public 0.87791, OOF 0.8741) + blendTFS (public 0.87641); their recipe is section 6.**
Sections 1-5 are its prerequisites (section 5 = blendRFS).

| File | Kaggle submission ID | Public AUC | OOF AUC |
|---|---|---|---|
| **`submission_blendXFS.csv`** | (submitted 2026-11-10 ~00:24 UAE) | **0.87791** (best; selected) | **0.8741** |
| `submission_blendTFS.csv` | (submitted 2026-11-09 ~13:06 UAE) | 0.87641 (selected) | 0.8724 |
| `submission_blendRFS.csv` | (submitted 2026-11-08 ~22:52 UAE) | 0.87550 | 0.8719 |
| `submission_blendNFS.csv` | 58123057 | 0.87416 | 0.8706 |
| `submission_blendOFS.csv` | 58126032 | 0.87356 | 0.8710 |

blendNFS and blendOFS extend the designated GPU route (blendG2 → blendK, see TRAINING_GPU.md and report_final.pdf) with four new ingredients.
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

## 5. blendRFS: Chronos-2 fine-tuned as a break classifier on synthetic data (GPU, every training step < 5 min on one RTX 4090)

### 5.1 Generator genS and fresh fakes
`genS` = `ku_gen_nuisld0.json` (nuisances inside break series, no level delay) with break-kind shares p_kind = [.36, .32, .07, .25]
(more shape breaks); file `src/outputs/factory/ku_gen_genS.json` (identical to `ku_gen_h_shape25d7.json`). It raised the zero-shot AUC of
the CNN from 0.8297 to 0.8330. Fakes (50,000 series per file; seed of file `sim_i` = BASE + 50,000·i, i = 0..11):
```
for i in $(seq 0 11); do python src/probe/sim_tensor.py 50000 src/outputs/factory/ku_gen_genS.json SIMDIR/sim_$i.npz $((BASE + i*50000)); done
```
A SIMDIR must hold exactly these 12 files. The training scripts read `sorted(glob(SIMDIR/sim_*.npz))[FOFF:FOFF+NFILE]`; the order is
lexicographic, so index k -> file sim_[0,1,10,11,2,3,4,5,6,7,8,9][k].
The genS CNNs of the zero-shot arm `zsgs`: `src/probe/gen_zs.sh` (fresh genS fakes + 1 epoch from `seqpre_mtw_c64.pt`); checkpoints
`ckpt/seq/seqpre_h_shape25d7_mt_c64.pt`, `seqpre_h_shape25b_mt_c64.pt` (replicate with new seeds), `seqpre_gsr_mt_c64.pt`
(shape25d7 + 1 epoch on genS fakes with `REVP=.5`). `python src/probe/make_zsgs.py` writes `outputs/arms/zsgs_{oof,test}.npy`.

### 5.2 Chronos-2 classifier (`src/probe/c2_ft.py`, two-variate version `src/probe/c2mv_ft.py`)
Public `amazon/chronos-2` (120M parameters, Apache-2.0; `pip install chronos-forecasting`), used as an encoder. Input: the window of
W = 512 points before and 512 after the boundary of the robust-z series (NaN outside the series). Pooled pre-boundary tokens, post-boundary
tokens, their difference, the REG token and the output token (5 × 768) feed a 2-layer head (break / no break). `MT=1` adds the
auxiliary loss on the hidden generator parameters (break kind, family, three break sizes, noise type, nuisances). `c2mv_ft.py` feeds the
z series and its AR(10) residual as two variates of one group (Chronos-2 group attention) and averages their tokens.
Labels are synthetic only; real labels are used only to report the zero-shot AUC.
```
python src/probe/c2_ft.py   SIMDIR FOFF NFILE OUT.pt [INIT.pt]      # env: WIN (512), MT, BLR (backbone lr, default 2e-5)
python src/probe/c2mv_ft.py SIMDIR FOFF NFILE OUT.pt [INIT.pt]      # bs 64
```
Lineage (each row is one < 5 min step; data = genS fakes, seed BASE):

| checkpoint | init | data (BASE, FOFF, NFILE) | env | zero-shot AUC (fwd) |
|---|---|---|---|---|
| c2ft_a / b / c | Chronos-2 / a / b | 72,000,000 (6 files in total over a-c: FOFF 1 6, 7 5) | WIN=256 | 0.806 (c) |
| w512 | c | 72,600,000, 6 4 | WIN=512 | 0.825 |
| **w512b** (features `c2b`) | w512 | 73,200,000, 0 4 | WIN=512 | 0.8252 |
| w512c | w512b | 73,200,000, 4 4 | WIN=512 | 0.8271 |
| mt → mt2 → **mt3** (features `c2m`) | w512c → mt → mt2 | 73,200,000, (0 4), 4 4, 8 4 | WIN=512 MT=1 | 0.8322 → 0.8343 → 0.8367 |
| c2mv_a | mt3 | 74,400,000, 0 1 | WIN=512 MT=1 (two-variate) | 0.8391 |
| c2mv_c | c2mv_a | 74,400,000, 3 2 | WIN=512 MT=1 BLR=1e-5 | 0.8398 |
| **c2mv_d** (features `c2v`) | c2mv_c | 74,400,000, 5 2 | WIN=512 MT=1 BLR=1e-5 | 0.8408 |

Honest note: the synthetic directories were regenerated several times during the day, so the data of the first steps (c2ft_a, mt) is
recorded only up to the seed block, and chunk-to-chunk zero-shot AUC varies by about ±0.005; retraining from scratch gives a model of the
same quality but not identical weights. **The exact three checkpoints used by blendRFS** are therefore archived (see 5.6) with SHA-256:
```
7089cd6eeaceb6bddf4057eb5f96a273c3d729281bd36150e61294ac66e9c3f6  c2ft_w512b.pt
c4c6a003c7c379c5d7e060bb645399f48c4ff29ae6fe658cf8b1fb6c6d9ba786  c2ft_mt3.pt
41bf6cc6d6329652f48f52be21e17c7531793035078b1a41d78d86992047e1bd  c2mv_d.pt
```
(The auxiliary head of the MT models was not in the optimizer, i.e. it stayed a fixed random projection that the backbone learned to fit;
we found this after blendRFS and keep it for reproducibility. `AUXOPT=1` trains it; it gave no clear gain, 0.8413 vs 0.8418.)

### 5.3 Label-free features from the classifiers (real train + test, original and reversed series)
```
python src/probe/c2_score.py   outputs/arms/c2ft_w512b.pt c2b     # -> zsc2b*, outputs/factory/seqembc2b.parquet (PCA32 + logit)
python src/probe/c2_score.py   outputs/arms/c2ft_mt3.pt   c2m     # -> zsc2m{,rev}_{oof,test}.npy, seqembc2m (PCA32 + logit + 20 aux)
python src/probe/c2mv_score.py outputs/arms/c2mv_d.pt     c2v     # -> zsc2v{,rev}_{oof,test}.npy, seqembc2v
```
(env `REV` = root of the reversed-series feature tree; the same PCA is applied to both views; ~3 min each.)

### 5.4 Three more TabPFN groups (same settings as section 3: `REV=REV SIDE=1 RAWV=1`; `FULL=1 ... K 0` for the test refits)
| Group (weight) | EXTRA | (K, TS) bags | group OOF |
|---|---|---|---|
| c2 (72) | `seqembc2b,timesfm` | (70,2) (55,6) (60,0) (50,1) (65,3) | 0.8682 |
| cm (48) | `seqembc2m,timesfm` (60,0)(70,2)(55,6) + `seqembmnr,seqembc2m,timesfm` (70,2) | | 0.8689 |
| cv (192) | `seqembc2v,timesfm` | (70,2) (55,6) (60,0) (50,1) | 0.8697 |
Example: `REV=REV SIDE=1 RAWV=1 EXTRA=seqembc2v,timesfm TS=2 TAG=sidec2v_2 python src/probe/tabpfn_top.py 70 0 1 2 3 4`, then
`TAG=sidec2v_2 python src/probe/tabpfn_top.py 70 merge`, and for the test predictions
`FULL=1 REV=REV SIDE=1 RAWV=1 EXTRA=seqembc2v,timesfm TS=2 TAG=sidec2v_2full python src/probe/tabpfn_top.py 70 0`.
The c2v group has correlation 0.983 with the rest of the blend (the most diverse group).

### 5.5 Blend and stacker
blendR = blendO groups + 72·c2 + 48·cm + 192·cv (exact weights in `src/probe/stack_tpf_sub.py`, dict `W`): OOF 0.8715 (`submission_blendRF.csv`).
Stacker as in section 4, with the 242 arms listed in `manifests/blendRFS_stacker_arms.txt` (5 label-free zero-shot arms + 237 OOF arm files;
every `tpfK<tag>_S` file is `tabpfn_top.py K` with TAG `<tag>_S`) and context = mn2 CNN outputs + c2m / c2v auxiliary outputs:
```
FULLT=1 ARMS=1 SW=1 \
XARMS=outputs/arms/zsgs_oof.npy,outputs/arms/zsc2m_oof.npy,outputs/arms/zsc2mrev_oof.npy,outputs/arms/zsc2v_oof.npy,outputs/arms/zsc2vrev_oof.npy \
CTXEXTRA=outputs/factory/seqembmn2.parquet,outputs/factory/seqembc2m.parquet,outputs/factory/seqembc2v.parquet \
OUT=outputs/submission_blendRFS.csv python src/probe/stack_tpf_sub.py
```
Outer 5-fold CV of the stacker on the OOF table (seeds 11, 13): 0.8719 / 0.8719 (linear blendR 0.8715). Final = rank(stacker) + rank(blendR), 1:1.
Kaggle public 0.87550 (blendNFS 0.87416), consistent with the OOF gain.

### 5.6 Where the Chronos-2 checkpoints are
The three fine-tuned checkpoints (~480 MB each, trained on synthetic data only) exceed GitHub's 100 MB file limit, so they are not in
this repository. They are kept by the team on the local workstation (`outputs/arms/`) and archived on the KU HPC storage
(`/dpc/kuin0094/hongxi/c2ckpt/`), and can be provided to the organisers on request; check them with the SHA-256 sums in 5.2.
Put them in `outputs/arms/` and run 5.3 → 5.5. Without them, retrain with the lineage in 5.2 (about 1 GPU-hour in total).

## 6. blendTFS → blendWFS → blendXFS (final selections: blendXFS + blendTFS)

| File | Public AUC | OOF AUC (stacker, outer CV seeds 11 / 13) | New ingredient |
|---|---|---|---|
| `submission_blendTFS.csv` | 0.87641 (selected) | 0.8724 / 0.8724 (3:1) | realistic generator genR10 → Chronos-2 classifier c2mv_R10 |
| `submission_blendWFS.csv` | still "Pending" on Kaggle | 0.8732 / 0.8732 (3:1) | genR10 semi-auto bags (bag8r) + 6 pair arms, more R10 / R10b arms |
| **`submission_blendXFS.csv`** | **0.87791** (selected, best) | **0.8741 / 0.8740 (3:1)** | segment noise classifier (`sn_` features) in 8 TabPFN arms and stacker context |

### 6.1 Generator genR10: fitted for realism, not for score
From genT (= genS with family shares p_struct [.49,.17,.10,.15,.09,0], `ku_gen_genT.json`) we changed the generator one setting at a
time and kept a change only if it made the fakes **harder to tell from real series**. The test is a real-vs-fake classifier (C2ST):
LightGBM on ~34 hand statistics of the pre / post segment and their change (`src/probe/gen_c2st.py`), mean of 3 fake seed sets
(`c2st3`), lower = more realistic; it was cross-checked with a Chronos-embedding discriminator (`src/probe/gen_disc.py`) and with
per-statistic KS gaps (`src/probe/gen_ks.py`). No real label is used for any of this (the C2ST compares real and fake series per class).
genT 0.662 → genR1 (bimodal noise only in WN) 0.649 → genR2 (GARCH and trend only in WN) 0.637 → … → genR5 (more, lighter-tailed
t noise; drift nuisance) 0.621 → genR6 (isolated spikes) → genR8 (smaller trends) 0.582 → **genR10** (bigger level spikes on random
walks) **0.580**. All steps are `src/outputs/factory/ku_gen_genR*.json`; new options are in `src/probe/ku_gen.py` (default off, so
older configs give identical output). genR11 (= genR10, level breaks ×1.1, from break-specific statistics, `src/probe/c2st_brk*.py`)
made the break statistics more real but did not change the classifier (zero-shot 0.8441 vs 0.8442), so it is not used in any blend.

### 6.2 blendTFS
1. Fakes: `python src/probe/sim_tensor.py 50000 src/outputs/factory/ku_gen_genR10.json SIMR10/sim_$i.npz $((91000000 + i*50000))`
   with `REVP=.5`, i = 0..3 (files 2-3 are used only for R10b).
2. Chronos-2 classifier chain (`src/probe/c2mv_ft.py`, `WIN=512 MT=1`; the R10 / R10b steps use `BLR=5e-6 ACC=4`), each step < 5 min:
   c2mv_d → c2mv_e, c2mv_f (+genS fakes) → **c2mv_r** (+100k genS fakes, `REVP=.5`, seeds 76.0M; features `c2vr`) →
   c2mv_t (+100k genT fakes) → **c2mv_R10** (+genR10 files 0-1; zero-shot fwd 0.8428, fwd+rev 0.8442) →
   **c2mv_R10b** (+genR10 files 2-3; fwd+rev 0.8455). Zero-shot AUC = AUC on the real training set without using its labels.
3. Features: `python src/probe/c2mv_score.py outputs/arms/c2mv_R10.pt c2vR10` (→ `zsc2vR10{,rev}`, `seqembc2vR10`), same for `c2vr`, `c2vR10b`.
4. TabPFN arms `EXTRA=seqembc2vR10,timesfm` with (K,TS) = (60,0) (70,2) (55,6) (50,1) (65,3) (80,4), settings as in 5.4, plus full refits.
5. Stacker (260 arms), `SW=3` = rank(stacker)·3 + rank(blend):
```
CB=outputs/factory/seqembmn2.parquet,outputs/factory/seqembc2m.parquet,outputs/factory/seqembc2vr.parquet,outputs/factory/seqembc2vR10.parquet
XB=outputs/arms/zsgs_oof.npy,outputs/arms/zsc2m_oof.npy,outputs/arms/zsc2mrev_oof.npy,outputs/arms/zsc2vr_oof.npy,outputs/arms/zsc2vrrev_oof.npy,outputs/arms/zsc2vR10_oof.npy,outputs/arms/zsc2vR10rev_oof.npy
HF_HUB_OFFLINE=1 FULLT=1 ARMS=1 SW=3 XARMS=$XB CTXEXTRA=$CB OUT=outputs/submission_blendTFS.csv python src/probe/stack_tpf_sub.py
```

### 6.3 blendWFS
- R10b arms: `EXTRA=seqembc2vR10b,timesfm`, same six (K,TS) bags; CNN `seqpre_mnR9_mt_c64.pt` (= mnr + 1 epoch on 600k genR9 fakes,
  zero-shot 0.8386) and `seqpre_mnR10_mt_c64.pt` (both in `ckpt/seq/`) → `seqembmnR9`, `seqembmnR10`; arms `EXTRA=seqembmnR10,chronos_2`.
- Semi-automatic generator-parameter estimates (`src/probe/semiauto_full.py`) trained on 2 extra realism-weighted genR10 bags
  (seeds 93.0M / 93.1M) and averaged with the 6 old bags: `bag8r = (6·bag6w + 2·r10w2)/8` (quick LightGBM test +0.0074, CI +0.0051..+0.0096,
  the largest single gain of the last day). Six pair arms rebuilt on it (`src/hpc/pair_any.sbatch`, `src/hpc/gnjbag8r.sbatch`).
- Stacker: 288 arms, CB and XB as in 6.2 plus `seqembc2vR10b` / `zsc2vR10b{,rev}`; `OUT=outputs/submission_blendWFS.csv`.

### 6.4 blendXFS: segment noise classifier
Oracle test on genR10 fakes (`src/probe/oracle_nuis.py`): telling LightGBM the true nuisance settings raises AUC 0.8154 → 0.8249,
and most of the gain is the noise type / degrees of freedom of the pre segment (+0.0070). So we estimate them:
`src/probe/seg_noise.py` fits AR(5) on the pre segment, whitens the series, computes residual-distribution statistics per segment and
trains LightGBM on genR10 fakes (pre segments + post segments of series without a shape break) to predict noise type (Gauss / t / bimodal),
1/df and GARCH; applied to the real pre and post segments (original and reversed series):
```
QV=1 BAGS=abef python src/probe/seg_noise.py outputs/factory/segnoise2.parquet          # 19 sn_ columns, ~1 min
QV=1 BAGS=abef REVSER=1 python src/probe/seg_noise.py REV/outputs/factory/segnoise2.parquet
```
On fakes: type accuracy 0.817, 1/df corr 0.735, GARCH accuracy 0.889. Used in 8 TabPFN arms
(`EXTRA=seqembc2vR10b,seqembmnR9,timesfm,segnoise2`, (K,TS) = (50,1) (60,0) (70,2) (80,4) (65,3) (55,6), best single arms 0.8729; plus two
`segnoise` arms) and as 5 stacker context columns (`CTXSN=1`: sn_tv, sn_didf, sn_ce, sn_pre_idf, sn_post_idf). 296 arms
(`manifests/blendXFS_stacker_arms.txt`, per-arm OOF AUC in `manifests/blendXFS_arms_table.csv`):
```
CB=outputs/factory/seqembmn2.parquet,outputs/factory/seqembc2m.parquet,outputs/factory/seqembc2vr.parquet,outputs/factory/seqembc2vR10.parquet,outputs/factory/seqembc2vR10b.parquet
XB=outputs/arms/zsgs_oof.npy,outputs/arms/zsc2m_oof.npy,outputs/arms/zsc2mrev_oof.npy,outputs/arms/zsc2vr_oof.npy,outputs/arms/zsc2vrrev_oof.npy,outputs/arms/zsc2vR10_oof.npy,outputs/arms/zsc2vR10rev_oof.npy,outputs/arms/zsc2vR10b_oof.npy,outputs/arms/zsc2vR10brev_oof.npy
HF_HUB_OFFLINE=1 CTXSN=1 CTXSNF=outputs/factory/segnoise2.parquet FULLT=1 ARMS=1 SW=3 XARMS=$XB CTXEXTRA=$CB \
  OUT=outputs/submission_blendXFS.csv python src/probe/stack_tpf_sub.py
```
Rank correlation 0.9987 with blendWFS, 0.998 with blendTFS. Kaggle public 0.87791.

### 6.5 Extra Chronos-2 checkpoints (not in git, same storage rule as 5.6)
```
3e9e081c01db3479983ebdcc2373b071cbddd69e529b5ed13be5b3a55f52c0c6  c2mv_r.pt
605ed0083240e49a1a9fe45b69b2cd20ded744325cd421f4a3631b307f8081b8  c2mv_R10.pt
97fd407d2f5af74249e5ca8c6607f7166e5cb538d2f70e3dd7da7101d97c8350  c2mv_R10b.pt
```
Honest notes: the Chronos steps have ±0.004 zero-shot noise between fake chunks (R10b is partly lucky); the stacker settings (SW=3,
which arms) were chosen on the same OOF table, so its OOF may be slightly optimistic; things that did not help in this phase are in
`REPORT_CHANGELOG.md` (real labels in Chronos fine-tuning, fakes in TabPFN context, TiRex / TOTO features, another CV split).

## Hardware and honesty notes
- GPU: 2 × RTX 4090 (TabPFN, CNN pretraining, Chronos-2, TimesFM); CPU feature pipeline on the KU HPC. The whole test-time
  inference of these blends needs one ≥ 16 GB GPU; a CPU-only LightGBM route reaches OOF 0.8651.
- Group weights after the LightGBM stage were chosen on the full OOF with a CI > 0 and 5/5-fold rule, so the OOF numbers may be
  slightly optimistic; public scores (0.87416 / 0.87356) are consistent with them (public-board sd ≈ 0.009).
- These submissions were made after the qualification freeze; the designated qualification submission is unchanged (README).
