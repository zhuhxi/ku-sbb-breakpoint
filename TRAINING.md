# TRAINING.md: how to rebuild every input of `release/predict.py`

This is the ordered recipe for regenerating the feature files and checkpoints that `predict.py` reads, starting from the raw
competition files `data/{train_series,test_series}.parquet`, `data/train_labels.csv` and `data/sample_submission.csv`.

How it was written: from the scripts in this project, the HPC batch files `hpc/*.sbatch`, the lab notebook `LESSONS.md` and
checks on the shipped outputs (column lists, file times, averages recomputed). Nothing was re-run.
Anything not backed by a script, a log line or a check is marked **UNCONFIRMED**.

Conventions
- Run every command from the project root (`ku_sbb/`) with `PYTHONPATH=.`. All paths are relative to it.
- Thread count: `NTHREADS` (default 30) for the feature scripts; LightGBM threads are set by `J`, `CONF_JOBS`, `SA_JOBS` or `NTHREADS`, depending on the script.
- Hardware: "HPC" means Madar, Slurm account kuin0094, 64 to 96 cores per job, `--exclusive`, venv `~/venvs/kusbb`.
  Local versions: lightgbm 4.6.0, scikit-learn 1.7.2, numpy 1.26.4, pandas 2.3.3, scipy 1.16.3. The HPC venv versions are **UNCONFIRMED**.
- Row sets: train has 24,000 series and test has 4,000. Files under `outputs/factory/` hold all 28,000 ids (train + test) in one parquet.
- Bit-exact reproduction: several scripts were edited after their outputs were made (see "Caveats"). Treat the shipped
  parquet files as the reference and compare any rebuild against them.

---

## Stage 1: base per-series features (deterministic, CPU)

1. **`outputs/feat_{train,test}.parquet`** (242 columns): `features.py::build_features`, cached by `train.py::cached_features`.
   - Command: `python train.py --data data --out outputs`. This also trains a small LightGBM baseline that you can ignore.
     To build the features only, run
     `python -c "from pathlib import Path; from train import cached_features as c; c(Path('data/train_series.parquet'), Path('outputs/feat_train.parquet')); c(Path('data/test_series.parquet'), Path('outputs/feat_test.parquet'))"`
   - Built 2026-10-29 00:05; `features.py` has not changed since. Runtime **UNCONFIRMED**.
2. **`outputs/magic_b_{train,test}.parquet`** (16 `mb_` float-bit columns): `python brute/magic_b.py` (imports `bits.py`).
   Deterministic; built 2026-10-29.
3. **`outputs/raw_{train,test}.parquet`** (108 raw-scale columns): `python rawstats.py`. Deterministic.
4. **`outputs/harvest_{train,test}.parquet`** (85 `H_` columns): `python harvest.py` (needs `pycatch22`; joblib n_jobs is fixed at 28).
5. **`outputs/harvest_keep.json`** (58 column names).
   - Checked: the list is exactly the harvest columns, in file order, minus the 27 `H_icss_*` and `H_cp_*` columns. That is the
     "H_rest" arm of `cv_harvest.py`, whose docstring wrongly says 48 columns.
   - The script that wrote the JSON was not found (**UNCONFIRMED**; it was probably written inline on 2026-10-29 18:24). Equivalent command:
     `python -c "import json,pandas as pd; H=pd.read_parquet('outputs/harvest_train.parquet'); json.dump([c for c in H if not c.startswith(('H_icss_','H_cp_'))], open('outputs/harvest_keep.json','w'))"`

## Stage 2: feature-factory families (`factory.py`, deterministic, train and test in one file)

`factory.py FAM [FAM ...] [TRANSFORM ...]` writes `outputs/factory/{fam}{_TRANSFORMS}.parquet`. The default transforms are RAW, DIFF and CUMSUM.
Each family takes 4 to 20 s on about 30 threads (`outputs/factory_build.log`). Built 2026-10-31.

6. `NTHREADS=30 python factory.py st RES10` produces **`st_RES10.parquet`**. RES10 is the AR(10) residual. The log shows 116 columns.
7. `NTHREADS=30 python factory.py st ARES10` produces **`st_ARES10.parquet`**. ARES10 is the absolute AR(10) residual; 113 columns.
8. `NTHREADS=30 python factory.py fe fe2 ar2 cost` produces **`fe.parquet`** (126 columns), **`fe2.parquet`**, **`ar2.parquet`** (285 columns) and **`cost.parquet`** (207 columns).
   `predict.py` keeps only the columns matching `PAT`.

## Stage 3: generator features (deterministic per series)

9. **`genA.parquet` and `genB.parquet`** (12 + 12 `g_` columns): `NTHREADS=64 python probe/genfeat_all.py --jobs 64`.
   - The script uses `probe/gen_feats.py`, `probe/model_feats.py`, `probe/close_loop.py` and `probe/mle_noise.py`.
   - This is the first line of `hpc/gen.sbatch`; the rest of that file is a confirm run and is not needed. About 2 min on 64 cores. Built 2026-11-01 23:23.
10. **`gnoise.parquet`** (4 columns, noise-type change at the boundary): `python probe/noise_lr.py --jobs 64` (uses `probe/close_loop.py`). Built 2026-11-03 01:39.
11. **`gjlr.parquet`** (3 columns, joint Gaussian likelihood ratio): `python probe/joint_lr.py`. Built 2026-11-03 02:00.

## Stage 4: semi-automatic ABC columns (31 `saf_` columns, HPC)

Pipeline for one "bag":
- `probe/sim_series_ku2.py N PARAMS.json TAG SEED0` writes the fakes to `outputs/aug/series_TAG.parquet` and their hidden parameters to `meta_TAG.parquet`. Series `s` uses seed `SEED0+s`.
- `aug_features.py TAG` computes the real model's columns for the fakes and writes `outputs/aug/X_TAG.parquet`.
- `probe/semiauto_full.py TAG OUT` fits one LightGBM per hidden parameter on the fakes and predicts it for the real train and test series.

It needs Stages 1 and 2 plus the frozen generator parameter files.

12. Frozen inputs, shipped as they are:
    - `outputs/factory/ku_gen_best2.json`: from the `probe/gen_search.py` climb, about 75 min (LESSONS 2026-11-05).
    - `outputs/factory/ku_gen_size_three-bin.json`: from `probe/size_form.py` (2026-11-05).
    - Regenerating either file is **UNCONFIRMED** (it is a stochastic search). Use the shipped JSON.
13. **`semiauto_full.parquet`** (bag simk2, unweighted):
    `sbatch hpc/semiauto_full.sbatch 30000`, which runs
    `sim_series_ku2.py 30000 outputs/factory/ku_gen_best2.json simk2 8000000`, then `aug_features.py simk2`, then `semiauto_full.py simk2`.
    The default output is `semiauto_full.parquet`. The run used `SA_JOBS` unset (n_jobs=-1). Job 1555158, about 14 min, built 2026-11-05.
    N=30000 is from LESSONS ("30k best2 fakes").
14. **simk3** (three-bin generator, same seeds):
    `sbatch hpc/semiauto_full3.sbatch 30000` (`... ku_gen_size_three-bin.json simk3 8000000`).
    The training step was then re-run alone with `sbatch hpc/semiauto_full3b.sbatch` (`SA_JOBS=32`). It writes `outputs/factory/semiauto_full3.parquet`.
15. **simk4 to simk7**: `sbatch hpc/semiauto_bag.sbatch PARAMS.json TAG SEED0`, with 30k fakes per bag, `SA_JOBS=32`, about 2 min per job, 4 jobs in parallel.
    It writes `outputs/factory/semiauto_TAG.parquet`. simk4 and simk6 use `ku_gen_best2.json`; simk5 and simk7 use `ku_gen_size_three-bin.json` (LESSONS 2026-11-06).
    **SEED0 for simk4 to simk7 is UNCONFIRMED.** LESSONS says only "new seeds", and the HPC logs are not in the local copy.
16. Realism-weighted versions: for TAG in simk2 to simk7, run `sbatch hpc/semiauto_w.sbatch TAG`
    (`SA_WEIGHT=1 SA_JOBS=32 python probe/semiauto_full.py TAG outputs/factory/semiauto_TAG_w.parquet`; about 40 s each).
    It needs `outputs/aug/X_TAG.parquet` and `meta_TAG.parquet` from steps 13 to 15. The discriminator split is `KFold(2, shuffle, random_state=0)`.
17. **`semiauto_bag6.parquet`** is the element-wise mean of `semiauto_full`, `semiauto_full3` and `semiauto_simk4` to `semiauto_simk7`.
    **`semiauto_bag6w.parquet`** is the mean of `semiauto_simk{2..7}_w`.
    - Checked: recomputing these means reproduces both shipped files exactly (max abs diff 0.0). `semiauto_bag4` = the first 4 bags, also exact.
    - The script that did the averaging was not found (**UNCONFIRMED**). Equivalent command:
      `python -c "import pandas as pd; L=lambda t: pd.read_parquet(f'outputs/factory/{t}.parquet'); A=[L(t) for t in ['semiauto_full','semiauto_full3']+[f'semiauto_simk{k}' for k in range(4,8)]]; (sum(a.reindex(A[0].index) for a in A)/6).to_parquet('outputs/factory/semiauto_bag6.parquet')"`
      Use the same command on `semiauto_simk{2..7}_w` to make `semiauto_bag6w.parquet`.

## Stage 5: interaction pool `combo10gnjx_{train,test}.parquet`

18. **`combo10x_{train,test}.parquet`** (615 columns): the magic_b columns, then `harvest_{part}[harvest_keep]`, then the columns of
    st_RES10, st_ARES10, fe, fe2, ar2 and cost that match `^fe_|^fe2_|^st_|_(?:ab|ba)_|_gain$`, de-duplicated, in that order, for train ids or test ids.
    - Checked: this recipe gives the shipped column list exactly. The writer script was not found (**UNCONFIRMED**; built 2026-10-31 17:18).
19. **`combo10gnjx_{train,test}.parquet`** (646 columns) = combo10x plus the 31 `g_` columns of genA, genB, gnoise and gjlr (LESSONS 2026-11-05).
    - The builder and the column order are **UNCONFIRMED**. The file is not in the local copy; it exists only on the HPC.
    - Column order does not matter for `predict.py`, which selects columns by the names stored in `top.json`.

## Stage 6: models and checkpoints (`ckpt/`)

20. **11 pair arms**: `probe/confirm_pair.py K TAG FAMILY` on HPC, 64 cores, about 2.5 min each.
    - Base set: everything from Stages 1 to 4, i.e. `probe/feats.py` FAMS[:-1] plus FAMILY.
    - Per fold: a quick lr 0.1 model picks the top-K features; pair rank differences and products are added; then an lr 0.02 model is trained with 3 seeds.
    - Fixed seeds: `StratifiedKFold(5, shuffle, SPLIT=42)`, inner split `random_state=k`, model seeds `100*k+s`.
    - Example (`hpc/pair60.sbatch`, run 2026-11-07 about 17:20): `J=64 python -W ignore probe/confirm_pair.py 60 pair60 semiauto_bag6w`.

    | arm | K | FAMILY | TOPSEED |
    |---|---|---|---|
    | pair40 | 40 | semiauto_bag6w | 0 |
    | pair60 | 60 | semiauto_bag6w | 0 |
    | pair60saf | 60 | semiauto_full | 0 |
    | pair80 | 80 | semiauto_bag6w | 0 |
    | pairb_50_1 | 50 | semiauto_bag6w | 1* |
    | pairb_70_2 | 70 | semiauto_bag6w | 2* |
    | pairb_60_3 | 60 | semiauto_bag6 | 3* |
    | pairb_55_5 | 55 | semiauto_bag6w | 5* |
    | pairb_65_6 | 65 | semiauto_bag6w | 6* |
    | pairb_60_7 | 60 | semiauto_full | 7* |
    | pairb_75_8 | 75 | semiauto_bag6w | 8* |

    - K and FAMILY come from `PAIR_ARMS` in `predict.py`. K is also confirmed by the length of `ckpt/*/fold0/top.npy`.
    - *The TOPSEED of the `pairb_K_T` arms is inferred from the name only (T = TOPSEED env; LESSONS: "TOPSEED bags"). **UNCONFIRMED.**
    - Checkpoints: `SAVE=ckpt TOPSEED=T J=64 python probe/confirm_pair.py K TAG_ck FAMILY` writes `ckpt/TAG_ck/fold{k}/{top.npy, seed{0,1,2}.txt}`.
      The `_ck` tag naming is inferred from the directory names. The checkpoints were written 2026-11-07 19:41, after `SAVE` was added to the script at 19:40.
      The exact checkpoint command is **UNCONFIRMED**.
21. **Interaction arm** (`brute/lgb_inter.py`, "top 800" arm; OOF 0.8553):
    - Run on HPC with `--exclusive`, 96 cores, 600 to 700 GB RAM, about 1 h. It does not finish on a shared node.
    - Original run 2026-11-05: `hpc/inter_gnj.sbatch 162 2 0 0.02 t162`, writing `outputs/brute/{oof,test}_lgb_inter_gnjt162.npy`. That batch file is not in the local copy (**UNCONFIRMED**).
      It is presumably the same as `hpc/inter2.sbatch` with `--extra outputs/factory/combo10gnjx_train.parquet` and `--tag _gnj$5`:
      `NTHREADS=96 LR=0.02 python -u brute/lgb_inter.py --extra outputs/factory/combo10gnjx_train.parquet --top 162 --seeds 2 --seed-offset 0 --tag _gnjt162`
    - The test pool is found by replacing `_train` with `_test` in the `--extra` path.
    - Pool: feat (242) + raw (108) + combo10gnjx. Per fold, the top 162 columns by gain give 91,449 interactions. Folds use `StratifiedKFold(5, shuffle, 42)`, the ranking model uses seed f, and the model seeds are `100*f+s`.
    - Checkpoint: `predict.py` reads `ckpt/intert162_ck/fold{k}/{top.json, inter.npz, keep_seed{s}.json, top800_seed{s}.txt}`.
      `lgb_inter.py` saves to `$SAVE/inter{--tag}`, so this needs `SAVE=ckpt ... --tag t162_ck`.
    - **`ckpt/intert162_ck` is missing from the local copy**, and the command that made it is **UNCONFIRMED**. `SAVE` was added to `lgb_inter.py` on 2026-11-07 19:43.
22. Blend (inside `predict.py`): `rank(2*rank(mean of the 11 pair-arm ranks) + 1*rank(inter top800))`, giving OOF 0.8606 and public 0.86077 (LESSONS 2026-11-07).

## Caveats for bit-exact reproduction (all UNCONFIRMED effects)

- **`factory.py` edited after the families were built.** It changed 2026-11-01 12:48; the families date from 2026-10-31. The edits were probably new families only.
- **`probe/close_loop.py` edited after genA/genB were built.** It changed 2026-11-03 to add `fit_noise(full=True)`; genA/genB date from 2026-11-01.
- **Generator scripts edited after the bags were made.** `probe/ku_gen.py` changed 2026-11-06 17:39 and `probe/sim_series_ku2.py` 15:53, after the bags of 2026-11-06 00:52 and 02:09.
  LESSONS says the new switches default to off and leave the series unchanged. This was not checked for simk2 to simk7.
- **LightGBM results depend on thread count.** The semiauto models ran with `n_jobs=-1` for simk2 and 32 threads for the rest, and the pair arms ran with `J=64`.
  With different thread counts, expect small numeric differences, not identical files.
- **Use the shipped files where they exist.** `ku_gen_*.json`, `harvest_keep.json` and all shipped parquet files are the reference. Regenerated files should match them in column names and closely in values.
