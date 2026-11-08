# AI-use disclosure — team BreakPoint

**Tool**: Anthropic Claude (Claude Code agent, used through the Claude desktop app), 2026-10-28 to 2026-11-08.

**What it did**
- Wrote and ran analysis / feature / model code on the team's own machines (local workstation, KU Madar HPC through our own account) under continuous human direction and review.
- Proposed and ran experiments (feature families, synthetic generator calibration, blends), reported results with confidence intervals, and kept the lab notebook.
- Drafted the report, README and these documents; the team reviewed and edited them.

**Data handling**
- No competition files, label files or prediction files were uploaded to any third-party AI service. Data stayed on the local workstation and the university HPC.
- The assistant's context contained code, summary statistics, model scores and, occasionally during debugging, a few printed values of individual series.
- Public pretrained models run locally on our own GPUs (no data sent anywhere): TabPFN 3.5 (used in the designated GPU-route submission blendG2 and in all later blends), Chronos-2 and TimesFM-2.5 (zero-shot forecasters, used for forecast-error features in blendNFS / blendOFS).

**Human responsibility**
- The team chose the directions, approved every Kaggle submission, and can explain every component of the designated pipeline (features, validation, pair interactions, blend).
