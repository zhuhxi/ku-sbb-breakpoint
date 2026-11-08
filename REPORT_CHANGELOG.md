# Final report change log (report_final.html / .pdf)

The qualification report (report.html/.pdf, tag `qualification`) is frozen. All later edits go to report_final.*.

- 2026-11-08 08:01 UAE — created from the qualification report; descriptive title; summary paragraph with best submission blendK (public 0.86971, OOF 0.8684) and the three main gains; results table extended with blendH/I/J/K and the TabPFN stacker (blendKS); A4 page setup, rendered with headless Chrome (3 pages).
- 2026-11-08 08:05 UAE — §3 rewritten for blendK: LightGBM vs TabPFN arms described separately, best TabPFN input stated, full blendK formula (10 groups with weights).
- 2026-11-08 08:10 UAE — Figure 1 added: inline-SVG pipeline diagram (data -> three feature sources -> reversal / per-fold selection / pair interactions -> LightGBM and TabPFN arms -> fixed-weight blend), caption states which steps use labels.
- 2026-11-08 08:15 UAE — ablation table added to §3 (new analysis, ku_sbb/probe/ablate_k.py): removing all TabPFN -0.0057, all reversal groups -0.0019, CNN-feature groups -0.0007 (all 5/5 folds); LightGBM groups add nothing at the final stage (+0.0001) — stated honestly with their remaining role.
- 2026-11-08 08:19 UAE — 'What did not help' turned into a compact negative-results table (14 ideas with their measured effect, incl. post-qualification ones: TabICL, TabPFN fine-tuning, second simulator, SSL detector, Chronos, stacking, negation/cropping).
- 2026-11-08 08:24 UAE — §6 reproducibility table per submission (blendD_r12 bit-exact CPU, blendG2 exact GPU, blendK: packaging into repo still to do before final freeze — stated honestly); rounding rule stated.
- 2026-11-08 08:29 UAE — §2: new bullet on the synthetic-pretrained CNN features (architecture, 600k fakes, multi-task heads, zero-shot AUC 0.81 without real labels, 53 label-free features incl. reversed copy, effect on TabPFN 0.8658 -> 0.8672).
- 2026-11-08 08:33 UAE — §5 Resources rewritten for blendK: CPU-only route OOF 0.8641 (gap 0.0043), explicit request for one >=16 GB GPU at the audit, what more GPU time would buy.
- 2026-11-08 08:38 UAE — §4: leaderboard-noise bullet (new computation on blendK OOF: sd 0.0089 at public size 1,520 rows, 0.0067 at private size 2,480; public 0.86971 consistent with OOF 0.8684).
- 2026-11-08 08:43 UAE — §6: hardware line updated (GPUs used for TabPFN and CNN pretraining); sources completed with full citations (TabPFN Nature 2025, Bai et al. 2018 TCN, Wen et al. 2021 time-series augmentation survey).
  (body font 10.5 -> 10.2 pt to stay at 3 pages.)
- 2026-11-08 08:48 UAE — visual QA of page 2; §3 heading 'designated' -> 'final'; blend-weight paragraph made accurate: CV grid only for LightGBM weights, later TabPFN group weights picked on full OOF with a CI>0 and 5/5-fold rule, possible slight optimism stated; §4 'final blend' wording.
- 2026-11-08 08:53 UAE — new §7 'What we learned' (3 one-line takeaways, as in winning write-ups); last limitation bullet shortened; margins 12/14 mm and line-height 1.24 to stay at 3 pages.
- 2026-11-08 08:57 UAE — §2 semi-automatic feature bullet tightened (same content, ~25% shorter) to free space.
- 2026-11-08 09:02 UAE — §1 leakage guards for the new components: reversed copies stay in their series' fold (never in training/TabPFN context of their own validation fold); CNN pretrained without real labels.
- 2026-11-08 09:07 UAE — layout QA: fixed orphaned '§3' heading at the bottom of page 1 (shorter Figure 1 caption); figure kept unsplit (break-inside: avoid); table-level avoid rules tried and dropped (pushed to 4 pages).
- 2026-11-08 09:12 UAE — summary claims made consistent with the ablation table: four ideas with measured gains (residual +0.008, TabPFN +0.006, reversal +0.002, CNN features +0.0007); earlier '+0.003 CNN' figure mixed several steps and was removed.
- 2026-11-08 09:16 UAE — §4 error analysis recomputed for blendK vs blendD (new computation): AUC per estimated family (RW 0.710 still weakest; SIN +0.012, LL +0.015) and per shorter-segment length (0.815 / 0.876 / 0.907); old pre-TabPFN numbers replaced.
- 2026-11-08 09:21 UAE — §4 ceiling bullet: blendK (0.8684) placed against the simulated plug-in oracle (0.86–0.87) vs true-parameter oracle (~0.91), with what that implies for further gains.
- 2026-11-08 09:26 UAE — §4 train/test shift: quantified blendK on the rescaled test group (17.9% of test): mean |rank-0.5| 0.19 vs 0.26, mean rank 0.53 vs 0.49 — less certain, slightly shifted (new computation).
- 2026-11-08 15:50 UAE — best submission updated to blendNFS (ID 58123057, public 0.87416, OOF 0.8706) and blendOFS (ID 58126032, OOF 0.8710, public 0.87356) in the summary and results table; reproducibility row points to TRAINING_LATEST.md / ckpt/seq / manifests; Chronos-2 and TimesFM-2.5 added to sources; body font 10.2 -> 10 pt to stay at 3 pages.
