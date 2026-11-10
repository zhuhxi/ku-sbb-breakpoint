# Team BreakPoint — final talk: rehearsal and Q&A prep

Deck: `BreakPoint_final.pptx` (9 slides, about 5 min 45 s; no backup slides — answer questions verbally). Every slide has a timed English script and Chinese cues in the speaker notes.

## Rehearsal plan
- Target 5:30 with a timer; 6:00 is the hard limit. If one slide runs long, shorten the explanation on slide 5 (validation), not the takeaway.
- Main idea in one sentence (say it on slides 1 and 9): *we rebuilt how the data was generated, and let synthetic data teach the model what 24,000 labels cannot.*
- Point at the visual on each slide before reading numbers. On slide 2, pause on the last panel ("can you see it?").
- No backup slides: for detail questions, point back to slide 4 (pipeline), 6 (oracle) or 8 (limitations).

## Likely judge questions (answer in 20–40 s each)

**1. How do you know the synthetic data does not leak test information or overfit the public leaderboard?**
The generator is fitted to statistics of the series themselves; no labels are involved beyond the training folds, and test labels are never available to us. Every label-dependent step is inside the training folds; we decide by out-of-fold AUC on 24,000 series, not by the public score (only ~1,520 series, noise ±0.009). OOF and public rose together across 8 submissions (slide 5).
*中文要点：生成器只拟合序列本身的统计量，不用测试标签；所有用标签的步骤都在折内；按 OOF 决策，公榜噪声 ±0.009；8 次提交 OOF 与公榜同步上升。*

**2. How did you reverse-engineer the generator?**
Fit candidate structures (white noise, AR, sine, random walk, local level) and noise families to each real series by maximum likelihood, read off the parameter distributions, then tuned the recipe until a real-vs-fake classifier was close to chance (AUC 0.66 → 0.58).
*中文要点：逐条最大似然拟合结构和噪声，读出参数分布，再调到真假判别接近随机。*

**3. Why is 0.58, not 0.50? What is still different?**
Synthetic breaks are slightly smaller at middle sizes, and real breaks raise the kurtosis more often. A classifier trained on synthetic data still reaches 0.84 on the real series without any real label, so the gap is small.
*中文要点：中等断点偏小、真实断点更常让尾部变重；只用假数据训练在真实数据上仍有 0.84。*

**4. What exactly is the oracle experiment? Isn't it cheating?**
It is only run on synthetic data, where the true hidden parameters are known. It is a diagnostic: it tells us which parameter is worth estimating better. The model we submit never sees any true parameter.
*中文要点：只在假数据上做，是诊断工具，告诉我们该把哪个量估准；提交的模型从不使用真值。*

**5. Why TabPFN? Is it reproducible?**
TabPFN is a public pretrained tabular model that is strong on ~20k-row tables and needs no tuning. Fixed seeds; arm lists in `manifests/`; full-train refits for test predictions. The GPU route reruns from scripts and fixed seeds (bit-exact GPU reproduction not verified); the CPU route (LightGBM, 0.861 public) is bit-exact in 31 s on 2 cores.
*中文要点：公开预训练模型、两万行规模强、免调参；种子固定、清单在 manifests；GPU 路线未验证逐位一致；CPU 路线逐位复现。*

**6. 296 arms — isn't that overkill?**
Yes, partly: the best single arm (0.8729) is within 0.0012 of the stacker (0.8741). The arms accumulated over the challenge; a pruned version with a dozen arms is the obvious next step. We kept the full set because it was validated end to end.
*中文要点：承认冗余，单分支只差 0.0012；精简是下一步；全量版本是完整验证过的。*

**7. What did not work?**
XGBoost/CatBoost blends, fine-tuning Chronos-2 on real labels, synthetic rows inside TabPFN's context, LOF features, model soups, per-type recalibration. Each was checked with fold-clean CV and gave no gain.
*中文要点：XGB/CatBoost 融合、真实标签微调 Chronos-2、TabPFN 上下文加假数据、LOF、模型汤、按类型再校准，都在折内验证过、没提升。*

**8. Where does the model fail?**
Random-walk series (AUC 0.71) and short post-boundary segments (0.82 vs 0.91). Random walks are equally hard on synthetic data, so this looks intrinsic (slide 8).
*中文要点：随机游走和短序列；假数据上同样难，像是本质难。*

**9. How long does training and inference take?**
CPU route: inference 31 s on 2 cores. GPU route: features and generator fakes in minutes on a 64-core node; each TabPFN arm about 1–2 min on one GPU; the full ensemble a few GPU hours.
*中文要点：CPU 推理 31 秒；GPU 路线单分支 1–2 分钟，全部几小时。*

**10. How was AI assistance used?**
Disclosed in `AI_DISCLOSURE.md`: an AI coding assistant wrote and ran code on our machines under our direction; no competition data, labels or predictions were sent to third-party AI services.
*中文要点：按 AI_DISCLOSURE.md 回答；数据从未上传第三方 AI。*
