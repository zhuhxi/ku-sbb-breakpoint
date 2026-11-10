# Final talk (11 Nov 2026, 6 min + 4 min Q&A)

- `BreakPoint_final.pptx` / `.pdf` — 9 slides; `QA_prep.md` — expected questions and short answers.
- `figures/<name>/figure.py` — self-contained matplotlib script for each slide figure (data inline; `data.txt` = the numbers it uses).
  All example series are synthetic (our genR10 generator); the AUC numbers are our logged OOF / public results. No competition data.
  `figures/hero/make_hero.py` draws the cover and closing figures.
- `build_deck.py` — builds the deck with python-pptx; it was run from the project root with the figures under `reports/final/`
  (`figmirror/<name>/` = `figures/<name>/` here, `hero/` = `figures/hero/`).
- Visual style follows the figures of Chronos (Ansari et al., TMLR 2024); reference images are not redistributed here.

## Checkpoint download (added 2026-11-11, slide update)
The six fine-tuned Chronos-2 checkpoints used by blendTFS / blendXFS (~480 MB each, too large for git) are in a KU OneDrive folder,
viewable by Khalifa University accounts: https://kuacae-my.sharepoint.com/:f:/g/personal/100065148_ku_ac_ae/IgCz8MctEnciQpDezcTuI0P7AblhRyTj8fYTHY3c4jFxyRI?e=84qE0h
Files: c2ft_w512b.pt, c2ft_mt3.pt, c2mv_d.pt, c2mv_r.pt, c2mv_R10.pt, c2mv_R10b.pt + SHA256SUMS.txt + README.txt.
Put them in `outputs/arms/` and run `sha256sum -c SHA256SUMS.txt`; the same hashes are in `TRAINING_LATEST.md` 5.2 / 6.5 (tag `final`).
The technical submission itself is unchanged since the freeze (tag `final` = fa1b979); this note and the last slide are the only additions.
