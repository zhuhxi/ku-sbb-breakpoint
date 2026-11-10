# Final talk (11 Nov 2026, 6 min + 4 min Q&A)

- `BreakPoint_final.pptx` / `.pdf` — 9 slides; `QA_prep.md` — expected questions and short answers.
- `figures/<name>/figure.py` — self-contained matplotlib script for each slide figure (data inline; `data.txt` = the numbers it uses).
  All example series are synthetic (our genR10 generator); the AUC numbers are our logged OOF / public results. No competition data.
  `figures/hero/make_hero.py` draws the cover and closing figures.
- `build_deck.py` — builds the deck with python-pptx; it was run from the project root with the figures under `reports/final/`
  (`figmirror/<name>/` = `figures/<name>/` here, `hero/` = `figures/hero/`).
- Visual style follows the figures of Chronos (Ansari et al., TMLR 2024); reference images are not redistributed here.
