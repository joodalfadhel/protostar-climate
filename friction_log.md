# Friction log

Add an entry whenever something slows you down: unclear docs, missing metadata, units, broken links, license questions, env issues, join problems, agent issues. This becomes the README's *Feedback to the Ecosystem* section (15% of judging). Be specific.

| Date | Who | What happened | Time lost | Suggested fix |
|---|---|---|---|---|
| 2026-11-05 | Jood | Space's About/Data tabs load via JS; the submission format isn't in the Space README or config.py, and config.py's commit history contradicts itself on whether a CSV is required | | Put the submission spec in the Space README |
| 2026-11-10 | David (with Claude Code) | `code/check_submission.py` fails with "`fsspec` is required to handle `hf://` URIs" on a fresh environment; neither `fsspec` nor `huggingface_hub` was in `requirements.txt` | ~10 min | Add both to the template's `requirements.txt` (done here) |
| 2026-11-10 | David (with Claude Code) | `code/baseline.py` imported `datasets` at the top, so it failed on a machine without it even though `data/annual_valid.csv` was present and the import was only needed when that file is missing | ~5 min | Import `datasets` lazily (done here) |
| 2026-11-10 | David (with Claude Code) | `Max Tmax` holds `-inf` in empty years, and a plain `mean()` returns `-inf`/NaN silently, so a backtest printed NaN for hottest day | ~15 min | Store missing as empty in `annual_indices`; document it |
| 2026-11-10 | David (with Claude Code) | The Space's `README.md` is a few lines; the real submission rules, scoring rule and data rule are in `content/data.md` and `content/about.md`, which can be read as raw files but are not linked from the README | ~20 min | Link `content/*.md` from the Space README |
| | | | | |
