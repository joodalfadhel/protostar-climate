# Code

Run from the dataset root (the folder with `submission.csv`):

| File | What it does |
|---|---|
| `baseline.py` | Loads `data/annual_valid.csv`, backtests the forecasting methods at five cutoffs (data ≤ 2018 only), picks one per metric and writes `submission.csv`. `--validate` prints the table only. Includes the departure-persistence methods `pooled15p`, `pooled15h`, `ensPp`, `ensPh` (`persistence_adjust`). |
| `build_features.py` | Builds `data/city_features.csv` (elevation, coast distance, ocean-current influence) from `data/raw/`. Run from inside `data/`. |
| `add_tree_canopy.py` | Adds NLCD tree canopy (2018) to `data/city_features.csv`. Needs internet (MRLC service) and `pillow`. Run from inside `data/`. |
| `policy_ranking.py` | Policy Idea analysis: CDD increase vs. canopy, written to `data/policy_ranking.csv`. |
| `check_submission.py` | The template's checker, the same checks as the Space: `python code/check_submission.py`. Needs `fsspec` and `huggingface_hub`. |
| `requirements.txt` | Python packages. |
| `AGENTS.md` | Data rules for AI agents. |

Steps: `pip install -r code/requirements.txt`, then `python code/baseline.py`, then `python code/check_submission.py`.
