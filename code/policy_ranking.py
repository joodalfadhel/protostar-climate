"""Rank cities for the Policy Idea: large forecast CDD increase AND low tree canopy.

ANALYSIS ONLY: not read by baseline.py; does not change any forecast.
Inputs: submission.csv, data/annual_valid.csv, data/city_features.csv (tree_canopy_frac).
Output: data/policy_ranking.csv
Usage: python code/policy_ranking.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

sub = pd.read_csv(ROOT / "submission.csv")
fc = sub[sub["metric"] == "cdd"].groupby("city_id")["predicted_value"].mean()
hist = pd.read_csv(ROOT / "data" / "annual_valid.csv")
hist = hist[hist["Year"].between(1989, 2018)].groupby("ID")["cdd"].mean()
f = pd.read_csv(ROOT / "data" / "city_features.csv").set_index("ID")

r = f[["Name", "tree_canopy_frac"]].copy()
r["canopy_pct"] = 100 * r.pop("tree_canopy_frac")
r["cdd_1989_2018"] = hist
r["cdd_forecast_2019_2025"] = fc
r["cdd_increase"] = r["cdd_forecast_2019_2025"] - r["cdd_1989_2018"]
r = r.dropna()
r["priority"] = (r["cdd_increase"].rank(pct=True) + (-r["canopy_pct"]).rank(pct=True)) / 2
r = r.sort_values("priority", ascending=False)
r.round(2).to_csv(ROOT / "data" / "policy_ranking.csv", index_label="city_id")

print(f"{len(r)} cities. Forecast CDD up vs 1989-2018 in {(r.cdd_increase > 0).sum()};"
      f" median increase {r.cdd_increase.median():.0f} F-days ({100 * (r.cdd_increase / r.cdd_1989_2018).median():.0f}%).")
print(f"Correlation canopy % vs CDD increase: {r.canopy_pct.corr(r.cdd_increase):.2f}")
print(r.head(10)[["Name", "canopy_pct", "cdd_1989_2018", "cdd_forecast_2019_2025", "cdd_increase"]].round(0).to_string())
