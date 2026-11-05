"""Baseline forecasts for the Protostar Climate Risk challenge.

Runs the three official reference baselines (last value, 30-year average,
30-year linear trend), validates them on a practice split (train <= 2010,
predict 2011-2018), picks the best per metric, and fills submission.csv.

Usage (from the dataset root):
    python code/baseline.py            # validate, then write submission.csv
    python code/baseline.py --validate # validation table only

Data rule: uses only the challenge dataset (records up to 2018-12-31).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset

REPO = "CMU-OSPO/protostar-climate-train-to-2018"
ROOT = Path(__file__).resolve().parent.parent
SUBMISSION = ROOT / "submission.csv"

# metric code -> column in annual_indices
METRICS = {
    "annual_avg_temp": "Avg Temp",
    "total_precip_in": "ToT Prcp",
    "hottest_day_temp": "Max Tmax",
    "cdd": "cdd",
}
WINDOW = 30
MAX_MISSING_DAYS = 10


def load_annual():
    df = load_dataset(REPO, "annual_indices", split="train").to_pandas()
    assert df["Year"].max() <= 2018, "Weather input dated after 2018: data rule"
    miss_cols = [c for c in df.columns if c.startswith("Miss")]
    if miss_cols:  # drop badly incomplete years, as the scorer does
        df = df[df[miss_cols].max(axis=1) <= MAX_MISSING_DAYS]
    return df


def forecast(series: pd.Series, last_year: int, years, method: str):
    """series: values indexed by Year, already cut at last_year."""
    hist = series.dropna()
    hist = hist[hist.index > last_year - WINDOW]
    if hist.empty:
        return pd.Series(np.nan, index=years)
    if method == "last":
        return pd.Series(hist.iloc[-1], index=years)
    if method == "avg30":
        return pd.Series(hist.mean(), index=years)
    if method == "trend30":
        if len(hist) < 10:
            return pd.Series(hist.mean(), index=years)
        slope, intercept = np.polyfit(hist.index.values, hist.values, 1)
        return pd.Series(intercept + slope * np.asarray(years), index=years)
    raise ValueError(method)


def predict_all(df, last_year, years, method, col):
    out = []
    for city, g in df[df["Year"] <= last_year].groupby("ID"):
        s = g.set_index("Year")[col]
        p = forecast(s, last_year, years, method)
        out.append(pd.DataFrame({"ID": city, "Year": years, "pred": p.values}))
    return pd.concat(out, ignore_index=True)


def validate(df):
    years = list(range(2011, 2019))
    rows = []
    for code, col in METRICS.items():
        actual = df[df["Year"].isin(years)][["ID", "Year", col]]
        for method in ("last", "avg30", "trend30"):
            pred = predict_all(df, 2010, years, method, col)
            m = actual.merge(pred, on=["ID", "Year"]).dropna()
            rows.append({"metric": code, "method": method,
                         "MAE": (m[col] - m["pred"]).abs().mean()})
    table = pd.DataFrame(rows).pivot(index="metric", columns="method", values="MAE")
    print("Validation MAE (train <= 2010, predict 2011-2018):")
    print(table.round(2))
    return table.idxmin(axis=1).to_dict()


def write_submission(df, best):
    sub = pd.read_csv(SUBMISSION)  # template: city_id, year, metric, predicted_value
    years = sorted(sub["year"].unique())
    preds = []
    for code, col in METRICS.items():
        p = predict_all(df, 2018, years, best[code], col)
        p["metric"] = code
        preds.append(p)
    preds = pd.concat(preds).rename(columns={"ID": "city_id", "Year": "year"})
    sub = sub.drop(columns="predicted_value").merge(
        preds, on=["city_id", "year", "metric"], how="left")
    missing = sub["pred"].isna().sum()
    if missing:
        print(f"Warning: {missing} rows had no history; filled with metric median")
        sub["pred"] = sub["pred"].fillna(sub.groupby("metric")["pred"].transform("median"))
    sub = sub.rename(columns={"pred": "predicted_value"})
    sub.to_csv(SUBMISSION, index=False)
    print(f"Wrote {SUBMISSION} ({len(sub)} rows). Methods used: {best}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()
    np.random.seed(42)
    data = load_annual()
    best = validate(data)
    if not args.validate:
        write_submission(data, best)
