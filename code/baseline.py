"""Forecasts for the Protostar Climate Risk challenge.

Runs the three official reference baselines (last value, 30-year average,
30-year linear trend) plus two added methods, backtests them at several
cutoff years (all using data <= 2018 only), picks the best per metric, and
fills submission.csv.

Added methods:
  pooled  (temperature, CDD): one warming line fitted to the average anomaly of
          all cities, added to each city's 30-year mean. For CDD the shift is
          scaled by the city's own CDD-per-degree sensitivity.
  pooledNN / pooled25 / pooled50: variants with a shorter baseline window (NN
          years) or some weight on each city's own slope.
  medNN, avgNN (precipitation, hottest day): median or mean of the last NN years.
Method choice uses backtests on data <= 2018 only, never leaderboard scores.

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
# pooled variants: name -> (baseline window in years, weight on the city's own slope)
POOLED = {"pooled": (30, 0.0), "pooled25": (30, 0.25), "pooled50": (30, 0.5),
          "pooled20": (20, 0.0), "pooled15": (15, 0.0), "pooled10": (10, 0.0)}
# methods tried per metric
_BASIC = ("last", "avg30", "trend30")
METHODS = {
    "annual_avg_temp": _BASIC + tuple(POOLED),
    "cdd": _BASIC + tuple(POOLED),
    "total_precip_in": _BASIC + ("med30", "avg20", "avg15", "avg10", "med15"),
    "hottest_day_temp": _BASIC + ("med30", "avg20", "avg15", "avg10", "med15"),
}
WINDOW = 30
TREND_WINDOW = 40  # years used to fit the pooled warming line
HORIZON = 7
ORIGINS = (1996, 2000, 2004, 2008, 2011)  # backtest cutoffs; each predicts the next 7 years
MAX_MISSING_DAYS = 10


def load_annual():
    df = load_dataset(REPO, "annual_indices", split="train").to_pandas()
    assert df["Year"].max() <= 2018, "Weather input dated after 2018: data rule"
    miss_cols = [c for c in df.columns if c.startswith("Miss")]
    if miss_cols:  # drop badly incomplete years, as the scorer does
        df = df[df[miss_cols].max(axis=1) <= MAX_MISSING_DAYS]
    return df


def national_shift(df, last_year, years, window=WINDOW):
    """Warming (deg F) expected in `years` relative to the 30-year baseline mean."""
    d = df[(df["Year"] <= last_year) & (df["Year"] > last_year - TREND_WINDOW)]
    temp = d.pivot_table(index="Year", columns="ID", values="Avg Temp")
    anomaly = (temp - temp.mean()).mean(axis=1).dropna()
    slope, intercept = np.polyfit(anomaly.index.values, anomaly.values, 1)
    base_years = np.arange(last_year - window + 1, last_year + 1)
    ref = (intercept + slope * base_years).mean()
    return pd.Series(intercept + slope * np.asarray(years) - ref, index=years), slope


def city_slopes(df, last_year):
    """Each city's own warming slope (deg F per year) over the trend window."""
    d = df[(df["Year"] <= last_year) & (df["Year"] > last_year - TREND_WINDOW)]
    out = {}
    for city, g in d.groupby("ID"):
        g = g[["Year", "Avg Temp"]].dropna()
        out[city] = np.polyfit(g["Year"], g["Avg Temp"], 1)[0] if len(g) >= 20 else np.nan
    return pd.Series(out)


def cdd_sensitivity(df, last_year):
    """CDD change per deg F of annual average temperature, per city."""
    d = df[(df["Year"] <= last_year) & (df["Year"] > last_year - TREND_WINDOW)]
    out = {}
    for city, g in d.groupby("ID"):
        g = g[["Avg Temp", "cdd"]].dropna()
        out[city] = np.polyfit(g["Avg Temp"], g["cdd"], 1)[0] if len(g) >= 15 else np.nan
    s = pd.Series(out)
    return s.fillna(s.median())


def forecast(series: pd.Series, last_year: int, years, method: str,
             shift=None, sens=1.0, clip=False, window=WINDOW):
    """series: values indexed by Year, already cut at last_year."""
    full = series.dropna()
    if method[:3] in ("avg", "med") and method[3:].isdigit():
        window = int(method[3:])  # e.g. avg15 = mean of the last 15 years
    hist = full[full.index > last_year - window]
    if hist.empty:
        return pd.Series(np.nan, index=years)
    if method == "last":
        return pd.Series(hist.iloc[-1], index=years)
    if method.startswith("avg"):
        return pd.Series(hist.mean(), index=years)
    if method.startswith("med"):
        return pd.Series(hist.median(), index=years)
    if method == "pooled":
        p = hist.mean() + sens * shift.reindex(years).values
        return pd.Series(np.maximum(p, 0) if clip else p, index=years)
    if method == "trend30":
        hist = full[full.index > last_year - WINDOW]
        if len(hist) < 10:
            return pd.Series(hist.mean(), index=years)
        slope, intercept = np.polyfit(hist.index.values, hist.values, 1)
        return pd.Series(intercept + slope * np.asarray(years), index=years)
    raise ValueError(method)


def predict_all(df, last_year, years, method, col):
    pooled = method in POOLED
    if pooled:
        window, w = POOLED[method]
        shift, nat_slope = national_shift(df, last_year, years, window)
        dev = (city_slopes(df, last_year) - nat_slope).fillna(0.0)
        sens = None if col == "Avg Temp" else cdd_sensitivity(df, last_year)
        t = np.asarray(years) - (last_year - (window - 1) / 2)
    out = []
    for city, g in df[df["Year"] <= last_year].groupby("ID"):
        s = g.set_index("Year")[col]
        if pooled:
            # national warming line, nudged toward the city's own slope
            city_shift = shift + w * dev.get(city, 0.0) * t
            k = 1.0 if sens is None else sens.get(city, sens.median())
            p = forecast(s, last_year, years, "pooled", shift=city_shift,
                         sens=k, clip=col == "cdd", window=window)
        else:
            p = forecast(s, last_year, years, method)
        out.append(pd.DataFrame({"ID": city, "Year": years, "pred": p.values}))
    return pd.concat(out, ignore_index=True)


def validate(df):
    """Mean skill vs the 30-year average over several cutoffs (higher is better)."""
    skill = {}
    for code, col in METRICS.items():
        for origin in ORIGINS:
            years = list(range(origin + 1, min(origin + HORIZON, 2018) + 1))
            actual = df[df["Year"].isin(years)][["ID", "Year", col]]
            mae = {}
            for method in METHODS[code]:
                pred = predict_all(df, origin, years, method, col)
                m = actual.merge(pred, on=["ID", "Year"]).dropna()
                mae[method] = (m[col] - m["pred"]).abs().mean()
            for method, v in mae.items():
                skill.setdefault((code, method), []).append(1 - v / mae["avg30"])
    table = pd.Series({k: np.mean(v) for k, v in skill.items()}).unstack()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print(f"Backtest skill vs 30-year average (mean over cutoffs {ORIGINS}):")
    print(table.round(3))
    best = table.idxmax(axis=1).to_dict()
    print("Chosen:", best, "| mean skill x100:",
          round(100 * np.mean([table.loc[c, best[c]] for c in best]), 1))
    return best


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