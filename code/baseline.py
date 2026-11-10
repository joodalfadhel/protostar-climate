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
# ensembles: name -> component methods, averaged with equal weight
_ENS_WARM = {"ensP": ("pooled", "pooled20", "pooled15", "pooled10"),
             "ensPT": ("pooled", "pooled10", "trend30", "avg10")}
_ENS_FLAT = {"ensA": ("avg30", "med30", "avg15", "avg10"),
             "ensB": ("avg20", "avg10", "med15")}
ENSEMBLES = {"annual_avg_temp": _ENS_WARM, "cdd": _ENS_WARM,
             "total_precip_in": _ENS_FLAT, "hottest_day_temp": _ENS_FLAT}
# methods tried per metric
_BASIC = ("last", "avg30", "trend30")
METHODS = {
    "annual_avg_temp": _BASIC + tuple(POOLED) + tuple(_ENS_WARM),
    "cdd": _BASIC + tuple(POOLED) + tuple(_ENS_WARM),
    "total_precip_in": _BASIC + ("med30", "avg20", "avg15", "avg10", "med15") + tuple(_ENS_FLAT),
    "hottest_day_temp": _BASIC + ("med30", "avg20", "avg15", "avg10", "med15") + tuple(_ENS_FLAT),
}
# departure persistence: name -> (base method, strength). The base forecast plus a faded copy of the city's
# latest departure from the national warming line (see persistence_adjust).
PERSIST = {"pooled15p": ("pooled15", 1.0), "pooled15h": ("pooled15", 0.5),
           "ensPp": ("ensP", 1.0), "ensPh": ("ensP", 0.5)}
for _m in ("annual_avg_temp", "cdd"):
    METHODS[_m] = METHODS[_m] + tuple(PERSIST)
COL_TO_CODE = {v: k for k, v in METRICS.items()}
# geography-shrunk trend: name -> (fit window in years, shrinkage k toward the geography slope)
GEO_METHODS = {"geo30": (30, 0.5), "geo40": (40, 0.5), "geo50": (50, 0.5), "geo30k8": (30, 0.8)}
for _m in METHODS:
    METHODS[_m] = METHODS[_m] + tuple(GEO_METHODS)
GEO_FEATURES = ROOT / "data" / "city_features.csv"


def geo_matrix():
    g = pd.read_csv(GEO_FEATURES).set_index("ID")
    return pd.DataFrame({
        "elev": g["elevation_m"] / 1000,
        "w_coast": g["w_coast"], "w_gulf": g["w_gulf_stream"], "w_calif": g["w_calif_current"],
        "lat": (g["Lat"] - g["Lat"].mean()) / g["Lat"].std(),
        "lon": (g["Lon"] - g["Lon"].mean()) / g["Lon"].std(),
    })


def predict_geo(df, last_year, years, col, window, k, lam=1.0):
    """Each city's trend, shrunk toward the slope predicted from its geography
    (ridge regression across cities on elevation, coast and ocean-current features)."""
    G = geo_matrix()
    fits = {}
    for city, g in df[(df["Year"] <= last_year) & (df["Year"] > last_year - window)].groupby("ID"):
        g = g[["Year", col]].dropna()
        if len(g) >= 10 and city in G.index:
            slope = np.polyfit(g["Year"], g[col], 1)[0]
            fits[city] = (slope, g[col].mean(), g["Year"].mean())
    s = pd.DataFrame(fits, index=["slope", "mean", "xm"]).T
    X = np.c_[np.ones(len(s)), G.loc[s.index].values]
    beta = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ s["slope"].values)
    s["geo_slope"] = X @ beta
    yrs = np.asarray(years)
    out = []
    for city in df[df["Year"] <= last_year]["ID"].unique():
        if city in s.index:
            r = s.loc[city]
            sl = (1 - k) * r["slope"] + k * r["geo_slope"]
            p = r["mean"] + sl * (yrs - r["xm"])
            if col in ("cdd", "ToT Prcp"):
                p = np.maximum(p, 0)
        else:  # too little history: fall back to the 30-year average
            h = df[(df["ID"] == city) & (df["Year"] <= last_year)].set_index("Year")[col]
            p = forecast(h, last_year, years, "avg30").values
        out.append(pd.DataFrame({"ID": city, "Year": years, "pred": p}))
    return pd.concat(out, ignore_index=True)
WINDOW = 30
TREND_WINDOW = 40  # years used to fit the pooled warming line
HORIZON = 7
ORIGINS = (1996, 2000, 2004, 2008, 2011)  # backtest cutoffs; each predicts the next 7 years
MAX_MISSING_DAYS = 10


CURATED = ROOT / "data" / "annual_valid.csv"


def load_annual():
    """Read the curated table in data/; build it from the challenge dataset if missing."""
    if CURATED.exists():
        df = pd.read_csv(CURATED)
    else:
        from datasets import load_dataset  # only needed when the curated table is missing
        df = load_dataset(REPO, "annual_indices", split="train").to_pandas()
        df = df.replace([np.inf, -np.inf], np.nan)  # Max Tmax is -inf in empty years
        miss_cols = [c for c in df.columns if c.startswith("Miss")]
        if miss_cols:  # drop badly incomplete years, as the scorer does
            df = df[df[miss_cols].max(axis=1) <= MAX_MISSING_DAYS]
        CURATED.parent.mkdir(exist_ok=True)
        df.to_csv(CURATED, index=False)
        print(f"Saved curated data to {CURATED}")
    assert df["Year"].max() <= 2018, "Weather input dated after 2018: data rule"
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


def _dist_300km(ids):
    """0/1 matrix: which cities lie within 300 km of each other (self included)."""
    g = pd.read_csv(GEO_FEATURES).set_index("ID")
    W = np.eye(len(ids))
    known = [i for i, c in enumerate(ids) if c in g.index]
    lat = np.radians(g.loc[[ids[i] for i in known], "Lat"].values)
    lon = np.radians(g.loc[[ids[i] for i in known], "Lon"].values)
    a = np.sin((lat[:, None] - lat[None]) / 2) ** 2 + np.cos(lat)[:, None] * np.cos(lat)[None] * np.sin((lon[:, None] - lon[None]) / 2) ** 2
    near = (2 * 6371.0 * np.arcsin(np.sqrt(a)) < 300).astype(float)
    W[np.ix_(known, known)] = near
    return W


def persistence_adjust(df, last_year, years, col, gamma):
    """Faded carry-over of each city's latest temperature departure (deg F, or CDD via sensitivity).

    Departure = observed temperature minus (city mean + national warming line), split into a regional part
    (mean of cities within 300 km) and a local remainder. Each part decays by its own year-to-year persistence,
    estimated across all cities from data up to `last_year`. Idea from a public submission's README
    (two-scale decay of the recent temperature departure); implemented and backtested here from scratch.
    """
    d = df[(df["Year"] <= last_year) & (df["Year"] > last_year - TREND_WINDOW)]
    temp = d.pivot_table(index="Year", columns="ID", values="Avg Temp")
    anom = temp - temp.mean()
    nat = anom.mean(axis=1).dropna()
    slope, icpt = np.polyfit(nat.index.values, nat.values, 1)
    E = (anom.sub(icpt + slope * temp.index.values, axis=0)).values
    ok = ~np.isnan(E)
    W = _dist_300km(list(temp.columns))
    reg = (np.nan_to_num(E) @ W.T) / np.maximum(ok.astype(float) @ W.T, 1)
    loc = E - reg

    def phi(Z):
        z0, z1 = np.nan_to_num(Z[:-1]), np.nan_to_num(Z[1:])
        return float(np.clip((z0 * z1).sum() / max((z0 ** 2).sum(), 1e-9), 0, 0.9))

    p_reg, p_loc = phi(reg), phi(loc)
    last = temp.index[-1] == last_year
    r_reg = reg[-1] if last else np.zeros(E.shape[1])
    r_loc = np.nan_to_num(loc[-1]) if last else np.zeros(E.shape[1])
    sens = cdd_sensitivity(df, last_year) if col == "cdd" else None
    rows = []
    for j, city in enumerate(temp.columns):
        k = 1.0 if sens is None else sens.get(city, sens.median())
        for y in years:
            h = y - last_year
            rows.append((city, y, gamma * k * (r_reg[j] * p_reg ** h + r_loc[j] * p_loc ** h)))
    return pd.DataFrame(rows, columns=["ID", "Year", "adj"])


def predict_all(df, last_year, years, method, col):
    if method in PERSIST:  # base forecast + faded departure
        base, gamma = PERSIST[method]
        p = predict_all(df, last_year, years, base, col).merge(
            persistence_adjust(df, last_year, years, col, gamma), on=["ID", "Year"], how="left")
        p["pred"] = p["pred"] + p["adj"].fillna(0.0)
        if col == "cdd":
            p["pred"] = p["pred"].clip(lower=0)
        return p[["ID", "Year", "pred"]]
    ens = ENSEMBLES[COL_TO_CODE[col]].get(method)
    if ens:  # equal-weight average of the component forecasts
        parts = [predict_all(df, last_year, years, m, col).set_index(["ID", "Year"])["pred"]
                 for m in ens]
        return pd.concat(parts, axis=1).mean(axis=1).rename("pred").reset_index()
    if method in GEO_METHODS:
        return predict_geo(df, last_year, years, col, *GEO_METHODS[method])
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