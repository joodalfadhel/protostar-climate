---
pretty_name: "Protostar Climate Challenge Submission: GW OSPO"
license: cc-by-4.0
task_categories:
- time-series-forecasting
tags:
- climate
- forecasting
- protostar-climate-2026
configs:
- config_name: default
  data_files:
  - split: test
    path: submission.csv
---

# GW OSPO: Protostar Climate Challenge Submission

**Negative result:** no

Forecasts of yearly weather for 210 U.S. cities, 2019–2025, made as of the end of 2018 (no data that reveals 2019–2025 weather), for the [Protostar Challenge: Climate Risk Intelligence](https://huggingface.co/spaces/CMU-OSPO/protostar-challenge-climate-risk-intelligence).

## Team

| Name | Hugging Face username | University, Program |
|---|---|---|
| Jood Alfadhel | @JoodAlfadhel | George Washington University, B.S. Computer Science |
| Eyouel Kibret | @eyouelk | George Washington University, [PROGRAM] |
| David Lippert | @david-lippert | George Washington University | 

## Method

One script, [`code/baseline.py`](code/baseline.py), builds every forecast from the challenge's `annual_indices` table (records up to 2018 only).

- **Model:** a menu of simple, explainable forecasts, with one chosen per metric by backtesting:
  - **Temperature: national warming line plus a fading departure (`pooled15p`).** We average all 210 cities' temperature anomalies, fit one warming line over the last 40 years, and add the expected warming to each city's recent 15-year mean (`pooled15`). Pooling across cities cancels out local noise that makes single-city trend lines unreliable. We then add a *departure term*: how far each city's last year ran above or below the national warming line, carried forward with a fade. The departure is split into a regional part (mean of cities within 300 km) and a local remainder, and each fades at its own year-to-year persistence, estimated from the data up to the forecast year (`persistence_adjust` in `code/baseline.py`). The two-scale idea comes from the README of a public submission to this challenge; we implemented and backtested it ourselves.
  - **Cooling degree days: ensemble of the national-warming forecasts plus the same fading departure (`ensPp`).** The same warming shift, converted to CDD with each city's own CDD-per-°F sensitivity, averaged over 10-, 15-, 20- and 30-year baselines (`ensP`). The temperature departure term is converted to CDD with the same sensitivity, so CDD stays consistent with the temperature forecast.
  - **Precipitation: ensemble of averages (`ensA`)**, the mean of the 30-year mean, 30-year median, 15-year mean and 10-year mean.
  - **Hottest day: 30-year mean (`avg30`).** No method beat it in backtests; hottest-day values are noisy and show little trend. We also tried raising it with the national warming, scaled by how much the hottest day moves with average temperature (about 0.5–0.7 °F per °F in the data), at full, half, 30% and fitted strength. Over the same five cutoffs the skill was −0.010, −0.000, +0.001 and −0.003 versus 0.000 for the plain mean. The expected warming over 2019–2025 is only a few tenths of a degree, small next to the typical hottest-day error of about 2.6 °F, so we kept the flat 30-year mean. This test was run in a scratch version of the script and is not kept in `code/`.
- **Features:** each city's own history of the four metrics, the national average temperature anomaly, and each city's CDD sensitivity to annual temperature. Years with more than 10 missing days are dropped, matching the scorer.
- **Geography + tree canopy (tested, not chosen):** we also tried a *geography-shrunk trend*: each city's own trend pulled toward the slope predicted, by ridge regression across cities, from its elevation, distance to the coast, Gulf Stream and California Current influence, and latitude/longitude (`geo30`, `geo40`, `geo50`, `geo30k8`). On a single 2011–2018 split it looked better than the 30-year average for all four metrics. But the window and shrinkage had been picked on that same split, and **in our 5-cutoff backtest it lost to the methods above on every metric** (e.g., precipitation −0.013 to −0.033 skill). We kept it in the code and the table so others can check, but it doesn't drive any forecast.
- **How we checked it:** rolling backtests from five cutoffs (1996, 2000, 2004, 2008, 2011). Each cutoff predicts the next 7 years using only earlier data, and is scored as skill vs. the 30-year average (the challenge's own scoring rule). Mean backtest skill: temperature +0.176, CDD +0.130, precipitation ≈0, hottest day ≈0 (overall 7.7 on the challenge's 0–100 scale; 7.3 before the departure term was added). The departure term is a small gain (+0.008 temperature, +0.007 CDD) and it helped in the mean over the five cutoffs, not necessarily at each one, so treat it as modest evidence. The half-strength version (`pooled15h`, `ensPh`) scored almost the same (+0.174, +0.127). **We chose methods from these backtests only, never from leaderboard scores**, so our choices aren't tuned to 2019–2021.

| Metric | Chosen method | Backtest skill vs. 30-yr average | Best plain baseline |
|---|---|---|---|
| Avg temperature | `pooled15p` (`pooled15` alone: +0.168) | +0.176 | linear trend +0.124 |
| Cooling degree days | `ensPp` (`ensP` alone: +0.123) | +0.130 | linear trend +0.064 |
| Precipitation | `ensA` | +0.000 | 30-yr average 0.000 |
| Hottest day | `avg30` | 0.000 | 30-yr average 0.000 |
| *(all, tested)* | *`geo40` (geography trend)* | *temp +0.145, CDD +0.111, hottest −0.020, precip −0.018* | — |

## AI Tools and Agents

- **Tools and models:** Claude (Anthropic) in the Claude app, model Claude Opus 5.5; and Claude Code (Anthropic), model Claude Sonnet 5.5, in a later session. [ADD ANY TOOLS EYOUEL USED]
- **What they did:** wrote the geography-feature scripts (`code/build_features.py`, `code/add_tree_canopy.py`) and fetched the static elevation, coastline and canopy data; read a public submission's README for ideas and, from that, wrote `persistence_adjust` and the `pooled15p`/`ensPp` methods in `code/baseline.py` (the departure term, backtested with the code's own 5-cutoff table); made the `datasets` import lazy and fixed `code/requirements.txt`; updated this README and the friction log; reviewed and re-tested the geography features (moving them from a single split to the 5-cutoff backtest); wrote `code/policy_ranking.py`; explained the challenge rules; set up the repository structure; wrote the first version of `code/baseline.py` (the three reference baselines and the backtest); added the ensemble methods; drafted this README and the data card from our results.
- **What a person reviewed:** we ran every script ourselves and read the backtest output. The model choices come from the code's backtest, not from the AI. No AI tool supplied or edited any `predicted_value`, and no tool downloaded weather data after 2018 (the AI's environment could not reach the dataset at all; all data loading ran on our laptops).
- **Agent instructions file:** [`code/AGENTS.md`](code/AGENTS.md) (the template's file, unchanged).

## Policy Idea

1. **What risk did we notice?** Cooling demand is rising almost everywhere. Our forecasts put 2019–2025 cooling degree days above the 1989–2018 average in **209 of 210 cities** (median +111 °F-days, about +9%). The biggest increases are in places that are already the hottest: Needles, Bakersfield, Galveston, El Paso, Phoenix, Corpus Christi (+220 to +355 °F-days a year). More CDD means more air-conditioning load, higher summer grid peaks, and more heat risk for households that can't afford cooling.
2. **What would we do about it, and who would do it?** A **heat-and-shade priority list run by electric utilities with city governments.** We combined our CDD forecast with tree canopy (NLCD 2018) around each station and ranked cities by *big CDD increase + little shade* ([`data/policy_ranking.csv`](data/policy_ranking.csv), from [`code/policy_ranking.py`](code/policy_ranking.py)). The top 10 all have under 2.5% canopy. Utilities would fund shade where it's missing, through their existing energy-efficiency budgets: cool roofs and shade structures in desert and dry-valley cities (Needles, Phoenix, Tucson, El Paso, Bakersfield, Fresno), where trees need scarce water, and street trees in the Gulf Coast and Texas cities (Galveston, Corpus Christi, Key West, Amarillo). The ranking is re-run each year as new data arrives.
3. **Why is it sustainable?** Shade and reflective roofs cut cooling load without burning fuel, so they lower emissions and bills together, and they shave the summer peaks that would otherwise need new power plants. Matching the fix to the climate (roofs where water is scarce, trees where rain supports them) avoids trading a heat problem for a water problem. Everything uses open data and open code, so any city can reproduce the ranking.

## Feedback to the Ecosystem

- **What got in the way:**
  - The submission rules lived only in the Space's JavaScript-rendered tabs. The Space's own `README.md` and `config.py` didn't describe the submission format, and the commit history contradicted itself ("Remove forecasting, CSV and leaderboard" four days before the CSV check was added). It took us time to establish what to submit.
  - The scorer drops station-years with more than 10 missing days, but the training data includes them, so each team has to rediscover and re-apply that filter (we did, using the `Miss *` columns).
  - Column names in `annual_indices` (`Avg Temp`, `ToT Prcp`, `Max Tmax`) don't match the metric codes in `submission.csv` (`annual_avg_temp`, `total_precip_in`, `hottest_day_temp`). The mapping is only in the Space text.
  - Tooling: on macOS with Python 3.9, `pip install huggingface_hub` puts the `hf` command outside the PATH. Logging in failed with "command not found", and git then fell back to an old saved credential ("Invalid username or password").
  - The `Max Tmax` column holds `-inf` in years with no data rather than an empty value, which silently breaks averages unless it's caught.
  - Backtests on one split are easy to over-read: our geography model looked like a win on 2011–2018 alone and lost on five cutoffs. Only one practice split is published.
  - No ready-made table of physical geography exists for the 210 stations. We had to build elevation (Open-Meteo API), coastline distance (Natural Earth) and canopy (MRLC web service) ourselves, and draw the ocean-current paths by hand, since we found no simple open dataset of current axes.
  - Many stations are at airports, so land cover around the station (e.g., canopy) doesn't describe the city where people live.
  - [ADD ANYTHING ELSE YOU HIT]
- **What would fix it:**
  - Put the submission spec, the scoring rule and the missing-day rule in the dataset card and the Space `README.md`, so they're in plain files that tools and people can read.
  - Ship a `valid` flag column (or a pre-filtered table) in `annual_indices`, and add a `metric_code` row to the column descriptions.
  - Publish a station-features table (elevation, coast distance, land cover) alongside `city_info`, and the practice backtest at several cutoffs, not one.
  - In the submission template, include a `requirements.txt` and a note on logging in with `git` credentials when `hf` isn't on the PATH.

## Files

| File | Content |
|---|---|
| `submission.csv` | Our forecasts: `city_id,year,metric,predicted_value` |
| `code/baseline.py` | Loads the data, runs the backtests, picks a method per metric and writes `submission.csv` |
| `code/check_submission.py` | The template's checker (same checks as the Space) |
| `code/build_features.py` | Builds `data/city_features.csv`: elevation, coast distance, ocean-current influence |
| `code/add_tree_canopy.py` | Adds tree canopy (NLCD 2018) to `data/city_features.csv` |
| `code/policy_ranking.py` | Policy Idea analysis: CDD increase vs canopy → `data/policy_ranking.csv` |
| `code/requirements.txt` | Python packages |
| `code/AGENTS.md` | Data rules for AI agents |
| `data/` | The curated data our code uses; described in its data card, [`data/README.md`](data/README.md) |

**Metric codes:** `annual_avg_temp` (°F), `total_precip_in` (inches), `hottest_day_temp` (°F), `cdd` (°F-days).
**City IDs:** the station `ID` from the `city_info` table of the training dataset (e.g., Pittsburgh = `USW00094823`).

## How to Reproduce

1. `pip install -r code/requirements.txt` (Python 3.9 or newer).
2. `python code/baseline.py` reads `data/annual_valid.csv` (or, if it's missing, downloads `annual_indices` from the challenge dataset and saves it there), prints the backtest table and writes `submission.csv`. It takes about a minute. The run is deterministic.
3. `python code/check_submission.py` confirms the file is complete.
4. Optional: `python code/policy_ranking.py` reproduces the Policy Idea ranking. To rebuild the geography features from scratch, run `python ../code/build_features.py` then `python ../code/add_tree_canopy.py` from inside `data/` (they read `raw/` and write `city_features.csv`; the canopy step downloads from MRLC).

## Data

- **External data** (all static geography, none of it responds to the weather; see the data card):
  - Elevation: [Open-Meteo Elevation API](https://open-meteo.com/en/docs/elevation-api) (Copernicus DEM). Used in the geography trend (tested, not chosen).
  - Coastline: [Natural Earth 1:50m coastline](https://www.naturalearthdata.com/), public domain. Same use.
  - Ocean-current paths (Gulf Stream, California Current, Labrador Current): hand-drawn approximations in `code/build_features.py`, not an official dataset. Same use.
  - Tree canopy: [NLCD / USFS Tree Canopy Cover, CONUS 2018, v2021-4](https://www.mrlc.gov/data), 30 m, mean over a ~10 km box around each station. **Analysis and policy only.**
- **Forecast inputs actually used:** only the challenge dataset, records up to 2018-12-31.
- **Who built the geography features:** [NAME, and their role if they're not on the team]
- **Curated data:** see the data card in [`data/README.md`](data/README.md) (each file, its source, license and preparation).

Training data: [`CMU-OSPO/protostar-climate-train-to-2018`](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), from:

- Lai, Y., & Dzombak, D. A. (2019). Compiled historical daily temperature and precipitation data for selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7890488.v6
- Lai, Y., & Dzombak, D. A. (2019). Historical changes of annual temperature and precipitation indices at selected 210 U.S. cities. Carnegie Mellon University. https://doi.org/10.1184/R1/7961012.v6
- Lai, Y., & Dzombak, D. A. (2019). *Use of historical data to assess regional climate change.* Journal of Climate, 32(14), 4299–4320. https://doi.org/10.1175/JCLI-D-18-0630.1
