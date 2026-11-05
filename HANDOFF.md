# Handoff

**Code pushed and runnable by Sunday night.** We submit Tuesday by 12pm (official deadline Tue Nov 10, 11:59 PM ET). Read the Space's *About* and *Data & submission* tabs once.

## The task
- Stand at the end of 2018. Forecast 2019–2025 for 210 cities × 4 metrics = 5,880 rows in `submission.csv` (the rows are pre-filled; we fill `predicted_value`).
- Metrics: `annual_avg_temp` (Avg Temp), `total_precip_in` (ToT Prcp), `hottest_day_temp` (Max Tmax), `cdd` (cdd), all from `annual_indices`.
- Score = MAE per metric relative to the 30-year-average baseline, averaged. Live score covers 2019–21, final score covers 2022–25.
- Organizers say most of the gain is in temp and CDD. For precip and hottest day, matching the 30-year average is already fine.

## Data rule (can disqualify us)
- Forecast inputs may only be weather up to **2018-12-31**, plus non-weather-driven data from any year (census, land cover, tree canopy, elevation).
- No 2019+ weather or anything driven by it (streamflow, air quality, energy demand). No values from LLMs or pretrained models that saw post-2018 data.
- 2019+ data **is** fine for the policy idea.
- If you use Claude/ChatGPT/Copilot, write down which tool, what for, and what you checked yourself. It goes in the README.

## Already in the repo
- `code/baseline.py` runs the 3 official baselines, validates them (train ≤2010, predict 2011–18), picks the best per metric, and writes a valid `submission.csv`. **Run this first** so we always have something submittable.
- `code/check_submission.py` (from the template) runs the same checks as the Space.

## Your tasks
1. `python code/baseline.py`, then `python code/check_submission.py`. Log anything that breaks in `friction_log.md`.
2. Improve temp + CDD. Ideas: trend fitted on a shorter or smoothed window, pooling trends across nearby cities, a model on lag features + city lat/lon. Always compare against the validation MAE from the baseline.
3. **Interoperability (20%):** add one external dataset that's allowed as an input (e.g. census population/density, NLCD land cover, tree canopy, elevation) and join it to `city_info`. Save the processed version in `data/` and describe it in `data/README.md` (source, license, how prepared).
4. Make the final model one script that reproduces `submission.csv` from `data/`.
5. Leave me short notes for the README: method in a few lines, validation MAE table, 1 figure.

## Rules
- `random_state=42` everywhere.
- Never commit tokens.
- Log friction as you go.
