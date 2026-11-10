# Data card: `data/`

The curated data our code reads. Together with `code/`, it reproduces `submission.csv` and the Policy Idea ranking.

## Files

| File | What it holds | Source | License | Years covered | How we prepared it | Used for |
|---|---|---|---|---|---|---|
| `annual_valid.csv` | One row per city and year: all `annual_indices` columns, including the four forecast metrics (`Avg Temp`, `ToT Prcp`, `Max Tmax`, `cdd`) and the `Miss *` missing-day counts | [Challenge dataset](https://huggingface.co/datasets/CMU-OSPO/protostar-climate-train-to-2018), `annual_indices` | As on the challenge dataset card | Each city's record, ending 2018 | Replaced `-inf` (empty years in `Max Tmax`) with missing; dropped city-years where any `Miss *` column exceeds 10 days (the scorer's rule). Written by `code/baseline.py`. | Forecast input |
| `city_features.csv` | One row per city: `Lat`, `Lon`, `elevation_m`, distances to the coast / Gulf Stream / California Current / Labrador Current (km), 0–1 ocean-influence weights, `tree_canopy_frac` (0–1) and `tree_canopy_valid_frac` (QC) | Elevation: Open-Meteo Elevation API (Copernicus DEM). Coastline: Natural Earth 1:50m (public domain). Current paths: hand-drawn approximations (see `code/build_features.py`). Canopy: NLCD/USFS Tree Canopy Cover CONUS 2018 v2021-4, via the MRLC WCS service | Natural Earth and NLCD: public domain; Open-Meteo: CC BY 4.0 | Static (canopy: 2018) | `code/build_features.py`, then `code/add_tree_canopy.py` (mean canopy over a ~10 km box around each station) | Geography trend: tested, not chosen. Canopy: **analysis and policy only** |
| `raw/cities_elev.csv` | City ID, name, lat/lon, elevation | Open-Meteo Elevation API | CC BY 4.0 | Static | One API call per station | Input to `build_features.py` |
| `raw/coast50.geojson` | Natural Earth 1:50m coastline | [Natural Earth](https://www.naturalearthdata.com/) | Public domain | Static | Unchanged | Input to `build_features.py` |
| `policy_ranking.csv` | Per city: canopy %, mean CDD 1989–2018, mean forecast CDD 2019–2025, increase, priority rank | Our forecasts + the files above | CC BY 4.0 (ours) | 1989–2025 | `code/policy_ranking.py` | Analysis and policy only |

## How the files fit together

- **Join key:** `ID` = the station ID in the challenge `city_info` table = `city_id` in `submission.csv`.
- **Units:** °F, inches, °F-days; elevation in m; distances in km; canopy as a 0–1 fraction (×100 = %).
- **Known limitations:** many stations are at airports, so canopy around the station is lower than in the city's neighborhoods. Ocean-current paths are approximate. Rows with too little history fall back to the 30-year average.

## Data rule check

- [x] Every forecast input is weather data up to 2018-12-31 (`code/baseline.py` stops if any year is after 2018) or static geography that doesn't respond to the weather.
- [x] No 2019–2025 data is used anywhere.
