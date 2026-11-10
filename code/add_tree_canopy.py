"""Add `tree_canopy_frac` (0-1) to city_features.csv.

Source: USGS/USFS NLCD Tree Canopy Cover, CONUS, 2018 (v2021-4), 30 m, % canopy 0-100,
served by MRLC WCS (layer nlcd_tcc_conus_2018_v2021-4). Canopy is non-weather-responsive land-cover data and
the vintage (2018) is within the challenge's 2018-12-31 cutoff.
Value = mean canopy over a ~10 km x 10 km box centred on the station (a point value is too noisy for an airport station).
Run from /Users/david/Hugging Face/features:  python3 -I add_tree_canopy.py
"""
import io
import math
import time
import urllib.request

import numpy as np
import pandas as pd
from PIL import Image

HALF_KM = 5.0
LAYER = "mrlc_download:nlcd_tcc_conus_2018_v2021-4"
URL = ("https://www.mrlc.gov/geoserver/mrlc_download/wcs?service=WCS&version=1.0.0&request=GetCoverage"
       "&coverage={layer}&crs=EPSG:4326&bbox={w},{s},{e},{n}&width=300&height=300&format=GeoTIFF")


def canopy(lat, lon):
    dlat = HALF_KM / 111.0
    dlon = HALF_KM / (111.0 * math.cos(math.radians(lat)))
    url = URL.format(layer=LAYER, w=lon - dlon, e=lon + dlon, s=lat - dlat, n=lat + dlat)
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            a = np.array(Image.open(io.BytesIO(data))).astype(float)
            valid = a <= 100  # >100 = nodata / non-CONUS
            return (a[valid].mean() / 100.0, valid.mean()) if valid.any() else (np.nan, 0.0)
        except Exception as ex:  # retry transient failures
            time.sleep(2 * (attempt + 1))
            err = ex
    print("FAILED", lat, lon, err)
    return np.nan, 0.0


if __name__ == "__main__":
    f = pd.read_csv("city_features.csv")
    f = f.drop(columns=[c for c in ("tree_canopy_frac", "tree_canopy_valid_frac") if c in f])
    res = [canopy(r.Lat, r.Lon) for r in f.itertuples()]
    f["tree_canopy_frac"] = [round(r[0], 4) if r[0] == r[0] else np.nan for r in res]
    f["tree_canopy_valid_frac"] = [round(r[1], 3) for r in res]  # share of box pixels with data (QC)
    f.to_csv("city_features.csv", index=False)
    print(f.tree_canopy_frac.describe())
    print("missing:", f.tree_canopy_frac.isna().sum(), "| low coverage (<0.9):", (f.tree_canopy_valid_frac < 0.9).sum())
    s = f.sort_values("tree_canopy_frac")
    print(s.head(4)[["Name", "tree_canopy_frac"]]); print(s.tail(4)[["Name", "tree_canopy_frac"]])
