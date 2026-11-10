"""Build static geography features for the 210 challenge cities.

Inputs  (raw/): cities_elev.csv  (ID, Name, Lat, Lon, elevation_m)  -- Open-Meteo elevation API (static DEM, no weather)
                coast50.geojson  -- Natural Earth 50m coastline (public domain)
Output: city_features.csv

All inputs are static geography, so they are allowed by the challenge rules.
The current axes are HAND-DRAWN APPROXIMATIONS (see below), not an official dataset.
"""
import json
import numpy as np
import pandas as pd

R = 6371.0


def hav(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def densify(pts, step_deg=0.05):
    out = []
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        n = max(2, int(max(abs(x1 - x0), abs(y1 - y0)) / step_deg))
        out += list(zip(np.linspace(x0, x1, n), np.linspace(y0, y1, n)))
    return np.array(out)  # columns: lon, lat


# Approximate axes, (lon, lat). Gulf Stream: Florida Straits -> Hatteras -> off New England.
GULF_STREAM = [(-80.0, 25.0), (-79.9, 27.0), (-79.5, 29.0), (-78.5, 31.0), (-76.5, 33.5),
               (-75.3, 35.2), (-72.0, 37.5), (-69.0, 39.5), (-65.0, 40.8)]
# California Current: cool southward flow ~100-300 km offshore, Washington -> Baja.
CALIFORNIA = [(-127.0, 48.0), (-127.0, 45.0), (-126.0, 42.0), (-125.0, 39.0),
              (-123.5, 36.5), (-121.5, 34.0), (-119.5, 32.0), (-117.5, 30.0)]
# Labrador Current (cold, southward) approximate shelf-edge path, useful for New England / Mid-Atlantic.
LABRADOR = [(-55.0, 50.0), (-52.0, 46.0), (-56.0, 43.0), (-62.0, 41.5), (-68.0, 40.0), (-72.0, 38.5)]


def min_dist(lat, lon, line):
    return float(hav(lat, lon, line[:, 1], line[:, 0]).min())


def main():
    cities = pd.read_csv("raw/cities_elev.csv")
    gj = json.load(open("raw/coast50.geojson"))
    segs = []
    for f in gj["features"]:
        g = f["geometry"]
        lines = [g["coordinates"]] if g["type"] == "LineString" else g["coordinates"]
        for ln in lines:
            a = np.array(ln)
            if len(a) > 1 and a[:, 0].max() > -140 and a[:, 0].min() < -50 and a[:, 1].max() > 10 and a[:, 1].min() < 60:
                segs.append(densify([tuple(p) for p in a]))
    coast = np.vstack(segs)
    coast = coast[(coast[:, 0] > -140) & (coast[:, 0] < -50) & (coast[:, 1] > 10) & (coast[:, 1] < 60)]

    gs, ca, lb = densify(GULF_STREAM), densify(CALIFORNIA), densify(LABRADOR)
    rows = []
    for _, c in cities.iterrows():
        d_gs = min_dist(c.Lat, c.Lon, gs)
        d_ca = min_dist(c.Lat, c.Lon, ca)
        d_lb = min_dist(c.Lat, c.Lon, lb)
        rows.append(dict(
            ID=c.ID, Name=c.Name, Lat=c.Lat, Lon=c.Lon,
            elevation_m=max(c.elevation_m, 0),
            dist_coast_km=min_dist(c.Lat, c.Lon, coast),
            dist_gulf_stream_km=d_gs,
            dist_california_current_km=d_ca,
            dist_labrador_km=d_lb,
        ))
    f = pd.DataFrame(rows)
    # Smooth 0-1 "ocean influence" weights (exponential decay, e-folding 150 km coast / 300 km currents).
    f["w_coast"] = np.exp(-f.dist_coast_km / 150)
    f["w_gulf_stream"] = np.exp(-f.dist_gulf_stream_km / 300) * (f.Lon > -90)  # Atlantic side only
    f["w_calif_current"] = np.exp(-f.dist_california_current_km / 300) * (f.Lon < -110)  # Pacific side only
    f.to_csv("city_features.csv", index=False)
    print(f.describe().T[["min", "50%", "max"]])
    print(f.sort_values("dist_coast_km").head(3)[["Name", "dist_coast_km"]])
    print(f.sort_values("dist_coast_km").tail(3)[["Name", "dist_coast_km"]])


if __name__ == "__main__":
    main()
