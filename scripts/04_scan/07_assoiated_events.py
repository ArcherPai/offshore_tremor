#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import timedelta
import numpy as np
import pandas as pd

# =========================
# USER SETTINGS
# =========================
CSV_PATH = Path("/home/bxd240002/scratch/Archer/OBS_test/outputs/merge_runs/EVENTS_merged_after_20260312T004541.csv")
OUT_DIR  = Path("/home/bxd240002/scratch/Archer/locating/data/csv")

EVENT_TYPE_KEEP = "TREMOR"

# association thresholds
MIN_TIME_OVERLAP_S   = 150.0   # if two 300-s windows overlap by >= 150 s
MAX_CENTER_DIFF_S    = 180.0   # OR if their centers differ by <= 180 s
MAX_STATION_DIST_KM  = 100.0   # spatial sanity check

# group filters
MIN_STATIONS_PER_GROUP = 3     # keep only groups with >= 3 unique stations

# centroid weighting
USE_PMAX_WEIGHT = True
PMAX_MIN_FLOOR  = 1e-6

# =========================
# HELPERS
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    """
    Great-circle distance (km).
    """
    R = 6371.0
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return R * c

class DSU:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

# =========================
# LOAD
# =========================
df = pd.read_csv(CSV_PATH)

required_cols = [
    "event_uid", "event_type", "event_start_iso", "window_s",
    "network", "station", "event_lat", "event_lon"
]
for c in required_cols:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# keep tremor only
df = df[df["event_type"] == EVENT_TYPE_KEEP].copy()

# parse times
df["t0"] = pd.to_datetime(df["event_start_iso"], errors="coerce")
df = df.dropna(subset=["t0", "event_lat", "event_lon", "network", "station"]).copy()

# window length
df["window_s"] = pd.to_numeric(df["window_s"], errors="coerce")
df = df.dropna(subset=["window_s"]).copy()

df["t1"] = df["t0"] + pd.to_timedelta(df["window_s"], unit="s")
df["t_center"] = df["t0"] + (df["t1"] - df["t0"]) / 2

# station id
df["station_id"] = df["network"].astype(str) + "." + df["station"].astype(str)

# pmax weight
if "pmax" in df.columns:
    df["pmax"] = pd.to_numeric(df["pmax"], errors="coerce").fillna(0.0)
else:
    df["pmax"] = 1.0

df = df.reset_index(drop=True)

if len(df) == 0:
    raise RuntimeError("No valid tremor rows found after filtering.")

print(f"[INFO] Input tremor detections: {len(df)}")

# =========================
# SORT BY TIME
# =========================
df = df.sort_values("t0").reset_index(drop=True)

# =========================
# BUILD ASSOCIATION GRAPH
# =========================
n = len(df)
dsu = DSU(n)

# We only compare nearby-in-time rows for efficiency
# Since windows are 300 s and threshold ~180 s, a few hours buffer is enough
TIME_BUFFER_S = max(df["window_s"].max(), MAX_CENTER_DIFF_S, MIN_TIME_OVERLAP_S) + 600.0

for i in range(n):
    ti0 = df.at[i, "t0"]
    ti1 = df.at[i, "t1"]
    tci = df.at[i, "t_center"]

    lati = df.at[i, "event_lat"]
    loni = df.at[i, "event_lon"]

    j = i + 1
    while j < n:
        tj0 = df.at[j, "t0"]

        # since sorted by t0, stop if too far in time
        if (tj0 - ti0).total_seconds() > TIME_BUFFER_S:
            break

        tj1 = df.at[j, "t1"]
        tcj = df.at[j, "t_center"]

        # time relation
        overlap_s = max(
            0.0,
            (min(ti1, tj1) - max(ti0, tj0)).total_seconds()
        )
        center_diff_s = abs((tcj - tci).total_seconds())

        time_ok = (overlap_s >= MIN_TIME_OVERLAP_S) or (center_diff_s <= MAX_CENTER_DIFF_S)
        if not time_ok:
            j += 1
            continue

        # spatial sanity check
        latj = df.at[j, "event_lat"]
        lonj = df.at[j, "event_lon"]
        dist_km = haversine_km(lati, loni, latj, lonj)

        if dist_km <= MAX_STATION_DIST_KM:
            dsu.union(i, j)

        j += 1

# =========================
# ASSIGN GROUP IDs
# =========================
roots = [dsu.find(i) for i in range(n)]
df["group_root"] = roots

# group root -> compact id
root_to_gid = {}
gid_list = []
counter = 1
for r in df["group_root"]:
    if r not in root_to_gid:
        root_to_gid[r] = f"CAND_{counter:06d}"
        counter += 1
    gid_list.append(root_to_gid[r])

df["candidate_id"] = gid_list

# =========================
# GROUP SUMMARY
# =========================
rows = []

for candidate_id, g in df.groupby("candidate_id"):
    station_ids = sorted(g["station_id"].unique())
    n_stations = len(station_ids)
    n_detections = len(g)

    if n_stations < MIN_STATIONS_PER_GROUP:
        continue

    group_start = g["t0"].min()
    group_end   = g["t1"].max()

    lats = g["event_lat"].to_numpy(dtype=float)
    lons = g["event_lon"].to_numpy(dtype=float)

    if USE_PMAX_WEIGHT:
        w = np.maximum(g["pmax"].to_numpy(dtype=float), PMAX_MIN_FLOOR)
    else:
        w = np.ones(len(g), dtype=float)

    centroid_lat = np.sum(w * lats) / np.sum(w)
    centroid_lon = np.sum(w * lons) / np.sum(w)

    # station spread relative to centroid
    dists = haversine_km(lats, lons, centroid_lat, centroid_lon)
    station_spread_km = float(np.max(dists)) if len(dists) else np.nan
    mean_station_dist_km = float(np.mean(dists)) if len(dists) else np.nan

    row = {
        "candidate_id": candidate_id,
        "group_start_iso": group_start.isoformat(),
        "group_end_iso": group_end.isoformat(),
        "duration_s": (group_end - group_start).total_seconds(),
        "n_detections": n_detections,
        "n_stations": n_stations,
        "stations": ";".join(station_ids),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "station_spread_km": station_spread_km,
        "mean_station_dist_km": mean_station_dist_km,
        "mean_pmax": float(g["pmax"].mean()),
        "max_pmax": float(g["pmax"].max()),
    }
    rows.append(row)

assoc = pd.DataFrame(rows)

# sort by start time
if len(assoc) > 0:
    assoc["group_start_iso_dt"] = pd.to_datetime(assoc["group_start_iso"])
    assoc = assoc.sort_values("group_start_iso_dt").drop(columns="group_start_iso_dt").reset_index(drop=True)

# keep only detections belonging to retained groups
keep_ids = set(assoc["candidate_id"]) if len(assoc) > 0 else set()
df_keep = df[df["candidate_id"].isin(keep_ids)].copy()

# =========================
# OUTPUT
# =========================
OUT_DIR.mkdir(parents=True, exist_ok=True)

out_assoc_csv = OUT_DIR / "associated_catalog.csv"
out_members_csv = OUT_DIR / "associated_catalog_members.csv"

assoc.to_csv(out_assoc_csv, index=False)
df_keep.to_csv(out_members_csv, index=False)

print("\n======================")
print("DONE")
print(f"Input CSV              : {CSV_PATH}")
print(f"Associated catalog     : {out_assoc_csv}")
print(f"Associated members     : {out_members_csv}")
print(f"Retained candidates    : {len(assoc)}")
print(f"Retained detections    : {len(df_keep)}")
print("======================\n")

if len(assoc) > 0:
    print(assoc.head(10).to_string(index=False))
else:
    print("[WARN] No groups survived the filters.")