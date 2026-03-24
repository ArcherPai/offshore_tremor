#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import OrderedDict
import json
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from obspy import UTCDateTime, Stream, Trace
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope as obspy_envelope


# =========================
# USER SETTINGS
# =========================
ASSOC_CSV   = Path("/home/bxd240002/scratch/Archer/locating/data/csv/associated_catalog.csv")
MEMBERS_CSV = Path("/home/bxd240002/scratch/Archer/locating/data/csv/associated_catalog_members.csv")

OUT_ROOT = Path("/home/bxd240002/scratch/Archer/locating/outputs/location_work_batch_4th/")

FDSN_PROVIDER = "https://service.earthscope.org"

# waveform window around associated group
PAD_BEFORE_S = 120
PAD_AFTER_S  = 120

# channel priority for Z-only first-pass location
Z_PRIORITY = ["HHZ", "BHZ", "EHZ", "SHZ"]

# preprocessing
BANDPASS_HZ = (2.0, 8.0)
ENVELOPE_SMOOTH_S = 5.0
TARGET_FS = 1.0

# response removal
TRY_REMOVE_RESPONSE = True
PRE_FILT = (0.5, 1.0, 15.0, 20.0)

# plotting
SAVE_PLOTS = True
PLOT_TIMEZONE_LABEL = "UTC"
ASSOC_SHADE_COLOR = "0.88"   # light gray

# batch behavior
SKIP_IF_DONE = True
MIN_STATIONS_WARN = 3

# caches
INVENTORY_CACHE_SIZE = 512
WAVEFORM_HOURLY_CACHE_SIZE = 2048

# =========================
# LOCATION SETTINGS
# =========================
DO_LOCATION = True

# relative lag measurement on 1-Hz envelopes
XCORR_MAX_LAG_S = 60
XCORR_MIN_OVERLAP_S = 15

# constant velocity first-pass location
LOC_VEL_KM_S = 3.0

# grid search around station centroid / footprint
GRID_MARGIN_KM = 80.0
GRID_DX_KM = 5.0
GRID_DY_KM = 5.0
GRID_Z_MIN_KM = 5.0
GRID_Z_MAX_KM = 60.0
GRID_DZ_KM = 5.0


# =========================
# LRU CACHE
# =========================
class LRUCache:
    def __init__(self, maxsize=256):
        self.maxsize = int(maxsize)
        self._d = OrderedDict()

    def get(self, key):
        if key not in self._d:
            return None
        self._d.move_to_end(key)
        return self._d[key]

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


INVENTORY_CACHE = LRUCache(INVENTORY_CACHE_SIZE)
WAVEFORM_CACHE = LRUCache(WAVEFORM_HOURLY_CACHE_SIZE)


# =========================
# BASIC HELPERS
# =========================
def moving_average(x, n):
    if n <= 1:
        return x.copy()
    kernel = np.ones(int(n), dtype=float) / float(n)
    return np.convolve(x, kernel, mode="same")


def candidate_folder_name(candidate_id, group_start):
    ts = pd.Timestamp(group_start).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{candidate_id}"


def hour_floor(t):
    ts = pd.Timestamp(t.datetime).floor("h")
    return UTCDateTime(ts.to_pydatetime())


def hour_ceil(t):
    ts = pd.Timestamp(t.datetime)
    if ts == ts.floor("h"):
        return UTCDateTime(ts.to_pydatetime())
    ts = ts.ceil("h")
    return UTCDateTime(ts.to_pydatetime())


def build_hour_windows(t1, t2):
    start = hour_floor(t1)
    end = hour_ceil(t2)
    cur = start
    out = []
    while cur < end:
        nxt = cur + 3600
        out.append((cur, nxt))
        cur = nxt
    return out


def inventory_cache_key(net, sta):
    return (str(net), str(sta))


def waveform_cache_key(net, sta, cha, h1, h2):
    return (str(net), str(sta), str(cha), str(h1), str(h2))


# =========================
# GEO HELPERS
# =========================
def latlon_to_local_xy_km(lat, lon, lat0, lon0):
    """
    Simple local tangent-plane approximation.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)

    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))

    x = (lon - lon0) * km_per_deg_lon
    y = (lat - lat0) * km_per_deg_lat
    return x, y


def local_xy_km_to_latlon(x, y, lat0, lon0):
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(lat0))

    lat = lat0 + y / km_per_deg_lat
    lon = lon0 + x / km_per_deg_lon
    return lat, lon


# =========================
# INVENTORY / DOWNLOAD
# =========================
def choose_best_z_channel(inv, net, sta):
    chans = []
    for network in inv:
        if network.code != net:
            continue
        for station in network:
            if station.code != sta:
                continue
            for ch in station.channels:
                code = ch.code.upper()
                if code.endswith("Z"):
                    chans.append(code)

    for pref in Z_PRIORITY:
        if pref in chans:
            return pref

    if chans:
        return sorted(chans)[0]
    return None


def extract_station_coords(inv, net, sta, cha):
    """
    Prefer channel coordinates; fallback to station coordinates.
    """
    for network in inv:
        if network.code != net:
            continue
        for station in network:
            if station.code != sta:
                continue

            for ch in station.channels:
                if ch.code.upper() == cha.upper():
                    lat = getattr(ch, "latitude", None)
                    lon = getattr(ch, "longitude", None)
                    elev = getattr(ch, "elevation", None)
                    dep = getattr(ch, "depth", None)
                    if lat is not None and lon is not None:
                        return float(lat), float(lon), float(elev or 0.0), float(dep or 0.0), "channel"

            lat = getattr(station, "latitude", None)
            lon = getattr(station, "longitude", None)
            elev = getattr(station, "elevation", None)
            if lat is not None and lon is not None:
                return float(lat), float(lon), float(elev or 0.0), np.nan, "station"

    return np.nan, np.nan, np.nan, np.nan, "missing"


def get_inventory_cached(client, net, sta, t1, t2):
    key = inventory_cache_key(net, sta)
    inv = INVENTORY_CACHE.get(key)
    if inv is not None:
        return inv

    inv = client.get_stations(
        network=net,
        station=sta,
        location="*",
        channel="*Z",
        starttime=t1,
        endtime=t2,
        level="response"
    )
    INVENTORY_CACHE.put(key, inv)
    return inv


def fetch_hour_trace_cached(client, net, sta, cha, h1, h2):
    key = waveform_cache_key(net, sta, cha, h1, h2)
    cached = WAVEFORM_CACHE.get(key)
    if cached is not None:
        return cached.copy() if cached is not None else None

    try:
        st = client.get_waveforms(net, sta, "*", cha, h1, h2, attach_response=True)
        if len(st) == 0:
            WAVEFORM_CACHE.put(key, None)
            return None

        st.merge(method=1, fill_value="interpolate")
        st.sort()
        if len(st) == 0:
            WAVEFORM_CACHE.put(key, None)
            return None

        tr = st[0]
        WAVEFORM_CACHE.put(key, tr.copy())
        return tr.copy()
    except Exception as e:
        print(f"[WARN] hourly get_waveforms failed for {net}.{sta}.{cha} {h1} -> {h2}: {e}")
        WAVEFORM_CACHE.put(key, None)
        return None


def fetch_trace_cached(client, net, sta, cha, t1, t2):
    parts = []
    for h1, h2 in build_hour_windows(t1, t2):
        tr_hour = fetch_hour_trace_cached(client, net, sta, cha, h1, h2)
        if tr_hour is None:
            continue
        try:
            tr_cut = tr_hour.copy().slice(starttime=t1, endtime=t2, nearest_sample=False)
            if tr_cut.stats.npts > 1:
                parts.append(tr_cut)
        except Exception as e:
            print(f"[WARN] slice failed for {net}.{sta}.{cha}: {e}")

    if len(parts) == 0:
        return None

    st = Stream(parts)
    try:
        st.merge(method=1, fill_value="interpolate")
        st.detrend("demean")
        st.detrend("linear")
        st.sort()
        if len(st) == 0:
            return None
        return st[0]
    except Exception as e:
        print(f"[WARN] merge failed for {net}.{sta}.{cha}: {e}")
        return None


# =========================
# PREPROCESS
# =========================
def preprocess_to_env_1hz(tr):
    x = tr.copy()

    if TRY_REMOVE_RESPONSE:
        try:
            x.remove_response(output="VEL", pre_filt=PRE_FILT, water_level=60)
        except Exception as e:
            print(f"[WARN] remove_response failed for {x.id}: {e}")

    try:
        x.detrend("demean")
        x.detrend("linear")
        x.taper(max_percentage=0.02, type="cosine")
        x.filter(
            "bandpass",
            freqmin=BANDPASS_HZ[0],
            freqmax=BANDPASS_HZ[1],
            corners=4,
            zerophase=True
        )
    except Exception as e:
        print(f"[WARN] bandpass failed for {x.id}: {e}")
        return None

    data = x.data.astype(np.float32)
    env = obspy_envelope(data)

    fs = float(x.stats.sampling_rate)
    smooth_n = max(1, int(round(ENVELOPE_SMOOTH_S * fs)))
    env = moving_average(env, smooth_n)

    y = Trace(data=env.astype(np.float32), header=x.stats)
    y.detrend("demean")

    try:
        cur_fs = float(y.stats.sampling_rate)
        if abs(cur_fs - TARGET_FS) > 1e-6:
            ratio = cur_fs / TARGET_FS
            decim = int(round(ratio))

            if decim > 1 and abs(ratio - decim) < 0.05:
                y.data = y.data[::decim]
                y.stats.sampling_rate = cur_fs / decim
                y.stats.npts = len(y.data)
            else:
                t0 = y.stats.starttime
                t1 = y.stats.endtime
                n_new = int(np.floor((t1 - t0) * TARGET_FS)) + 1
                old_t = np.arange(y.stats.npts) / cur_fs
                new_t = np.arange(n_new) / TARGET_FS
                if len(old_t) < 2 or len(new_t) < 2:
                    return None
                new_data = np.interp(new_t, old_t, y.data).astype(np.float32)
                y.data = new_data
                y.stats.sampling_rate = TARGET_FS
                y.stats.npts = len(new_data)
    except Exception as e:
        print(f"[WARN] downsample/interp failed for {y.id}: {e}")
        return None

    mu = np.nanmean(y.data)
    sd = np.nanstd(y.data)
    y.data = ((y.data - mu) / (sd + 1e-9)).astype(np.float32)

    return y


def preprocess_to_bp_only(tr):
    x = tr.copy()

    if TRY_REMOVE_RESPONSE:
        try:
            x.remove_response(output="VEL", pre_filt=PRE_FILT, water_level=60)
        except Exception as e:
            print(f"[WARN] remove_response failed for {x.id}: {e}")

    try:
        x.detrend("demean")
        x.detrend("linear")
        x.taper(max_percentage=0.02, type="cosine")
        x.filter(
            "bandpass",
            freqmin=BANDPASS_HZ[0],
            freqmax=BANDPASS_HZ[1],
            corners=4,
            zerophase=True
        )
    except Exception as e:
        print(f"[WARN] bandpass-only failed for {x.id}: {e}")
        return None

    return x


# =========================
# PLOTTING
# =========================
def _setup_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M:%S"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def plot_envelopes_preview(env_traces, group_start, group_end, out_png):
    if len(env_traces) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    keys = sorted(env_traces.keys())
    offset = 0.0

    for k in keys:
        tr = env_traces[k]
        fs = float(tr.stats.sampling_rate)
        t0 = pd.Timestamp(tr.stats.starttime.datetime)
        tt = t0 + pd.to_timedelta(np.arange(tr.stats.npts) / fs, unit="s")
        ax.plot(tt, tr.data + offset, lw=0.8, label=k)
        offset += 4.0

    ax.axvspan(group_start, group_end, color=ASSOC_SHADE_COLOR, alpha=1.0, label="associated window")
    ax.set_xlabel(f"Time ({PLOT_TIMEZONE_LABEL})")
    ax.set_ylabel("Envelope amplitude (offset)")
    ax.set_title(
        "Preprocessed 1-Hz envelopes\n"
        f"Window: {group_start} to {group_end} ({PLOT_TIMEZONE_LABEL})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    _setup_time_axis(ax)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bandpassed_waveforms_preview(bp_traces, group_start, group_end, out_png, normalize=True):
    if len(bp_traces) == 0:
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    keys = sorted(bp_traces.keys())
    offset = 0.0

    for k in keys:
        tr = bp_traces[k]
        fs = float(tr.stats.sampling_rate)
        t0 = pd.Timestamp(tr.stats.starttime.datetime)
        tt = t0 + pd.to_timedelta(np.arange(tr.stats.npts) / fs, unit="s")

        y = tr.data.astype(np.float32)
        if normalize:
            amp = np.nanmax(np.abs(y))
            if np.isfinite(amp) and amp > 0:
                y = y / amp

        ax.plot(tt, y + offset, lw=0.6, label=k)
        offset += 3.0

    ax.axvspan(group_start, group_end, color=ASSOC_SHADE_COLOR, alpha=1.0, label="associated window")
    ax.set_xlabel(f"Time ({PLOT_TIMEZONE_LABEL})")
    ax.set_ylabel("Bandpassed waveform (offset)")
    ax.set_title(
        "2–8 Hz bandpassed waveforms\n"
        f"Window: {group_start} to {group_end} ({PLOT_TIMEZONE_LABEL})"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    _setup_time_axis(ax)

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def plot_bandpassed_waveforms_preview_bw(bp_traces, group_start, group_end, out_png, normalize=True):
    """
    Black-white version:
    - black traces
    - station names on the far right
    - right-side y-axis style labels
    - associated window in light gray
    """
    if len(bp_traces) == 0:
        return

    fig, ax = plt.subplots(figsize=(15, 8))
    keys = sorted(bp_traces.keys())
    offsets = []
    x_right = None

    for i, k in enumerate(keys):
        tr = bp_traces[k]
        fs = float(tr.stats.sampling_rate)
        t0 = pd.Timestamp(tr.stats.starttime.datetime)
        tt = t0 + pd.to_timedelta(np.arange(tr.stats.npts) / fs, unit="s")

        y = tr.data.astype(np.float32)
        if normalize:
            amp = np.nanmax(np.abs(y))
            if np.isfinite(amp) and amp > 0:
                y = y / amp

        offset = i * 2.5
        offsets.append(offset)
        ax.plot(tt, y + offset, lw=0.55, color="black")

        if x_right is None or tt[-1] > x_right:
            x_right = tt[-1]

    ax.axvspan(group_start, group_end, color=ASSOC_SHADE_COLOR, alpha=1.0)

    # right-side labels
    x_text = x_right + pd.Timedelta(seconds=8)
    for k, off in zip(keys, offsets):
        ax.text(
            x_text, off, k,
            va="center", ha="left",
            fontsize=8, color="black"
        )

    # hide left y-axis ticks
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)

    # keep right spine visible to mimic right y-axis
    ax.spines["right"].set_visible(True)
    ax.yaxis.set_ticks_position("right")

    # expand xlim so right labels fit
    xmin = min(pd.Timestamp(bp_traces[k].stats.starttime.datetime) for k in keys)
    xmax = x_right + pd.Timedelta(seconds=35)
    ax.set_xlim(xmin, xmax)

    ax.set_xlabel(f"Time ({PLOT_TIMEZONE_LABEL})")
    ax.set_ylabel("")
    ax.set_title(
        "2–8 Hz bandpassed waveforms\n"
        f"Window: {group_start} to {group_end} ({PLOT_TIMEZONE_LABEL})"
    )
    ax.grid(alpha=0.2)
    _setup_time_axis(ax)

    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# LOCATION HELPERS
# =========================
def build_common_env_dataframe(env_traces):
    """
    Align all 1-Hz envelope traces onto a common UTC second grid.
    """
    if len(env_traces) == 0:
        return None

    starts = [pd.Timestamp(tr.stats.starttime.datetime) for tr in env_traces.values()]
    ends = []
    for tr in env_traces.values():
        fs = float(tr.stats.sampling_rate)
        ends.append(pd.Timestamp(tr.stats.starttime.datetime) + pd.to_timedelta((tr.stats.npts - 1) / fs, unit="s"))

    tmin = max(starts)
    tmax = min(ends)
    if tmax <= tmin:
        return None

    index = pd.date_range(tmin, tmax, freq="1S")
    df = pd.DataFrame(index=index)

    for sta, tr in env_traces.items():
        fs = float(tr.stats.sampling_rate)
        tt = pd.Timestamp(tr.stats.starttime.datetime) + pd.to_timedelta(np.arange(tr.stats.npts) / fs, unit="s")
        s = pd.Series(tr.data.astype(float), index=pd.DatetimeIndex(tt))
        s = s[~s.index.duplicated(keep="first")]
        s = s.reindex(index)
        df[sta] = s.values

    return df


def normalized_xcorr_lag(x, y, max_lag_s):
    """
    x, y on same 1-Hz grid. Return lag in seconds where y lags x.
    Positive lag means y arrives later than x.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < max(10, XCORR_MIN_OVERLAP_S):
        return np.nan, np.nan

    x = x - np.mean(x)
    y = y - np.mean(y)

    sx = np.std(x)
    sy = np.std(y)
    if sx <= 0 or sy <= 0:
        return np.nan, np.nan

    x = x / sx
    y = y / sy

    lags = np.arange(-max_lag_s, max_lag_s + 1)
    vals = np.full(len(lags), np.nan, dtype=float)

    for i, lag in enumerate(lags):
        if lag < 0:
            xx = x[-lag:]
            yy = y[:len(y) + lag]
        elif lag > 0:
            xx = x[:-lag]
            yy = y[lag:]
        else:
            xx = x
            yy = y

        if len(xx) < XCORR_MIN_OVERLAP_S:
            continue

        vals[i] = np.mean(xx * yy)

    if not np.any(np.isfinite(vals)):
        return np.nan, np.nan

    imax = np.nanargmax(vals)
    best_lag = lags[imax]
    best_cc = vals[imax]
    return float(best_lag), float(best_cc)


def estimate_relative_lags(env_df, group_start, group_end, station_meta):
    """
    Use associated window only for lag estimation.
    Select reference station by highest pmax among available traces.
    """
    if env_df is None or env_df.shape[1] < 2:
        return None, None

    win = env_df.loc[group_start:group_end].copy()
    cols = [c for c in win.columns if win[c].notna().sum() >= XCORR_MIN_OVERLAP_S]
    if len(cols) < 2:
        return None, None

    win = win[cols]

    meta = station_meta.set_index("station_id").copy()
    avail_meta = meta.loc[meta.index.intersection(cols)].copy()
    if len(avail_meta) < 2:
        return None, None

    if "pmax" in avail_meta.columns and avail_meta["pmax"].notna().any():
        ref_station = avail_meta["pmax"].astype(float).sort_values(ascending=False).index[0]
    else:
        ref_station = cols[0]

    rows = []
    xref = win[ref_station].values

    for sta in cols:
        lag_s, cc = normalized_xcorr_lag(xref, win[sta].values, XCORR_MAX_LAG_S)
        rows.append({
            "station_id": sta,
            "ref_station": ref_station,
            "lag_to_ref_s": lag_s,
            "xcorr_coeff": cc,
            "used_for_location": bool(np.isfinite(lag_s))
        })

    lag_df = pd.DataFrame(rows)
    lag_df.loc[lag_df["station_id"] == ref_station, "lag_to_ref_s"] = 0.0
    lag_df.loc[lag_df["station_id"] == ref_station, "xcorr_coeff"] = 1.0
    lag_df["used_for_location"] = lag_df["lag_to_ref_s"].notna()

    return ref_station, lag_df


def run_constant_velocity_location(station_meta, lag_df, vel_km_s=3.0):
    """
    Fit observed relative lags to predicted relative travel times
    using a 3D grid search.
    """
    if lag_df is None or len(lag_df) < 3:
        return None, None

    use = lag_df[lag_df["used_for_location"]].copy()
    if len(use) < 3:
        return None, None

    meta = station_meta.set_index("station_id").copy()
    use = use.join(meta, on="station_id", how="left")

    use = use[
        use["station_lat"].notna() &
        use["station_lon"].notna() &
        use["lag_to_ref_s"].notna()
    ].copy()

    if len(use) < 3:
        return None, None

    ref_station = use["ref_station"].iloc[0]
    if ref_station not in use["station_id"].values:
        return None, None

    lat0 = float(use["station_lat"].mean())
    lon0 = float(use["station_lon"].mean())

    x_sta, y_sta = latlon_to_local_xy_km(
        use["station_lat"].values,
        use["station_lon"].values,
        lat0, lon0
    )
    z_sta = np.zeros(len(use), dtype=float)

    # OBS elevation could be used, but for first-pass keep stations at z=0 reference
    # because channel depth/elevation conventions can vary in sign and confuse quick search

    obs_lag = use["lag_to_ref_s"].values.astype(float)
    sta_ids = use["station_id"].values.astype(str)
    cc = use["xcorr_coeff"].fillna(0.0).values.astype(float)

    # weights from xcorr
    w = np.clip(cc, 0.05, 1.0)

    ref_idx = np.where(sta_ids == ref_station)[0]
    if len(ref_idx) != 1:
        return None, None
    ref_idx = ref_idx[0]

    xmin = np.nanmin(x_sta) - GRID_MARGIN_KM
    xmax = np.nanmax(x_sta) + GRID_MARGIN_KM
    ymin = np.nanmin(y_sta) - GRID_MARGIN_KM
    ymax = np.nanmax(y_sta) + GRID_MARGIN_KM

    xs = np.arange(xmin, xmax + 0.1, GRID_DX_KM)
    ys = np.arange(ymin, ymax + 0.1, GRID_DY_KM)
    zs = np.arange(GRID_Z_MIN_KM, GRID_Z_MAX_KM + 0.1, GRID_DZ_KM)

    rows = []
    best = None

    for x in xs:
        for y in ys:
            for z in zs:
                dist = np.sqrt((x_sta - x)**2 + (y_sta - y)**2 + (z_sta - z)**2)
                tpred = dist / vel_km_s
                pred_lag = tpred - tpred[ref_idx]

                resid = obs_lag - pred_lag
                rms = np.sqrt(np.average(resid**2, weights=w))
                mad = np.average(np.abs(resid), weights=w)

                row = {
                    "x_km": float(x),
                    "y_km": float(y),
                    "z_km": float(z),
                    "rms_s": float(rms),
                    "mad_s": float(mad)
                }
                rows.append(row)

                if (best is None) or (rms < best["rms_s"]):
                    best = dict(row)
                    best["pred_lag_s"] = pred_lag.copy()
                    best["residual_s"] = resid.copy()
                    best["ref_station"] = ref_station
                    best["station_ids"] = sta_ids.copy()
                    best["obs_lag_s"] = obs_lag.copy()
                    best["weights"] = w.copy()

    grid_df = pd.DataFrame(rows).sort_values("rms_s").reset_index(drop=True)

    best_lat, best_lon = local_xy_km_to_latlon(best["x_km"], best["y_km"], lat0, lon0)

    location_result = {
        "ref_station": best["ref_station"],
        "velocity_km_s": float(vel_km_s),
        "origin_model": "relative-lag constant-velocity grid search",
        "grid_dx_km": float(GRID_DX_KM),
        "grid_dy_km": float(GRID_DY_KM),
        "grid_dz_km": float(GRID_DZ_KM),
        "best_x_km": float(best["x_km"]),
        "best_y_km": float(best["y_km"]),
        "best_z_km": float(best["z_km"]),
        "best_lat": float(best_lat),
        "best_lon": float(best_lon),
        "rms_s": float(best["rms_s"]),
        "mad_s": float(best["mad_s"]),
        "n_stations_used": int(len(use)),
        "local_projection_center_lat": float(lat0),
        "local_projection_center_lon": float(lon0)
    }

    fit_df = pd.DataFrame({
        "station_id": best["station_ids"],
        "obs_lag_s": best["obs_lag_s"],
        "pred_lag_s": best["pred_lag_s"],
        "residual_s": best["residual_s"],
        "weight": best["weights"]
    })

    return location_result, grid_df, fit_df


def plot_location_map(station_meta, location_result, out_png):
    use = station_meta[
        station_meta["station_lat"].notna() &
        station_meta["station_lon"].notna()
    ].copy()

    if len(use) == 0 or location_result is None:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(use["station_lon"], use["station_lat"], marker="^", s=60, label="Stations")
    for _, r in use.iterrows():
        ax.text(r["station_lon"] + 0.01, r["station_lat"], r["station_id"], fontsize=8)

    ax.scatter(location_result["best_lon"], location_result["best_lat"], marker="*", s=180, label="Best location")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Initial location map\n"
        f"lat={location_result['best_lat']:.4f}, lon={location_result['best_lon']:.4f}, "
        f"z={location_result['best_z_km']:.1f} km, rms={location_result['rms_s']:.2f} s"
    )
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_lag_fit(fit_df, out_png):
    if fit_df is None or len(fit_df) == 0:
        return

    fit_df = fit_df.sort_values("obs_lag_s").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fit_df["obs_lag_s"], fit_df["pred_lag_s"])
    lo = min(fit_df["obs_lag_s"].min(), fit_df["pred_lag_s"].min()) - 1
    hi = max(fit_df["obs_lag_s"].max(), fit_df["pred_lag_s"].max()) + 1
    ax.plot([lo, hi], [lo, hi], lw=1)

    for _, r in fit_df.iterrows():
        ax.text(r["obs_lag_s"], r["pred_lag_s"], r["station_id"], fontsize=8)

    ax.set_xlabel("Observed lag to reference (s)")
    ax.set_ylabel("Predicted lag to reference (s)")
    ax.set_title("Lag fit for initial location")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


# =========================
# DONE CHECK
# =========================
def is_candidate_done(cand_dir):
    needed = [
        cand_dir / "station_metadata.csv",
        cand_dir / "candidate_meta.json",
    ]
    return all(p.exists() for p in needed)


# =========================
# CORE PER-CANDIDATE
# =========================
def process_one_candidate(client, assoc_row, mem_sub, out_root):
    candidate_id = str(assoc_row["candidate_id"])
    group_start = pd.to_datetime(assoc_row["group_start_iso"])
    group_end   = pd.to_datetime(assoc_row["group_end_iso"])

    folder_name = candidate_folder_name(candidate_id, group_start)
    cand_dir = out_root / folder_name
    cand_dir.mkdir(parents=True, exist_ok=True)

    if SKIP_IF_DONE and is_candidate_done(cand_dir):
        print(f"[SKIP] {folder_name} already done.")
        return {
            "candidate_id": candidate_id,
            "folder_name": folder_name,
            "status": "skipped",
            "group_start_iso": str(group_start),
            "group_end_iso": str(group_end),
            "n_member_rows": int(len(mem_sub)),
            "n_unique_stations": np.nan,
            "n_usable_stations": np.nan,
            "location_status": "",
            "error": ""
        }

    t1 = UTCDateTime((group_start - pd.Timedelta(seconds=PAD_BEFORE_S)).to_pydatetime())
    t2 = UTCDateTime((group_end   + pd.Timedelta(seconds=PAD_AFTER_S)).to_pydatetime())

    print("\n" + "=" * 90)
    print(f"[INFO] Candidate      : {candidate_id}")
    print(f"[INFO] Folder         : {folder_name}")
    print(f"[INFO] Group start    : {group_start}")
    print(f"[INFO] Group end      : {group_end}")
    print(f"[INFO] Download window: {t1} -> {t2}")
    print(f"[INFO] Member rows    : {len(mem_sub)}")

    mem = mem_sub.copy()
    mem["station_id"] = mem["network"].astype(str) + "." + mem["station"].astype(str)

    sta_df = (
        mem.groupby("station_id", as_index=False)
           .agg(
               network=("network", "first"),
               station=("station", "first"),
               pmax=("pmax", "max")
           )
    )

    print(f"[INFO] Unique stations: {len(sta_df)}")

    env_traces = {}
    bp_traces = {}
    station_rows = []

    for _, r in sta_df.iterrows():
        net = str(r["network"])
        sta = str(r["station"])
        station_id = f"{net}.{sta}"

        print(f"\n[INFO] Query inventory for {station_id}")
        try:
            inv = get_inventory_cached(client, net, sta, t1, t2)
        except Exception as e:
            print(f"[WARN] get_stations failed for {station_id}: {e}")
            continue

        cha = choose_best_z_channel(inv, net, sta)
        if cha is None:
            print(f"[WARN] No Z channel found for {station_id}")
            continue

        lat, lon, elev_m, depth_m, coord_source = extract_station_coords(inv, net, sta, cha)

        print(f"[INFO] Selected channel: {cha}")
        print(f"[INFO] Station coords  : lat={lat}, lon={lon}, elev_m={elev_m}, depth_m={depth_m}, source={coord_source}")

        tr = fetch_trace_cached(client, net, sta, cha, t1, t2)
        if tr is None:
            continue

        bp = preprocess_to_bp_only(tr)
        if bp is not None:
            bp_traces[station_id] = bp

        env = preprocess_to_env_1hz(tr)
        if env is None:
            continue

        env_traces[station_id] = env

        station_rows.append({
            "station_id": station_id,
            "network": net,
            "station": sta,
            "channel": cha,
            "station_lat": lat,
            "station_lon": lon,
            "station_elevation_m": elev_m,
            "channel_depth_m": depth_m,
            "coord_source": coord_source,
            "pmax": float(r["pmax"]) if pd.notna(r["pmax"]) else np.nan,
            "starttime": str(env.stats.starttime),
            "endtime": str(env.stats.endtime),
            "sampling_rate_hz": float(env.stats.sampling_rate),
            "npts": int(env.stats.npts)
        })

    if len(env_traces) < MIN_STATIONS_WARN:
        print(f"[WARN] Only {len(env_traces)} usable stations. ACC/location may be weak.")

    station_meta = pd.DataFrame(station_rows)
    station_meta_csv = cand_dir / "station_metadata.csv"
    station_meta.to_csv(station_meta_csv, index=False)

    # candidate meta
    candidate_meta = {
        "candidate_id": candidate_id,
        "folder_name": folder_name,
        "group_start_iso": str(group_start),
        "group_end_iso": str(group_end),
        "download_start_utc": str(t1),
        "download_end_utc": str(t2),
        "n_member_rows": int(len(mem)),
        "n_unique_stations": int(len(sta_df)),
        "n_usable_stations": int(len(station_meta)),
        "bandpass_hz": BANDPASS_HZ,
        "envelope_smooth_s": ENVELOPE_SMOOTH_S,
        "target_fs_hz": TARGET_FS,
        "try_remove_response": TRY_REMOVE_RESPONSE,
        "pre_filt": PRE_FILT,
        "inventory_cache_size": INVENTORY_CACHE_SIZE,
        "waveform_hourly_cache_size": WAVEFORM_HOURLY_CACHE_SIZE
    }
    with open(cand_dir / "candidate_meta.json", "w") as f:
        json.dump(candidate_meta, f, indent=2)

    # plots
    if SAVE_PLOTS and len(env_traces) > 0:
        plot_envelopes_preview(
            env_traces=env_traces,
            group_start=group_start,
            group_end=group_end,
            out_png=cand_dir / "envelopes_preview.png"
        )

    if SAVE_PLOTS and len(bp_traces) > 0:
        plot_bandpassed_waveforms_preview(
            bp_traces=bp_traces,
            group_start=group_start,
            group_end=group_end,
            out_png=cand_dir / "bandpassed_waveforms_preview.png",
            normalize=True
        )
        plot_bandpassed_waveforms_preview_bw(
            bp_traces=bp_traces,
            group_start=group_start,
            group_end=group_end,
            out_png=cand_dir / "bandpassed_waveforms_preview_bw.png",
            normalize=True
        )

    # location
    location_status = "not_run"
    if DO_LOCATION and len(env_traces) >= 3 and len(station_meta) >= 3:
        try:
            env_df = build_common_env_dataframe(env_traces)
            ref_station, lag_df = estimate_relative_lags(env_df, group_start, group_end, station_meta)

            if lag_df is not None:
                lag_df.to_csv(cand_dir / "xcorr_lags.csv", index=False)

            if lag_df is not None and lag_df["used_for_location"].sum() >= 3:
                loc_result, grid_df, fit_df = run_constant_velocity_location(
                    station_meta=station_meta,
                    lag_df=lag_df,
                    vel_km_s=LOC_VEL_KM_S
                )

                if loc_result is not None:
                    with open(cand_dir / "initial_location.json", "w") as f:
                        json.dump(loc_result, f, indent=2)

                    grid_df.head(200).to_csv(cand_dir / "location_grid_search_summary.csv", index=False)
                    fit_df.to_csv(cand_dir / "location_lag_fit.csv", index=False)

                    plot_location_map(station_meta, loc_result, cand_dir / "location_map.png")
                    plot_lag_fit(fit_df, cand_dir / "location_lagfit.png")

                    location_status = "ok"
                else:
                    location_status = "failed_grid"
            else:
                location_status = "insufficient_lags"
        except Exception as e:
            print(f"[WARN] location failed for {candidate_id}: {e}")
            location_status = f"error:{type(e).__name__}"

    return {
        "candidate_id": candidate_id,
        "folder_name": folder_name,
        "status": "ok",
        "group_start_iso": str(group_start),
        "group_end_iso": str(group_end),
        "n_member_rows": int(len(mem)),
        "n_unique_stations": int(len(sta_df)),
        "n_usable_stations": int(len(station_meta)),
        "location_status": location_status,
        "error": ""
    }


# =========================
# MAIN
# =========================
def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    assoc = pd.read_csv(ASSOC_CSV)
    members = pd.read_csv(MEMBERS_CSV)

    if "candidate_id" not in assoc.columns:
        raise RuntimeError("associated_catalog.csv missing column: candidate_id")
    if "candidate_id" not in members.columns:
        raise RuntimeError("associated_catalog_members.csv missing column: candidate_id")

    assoc = assoc.copy()
    assoc["group_start_iso"] = pd.to_datetime(assoc["group_start_iso"])
    assoc["group_end_iso"] = pd.to_datetime(assoc["group_end_iso"])
    assoc = assoc.sort_values("group_start_iso").reset_index(drop=True)

    client = Client(FDSN_PROVIDER)

    summaries = []
    total = len(assoc)

    print(f"[INFO] Total candidates in assoc CSV: {total}")
    print(f"[INFO] Output root: {OUT_ROOT}")

    for i, (_, row) in enumerate(assoc.iterrows(), start=1):
        candidate_id = str(row["candidate_id"])
        print("\n" + "#" * 100)
        print(f"[INFO] Processing {i}/{total}: {candidate_id}")

        mem_sub = members[members["candidate_id"] == candidate_id].copy()
        if len(mem_sub) == 0:
            folder_name = candidate_folder_name(candidate_id, row["group_start_iso"])
            summaries.append({
                "candidate_id": candidate_id,
                "folder_name": folder_name,
                "status": "no_members",
                "group_start_iso": str(row["group_start_iso"]),
                "group_end_iso": str(row["group_end_iso"]),
                "n_member_rows": 0,
                "n_unique_stations": np.nan,
                "n_usable_stations": np.nan,
                "location_status": "not_run",
                "error": "No member rows found"
            })
            continue

        try:
            result = process_one_candidate(
                client=client,
                assoc_row=row,
                mem_sub=mem_sub,
                out_root=OUT_ROOT
            )
            summaries.append(result)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            folder_name = candidate_folder_name(candidate_id, row["group_start_iso"])
            print(f"[ERROR] Failed for {folder_name}: {err}")
            print(traceback.format_exc())

            cand_dir = OUT_ROOT / folder_name
            cand_dir.mkdir(parents=True, exist_ok=True)
            with open(cand_dir / "FAILED.txt", "w") as f:
                f.write(err + "\n\n")
                f.write(traceback.format_exc())

            summaries.append({
                "candidate_id": candidate_id,
                "folder_name": folder_name,
                "status": "failed",
                "group_start_iso": str(row["group_start_iso"]),
                "group_end_iso": str(row["group_end_iso"]),
                "n_member_rows": int(len(mem_sub)),
                "n_unique_stations": np.nan,
                "n_usable_stations": np.nan,
                "location_status": "not_run",
                "error": err
            })

    summary_df = pd.DataFrame(summaries)
    summary_csv = OUT_ROOT / "batch_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print("\n" + "=" * 100)
    print("[DONE] Batch processing finished.")
    print(f"[DONE] Summary saved to: {summary_csv}")
    print(summary_df["status"].value_counts(dropna=False))
    if "location_status" in summary_df.columns:
        print(summary_df["location_status"].value_counts(dropna=False))
    print(f"[INFO] Inventory cache size used: {len(INVENTORY_CACHE)}")
    print(f"[INFO] Waveform cache size used : {len(WAVEFORM_CACHE)}")


if __name__ == "__main__":
    main()