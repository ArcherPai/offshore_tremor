#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Set, Dict, List
import os
import time
import math
import hashlib
import numpy as np
import pandas as pd
import h5py
from obspy import UTCDateTime, Trace, Stream, read
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope
import argparse
import calendar
from tempfile import NamedTemporaryFile


#------------------------
# Config
#------------------------

CHANNELS_CSV = "channels_epoch_2009_2025Sep.csv"
EVENTS_CSV = "tremor_events-2009-08-06T00_00_00-2025-09-10T23_59_59.csv"

MASTER_YEAR = 2020
OUT_DIR = f"./tremor_{MASTER_YEAR:04d}"
CACHE_DIR = "./waveform_cache"

STORAGE_MODE = "both"
SKIP_PER_EVENT_IF_EXISTS = True

MASTER_H5_PATH = f"{OUT_DIR}/tremor_raw_master_{MASTER_YEAR}.hdf5"
MASTER_LOCKFILE = f"{OUT_DIR}/tremor_raw_master_{MASTER_YEAR}.lock"
MASTER_CSV_PATH = f"{OUT_DIR}/tremor_channels_master_{MASTER_YEAR}.csv"
MASTER_CSV_LOCK = f"{OUT_DIR}/tremor_channels_master_{MASTER_YEAR}.csv.lock"

CHANNEL_CSV_COLUMNS = [
    "event_uid", "event_start_iso", "event_lat", "event_lon",
    "t_abs0_iso", "t_abs1_iso", "window_s",
    "storage_mode", "per_event_h5", "per_event_dir",
    "network", "station", "band",
    "chE", "chN", "chZ",
    "latitude", "longitude", "dist_km_from_event",
    "sampling_rate_hz", "slice_start_iso", "slice_end_iso", "fill_value",
    "hit_count", "top_hit_partner", "top_hit_corr", "top_hit_lag_s",
    "hit_partners",
]

EVENT_TIME_RANGE: Optional[Tuple[str, str]] = ("2020-09-18T00:00:00", "2020-09-19T00:00:00")
MAX_EVENTS: Optional[int] = None

EVENT_WIN_S = 300.0
PAD_BEFORE = 150.0
PAD_AFTER = 150.0
FILL_VALUE = np.nan
OVERLAP_MAX_FRAC = 0.0

MAX_DIST_KM = 100.0
MAX_STATIONS = None
BAND_PRIORITY = ["HH", "BH", "LH", "EH", "EN", "HN", "BN", "SN"]

BANDPASS_HZ = (1.0, 8.0)
ENVELOPE_LP_HZ = 0.1
DS_RATE_HZ = 1.0

XCORR_MAX_LAG_S = 60.0
HIT_THRESHOLD = 0.7

VMIN_KMPS = 3.0
LAG_CAP_MAX_S = 10.0

FDSN_CLIENTS = ["IRIS"]

SAVE_ONLY_HIT_STATIONS = True
PAIRS_CSV_MODE = "hits_only"

TWO_STAGE_FETCH = True
XCORR_COMPONENT = "Z"
KEEP_ONLY_HIT_STATIONS = SAVE_ONLY_HIT_STATIONS


#------------------------
# Argument Parsing
#------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Tremor cross-correlation downloader (raw-only)")
    ap.add_argument("--year", type=int, help="Target year, e.g., 2018")
    ap.add_argument("--month", type=int, choices=range(1, 13), help="Target month 1..12 (use with --year)")
    ap.add_argument("--start", type=str, help="ISO start (UTC), e.g., 2018-06-01T00:00:00")
    ap.add_argument("--end", type=str, help="ISO end (UTC), e.g., 2018-07-01T00:00:00")
    ap.add_argument("--storage-mode", choices=["per_event", "master", "both"], help="Override STORAGE_MODE")
    ap.add_argument("--outdir", type=str, help="Override OUT_DIR")
    ap.add_argument("--cache-dir", type=str, help="Override CACHE_DIR")
    ap.add_argument("--master-year", type=int, help="Override MASTER_YEAR (default = --year if set)")
    return ap.parse_args()


def configure_from_args(args):
    global EVENT_TIME_RANGE, MASTER_YEAR, MASTER_H5_PATH, MASTER_LOCKFILE, MASTER_CSV_PATH, MASTER_CSV_LOCK
    global STORAGE_MODE, OUT_DIR, CACHE_DIR

    if args.storage_mode:
        STORAGE_MODE = args.storage_mode
    if args.cache_dir:
        CACHE_DIR = args.cache_dir

    if args.start and args.end:
        EVENT_TIME_RANGE = (args.start, args.end)
    elif args.year and args.month:
        y, m = args.year, args.month
        start = f"{y:04d}-{m:02d}-01T00:00:00"
        end_m = m + 1 if m < 12 else 1
        end_y = y + 1 if m == 12 else y
        end = f"{end_y:04d}-{end_m:02d}-01T00:00:00"
        EVENT_TIME_RANGE = (start, end)
    elif args.year and not args.month:
        start = f"{args.year:04d}-01-01T00:00:00"
        end = f"{args.year + 1:04d}-01-01T00:00:00"
        EVENT_TIME_RANGE = (start, end)

    if args.master_year:
        MASTER_YEAR = args.master_year
    elif args.year:
        MASTER_YEAR = args.year

    if args.outdir:
        OUT_DIR = args.outdir
    else:
        OUT_DIR = f"./tremor_{MASTER_YEAR:04d}"

    MASTER_H5_PATH = f"{OUT_DIR}/tremor_raw_master_{MASTER_YEAR}.hdf5"
    MASTER_LOCKFILE = f"{OUT_DIR}/tremor_raw_master_{MASTER_YEAR}.lock"
    MASTER_CSV_PATH = f"{OUT_DIR}/tremor_channels_master_{MASTER_YEAR}.csv"
    MASTER_CSV_LOCK = f"{OUT_DIR}/tremor_channels_master_{MASTER_YEAR}.csv.lock"


#------------------------
# Time and Geometry Helpers
#------------------------

def to_utc(s: str) -> datetime:
    s = str(s).strip()
    if s.startswith("T"):
        s = s[1:]
    if s.endswith("Z"):
        s = s[:-1]
    s = s.replace(" ", "T")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        dt = pd.to_datetime(s, utc=True, errors="coerce")
        if pd.isna(dt):
            return datetime.utcnow().replace(tzinfo=timezone.utc)
        dt = dt.to_pydatetime()
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.asin(math.sqrt(a))


def event_uid(t0_iso: str, lat: float, lon: float, dur: float) -> str:
    key = f"{t0_iso}|{lat:.4f}|{lon:.4f}|{dur:.1f}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()[:10]


def strip_part_suffix(p: Path) -> Path:
    s = str(p)
    return Path(s[:-5]) if s.endswith(".part") else Path(s)


def dedup_event_windows(ev_df: pd.DataFrame, event_win_s: float, overlap_max_frac: float = 0.0) -> pd.DataFrame:
    if ev_df.empty:
        return ev_df
    ev = ev_df.sort_values("t0").copy()
    kept_idx: List[int] = []
    for idx, row in ev.iterrows():
        if not kept_idx:
            kept_idx.append(idx)
            continue
        last = ev.loc[kept_idx[-1]]
        min_start = last["t0"] + timedelta(seconds=(1.0 - overlap_max_frac) * event_win_s)
        if row["t0"] >= min_start:
            kept_idx.append(idx)
    return ev.loc[kept_idx].reset_index(drop=True)


#------------------------
# Station Selection
#------------------------

def pick_stations_3c(ch_df: pd.DataFrame, ev_lat: float, ev_lon: float) -> pd.DataFrame:
    df = ch_df.dropna(subset=["latitude", "longitude", "channel"]).copy()
    df["dist_km"] = np.vectorize(haversine_km)(ev_lat, ev_lon, df["latitude"].values, df["longitude"].values)
    df = df[df["dist_km"] <= MAX_DIST_KM].copy()
    df["band"] = df["channel"].astype(str).str[:2].str.upper()
    df["comp"] = df["channel"].astype(str).str[-1].str.upper()

    rows = []
    for (net, sta), g in df.groupby(["network", "station"]):
        bands: Dict[str, Dict[str, Optional[str]]] = {}
        lat0 = float(g["latitude"].iloc[0])
        lon0 = float(g["longitude"].iloc[0])
        dist0 = float(g["dist_km"].min())

        for _, r in g.iterrows():
            b = str(r["band"]).upper()
            c = str(r["comp"]).upper()
            if c not in ("E", "N", "Z"):
                continue
            d = bands.setdefault(b, {"E": None, "N": None, "Z": None})
            if d[c] is None:
                d[c] = str(r["channel"])

        chosen = None
        for b in BAND_PRIORITY:
            if b in bands and all(bands[b][c] is not None for c in ("E", "N", "Z")):
                chosen = (b, bands[b])
                break
        if chosen is None:
            continue

        b, info = chosen
        rows.append({
            "network": net,
            "station": sta,
            "latitude": lat0,
            "longitude": lon0,
            "chE": info["E"],
            "chN": info["N"],
            "chZ": info["Z"],
            "dist_km": dist0,
            "band": b,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("dist_km").head(MAX_STATIONS).reset_index(drop=True)


#------------------------
# Cache Helpers
#------------------------

def _hours_between(t1: UTCDateTime, t2: UTCDateTime) -> List[str]:
    h0 = UTCDateTime(year=t1.year, month=t1.month, day=t1.day, hour=t1.hour)
    hours = []
    h = h0
    while h < t2:
        hours.append(f"{h.year:04d}-{h.month:02d}-{h.day:02d}T{h.hour:02d}")
        h += 3600
    return hours


def _cache_hour_path(net: str, sta: str, chan: str, hour_utc: str) -> Path:
    return Path(CACHE_DIR) / "raw" / f"{net}.{sta}" / chan / f"{hour_utc}.mseed"


#------------------------
# Waveform Download
#------------------------

def fdsn_fetch(net: str, sta: str, ch: str, t1: UTCDateTime, t2: UTCDateTime) -> Optional[Trace]:
    for prov in FDSN_CLIENTS:
        try:
            cli = Client(prov)
            st = cli.get_waveforms(net, sta, "*", ch, t1, t2, attach_response=False)
            if len(st) == 0:
                continue
            st.sort()
            st.detrend("demean")
            st.merge(method=1, fill_value="interpolate")
            tr = st[0]
            tr.stats.network = net
            tr.stats.station = sta
            return tr
        except Exception:
            continue
    return None


def fetch_raw_with_cache(net: str, sta: str, chan: str, t1: UTCDateTime, t2: UTCDateTime) -> Optional[Trace]:
    assert t2 > t1
    needed_hours = _hours_between(t1, t2)

    for hour in needed_hours:
        p = _cache_hour_path(net, sta, chan, hour)
        if p.exists():
            continue
        h0 = UTCDateTime(f"{hour}:00:00")
        h1 = h0 + 3600
        tr = fdsn_fetch(net, sta, chan, h0, h1)
        p.parent.mkdir(parents=True, exist_ok=True)
        if tr is None or len(tr.data) == 0:
            try:
                Stream().write(str(p), format="MSEED")
            except Exception:
                pass
            continue
        Stream(traces=[tr]).write(str(p), format="MSEED")

    stream_all = Stream()
    for hour in needed_hours:
        p = _cache_hour_path(net, sta, chan, hour)
        if p.exists():
            try:
                stream_all += read(str(p), format="MSEED")
            except Exception:
                pass
    if len(stream_all) == 0:
        return None
    stream_all.merge(method=1, fill_value="interpolate")
    st_slice = stream_all.slice(t1, t2)
    if len(st_slice) == 0:
        return None
    st_slice.sort()
    tr = st_slice[0]
    tr.stats.network = net
    tr.stats.station = sta
    tr.stats.channel = chan
    return tr


def get_waveforms_for_event(
    sta_rows: pd.DataFrame,
    t0: datetime,
    comps: Tuple[str, ...] = ("E", "N", "Z")
) -> Dict[Tuple[str, str], Dict[str, Trace]]:
    t_abs0 = UTCDateTime(t0 - timedelta(seconds=PAD_BEFORE))
    t_abs1 = UTCDateTime(t0 + timedelta(seconds=EVENT_WIN_S + PAD_AFTER))
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]] = {}
    for _, r in sta_rows.iterrows():
        net, sta = str(r["network"]), str(r["station"])
        trio: Dict[str, Trace] = {}
        for comp, col in (("E", "chE"), ("N", "chN"), ("Z", "chZ")):
            if comp not in comps:
                continue
            ch = r.get(col)
            if isinstance(ch, str) and ch:
                tr = fetch_raw_with_cache(net, sta, ch, t_abs0, t_abs1)
                if tr is not None:
                    trio[comp] = tr

        need_full_3c = set(comps) == {"E", "N", "Z"}
        ok = all(k in trio for k in ("E", "N", "Z")) if need_full_3c else (XCORR_COMPONENT in trio or len(trio) > 0)
        if ok:
            sta2tr[(net, sta)] = trio
    return sta2tr


#------------------------
# Preprocessing
#------------------------

def preprocess_env1hz_wech(tr: Trace) -> Optional[Trace]:
    x = tr.copy()
    try:
        x.filter("bandpass", freqmin=BANDPASS_HZ[0], freqmax=BANDPASS_HZ[1], corners=4, zerophase=True)
    except Exception:
        return None
    envd = envelope(x.data.astype(np.float32))
    out = Trace(envd, header=x.stats)
    try:
        out.filter("lowpass", freq=ENVELOPE_LP_HZ, corners=2, zerophase=True)
    except Exception:
        return None
    fs = float(out.stats.sampling_rate)
    dec = max(1, int(round(fs / DS_RATE_HZ)))
    out.data = out.data[::dec]
    out.stats.sampling_rate = fs / dec
    mu, sd = np.nanmean(out.data), np.nanstd(out.data)
    out.data = (out.data - mu) / (sd + 1e-9)
    return out


def choose_Z_or_EN(trio: Dict[str, Trace]) -> Optional[Trace]:
    return trio.get("Z") or trio.get("E") or trio.get("N")


def slice_fixed_or_pad(tr: Trace, t_start: datetime, t_end: datetime, fill_value: float = float("nan")) -> np.ndarray:
    fs = float(tr.stats.sampling_rate)
    n_target = int(round((UTCDateTime(t_end) - UTCDateTime(t_start)) * fs))
    if n_target <= 0:
        return np.zeros(0, dtype=np.float32)
    seg = tr.slice(UTCDateTime(t_start), UTCDateTime(t_end), nearest_sample=True)
    y = seg.data.astype(np.float32, copy=False) if seg and len(seg.data) else np.empty(0, np.float32)
    if y.size == 0:
        return np.full(n_target, fill_value, np.float32)
    lead = int(round((seg.stats.starttime - UTCDateTime(t_start)) * fs))
    lead = max(0, min(lead, n_target))
    y = y[:max(0, n_target - lead)]
    pad_lead = lead
    pad_trail = n_target - pad_lead - y.size
    if pad_trail < 0:
        y = y[:y.size + pad_trail]
        pad_trail = 0
    return np.concatenate([
        np.full(pad_lead, fill_value, np.float32),
        y,
        np.full(pad_trail, fill_value, np.float32)
    ])


#------------------------
# Cross-Correlation
#------------------------

def xcorr_norm_1hz(a: np.ndarray, b: np.ndarray, max_lag_s: float):
    l = min(len(a), len(b))
    if l < 5:
        return np.array([0.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 0.0
    a = a[:l].astype(np.float32)
    b = b[:l].astype(np.float32)
    am, asd = np.nanmean(a), np.nanstd(a) + 1e-9
    bm, bsd = np.nanmean(b), np.nanstd(b) + 1e-9
    a = np.where(np.isfinite(a), (a - am) / asd, 0.0)
    b = np.where(np.isfinite(b), (b - bm) / bsd, 0.0)
    c = np.correlate(a, b, mode="full").astype(np.float32) / max(l - 1, 1)
    lags = np.arange(-l + 1, l, dtype=np.int32)
    m = (lags >= -int(max_lag_s)) & (lags <= int(max_lag_s))
    c = c[m]
    lags = lags[m]
    cmax = float(np.max(c)) if c.size else 0.0
    return lags.astype(np.float32), c, cmax


def compute_all_pairs_xcorr(env_traces: Dict[Tuple[str, str], Trace], max_lag_s: float, t_ev0: datetime, t_ev1: datetime):
    keys = list(env_traces.keys())
    xcorr_results = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            ti, tj = env_traces[ki], env_traces[kj]
            fs = float(ti.stats.sampling_rate)
            assert abs(fs - 1.0) < 1e-3, "envelope traces must be 1 Hz"
            tmin = max(ti.stats.starttime, tj.stats.starttime, UTCDateTime(t_ev0))
            tmax = min(ti.stats.endtime, tj.stats.endtime, UTCDateTime(t_ev1))
            if tmax <= tmin:
                continue
            win_len_s = float(tmax - tmin)
            max_lag_used = max(1.0, min(max_lag_s, 0.5 * win_len_s))
            ai = ti.slice(tmin, tmax, nearest_sample=True).data
            bj = tj.slice(tmin, tmax, nearest_sample=True).data
            lags, corr, cmax = xcorr_norm_1hz(ai, bj, max_lag_used)
            lag_at_max = float(lags[int(np.argmax(corr))]) if corr.size else 0.0
            xcorr_results[(ki, kj)] = {
                "lags": lags,
                "correlation": corr,
                "max_correlation": cmax,
                "lag_at_max": lag_at_max,
                "max_lag_used": float(max_lag_used),
                "station_pair": f"{ki[0]}.{ki[1]} - {kj[0]}.{kj[1]}",
            }
    return xcorr_results


def identify_hits(xcorr_results: Dict, threshold: float = HIT_THRESHOLD) -> Tuple[Dict, Set]:
    hits = {}
    all_hit_stations = set()
    for (ki, kj), r in xcorr_results.items():
        if r["max_correlation"] >= threshold:
            hits[(ki, kj)] = r
            all_hit_stations.add(ki)
            all_hit_stations.add(kj)
    return hits, all_hit_stations


#------------------------
# Hit Summaries
#------------------------

def _summarize_hits_per_station(
    sta_df: pd.DataFrame,
    xcorr_results: Dict,
    hits: Dict
) -> Dict[Tuple[str, str], Dict[str, object]]:
    partners: Dict[Tuple[str, str], List[Tuple[str, float, float]]] = {}
    for (ki, kj), r in hits.items():
        for a, b in [(ki, kj), (kj, ki)]:
            partners.setdefault(a, [])
            partners[a].append((
                f"{b[0]}.{b[1]}",
                float(r.get("max_correlation", np.nan)),
                float(r.get("lag_at_max", np.nan))
            ))

    out: Dict[Tuple[str, str], Dict[str, object]] = {}
    sta_keys = [(str(r["network"]), str(r["station"])) for _, r in sta_df.iterrows()]
    for key in sta_keys:
        plist = partners.get(key, [])
        if plist:
            plist_sorted = sorted(plist, key=lambda x: (x[1], -abs(x[2])), reverse=True)
            top_partner, top_corr, top_lag = plist_sorted[0]
            hp_str = ";".join([f"{p}:{c:.3f}@{l:+.0f}" for (p, c, l) in plist_sorted])
            out[key] = dict(
                hit_count=len(plist_sorted),
                top_hit_partner=top_partner,
                top_hit_corr=float(top_corr),
                top_hit_lag_s=float(top_lag),
                hit_partners=hp_str,
            )
        else:
            out[key] = dict(
                hit_count=0,
                top_hit_partner="",
                top_hit_corr=np.nan,
                top_hit_lag_s=np.nan,
                hit_partners="",
            )
    return out


def _build_band_rows_for_event(
    storage_mode: str,
    per_event_h5_path: Path,
    per_event_dir: Path,
    uid: str,
    ev_meta: Dict,
    sta_used_df: pd.DataFrame,
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]],
    t_abs0: datetime,
    t_abs1: datetime,
    hits_summary_by_station: Dict[Tuple[str, str], Dict[str, object]],
) -> List[Dict[str, object]]:
    rows = []
    t_abs0_iso = t_abs0.strftime("%Y-%m-%dT%H:%M:%S")
    t_abs1_iso = t_abs1.strftime("%Y-%m-%dT%H:%M:%S")

    for (net, sta), trio in sta2tr.items():
        chE = chN = chZ = ""
        fsE = fsN = fsZ = np.nan

        trE = trio.get("E")
        if trE is not None:
            chE = str(getattr(trE.stats, "channel", "") or "")
            fsE = float(getattr(trE.stats, "sampling_rate", np.nan))

        trN = trio.get("N")
        if trN is not None:
            chN = str(getattr(trN.stats, "channel", "") or "")
            fsN = float(getattr(trN.stats, "sampling_rate", np.nan))

        trZ = trio.get("Z")
        if trZ is not None:
            chZ = str(getattr(trZ.stats, "channel", "") or "")
            fsZ = float(getattr(trZ.stats, "sampling_rate", np.nan))

        sampling_rate = np.nan
        for fs in (fsZ, fsE, fsN):
            if not (isinstance(fs, float) and np.isnan(fs)):
                sampling_rate = float(fs)
                break

        row_sta = sta_used_df[
            (sta_used_df["network"].astype(str) == net) &
            (sta_used_df["station"].astype(str) == sta)
        ]
        if len(row_sta) == 1:
            lat = float(row_sta["latitude"].iloc[0])
            lon = float(row_sta["longitude"].iloc[0])
            dist = float(row_sta["dist_km"].iloc[0])
            band = str(row_sta["band"].iloc[0]) if "band" in row_sta.columns else (chZ[:2] or chE[:2] or chN[:2])
        else:
            lat = lon = dist = np.nan
            band = chZ[:2] or chE[:2] or chN[:2]

        hs = hits_summary_by_station.get((net, sta), {})
        rows.append({
            "event_uid": uid,
            "event_start_iso": ev_meta.get("event_start_iso", ""),
            "event_lat": float(ev_meta.get("event_lat", np.nan)),
            "event_lon": float(ev_meta.get("event_lon", np.nan)),
            "t_abs0_iso": t_abs0_iso,
            "t_abs1_iso": t_abs1_iso,
            "window_s": float(ev_meta.get("event_window_s", np.nan)),
            "storage_mode": storage_mode,
            "per_event_h5": str(strip_part_suffix(per_event_h5_path)) if per_event_h5_path else "",
            "per_event_dir": str(per_event_dir) if per_event_dir else "",
            "network": net,
            "station": sta,
            "band": band,
            "chE": chE,
            "chN": chN,
            "chZ": chZ,
            "latitude": lat,
            "longitude": lon,
            "dist_km_from_event": dist,
            "sampling_rate_hz": sampling_rate,
            "slice_start_iso": t_abs0_iso,
            "slice_end_iso": t_abs1_iso,
            "fill_value": FILL_VALUE,
            "hit_count": int(hs.get("hit_count", 0)) if not pd.isna(hs.get("hit_count", 0)) else 0,
            "top_hit_partner": str(hs.get("top_hit_partner", "")),
            "top_hit_corr": float(hs.get("top_hit_corr", np.nan)) if hs.get("top_hit_partner", "") else np.nan,
            "top_hit_lag_s": float(hs.get("top_hit_lag_s", np.nan)) if hs.get("top_hit_partner", "") else np.nan,
            "hit_partners": str(hs.get("hit_partners", "")),
        })
    return rows


#------------------------
# Master CSV Update
#------------------------

def append_or_update_master_channels_csv(rows: List[Dict[str, object]]):
    df_new = pd.DataFrame(
        [{k: r.get(k, "") for k in CHANNEL_CSV_COLUMNS} for r in rows],
        columns=CHANNEL_CSV_COLUMNS
    )

    if "band" not in df_new.columns:
        df_new["band"] = ""

    if df_new["band"].eq("").any():
        def _infer_band_new(row):
            for key in ("chZ", "chE", "chN"):
                ch = str(row.get(key, "") or "")
                if len(ch) >= 2:
                    return ch[:2].upper()
            return ""
        df_new.loc[df_new["band"].eq(""), "band"] = df_new[df_new["band"].eq("")].apply(_infer_band_new, axis=1)

    df_new = df_new[df_new["network"].astype(str).str.len() > 0]
    df_new = df_new[df_new["station"].astype(str).str.len() > 0]

    dest_dir = str(Path(MASTER_CSV_PATH).parent)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    key_cols_station = ["event_uid", "network", "station"]

    with SimpleFileLock(MASTER_CSV_LOCK, timeout=120.0):
        if not Path(MASTER_CSV_PATH).exists():
            tmp = NamedTemporaryFile(delete=False, dir=dest_dir)
            tmp_path = Path(tmp.name)
            tmp.close()
            df_new.to_csv(tmp_path, index=False)
            os.replace(tmp_path, MASTER_CSV_PATH)
            return

        df_old = pd.read_csv(MASTER_CSV_PATH)

        if "band" not in df_old.columns:
            df_old["band"] = ""
        else:
            df_old["band"] = df_old["band"].fillna("")

        def _infer_band_from_channels(row):
            for key in ("chZ", "chE", "chN"):
                ch = str(row.get(key, "") or "")
                if len(ch) >= 2:
                    return ch[:2].upper()
            return ""

        if df_old["band"].eq("").any():
            df_old.loc[df_old["band"].eq(""), "band"] = df_old[df_old["band"].eq("")].apply(_infer_band_from_channels, axis=1)

        if "component" in df_old.columns and "band" in df_old.columns:
            keep_cols = [c for c in df_old.columns if c != "component"]
            df_old = df_old[keep_cols].drop_duplicates(subset=key_cols_station, keep="first")

        all_cols = list(dict.fromkeys(list(df_old.columns) + CHANNEL_CSV_COLUMNS))
        df_old = df_old.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        mk_old = df_old[key_cols_station].astype(str).agg("||".join, axis=1)
        mk_new = df_new[key_cols_station].astype(str).agg("||".join, axis=1)
        df_out = pd.concat([df_old[~mk_old.isin(set(mk_new))], df_new], ignore_index=True)

        tmp = NamedTemporaryFile(delete=False, dir=dest_dir)
        tmp_path = Path(tmp.name)
        tmp.close()
        sort_cols = [c for c in ["event_uid", "network", "station"] if c in df_out.columns]
        if sort_cols:
            df_out = df_out.sort_values(sort_cols)
        df_out.to_csv(tmp_path, index=False)
        os.replace(tmp_path, MASTER_CSV_PATH)


#------------------------
# File Lock
#------------------------

class SimpleFileLock:
    def __init__(self, path: str, timeout: float = 120.0, poll: float = 0.2):
        self.path = path
        self.timeout = timeout
        self.poll = poll

    def __enter__(self):
        t0 = time.time()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if time.time() - t0 > self.timeout:
                    raise TimeoutError(f"Lock timeout: {self.path}")
                time.sleep(self.poll)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            os.remove(self.path)
        except FileNotFoundError:
            pass


#------------------------
# HDF5 and CSV Writing
#------------------------

def save_event_to_hdf5_and_csv(
    h5_temp_path: Path,
    event_outdir: Path,
    dt_str: str,
    uid: str,
    ev_meta: Dict,
    sta_df: pd.DataFrame,
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]],
    xcorr_results: Dict,
    hits: Dict,
    t_abs0: datetime,
    t_abs1: datetime
):
    with h5py.File(h5_temp_path, "w") as h5:
        for k, v in ev_meta.items():
            try:
                h5.attrs[k] = v
            except Exception:
                h5.attrs[k] = str(v)
        h5.attrs["event_uid"] = uid

        hit_pairs_list = [f"{ki[0]}.{ki[1]}__{kj[0]}.{kj[1]}" for (ki, kj) in hits.keys()]
        h5.attrs["n_stations"] = int(len(sta_df))
        h5.attrs["n_hit_pairs"] = int(len(hit_pairs_list))
        h5.attrs["station_keys"] = np.array([f"{r['network']}.{r['station']}" for _, r in sta_df.iterrows()], dtype="S")
        h5.attrs["hit_pairs"] = np.array(hit_pairs_list, dtype="S")
        h5.attrs["xcorr_band_min"] = float(BANDPASS_HZ[0])
        h5.attrs["xcorr_band_max"] = float(BANDPASS_HZ[1])
        h5.attrs["env_lowpass_hz"] = float(ENVELOPE_LP_HZ)
        h5.attrs["env_ds_hz"] = float(DS_RATE_HZ)
        h5.attrs["pad_before_s"] = float(PAD_BEFORE)
        h5.attrs["pad_after_s"] = float(PAD_AFTER)
        h5.attrs["window_total_s"] = float(PAD_BEFORE + EVENT_WIN_S + PAD_AFTER)

        g_sta = h5.create_group("stations")
        for _, r in sta_df.iterrows():
            key = f"{r['network']}.{r['station']}"
            g = g_sta.create_group(key)
            g.attrs["network"] = str(r["network"])
            g.attrs["station"] = str(r["station"])
            g.attrs["latitude"] = float(r["latitude"])
            g.attrs["longitude"] = float(r["longitude"])
            g.attrs["dist_km_from_event"] = float(r["dist_km"])
            g.attrs["chE"] = str(r["chE"])
            g.attrs["chN"] = str(r["chN"])
            g.attrs["chZ"] = str(r["chZ"])

        g_raw = h5.create_group("raw_waveforms")
        for (net, sta), trio in sta2tr.items():
            gg = g_raw.create_group(f"{net}.{sta}")
            for comp, tr in trio.items():
                arr = slice_fixed_or_pad(tr, t_start=t_abs0, t_end=t_abs1, fill_value=FILL_VALUE)
                d = gg.create_dataset(comp, data=arr, compression="gzip", compression_opts=3)
                d.attrs["sampling_rate"] = float(tr.stats.sampling_rate)
                d.attrs["starttime_iso"] = t_abs0.strftime("%Y-%m-%dT%H:%M:%S")
                d.attrs["endtime_iso"] = t_abs1.strftime("%Y-%m-%dT%H:%M:%S")
                d.attrs["network"] = net
                d.attrs["station"] = sta
                d.attrs["channel"] = str(tr.stats.channel)
                d.attrs["location"] = str(getattr(tr.stats, "location", "") or "")
                d.attrs["fill_value"] = FILL_VALUE

        g_xc = h5.create_group("xcorr")
        for (ki, kj), r in xcorr_results.items():
            if PAIRS_CSV_MODE == "hits_only" and (ki, kj) not in hits:
                continue
            pair_key = f"{ki[0]}.{ki[1]}__{kj[0]}.{kj[1]}"
            gg = g_xc.create_group(pair_key)
            gg.create_dataset("lags_s", data=r["lags"].astype(np.float32), compression="gzip", compression_opts=3)
            gg.create_dataset("corr", data=r["correlation"].astype(np.float32), compression="gzip", compression_opts=3)
            gg.attrs["max_correlation"] = float(r["max_correlation"])
            gg.attrs["lag_at_max_s"] = float(r["lag_at_max"])
            gg.attrs["max_lag_used_s"] = float(r["max_lag_used"])
            gg.attrs["is_hit"] = bool((ki, kj) in hits)

        g_hits = h5.create_group("hits")
        g_hits.attrs["threshold"] = float(HIT_THRESHOLD)
        hit_names = [f"{ki[0]}.{ki[1]}__{kj[0]}.{kj[1]}" for (ki, kj) in hits.keys()]
        g_hits.create_dataset("pairs", data=np.array(hit_names, dtype="S"), compression="gzip", compression_opts=3)

    hits_summary = _summarize_hits_per_station(sta_df=sta_df, xcorr_results=xcorr_results, hits=hits)
    rows = _build_band_rows_for_event(
        storage_mode=STORAGE_MODE,
        per_event_h5_path=h5_temp_path,
        per_event_dir=event_outdir,
        uid=uid,
        ev_meta=ev_meta,
        sta_used_df=sta_df,
        sta2tr=sta2tr,
        t_abs0=t_abs0,
        t_abs1=t_abs1,
        hits_summary_by_station=hits_summary
    )
    df_evt = pd.DataFrame(rows, columns=CHANNEL_CSV_COLUMNS)
    df_evt.sort_values(["event_uid", "network", "station"], inplace=True)
    csv_channels = event_outdir / f"tremor_{dt_str}_{uid}_channels.csv"
    df_evt.to_csv(csv_channels, index=False)
    return csv_channels


def validate_event_outputs(h5_path: Path, csv_channels: Path, sta2tr: Dict[Tuple[str, str], Dict[str, Trace]]):
    assert h5_path.exists(), f"HDF5 not written: {h5_path}"
    assert csv_channels.exists(), f"channels.csv not written: {csv_channels}"

    df = pd.read_csv(csv_channels)
    n_expected_rows = int(len(sta2tr))
    assert len(df) == n_expected_rows, f"channels.csv rows {len(df)} != expected {n_expected_rows}"

    dur_full = PAD_BEFORE + EVENT_WIN_S + PAD_AFTER
    with h5py.File(h5_path, "r") as h5:
        assert "raw_waveforms" in h5, "HDF5 missing 'raw_waveforms'"
        assert len(h5["raw_waveforms"].keys()) == len(sta2tr), "HDF5 raw_waveforms group count mismatch"
        for sta_grp in h5["raw_waveforms"].values():
            for comp_ds in sta_grp.values():
                fs = float(comp_ds.attrs["sampling_rate"])
                n = int(comp_ds.shape[0])
                n_exp = int(round(dur_full * fs))
                assert abs(n - n_exp) <= 1, f"raw length mismatch: got {n}, expect {n_exp}"


def append_event_to_master_h5(
    master_path: str,
    uid: str,
    t0_iso: str,
    t_abs0: datetime,
    t_abs1: datetime,
    ev_lat: float,
    ev_lon: float,
    sta_df: pd.DataFrame,
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]],
    fill_value: float = np.nan
):
    with h5py.File(master_path, "a") as h5:
        h5.attrs.setdefault("kind", "tremor_raw_master")
        h5.attrs.setdefault("schema_version", "1.0")
        h5.attrs["time_window_seconds"] = float((UTCDateTime(t_abs1) - UTCDateTime(t_abs0)))
        root = h5.require_group("events")
        if uid in root:
            return
        ge = root.create_group(uid)

        station_keys = [f"{r['network']}.{r['station']}" for _, r in sta_df.iterrows()]
        ge.attrs["n_stations"] = int(len(station_keys))
        ge.attrs["station_keys"] = np.array(station_keys, dtype="S")
        ge.attrs["pad_before_s"] = float((UTCDateTime(t0_iso) - UTCDateTime(t_abs0)))
        ge.attrs["pad_after_s"] = float((UTCDateTime(t_abs1) - UTCDateTime(t0_iso)))
        ge.attrs["event_start_iso"] = t0_iso
        ge.attrs["event_lat"] = float(ev_lat)
        ge.attrs["event_lon"] = float(ev_lon)
        ge.attrs["t_abs0_iso"] = t_abs0.strftime("%Y-%m-%dT%H:%M:%S")
        ge.attrs["t_abs1_iso"] = t_abs1.strftime("%Y-%m-%dT%H:%M:%S")

        g_sta = ge.create_group("stations")
        for _, r in sta_df.iterrows():
            key = f"{r['network']}.{r['station']}"
            g = g_sta.create_group(key)
            g.attrs["network"] = str(r["network"])
            g.attrs["station"] = str(r["station"])
            g.attrs["latitude"] = float(r["latitude"])
            g.attrs["longitude"] = float(r["longitude"])
            g.attrs["dist_km_from_event"] = float(r["dist_km"])
            g.attrs["chE"] = str(r["chE"])
            g.attrs["chN"] = str(r["chN"])
            g.attrs["chZ"] = str(r["chZ"])

        g_raw = ge.create_group("raw_waveforms")
        for (net, sta), trio in sta2tr.items():
            gg = g_raw.create_group(f"{net}.{sta}")
            for comp, tr in trio.items():
                arr = slice_fixed_or_pad(tr, t_start=t_abs0, t_end=t_abs1, fill_value=fill_value)
                d = gg.create_dataset(comp, data=arr, compression="gzip", compression_opts=3)
                d.attrs["sampling_rate"] = float(tr.stats.sampling_rate)
                d.attrs["starttime_iso"] = t_abs0.strftime("%Y-%m-%dT%H:%M:%S")
                d.attrs["endtime_iso"] = t_abs1.strftime("%Y-%m-%dT%H:%M:%S")
                d.attrs["network"] = net
                d.attrs["station"] = sta
                d.attrs["channel"] = str(tr.stats.channel)
                d.attrs["location"] = str(getattr(tr.stats, "location", "") or "")
                d.attrs["fill_value"] = fill_value


#------------------------
# Event Processing
#------------------------

def process_tremor_event(ev_row: pd.Series, ch_df: pd.DataFrame, outdir: Path) -> None:
    ev_lat = float(pd.to_numeric(ev_row.get("lat"), errors="coerce")) if "lat" in ev_row else np.nan
    ev_lon = float(pd.to_numeric(ev_row.get("lon"), errors="coerce")) if "lon" in ev_row else np.nan
    t0 = to_utc(str(ev_row.get("starttime")))
    if any([pd.isna(ev_lat), pd.isna(ev_lon)]) or t0 is None:
        print("[SKIP] Missing lat/lon/starttime in event")
        return

    t_abs0 = t0 - timedelta(seconds=PAD_BEFORE)
    t_abs1 = t0 + timedelta(seconds=EVENT_WIN_S + PAD_AFTER)
    t_ev0 = t0
    t_ev1 = t0 + timedelta(seconds=EVENT_WIN_S)

    dt_str = t0.strftime("%Y-%m-%dT%H_%M_%S")
    uid = event_uid(t0.strftime("%Y-%m-%dT%H:%M:%S"), ev_lat, ev_lon, EVENT_WIN_S)
    event_info = f"{t0.strftime('%Y-%m-%d %H:%M:%S')} (fixed {EVENT_WIN_S:.0f}s)"
    print(f"\n[PROCESSING] {event_info}  uid={uid}")

    event_outdir = outdir / f"tremor_hit_{dt_str}_{uid}"
    h5_temp = event_outdir / f"tremor_{dt_str}_{uid}.hdf5.part"
    h5_final = event_outdir / f"tremor_{dt_str}_{uid}.hdf5"
    if STORAGE_MODE in ("per_event", "both"):
        if SKIP_PER_EVENT_IF_EXISTS and h5_final.exists():
            print(f"[SKIP] Per-event HDF5 exists: {h5_final}")
            return

    sta_df = pick_stations_3c(ch_df, ev_lat, ev_lon)
    if len(sta_df) < 2:
        print(f"[SKIP] Not enough 3C stations ({len(sta_df)})")
        return
    print(f"[INFO] Selected {len(sta_df)} stations")

    if TWO_STAGE_FETCH:
        print("[INFO] Downloading QC component only (hourly cache)...")
        sta2tr_qc = get_waveforms_for_event(sta_df, t0, comps=(XCORR_COMPONENT,))
        if len(sta2tr_qc) < 2:
            print(f"[SKIP] Not enough stations for QC ({len(sta2tr_qc)})")
            return

        print("[INFO] Preprocessing envelopes for xcorr (in-memory, QC component)...")
        env_traces: Dict[Tuple[str, str], Trace] = {}
        t_abs0_utc = UTCDateTime(t_abs0)
        t_abs1_utc = UTCDateTime(t_abs1)
        for key, trio in sta2tr_qc.items():
            tr0 = choose_Z_or_EN(trio)
            if tr0 is None:
                continue
            trw = tr0.copy().slice(t_abs0_utc, t_abs1_utc, nearest_sample=True)
            env = preprocess_env1hz_wech(trw)
            if env is not None:
                env_traces[key] = env
        if len(env_traces) < 2:
            print(f"[SKIP] Not enough envelope traces for xcorr ({len(env_traces)})")
            return

        print("[INFO] Computing cross-correlations...")
        xcorr_results = compute_all_pairs_xcorr(env_traces, XCORR_MAX_LAG_S, t_ev0, t_ev1)
        if not xcorr_results:
            print("[SKIP] No cross-correlations computed")
            return

        hits, _ = identify_hits(xcorr_results, HIT_THRESHOLD)
        print("[INFO] Applying physical lag filtering...")
        sta_ll_all = {
            (str(r["network"]), str(r["station"])): (float(r["latitude"]), float(r["longitude"]))
            for _, r in sta_df.iterrows()
        }
        phys_hits = {}
        for (ki, kj), r in hits.items():
            lag_at_max = float(r.get("lag_at_max", 0.0))
            lat_i, lon_i = sta_ll_all.get(ki, (None, None))
            lat_j, lon_j = sta_ll_all.get(kj, (None, None))
            if lat_i is None or lat_j is None:
                continue
            d_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
            lag_cap = min(LAG_CAP_MAX_S, d_km / VMIN_KMPS)
            if abs(lag_at_max) <= lag_cap:
                phys_hits[(ki, kj)] = r

        if not phys_hits:
            print(f"[FILTER] Physical lag filter removed all {len(hits)} pairs")
            print("[SKIP] No hits found after physical lag filter")
            return

        print(f"[FILTER] Physical lag filter kept {len(phys_hits)} / {len(hits)} pairs")
        hits = phys_hits

        hit_station_keys = set([k for pair in hits.keys() for k in pair])
        if len(hit_station_keys) < 2:
            print("[SKIP] Not enough unique hit stations")
            return

        sta_used_df = sta_df[
            sta_df.apply(lambda r: (str(r["network"]), str(r["station"])) in hit_station_keys, axis=1)
        ].reset_index(drop=True)
        print(f"[INFO] Re-download 3C for hit stations only: {len(sta_used_df)} stations")

        print("[INFO] Downloading raw 3C for hits (hourly cache)...")
        sta2tr = get_waveforms_for_event(sta_used_df, t0, comps=("E", "N", "Z"))
        if len(sta2tr) < 2:
            print(f"[SKIP] Not enough 3C stations among hits ({len(sta2tr)})")
            return

    else:
        print("[INFO] Downloading full 3C (hourly cache)...")
        sta2tr = get_waveforms_for_event(sta_df, t0, comps=("E", "N", "Z"))
        if len(sta2tr) < 2:
            print(f"[SKIP] Not enough 3C stations ({len(sta2tr)})")
            return

        print("[INFO] Preprocessing envelopes for xcorr (in-memory)...")
        env_traces: Dict[Tuple[str, str], Trace] = {}
        t_abs0_utc = UTCDateTime(t_abs0)
        t_abs1_utc = UTCDateTime(t_abs1)
        for key, trio in sta2tr.items():
            tr0 = choose_Z_or_EN(trio)
            if tr0 is None:
                continue
            trw = tr0.copy().slice(t_abs0_utc, t_abs1_utc, nearest_sample=True)
            env = preprocess_env1hz_wech(trw)
            if env is not None:
                env_traces[key] = env
        if len(env_traces) < 2:
            print(f"[SKIP] Not enough envelope traces ({len(env_traces)})")
            return

        print("[INFO] Computing cross-correlations...")
        xcorr_results = compute_all_pairs_xcorr(env_traces, XCORR_MAX_LAG_S, t_ev0, t_ev1)
        if not xcorr_results:
            print("[SKIP] No cross-correlations computed")
            return

        hits, _ = identify_hits(xcorr_results, HIT_THRESHOLD)
        print("[INFO] Applying physical lag filtering...")
        sta_ll_all = {
            (str(r["network"]), str(r["station"])): (float(r["latitude"]), float(r["longitude"]))
            for _, r in sta_df.iterrows()
        }
        phys_hits = {}
        for (ki, kj), r in hits.items():
            lag_at_max = float(r.get("lag_at_max", 0.0))
            lat_i, lon_i = sta_ll_all.get(ki, (None, None))
            lat_j, lon_j = sta_ll_all.get(kj, (None, None))
            if lat_i is None or lat_j is None:
                continue
            d_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
            lag_cap = min(LAG_CAP_MAX_S, d_km / VMIN_KMPS)
            if abs(lag_at_max) <= lag_cap:
                phys_hits[(ki, kj)] = r

        if not phys_hits:
            print(f"[FILTER] Physical lag filter removed all {len(hits)} pairs")
            print("[SKIP] No hits found after physical lag filter")
            return

        print(f"[FILTER] Physical lag filter kept {len(phys_hits)} / {len(hits)} pairs")
        hits = phys_hits

        xcorr_results = {k: v for k, v in xcorr_results.items() if k in hits}
        hit_station_keys = set([k for pair in hits.keys() for k in pair])

        if KEEP_ONLY_HIT_STATIONS:
            sta_used_df = sta_df[
                sta_df.apply(lambda r: (str(r["network"]), str(r["station"])) in hit_station_keys, axis=1)
            ].reset_index(drop=True)
            sta2tr = {k: v for k, v in sta2tr.items() if k in hit_station_keys}
        else:
            sta_used_df = sta_df.loc[
                sta_df.apply(lambda r: (str(r["network"]), str(r["station"])) in set(sta2tr.keys()), axis=1)
            ].reset_index(drop=True)

    ev_meta = {
        "event_start_iso": t0.strftime("%Y-%m-%dT%H:%M:%S"),
        "event_window_s": float(EVENT_WIN_S),
        "event_lat": float(ev_lat),
        "event_lon": float(ev_lon),
        "pad_before_s": float(PAD_BEFORE),
        "pad_after_s": float(PAD_AFTER),
        "bandpass_hz_min": float(BANDPASS_HZ[0]),
        "bandpass_hz_max": float(BANDPASS_HZ[1]),
        "envelope_lowpass_hz": float(ENVELOPE_LP_HZ),
        "env_sampling_rate_hz": float(DS_RATE_HZ),
        "xcorr_max_lag_s": float(XCORR_MAX_LAG_S),
        "hit_threshold": float(HIT_THRESHOLD),
    }

    per_event_csv = None
    if STORAGE_MODE in ("per_event", "both"):
        event_outdir.mkdir(parents=True, exist_ok=True)

        per_event_csv = save_event_to_hdf5_and_csv(
            h5_temp_path=h5_temp,
            event_outdir=event_outdir,
            dt_str=dt_str,
            uid=uid,
            ev_meta=ev_meta,
            sta_df=sta_used_df,
            sta2tr=sta2tr,
            xcorr_results=xcorr_results,
            hits=hits,
            t_abs0=t_abs0,
            t_abs1=t_abs1
        )

        try:
            validate_event_outputs(
                h5_path=h5_temp,
                csv_channels=per_event_csv,
                sta2tr=sta2tr
            )
        except Exception as e:
            if h5_temp.exists():
                try:
                    h5_temp.unlink()
                except Exception:
                    pass
            if per_event_csv and per_event_csv.exists():
                try:
                    per_event_csv.unlink()
                except Exception:
                    pass
            try:
                if event_outdir.exists() and not any(event_outdir.iterdir()):
                    event_outdir.rmdir()
            except Exception:
                pass
            raise RuntimeError(f"Validation failed for event {dt_str}_{uid}: {e}")

        os.replace(h5_temp, h5_final)
        print(f"[OK] Per-event HDF5 written (atomic): {h5_final}")
        print(f"[OK] Per-event channels CSV: {per_event_csv}")

    try:
        hits_summary = _summarize_hits_per_station(sta_used_df, xcorr_results, hits)
        rows_master = _build_band_rows_for_event(
            storage_mode=STORAGE_MODE,
            per_event_h5_path=h5_final if h5_final.exists() else strip_part_suffix(h5_temp),
            per_event_dir=event_outdir,
            uid=uid,
            ev_meta=ev_meta,
            sta_used_df=sta_used_df,
            sta2tr=sta2tr,
            t_abs0=t_abs0,
            t_abs1=t_abs1,
            hits_summary_by_station=hits_summary
        )

        rows_master_df = pd.DataFrame(rows_master, columns=CHANNEL_CSV_COLUMNS).sort_values(["event_uid", "network", "station"])
        append_or_update_master_channels_csv(rows_master_df.to_dict(orient="records"))
        print(f"[OK] Updated master channels CSV: {MASTER_CSV_PATH}  (rows+={len(rows_master)})")
    except Exception as e:
        print(f"[WARN] Failed updating master channels CSV for uid={uid}: {e}")

    if STORAGE_MODE in ("master", "both"):
        try:
            with SimpleFileLock(MASTER_LOCKFILE, timeout=120.0):
                append_event_to_master_h5(
                    master_path=MASTER_H5_PATH,
                    uid=uid,
                    t0_iso=t0.strftime("%Y-%m-%dT%H:%M:%S"),
                    t_abs0=t_abs0,
                    t_abs1=t_abs1,
                    ev_lat=ev_lat,
                    ev_lon=ev_lon,
                    sta_df=sta_used_df,
                    sta2tr=sta2tr,
                    fill_value=FILL_VALUE
                )
            print(f"[OK] Appended raw to master HDF5: {MASTER_H5_PATH}  (event uid={uid})")
        except Exception as e:
            print(f"[WARN] Failed appending to master HDF5 for uid={uid}: {e}")


#------------------------
# Main
#------------------------

def main():
    args = parse_args()
    configure_from_args(args)
    assert STORAGE_MODE in ("per_event", "master", "both"), "STORAGE_MODE must be 'per_event'|'master'|'both'"
    print(f"=== Tremor Downloader (raw-only, hourly cache, fixed windows, STORAGE_MODE={STORAGE_MODE}) ===")
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    outdir = Path(OUT_DIR)

    print("[INFO] Loading channel and event data...")
    ch = pd.read_csv(CHANNELS_CSV)
    ev = pd.read_csv(EVENTS_CSV, low_memory=False)

    ch.columns = ch.columns.str.strip()
    ev.columns = ev.columns.str.strip()
    ch = ch.loc[:, ~ch.columns.duplicated(keep="first")]
    ev = ev.loc[:, ~ev.columns.duplicated(keep="first")]

    for col in ["lat", "lon", "depth"]:
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")
    ev["t0"] = ev["starttime"].apply(to_utc)

    if EVENT_TIME_RANGE is not None:
        tmin = to_utc(EVENT_TIME_RANGE[0])
        tmax = to_utc(EVENT_TIME_RANGE[1])
        ev = ev[(ev["t0"] >= tmin) & (ev["t0"] < tmax)].copy()
        print(f"[INFO] Filtered to time range: {tmin} to {tmax}")

    ev.sort_values("t0", inplace=True)

    if MAX_EVENTS is not None and len(ev) > MAX_EVENTS:
        ev = ev.head(MAX_EVENTS).copy()
        print(f"[INFO] Limited to {MAX_EVENTS} events")

    ev_dedup = dedup_event_windows(ev, event_win_s=EVENT_WIN_S, overlap_max_frac=OVERLAP_MAX_FRAC)
    if len(ev_dedup) < len(ev):
        print(f"[INFO] Deduplicated by event-window: kept {len(ev_dedup)} / {len(ev)}")
    ev = ev_dedup

    print(f"[INFO] Processing {len(ev)} events")

    for i, (idx, row) in enumerate(ev.iterrows()):
        print(f"\n--- Event {i + 1}/{len(ev)} (index {idx}) ---")
        try:
            process_tremor_event(row, ch, outdir)
        except Exception as e:
            print(f"[ERROR] Failed to process event {idx}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[COMPLETED] Results saved to: {outdir}")
    if STORAGE_MODE in ("master", "both"):
        print(f"[COMPLETED] Master HDF5: {MASTER_H5_PATH}")
    print(f"[COMPLETED] Master channels CSV: {MASTER_CSV_PATH}")


if __name__ == "__main__":
    main()