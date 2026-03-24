#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OBS NETSCAN:
- Read station deployment list file (gmap-stations1.txt style, no channel info)
- For each station: pick best available Z channel by priority (HHZ > BHZ > EHZ > BLZ > ELZ)
- Exclude first day & last day (both global scan range and per-station deployment range)
- Exclude airgun shot days/intervals from CSV
- Run trained 3-class Conv+BiLSTM model -> scan -> event (300s) plots + CSV

Deps:
  obspy torch numpy scipy matplotlib pywavelets(optional)
"""

import os
import csv
import json
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, stft as scipy_stft

import torch
import torch.nn as nn

from obspy import UTCDateTime, Stream
from obspy.clients.fdsn import Client


# ============================================================
# USER EDIT
# ============================================================

@dataclass
class USER:
    #CKPT_PATH: str = "/home/bxd240002/scratch/Archer/offshore_tremor/results/results_three_class_stft/gpu_train_20260202_141146/saved_models/final_model.pt"
    CKPT_PATH: str = "/home/bxd240002/scratch/Archer/offshore_tremor/results/results_three_class_stft/gpu_train_20260114_200039/checkpoints/best_model.pt"
    FDSN_PROVIDER: str = "IRIS"

    # ---- station list file (your gmap-stations1.txt) ----
    USE_STATION_FILE: bool = True
    STATION_FILE: str = "/home/bxd240002/scratch/Archer/OBS_test/data/stations/gmap-stations1.txt"

    # If not using station file, fall back to these:
    AUTO_DISCOVER: bool = False
    NET: str = ""
    STA: str = ""
    STA_LIST: Tuple[str, ...] = ()

    LOC: str = ""  # empty => wildcard

    # ---- channel priority (Z only) ----
    # Your preference order: HHZ > BHZ > EHZ > BLZ > ELZ
    CHANNEL_PRIORITY: Tuple[str, ...] = ("HHZ", "BHZ", "EHZ", "BLZ", "ELZ")

    # If you want to force a specific channel (override auto-pick), set this to e.g. ("BHZ",)
    FORCE_CHANNELS: Tuple[str, ...] = ()

    # ---- scan range (global) ----
    START_UTC = "2013-12-25T00:00:00"
    END_UTC   = "2013-12-28T00:00:00"

    # ---- exclude edges ----
    EXCLUDE_EDGE_DAYS: bool = True
    EDGE_DAYS: int = 1  # exclude first/last N days (default 1)

    # Sliding step (inference)
    STEP_SEC: float = 10.0

    # ---- detection thresholds ----
    TREMOR_PROB_THRESHOLD: float = 0.50
    EQ_PROB_THRESHOLD: float = 0.50

    # "max" => any window; "mean" => average over scanned windows
    AGG_MODE: str = "max"      # tremor
    EQ_AGG_MODE: str = "max"   # EQ

    # enable/disable each detector
    DO_TREMOR: bool = True
    DO_EQ: bool = False
    DO_UNKNOWN: bool = False
    DO_SHIPTONAL: bool = True

    REMOVE_RESPONSE: bool = False
    RESPONSE_OUTPUT: str = "VEL"

    OUT_DIR: str = "/home/bxd240002/scratch/Archer/OBS_test/outputs/obs_tremor_scan_out"
    SAVE_FMT: str = "png"

    # fixed visualization window size (seconds)
    EVENT_WINDOW_SEC: float = 300.0

    # limit plot count (-1 for all)
    MAX_PLOTS: int = -1

    # chunk download to avoid huge memory
    CHUNK_SEC: float = 3600.0
    CHUNK_OVERLAP_SEC: float = 200.0

    # Display STFT
    DISP_STFT_MAX_FREQ: float = 20.0
    DISP_STFT_NPERSEG: int = 256
    DISP_STFT_NOVERLAP: int = 192

    # Display bandpass for tremor
    BP_FMIN: float = 2.0
    BP_FMAX: float = 8.0

    # EQ display band
    EQ_BP_FMIN: float = 1.0
    EQ_BP_FMAX: float = 15.0

    # ---- exclude airgun shot days ----
    EXCLUDE_SHOT_DAYS: bool = True
    SHOT_DAYS_CSV: str = "/home/bxd240002/scratch/Archer/OBS_test/data/shotlog/merged/shot_days_utc.csv"

    PER_STATION_SUBDIR: bool = True
    WRITE_NETWORK_CSV: bool = True

    # ============================================================
    # Tremor low-frequency gate
    # ============================================================
    LOWFREQ_GATE: bool = True
    LOWFREQ_FLOOR_HZ: float = 0.2
    LOWFREQ_CEIL_HZ: float = 2.5
    GATE_TREMOR_MIN_HZ: float = 2.5
    GATE_TREMOR_MAX_HZ: float = 8.0
    LOWFREQ_RATIO_THR: float = 0.20
    DEBUG_GATE_EVERY: int = 0

    # ============================================================
    # Spike gate -> UNKNOWN
    # ============================================================
    SPIKE_GATE: bool = True
    SPIKE_BAND_MODE: str = "tremor"
    SPIKE_K: float = 10.0
    SPIKE_RATIO_THR: float = 0.01
    SPIKE_CREST_THR: float = 25.0
    UNKNOWN_SCORE_THR: float = 0.50
    UNKNOWN_BP_FMIN: float = 2.0
    UNKNOWN_BP_FMAX: float = 15.0

    # ============================================================
    # EQ low-frequency gate
    # ============================================================
    EQ_LOWFREQ_GATE: bool = True
    EQ_LOWFREQ_FLOOR_HZ: float = 0.2
    EQ_LOWFREQ_CEIL_HZ: float = 2.0
    EQ_GATE_MIN_HZ: float = 2.0
    EQ_GATE_MAX_HZ: float = 15.0
    EQ_LOWFREQ_RATIO_THR: float = 0.20
    DEBUG_EQ_GATE_EVERY: int = 0

    # ============================================================
    # SHIPTONAL narrowband gate
    # ============================================================
    TONAL_GATE: bool = True
    TONAL_FMIN: float = 2.0
    TONAL_FMAX: float = 10.0
    TONAL_BW_HZ: float = 0.5
    TONAL_RATIO_THR: float = 0.60
    TONAL_VETO_TREMOR: bool = True
    TONAL_VETO_EQ: bool = False
    TONAL_PLOT_FMIN: float = 1.0
    TONAL_PLOT_FMAX: float = 15.0
    SHIPTONAL_SCORE_THR: float = 0.50


U = USER()

LABEL_NAMES = {0: "NOISE", 1: "EARTHQUAKE", 2: "TREMOR", 3: "UNKNOWN"}


# ============================================================
# Utilities
# ============================================================

def _safe_loc(loc: str) -> str:
    return loc if (loc is not None and str(loc).strip() != "") else "*"


def _nan0(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _maybe_override_threshold_from_env() -> None:
    v = os.environ.get("TREMOR_P", "").strip()
    if v:
        try:
            U.TREMOR_PROB_THRESHOLD = float(v)
            print(f"[INFO] Override tremor threshold from env: TREMOR_P={U.TREMOR_PROB_THRESHOLD}")
        except Exception:
            print(f"[WARN] Invalid env TREMOR_P='{v}', ignore.")

    v = os.environ.get("EQ_P", "").strip()
    if v:
        try:
            U.EQ_PROB_THRESHOLD = float(v)
            print(f"[INFO] Override EQ threshold from env: EQ_P={U.EQ_PROB_THRESHOLD}")
        except Exception:
            print(f"[WARN] Invalid env EQ_P='{v}', ignore.")


def _maybe_override_run_cfg_from_env() -> None:
    def _get(k: str) -> str:
        return os.environ.get(k, "").strip()

    def _parse_bool(s: str) -> Optional[bool]:
        if not s:
            return None
        s = s.strip().lower()
        if s in ("1", "true", "t", "yes", "y", "on"):
            return True
        if s in ("0", "false", "f", "no", "n", "off"):
            return False
        return None

    v = _get("START_UTC")
    if v:
        U.START_UTC = v
        print(f"[INFO] Override START_UTC from env: {U.START_UTC}")

    v = _get("END_UTC")
    if v:
        U.END_UTC = v
        print(f"[INFO] Override END_UTC from env: {U.END_UTC}")

    v = _get("NET")
    if v:
        U.NET = v
        print(f"[INFO] Override NET from env: {U.NET}")

    v = _get("STA")
    if v:
        U.STA = v
        print(f"[INFO] Override STA from env: {U.STA}")

    # ---- station file override ----
    v = _get("STATION_FILE")
    if v:
        U.STATION_FILE = v
        print(f"[INFO] Override STATION_FILE from env: {U.STATION_FILE}")

    vb = _parse_bool(_get("USE_STATION_FILE"))
    if vb is not None:
        U.USE_STATION_FILE = bool(vb)
        print(f"[INFO] Override USE_STATION_FILE from env: {U.USE_STATION_FILE}")
    v = _get("LOC")
    if v or v == "":
        if "LOC" in os.environ:
            U.LOC = v
            print(f"[INFO] Override LOC from env: {U.LOC!r}")

    v = _get("FORCE_CHANNELS") or _get("CHANNELS")
    if v:
        chans = tuple([c.strip().upper() for c in v.split(",") if c.strip()])
        if chans:
            U.FORCE_CHANNELS = chans
            print(f"[INFO] Override FORCE_CHANNELS from env: {U.FORCE_CHANNELS}")

    v = _get("STA_LIST")
    if v:
        stas = tuple([s.strip().upper() for s in v.split(",") if s.strip()])
        if stas:
            U.STA_LIST = stas
            print(f"[INFO] Override STA_LIST from env: {U.STA_LIST}")

    vb = _parse_bool(_get("AUTO_DISCOVER"))
    if vb is not None:
        U.AUTO_DISCOVER = bool(vb)
        print(f"[INFO] Override AUTO_DISCOVER from env: {U.AUTO_DISCOVER}")

    for key in ("DO_TREMOR", "DO_EQ", "DO_UNKNOWN", "DO_SHIPTONAL"):
        vb = _parse_bool(_get(key))
        if vb is not None:
            setattr(U, key, bool(vb))
            print(f"[INFO] Override {key} from env: {getattr(U, key)}")


def make_run_subdir_short(base_out_dir: str, cfg_sr: float) -> str:
    now = UTCDateTime()
    run_tag = now.strftime("RUN%Y%m%dT%H%M%S")

    loc_tag = (U.LOC.strip() if U.LOC and U.LOC.strip() else "--")
    sta_tag = f"{U.NET}.NETSCAN.{loc_tag}"

    chans_tag = "AUTOCHZ" if (not U.FORCE_CHANNELS) else "-".join([c.strip().upper() for c in U.FORCE_CHANNELS])

    start = UTCDateTime(U.START_UTC).strftime("S%Y%m%dT%H%M%S")
    end = UTCDateTime(U.END_UTC).strftime("E%Y%m%dT%H%M%S")

    thr_tag = f"tp{int(round(float(U.TREMOR_PROB_THRESHOLD) * 100)):02d}_ep{int(round(float(U.EQ_PROB_THRESHOLD) * 100)):02d}"
    step_tag = f"st{int(round(float(U.STEP_SEC))):02d}"
    ev_tag = f"ev{int(round(float(U.EVENT_WINDOW_SEC))):03d}"
    sr_tag = f"sr{int(round(float(cfg_sr)))}"
    maxp_tag = f"max{U.MAX_PLOTS}" if U.MAX_PLOTS != -1 else "maxALL"

    gate_tag = "gateOFF"
    if bool(U.LOWFREQ_GATE):
        gate_tag = f"gateLF{U.LOWFREQ_CEIL_HZ:.1f}_r{int(round(U.LOWFREQ_RATIO_THR*100)):02d}"
    eq_gate_tag = "eqgateOFF"
    if bool(getattr(U, "EQ_LOWFREQ_GATE", False)):
        eq_gate_tag = f"eqgateLF{U.EQ_LOWFREQ_CEIL_HZ:.1f}_r{int(round(U.EQ_LOWFREQ_RATIO_THR*100)):02d}"

    sub = f"{run_tag}__{sta_tag}__{chans_tag}__{start}_{end}__{thr_tag}__{step_tag}__{ev_tag}__{sr_tag}__{gate_tag}__{eq_gate_tag}__{maxp_tag}"
    sub = sub.replace("/", "_").replace(" ", "_")
    return os.path.join(base_out_dir, sub)


def bandpass_np(x: np.ndarray, fs: float, fmin: float, fmax: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * float(fs)
    fmin = float(fmin)
    fmax = float(fmax)

    if fmin <= 0.0:
        fmin = 1e-3
    if fmax >= nyq:
        fmax = nyq - 1e-3
    if fmax <= fmin:
        raise ValueError(f"Invalid bandpass: fmin={fmin}, fmax={fmax}, nyq={nyq}")

    lo = fmin / nyq
    hi = fmax / nyq
    sos = butter(order, [lo, hi], btype="bandpass", output="sos")

    if x.ndim == 1:
        return sosfiltfilt(sos, x).astype(np.float32)

    y = np.zeros_like(x, dtype=np.float32)
    for c in range(x.shape[0]):
        y[c] = sosfiltfilt(sos, x[c]).astype(np.float32)
    return y


def detrend_basic(st: Stream) -> Stream:
    st = st.copy()
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.01, type="cosine")
    return st


def resample_stream(st: Stream, target_fs: float) -> Stream:
    st = st.copy()
    for tr in st:
        sr = float(tr.stats.sampling_rate)
        if abs(sr - target_fs) > 1e-6:
            tr.interpolate(sampling_rate=target_fs, method="lanczos", a=12)
    st.merge(method=1, fill_value="interpolate")
    return st


def stream_to_C_T(st: Stream, channels: Tuple[str, ...], C_target: int):
    by_cha = {}
    for tr in st:
        ch = str(tr.stats.channel).upper()
        if ch not in by_cha:
            by_cha[ch] = tr

    avail = sorted(by_cha.keys())

    picked = []
    for ch in channels:
        ch_u = str(ch).upper()
        picked.append(by_cha.get(ch_u, None))

    ref_tr = next((tr for tr in picked if tr is not None), None)
    if ref_tr is None:
        raise RuntimeError(f"No requested channels available. Requested={channels}, Available={avail}")

    ref = ref_tr.data.astype(np.float32)
    T = len(ref)

    X_list = []
    chan_names = []

    for ch, tr in zip(channels, picked):
        ch_u = str(ch).upper()
        if tr is None:
            x = np.zeros(T, dtype=np.float32)
            chan_names.append(f"{ch_u}(ZERO)")
        else:
            x = tr.data.astype(np.float32)
            if len(x) != T:
                if len(x) > T:
                    x = x[:T]
                else:
                    x = np.pad(x, (0, T - len(x)))
            chan_names.append(ch_u)
        X_list.append(x)

    X = np.stack(X_list, axis=0)

    if X.shape[0] < C_target:
        pad = np.zeros((C_target - X.shape[0], T), dtype=np.float32)
        X = np.concatenate([X, pad], axis=0)
        chan_names += ["PAD_ZERO"] * (C_target - len(chan_names))
    elif X.shape[0] > C_target:
        X = X[:C_target]
        chan_names = chan_names[:C_target]

    return X, chan_names


def map_ENZ_from_chan_names(chan_names: List[str]) -> Tuple[str, str, str]:
    chE = ""
    chN = ""
    chZ = ""

    def base_name(s: str) -> str:
        return s.split("(")[0].strip()

    for nm in chan_names:
        b = base_name(nm).upper()
        if b.endswith("Z"):
            if not chZ:
                chZ = nm
        elif b.endswith("E") or b.endswith("1"):
            if not chE:
                chE = nm
        elif b.endswith("N") or b.endswith("2"):
            if not chN:
                chN = nm

    if (not chZ) and len(chan_names) == 1:
        chZ = chan_names[0]

    return chE, chN, chZ


def _overlap(a0: UTCDateTime, a1: UTCDateTime, b0: UTCDateTime, b1: UTCDateTime) -> bool:
    return (a0 < b1) and (b0 < a1)


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def _mad_abs(x: np.ndarray) -> float:
    xa = np.abs(np.asarray(x, dtype=np.float32))
    med = float(np.median(xa))
    return float(np.median(np.abs(xa - med)) + 1e-12)


def spike_metrics(x_1d: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(x_1d, dtype=np.float32)
    mad = _mad_abs(x)
    thr = float(U.SPIKE_K) * mad
    spike_ratio = float(np.mean(np.abs(x) > thr))
    crest = float(np.max(np.abs(x)) / (_rms(x) + 1e-12))
    return spike_ratio, crest


def lowfreq_gate_ratio(
    x_1d: np.ndarray,
    fs: float,
    low_floor: float,
    low_ceil: float,
    trem_min: float,
    trem_max: float,
) -> float:
    x_low = bandpass_np(x_1d, fs, low_floor, low_ceil)
    x_tr  = bandpass_np(x_1d, fs, trem_min, trem_max)
    r_low = _rms(x_low)
    r_tr  = _rms(x_tr)
    return float(r_tr / (r_low + 1e-12))


def tonal_narrowband_ratio(
    x_1d: np.ndarray,
    fs: float,
    fmin: float,
    fmax: float,
    bw_hz: float,
) -> float:
    x = np.asarray(x_1d, dtype=np.float32)
    x_bb = bandpass_np(x, fs, float(fmin), float(fmax))
    rms_bb = _rms(x_bb)

    bw = float(bw_hz) if float(bw_hz) > 0 else 0.5
    freqs = np.arange(float(fmin), float(fmax), bw)
    if freqs.size == 0:
        freqs = np.array([float(fmin)], dtype=np.float32)

    mx = 0.0
    for f0 in freqs:
        f1 = min(float(f0) + bw, float(fmax))
        if f1 <= f0:
            continue
        x_nb = bandpass_np(x, fs, float(f0), float(f1))
        mx = max(mx, _rms(x_nb))

    return float(mx / (rms_bb + 1e-12))


def load_exclude_intervals_from_shot_days(csv_path: str) -> List[Tuple[UTCDateTime, UTCDateTime]]:
    if (not csv_path) or (not os.path.exists(csv_path)):
        print(f"[WARN] SHOT_DAYS_CSV not found: {csv_path}")
        return []

    intervals: List[Tuple[UTCDateTime, UTCDateTime]] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        if r.fieldnames is None:
            print(f"[WARN] shot days csv has no header: {csv_path}")
            return []

        fns = [x.strip() for x in r.fieldnames]
        lower_map = {x.lower(): x for x in fns}

        start_key = None
        end_key = None
        for k in ["start_utc", "start", "t0", "begin_utc", "shot_start_utc"]:
            if k in lower_map:
                start_key = lower_map[k]
                break
        for k in ["end_utc", "end", "t1", "stop_utc", "shot_end_utc"]:
            if k in lower_map:
                end_key = lower_map[k]
                break

        day_key = None
        for k in ["day", "date", "utc_day", "shot_day", "yyyymmdd"]:
            if k in lower_map:
                day_key = lower_map[k]
                break

        for row in r:
            if start_key and end_key and row.get(start_key, "").strip() and row.get(end_key, "").strip():
                try:
                    t0 = UTCDateTime(row[start_key].strip())
                    t1 = UTCDateTime(row[end_key].strip())
                    if t1 > t0:
                        intervals.append((t0, t1))
                    continue
                except Exception:
                    pass

            s = ""
            if day_key and row.get(day_key, "").strip():
                s = row[day_key].strip()
            else:
                for v in row.values():
                    if v is not None and str(v).strip():
                        s = str(v).strip()
                        break

            if not s:
                continue

            try:
                if len(s) == 8 and s.isdigit():
                    day0 = UTCDateTime(f"{s[0:4]}-{s[4:6]}-{s[6:8]}T00:00:00")
                else:
                    day0 = UTCDateTime(f"{s.split()[0]}T00:00:00")
                day1 = day0 + 86400.0
                intervals.append((day0, day1))
            except Exception:
                print(f"[WARN] cannot parse shot day row: {row}")

    intervals.sort(key=lambda x: x[0].timestamp)
    merged: List[Tuple[UTCDateTime, UTCDateTime]] = []
    for t0, t1 in intervals:
        if not merged:
            merged.append((t0, t1))
            continue
        p0, p1 = merged[-1]
        if t0 <= p1:
            merged[-1] = (p0, max(p1, t1))
        else:
            merged.append((t0, t1))

    print(f"[INFO] Loaded exclude intervals from shot days: {len(merged)}")
    if len(merged) <= 10:
        for (a, b) in merged:
            print(f"  - EXCLUDE {a.isoformat()} -> {b.isoformat()}")
    return merged


# ============================================================
# Station list parsing + channel picking + edge trimming
# ============================================================

@dataclass
class StationRow:
    net: str
    sta: str
    lat: float
    lon: float
    elev: float
    sitename: str
    start: UTCDateTime
    end: UTCDateTime


def read_station_file(path: str) -> List[StationRow]:
    """
    Parse gmap-stations1.txt-like file:
    Lines like:
      7D|FN06C|46.922001|-124.7313|-137.0|...|2013-09-01T00:00:00|2014-06-29T23:59:59
    Skips # comments and blank lines.
    """
    if (not path) or (not os.path.exists(path)):
        raise FileNotFoundError(f"STATION_FILE not found: {path}")

    rows: List[StationRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            s = ln.strip()
            if (not s) or s.startswith("#"):
                continue
            parts = s.split("|")
            if len(parts) < 8:
                continue
            try:
                net = parts[0].strip()
                sta = parts[1].strip()
                lat = float(parts[2])
                lon = float(parts[3])
                elev = float(parts[4])
                sitename = parts[5].strip()
                t0 = UTCDateTime(parts[6].strip())
                t1 = UTCDateTime(parts[7].strip())
                if t1 <= t0:
                    continue
                rows.append(StationRow(net, sta, lat, lon, elev, sitename, t0, t1))
            except Exception:
                continue

    rows.sort(key=lambda r: (r.net, r.sta, r.start.timestamp))
    print(f"[INFO] Loaded {len(rows)} station rows from {path}")
    return rows


def trim_edges(t0: UTCDateTime, t1: UTCDateTime, n_days: int) -> Tuple[UTCDateTime, UTCDateTime]:
    if n_days <= 0:
        return t0, t1
    dt = float(n_days) * 86400.0
    a = t0 + dt
    b = t1 - dt
    return a, b


def intersect(a0: UTCDateTime, a1: UTCDateTime, b0: UTCDateTime, b1: UTCDateTime) -> Optional[Tuple[UTCDateTime, UTCDateTime]]:
    t0 = max(a0, b0)
    t1 = min(a1, b1)
    if t1 <= t0:
        return None
    return t0, t1


def pick_best_z_channel(
    client: Client,
    net: str,
    sta: str,
    loc: str,
    t0: UTCDateTime,
    t1: UTCDateTime,
    priority: Tuple[str, ...],
) -> Optional[str]:
    """
    Inventory-based channel availability check (fast).
    Returns best channel code (e.g., 'BHZ') or None.
    """
    locq = _safe_loc(loc)
    for ch in priority:
        chq = str(ch).upper()
        try:
            inv = client.get_stations(
                network=net,
                station=sta,
                location=locq,
                channel=chq,
                starttime=t0,
                endtime=t1,
                level="channel",
            )
            # if any channel present => accept
            ok = False
            for _net in inv:
                for _sta in _net:
                    for _ch in _sta:
                        if str(_ch.code).upper() == chq:
                            ok = True
                            break
                    if ok:
                        break
                if ok:
                    break
            if ok:
                return chq
        except Exception:
            continue
    return None


# ============================================================
# Model + Config
# ============================================================

_HAS_PYWT = False
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    pywt = None
    _HAS_PYWT = False


def apply_wavelet_transform(signal_1d: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    if pywt is None:
        raise RuntimeError("pywt not installed but USE_WAVELET=True")
    coeffs = pywt.wavedec(signal_1d, wavelet, level=level)
    reconstructed = []
    for i, c_i in enumerate(coeffs):
        zeros = [np.zeros_like(c) for c in coeffs]
        zeros[i] = c_i
        rec = pywt.waverec(zeros, wavelet)
        reconstructed.append(rec)
    min_len = min(len(r) for r in reconstructed)
    reconstructed = [r[:min_len] for r in reconstructed]
    return np.asarray(reconstructed)


@dataclass
class Config:
    INPUT_CHANNELS: int = 3
    SAMPLE_RATE: int = 100
    TARGET_LENGTH: int = 15000

    USE_WAVELET: bool = False
    WAVELET_TYPE: str = "db4"
    WAVELET_LEVEL: int = 3

    USE_SPECTROGRAM: bool = True
    STFT_N_FFT: int = 256
    STFT_HOP_LENGTH: int = 128

    DROPOUT: float = 0.3
    CONV_CHANNELS: Tuple[int, ...] = (8, 16, 32)
    LSTM_HIDDEN: int = 32
    LSTM_LAYERS: int = 1
    BIDIRECTIONAL: bool = True

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class ConvLSTMClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        if cfg.USE_SPECTROGRAM and cfg.USE_WAVELET:
            freq_bins = cfg.STFT_N_FFT // 2 + 1
            in_channels = cfg.INPUT_CHANNELS * ((cfg.WAVELET_LEVEL + 1) + freq_bins)
        elif cfg.USE_SPECTROGRAM:
            freq_bins = cfg.STFT_N_FFT // 2 + 1
            in_channels = cfg.INPUT_CHANNELS * freq_bins
        elif cfg.USE_WAVELET:
            in_channels = cfg.INPUT_CHANNELS * (cfg.WAVELET_LEVEL + 2)
        else:
            in_channels = cfg.INPUT_CHANNELS

        conv_layers = []
        in_ch = in_channels
        for out_ch in cfg.CONV_CHANNELS:
            conv_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(cfg.DROPOUT),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size=cfg.CONV_CHANNELS[-1],
            hidden_size=cfg.LSTM_HIDDEN,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,
            bidirectional=cfg.BIDIRECTIONAL,
            dropout=cfg.DROPOUT if cfg.LSTM_LAYERS > 1 else 0.0,
        )

        lstm_out_dim = cfg.LSTM_HIDDEN * (2 if cfg.BIDIRECTIONAL else 1)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, cfg.LSTM_HIDDEN),
            nn.ReLU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(cfg.LSTM_HIDDEN, 3),
        )

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.transpose(1, 2)
        _, (h_n, _) = self.lstm(conv_out)
        if self.cfg.BIDIRECTIONAL:
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        logits = self.classifier(h)
        return logits, None, None


def config_from_ckpt(ckpt_obj: Dict[str, Any]) -> Config:
    cfgd = ckpt_obj.get("config", {}) if isinstance(ckpt_obj, dict) else {}

    def g(k, default):
        return cfgd.get(k, default)

    return Config(
        INPUT_CHANNELS=int(g("INPUT_CHANNELS", 3)),
        SAMPLE_RATE=int(g("SAMPLE_RATE", 100)),
        TARGET_LENGTH=int(g("TARGET_LENGTH", 15000)),
        USE_WAVELET=bool(g("USE_WAVELET", False)),
        WAVELET_TYPE=str(g("WAVELET_TYPE", "db4")),
        WAVELET_LEVEL=int(g("WAVELET_LEVEL", 3)),
        USE_SPECTROGRAM=bool(g("USE_SPECTROGRAM", False)),
        STFT_N_FFT=int(g("STFT_N_FFT", 256)),
        STFT_HOP_LENGTH=int(g("STFT_HOP_LENGTH", 128)),
        DROPOUT=float(g("DROPOUT", 0.3)),
        CONV_CHANNELS=tuple(g("CONV_CHANNELS", [8, 16, 32])),
        LSTM_HIDDEN=int(g("LSTM_HIDDEN", 32)),
        LSTM_LAYERS=int(g("LSTM_LAYERS", 1)),
        BIDIRECTIONAL=bool(g("BIDIRECTIONAL", True)),
        DEVICE=("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    def strip_once(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
        if not sd:
            return sd
        if all(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix):]: v for k, v in sd.items()}
        return sd

    sd = state_dict
    sd = strip_once(sd, "module.")
    sd = strip_once(sd, "_orig_mod.")
    sd = strip_once(sd, "module.")
    return sd


def load_model(ckpt_path: str) -> Tuple[ConvLSTMClassifier, Config, Dict[str, Any]]:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location="cpu")

    cfg = config_from_ckpt(ckpt)

    if cfg.USE_WAVELET and (not _HAS_PYWT):
        raise RuntimeError("Checkpoint config has USE_WAVELET=True but pywt not installed.")

    model = ConvLSTMClassifier(cfg)

    sd = ckpt.get("model_state_dict", None)
    if sd is None:
        sd = ckpt.get("model", None)
    if sd is None:
        raise RuntimeError("Cannot find model_state_dict in checkpoint.")

    sd = _strip_state_dict_prefix(sd)
    model.load_state_dict(sd, strict=True)

    p0 = next(model.parameters()).detach().cpu()
    print(f"[INFO] state_dict loaded OK, first_param mean={p0.mean().item():.6g}, std={p0.std().item():.6g}")

    device = torch.device(cfg.DEVICE)
    model = model.to(device)
    model.eval()
    return model, cfg, ckpt


# ============================================================
# Feature construction
# ============================================================

def normalize_per_channel_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mean = np.mean(x, axis=1, keepdims=True)
    std = np.std(x, axis=1, keepdims=True) + 1e-6
    return _nan0((x - mean) / std)


def apply_stft_to_channels(x_win_normed: np.ndarray, cfg: Config) -> np.ndarray:
    n_fft = int(cfg.STFT_N_FFT)
    hop = int(cfg.STFT_HOP_LENGTH if cfg.STFT_HOP_LENGTH and cfg.STFT_HOP_LENGTH > 0 else n_fft // 2)

    window = torch.hann_window(n_fft)
    feats = []
    for c in range(x_win_normed.shape[0]):
        sig = torch.tensor(x_win_normed[c], dtype=torch.float32)
        spec = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True)
        feats.append(spec.abs().numpy().astype(np.float32))

    if not feats:
        return np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)

    return _nan0(np.vstack(feats))


def apply_wavelet_to_channels(x_win_normed: np.ndarray, cfg: Config) -> np.ndarray:
    feats = []
    for c in range(x_win_normed.shape[0]):
        recs = apply_wavelet_transform(
            x_win_normed[c].astype(np.float32, copy=False),
            wavelet=cfg.WAVELET_TYPE,
            level=int(cfg.WAVELET_LEVEL)
        )
        feats.append(recs.astype(np.float32, copy=False))
    x_wav = np.concatenate(feats, axis=0) if feats else np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)
    return _nan0(x_wav)


def build_model_input(x_win: np.ndarray, cfg: Config) -> np.ndarray:
    x_win = _nan0(x_win)
    x_win_normed = normalize_per_channel_z(x_win)

    x_wav = None
    x_spec = None

    if cfg.USE_WAVELET:
        x_wav = apply_wavelet_to_channels(x_win_normed, cfg)
    if cfg.USE_SPECTROGRAM:
        x_spec = apply_stft_to_channels(x_win_normed, cfg)

    if cfg.USE_WAVELET and cfg.USE_SPECTROGRAM:
        L_min = min(x_wav.shape[1], x_spec.shape[1])
        x_wav = x_wav[:, :L_min]
        x_spec = x_spec[:, :L_min]
        x_combined = np.vstack([x_spec, x_wav])
    elif cfg.USE_SPECTROGRAM:
        x_combined = x_spec
    elif cfg.USE_WAVELET:
        Lw = x_wav.shape[1]
        if x_win_normed.shape[1] != Lw:
            x_win_normed = x_win_normed[:, :Lw]
        x_combined = np.vstack([x_win_normed, x_wav])
    else:
        x_combined = x_win_normed

    return _nan0(x_combined).astype(np.float32)


# ============================================================
# Inference scan on a continuous record array
# ============================================================

@torch.no_grad()
def scan_record(
    model: nn.Module,
    cfg: Config,
    X_full: np.ndarray,
    step_sec: float,
) -> Dict[str, Any]:
    device = torch.device(cfg.DEVICE)
    _, Ttot = X_full.shape
    win = int(cfg.TARGET_LENGTH)
    step = int(round(step_sec * cfg.SAMPLE_RATE))
    if Ttot < win:
        raise RuntimeError(f"Record too short: {Ttot} < win={win} samples")

    starts = list(range(0, Ttot - win + 1, step))
    probs = np.zeros((len(starts), 3), dtype=np.float32)

    spiky_flags = np.zeros((len(starts),), dtype=np.bool_)
    spike_ratio_arr = np.full((len(starts),), np.nan, dtype=np.float32)
    spike_crest_arr = np.full((len(starts),), np.nan, dtype=np.float32)
    pre_suppress_score = np.full((len(starts),), np.nan, dtype=np.float32)

    gate_ratio_arr = None
    if bool(U.LOWFREQ_GATE):
        gate_ratio_arr = np.full((len(starts),), np.nan, dtype=np.float32)

    eq_gate_ratio_arr = None
    if bool(getattr(U, "EQ_LOWFREQ_GATE", False)):
        eq_gate_ratio_arr = np.full((len(starts),), np.nan, dtype=np.float32)

    tonal_ratio_arr = None
    tonal_flags = None
    if bool(getattr(U, "TONAL_GATE", False)):
        tonal_ratio_arr = np.full((len(starts),), np.nan, dtype=np.float32)
        tonal_flags = np.zeros((len(starts),), dtype=np.bool_)

    for i, s in enumerate(starts):
        x_win = X_full[:, s:s+win]

        # spike
        is_spiky = False
        if bool(getattr(U, "SPIKE_GATE", False)):
            x_ch0 = x_win[0].astype(np.float32, copy=False)
            mode = str(getattr(U, "SPIKE_BAND_MODE", "tremor")).lower()
            if mode == "tremor":
                x_for_spike = bandpass_np(x_ch0, float(cfg.SAMPLE_RATE), float(U.BP_FMIN), float(U.BP_FMAX))
            elif mode == "eq":
                x_for_spike = bandpass_np(x_ch0, float(cfg.SAMPLE_RATE), float(U.EQ_BP_FMIN), float(U.EQ_BP_FMAX))
            else:
                x_for_spike = x_ch0
            sp_ratio, sp_crest = spike_metrics(x_for_spike)
            if (sp_ratio >= float(U.SPIKE_RATIO_THR)) or (sp_crest >= float(U.SPIKE_CREST_THR)):
                is_spiky = True
            spiky_flags[i] = bool(is_spiky)
            spike_ratio_arr[i] = float(sp_ratio)
            spike_crest_arr[i] = float(sp_crest)

        # tremor gate ratio
        if bool(U.LOWFREQ_GATE):
            x_ch0 = x_win[0].astype(np.float32, copy=False)
            ratio = lowfreq_gate_ratio(
                x_ch0,
                fs=float(cfg.SAMPLE_RATE),
                low_floor=float(U.LOWFREQ_FLOOR_HZ),
                low_ceil=float(U.LOWFREQ_CEIL_HZ),
                trem_min=float(U.GATE_TREMOR_MIN_HZ),
                trem_max=float(U.GATE_TREMOR_MAX_HZ),
            )
            gate_ratio_arr[i] = float(ratio)

        # eq gate ratio
        if bool(getattr(U, "EQ_LOWFREQ_GATE", False)):
            x_ch0 = x_win[0].astype(np.float32, copy=False)
            eq_ratio = lowfreq_gate_ratio(
                x_ch0,
                fs=float(cfg.SAMPLE_RATE),
                low_floor=float(U.EQ_LOWFREQ_FLOOR_HZ),
                low_ceil=float(U.EQ_LOWFREQ_CEIL_HZ),
                trem_min=float(U.EQ_GATE_MIN_HZ),
                trem_max=float(U.EQ_GATE_MAX_HZ),
            )
            eq_gate_ratio_arr[i] = float(eq_ratio)

        # tonal
        if bool(getattr(U, "TONAL_GATE", False)):
            x_ch0 = x_win[0].astype(np.float32, copy=False)
            tr = tonal_narrowband_ratio(
                x_ch0,
                fs=float(cfg.SAMPLE_RATE),
                fmin=float(getattr(U, "TONAL_FMIN", 2.0)),
                fmax=float(getattr(U, "TONAL_FMAX", 10.0)),
                bw_hz=float(getattr(U, "TONAL_BW_HZ", 0.5)),
            )
            tonal_ratio_arr[i] = float(tr)
            tonal_flags[i] = bool(tr >= float(getattr(U, "TONAL_RATIO_THR", 0.6)))

        x_in = build_model_input(x_win, cfg)
        xt = torch.from_numpy(x_in[None, ...]).to(device)
        logits, _, _ = model(xt)
        p = torch.softmax(logits, dim=1).cpu().numpy()[0].astype(np.float32)
        pre_suppress_score[i] = float(max(p[1], p[2]))

        # tremor suppress
        if bool(U.LOWFREQ_GATE):
            ratio = float(gate_ratio_arr[i])
            if ratio < float(U.LOWFREQ_RATIO_THR):
                p = p.copy()
                p[2] = 0.0
                p = p / (np.sum(p) + 1e-12)

        # eq suppress
        if bool(getattr(U, "EQ_LOWFREQ_GATE", False)) and bool(getattr(U, "DO_EQ", True)):
            eq_ratio = float(eq_gate_ratio_arr[i])
            if eq_ratio < float(U.EQ_LOWFREQ_RATIO_THR):
                p = p.copy()
                p[1] = 0.0
                p = p / (np.sum(p) + 1e-12)

        # tonal veto
        if bool(getattr(U, "TONAL_GATE", False)) and (tonal_flags is not None) and bool(tonal_flags[i]):
            p = p.copy()
            if bool(getattr(U, "TONAL_VETO_TREMOR", True)):
                p[2] = 0.0
            if bool(getattr(U, "TONAL_VETO_EQ", False)):
                p[1] = 0.0
            p = p / (np.sum(p) + 1e-12)

        # unknown (spike) override
        if bool(getattr(U, "SPIKE_GATE", False)) and bool(spiky_flags[i]):
            p = p.copy()
            p[1] = 0.0
            p[2] = 0.0
            p = p / (np.sum(p) + 1e-12)

        probs[i] = p

    tremor_p = probs[:, 2]
    eq_p = probs[:, 1]
    pred_idx = np.argmax(probs, axis=1).astype(np.int64)

    def _agg(x: np.ndarray, mode: str) -> float:
        if mode == "max":
            return float(np.max(x))
        if mode == "mean":
            return float(np.mean(x))
        raise ValueError("AGG_MODE must be 'max' or 'mean'")

    tremor_thr = float(U.TREMOR_PROB_THRESHOLD)
    eq_thr = float(U.EQ_PROB_THRESHOLD)

    tremor_intervals = []
    eq_intervals = []
    unknown_intervals = []
    shiptonal_intervals = []

    for i, s in enumerate(starts):
        a = s / float(cfg.SAMPLE_RATE)
        b = (s + win) / float(cfg.SAMPLE_RATE)

        is_tonal = (
            bool(getattr(U, "DO_SHIPTONAL", True))
            and bool(getattr(U, "TONAL_GATE", False))
            and (tonal_flags is not None)
            and bool(tonal_flags[i])
        )
        if is_tonal:
            shiptonal_intervals.append((a, b, float(tonal_ratio_arr[i]) if tonal_ratio_arr is not None else 1.0))
            continue

        if bool(getattr(U, "DO_TREMOR", True)) and float(tremor_p[i]) >= tremor_thr:
            tremor_intervals.append((a, b, float(tremor_p[i])))

        if bool(getattr(U, "DO_EQ", True)) and float(eq_p[i]) >= eq_thr:
            eq_intervals.append((a, b, float(eq_p[i])))

        if (
            bool(getattr(U, "SPIKE_GATE", False))
            and bool(getattr(U, "DO_UNKNOWN", False))
            and bool(spiky_flags[i])
        ):
            if float(pre_suppress_score[i]) >= float(getattr(U, "UNKNOWN_SCORE_THR", 0.5)):
                unknown_intervals.append((a, b, float(pre_suppress_score[i])))

    out = {
        "starts_samples": np.array(starts, dtype=np.int64),
        "probs": probs,
        "pred_idx": pred_idx,
        "tremor_p": tremor_p,
        "eq_p": eq_p,

        "tremor_agg_score": _agg(tremor_p, str(getattr(U, "AGG_MODE", "max"))),
        "eq_agg_score": _agg(eq_p, str(getattr(U, "EQ_AGG_MODE", "max"))),

        "tremor_present": (_agg(tremor_p, str(getattr(U, "AGG_MODE", "max"))) >= tremor_thr),
        "eq_present": (_agg(eq_p, str(getattr(U, "EQ_AGG_MODE", "max"))) >= eq_thr),

        "tremor_intervals": tremor_intervals,
        "eq_intervals": eq_intervals,

        "win_samples": win,
        "step_samples": step,
        "best_tr_idx": int(np.argmax(tremor_p)) if len(tremor_p) else 0,
        "best_eq_idx": int(np.argmax(eq_p)) if len(eq_p) else 0,

        "unknown_intervals": unknown_intervals,
        "spiky_flags": spiky_flags,
        "spike_ratio": spike_ratio_arr,
        "spike_crest": spike_crest_arr,
        "pre_suppress_score": pre_suppress_score,
        "shiptonal_intervals": shiptonal_intervals,
    }
    if gate_ratio_arr is not None:
        out["gate_ratio"] = gate_ratio_arr
    if eq_gate_ratio_arr is not None:
        out["eq_gate_ratio"] = eq_gate_ratio_arr
    if tonal_ratio_arr is not None:
        out["tonal_ratio"] = tonal_ratio_arr
    if tonal_flags is not None:
        out["tonal_flags"] = tonal_flags

    return out


def merge_intervals(intervals: List[Tuple[float, float, float]], gap: float = 0.0) -> List[Tuple[float, float, float]]:
    if not intervals:
        return []
    xs = sorted(intervals, key=lambda x: x[0])
    out = []
    cur_a, cur_b, cur_p = xs[0]
    for a, b, p in xs[1:]:
        if a <= cur_b + gap:
            cur_b = max(cur_b, b)
            cur_p = max(cur_p, p)
        else:
            out.append((cur_a, cur_b, cur_p))
            cur_a, cur_b, cur_p = a, b, p
    out.append((cur_a, cur_b, cur_p))
    return out


# ============================================================
# Plotting
# ============================================================

def compute_display_stft(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, tt, Z = scipy_stft(
        x, fs=fs,
        nperseg=int(U.DISP_STFT_NPERSEG),
        noverlap=int(U.DISP_STFT_NOVERLAP),
        detrend=False,
        padded=False,
        boundary=None,
    )
    S = np.abs(Z)
    return f, tt, S


def plot_paper_style_summary_event(
    t_rel: np.ndarray,
    x_raw_ch0: np.ndarray,
    x_bp_ch0: np.ndarray,
    fs: float,
    header: str,
    merged_intervals_rel: List[Tuple[float, float, float]],
    outpath: str,
    label_prefix: str = "p",
):
    f, tt, S = compute_display_stft(x_raw_ch0, fs)
    fmask = f <= float(U.DISP_STFT_MAX_FREQ)
    f2 = f[fmask]
    S2 = S[fmask, :]

    S2_db = 20.0 * np.log10(S2 + 1e-12)
    vmax = float(np.percentile(S2_db, 99.5))
    dyn_range_db = 50.0
    vmin = vmax - dyn_range_db

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1.3], hspace=0.10)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(t_rel, x_raw_ch0, linewidth=0.8)
    ax0.set_ylabel("Raw (Ch0)")
    ax0.set_title(header, fontsize=11)

    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    ax1.plot(t_rel, x_bp_ch0, linewidth=0.8)
    ax1.set_ylabel("Bandpassed (Ch0)")

    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    im = ax2.pcolormesh(tt + t_rel[0], f2, S2_db, shading="auto", vmin=vmin, vmax=vmax)
    ax2.set_ylabel("Freq (Hz)")
    ax2.set_xlabel("Time (s, rel to window start)")
    cbar = fig.colorbar(im, ax=ax2, pad=0.01)
    cbar.set_label("STFT amplitude (dB)")

    for (a, b, p) in merged_intervals_rel:
        a2 = max(0.0, float(a))
        b2 = min(float(U.EVENT_WINDOW_SEC), float(b))
        if b2 <= a2:
            continue
        for ax in (ax0, ax1, ax2):
            ax.axvspan(a2, b2, alpha=0.18)
        ax0.text(
            (a2 + b2) / 2.0, 0.95, f"{label_prefix}={p:.2f}",
            transform=ax0.get_xaxis_transform(),
            ha="center", va="top", fontsize=8
        )

    for ax in (ax0, ax1, ax2):
        ax.set_xlim(0.0, float(U.EVENT_WINDOW_SEC))

    fig.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08, hspace=0.18)
    fig.savefig(outpath, dpi=250)
    plt.close(fig)


# ============================================================
# Download helpers
# ============================================================

def get_station_meta(client: Client, t0: UTCDateTime, t1: UTCDateTime) -> Tuple[Dict[str, Any], Any]:
    locq = _safe_loc(U.LOC)
    chaq = ",".join(U.CHANNELS)
    meta: Dict[str, Any] = {}
    inv = None
    try:
        inv = client.get_stations(
            network=U.NET, station=U.STA, location=locq, channel=chaq,
            starttime=t0, endtime=t1, level="channel"
        )
        net0 = inv[0]
        sta0 = net0[0]
        meta["network"] = net0.code
        meta["station"] = sta0.code
        meta["latitude"] = float(sta0.latitude)
        meta["longitude"] = float(sta0.longitude)
        meta["elevation_m"] = float(sta0.elevation)
    except Exception as e:
        print(f"[WARN] get_stations failed: {e}")
    return meta, inv


def download_window(
    client: Client,
    t0: UTCDateTime,
    t1: UTCDateTime,
    attach_response: bool,
) -> Stream:
    locq = _safe_loc(U.LOC)
    chaq = ",".join(U.CHANNELS)
    st = client.get_waveforms(
        network=U.NET, station=U.STA, location=locq, channel=chaq,
        starttime=t0, endtime=t1, attach_response=bool(attach_response)
    )
    st.merge(method=1, fill_value="interpolate")
    st = detrend_basic(st)
    return st


# ============================================================
# Event bucketing logic (fixed 300s windows)
# ============================================================

def bucket_event_windows(
    detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]],
    scan_start: UTCDateTime,
    event_window_sec: float,
) -> Dict[int, Dict[str, Any]]:
    W = float(event_window_sec)
    bins: Dict[int, Dict[str, Any]] = {}

    for a, b, p in detections_abs:
        center = a + (b - a) * 0.5
        k = int(np.floor((center - scan_start) / W))
        if k not in bins:
            t0 = scan_start + k * W
            t1 = t0 + W
            bins[k] = {"t0": t0, "t1": t1, "detections": [], "p_max": -1.0}
        bins[k]["detections"].append((a, b, float(p)))
        bins[k]["p_max"] = max(bins[k]["p_max"], float(p))

    return bins


def detections_to_rel(
    detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]],
    win_start: UTCDateTime,
    win_sec: float,
) -> List[Tuple[float, float, float]]:
    out = []
    for a, b, p in detections_abs:
        ar = float(a - win_start)
        br = float(b - win_start)

        ar2 = max(0.0, ar)
        br2 = min(float(win_sec), br)

        if br2 <= 0.0 or ar2 >= float(win_sec) or br2 <= ar2:
            continue

        out.append((ar2, br2, float(p)))
    return out


def run_one_station(
    client: Client,
    model: nn.Module,
    cfg: Config,
    ckpt: Dict[str, Any],
    scan_start: UTCDateTime,
    scan_end: UTCDateTime,
    station_code: str,
    base_out_dir: str,
) -> List[List[str]]:
    U.STA = station_code

    station_dir = base_out_dir
    if bool(getattr(U, "PER_STATION_SUBDIR", True)):
        loc_disp = U.LOC.strip() if U.LOC and U.LOC.strip() else "--"
        station_dir = os.path.join(base_out_dir, f"{U.NET}.{U.STA}.{loc_disp}")
        os.makedirs(station_dir, exist_ok=True)

    prev_out = U.OUT_DIR
    U.OUT_DIR = station_dir

    try:
        meta, inv = get_station_meta(client, scan_start, scan_start + 60.0)
        event_lat = meta.get("latitude", None)
        event_lon = meta.get("longitude", None)

        tremor_detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]] = []
        eq_detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]] = []
        unknown_detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]] = []
        shiptonal_detections_abs: List[Tuple[UTCDateTime, UTCDateTime, float]] = []

        chunk = float(U.CHUNK_SEC)
        overlap = float(U.CHUNK_OVERLAP_SEC)
        fs = float(cfg.SAMPLE_RATE)

        cur0 = scan_start
        idx_chunk = 0

        exclude_intervals: List[Tuple[UTCDateTime, UTCDateTime]] = []
        if bool(getattr(U, "EXCLUDE_SHOT_DAYS", False)):
            exclude_intervals = load_exclude_intervals_from_shot_days(getattr(U, "SHOT_DAYS_CSV", ""))

        while cur0 < scan_end:
            idx_chunk += 1
            cur1 = min(cur0 + chunk, scan_end)
            dl0 = max(scan_start, cur0 - overlap)
            dl1 = min(scan_end, cur1 + overlap)

            if exclude_intervals:
                skip = any(_overlap(cur0, cur1, ex0, ex1) for (ex0, ex1) in exclude_intervals)
                if skip:
                    print(f"[SKIP] {U.NET}.{U.STA} chunk#{idx_chunk} overlaps shot-day exclusion: {cur0.isoformat()} -> {cur1.isoformat()}")
                    cur0 = cur1
                    continue

            print(f"[SCAN] {U.NET}.{U.STA} chunk#{idx_chunk} request {dl0.isoformat()} -> {dl1.isoformat()}  CH={','.join(U.CHANNELS)}")

            try:
                st = download_window(client, dl0, dl1, attach_response=bool(U.REMOVE_RESPONSE))
            except Exception as e:
                print(f"[WARN] {U.NET}.{U.STA} chunk download failed ({dl0} -> {dl1}): {e}")
                cur0 = cur1
                continue

            if U.REMOVE_RESPONSE and inv is not None:
                try:
                    st.remove_response(inventory=inv, output=U.RESPONSE_OUTPUT, zero_mean=False, taper=True, pre_filt=None)
                except Exception as e:
                    print(f"[WARN] {U.NET}.{U.STA} remove_response failed, continue with raw counts: {e}")

            st = resample_stream(st, float(cfg.SAMPLE_RATE))

            try:
                X, chan_names = stream_to_C_T(st, U.CHANNELS, C_target=int(cfg.INPUT_CHANNELS))
            except Exception as e:
                print(f"[WARN] {U.NET}.{U.STA} channel pack failed for chunk#{idx_chunk}: {e}")
                cur0 = cur1
                continue

            try:
                res = scan_record(model, cfg, X, step_sec=float(U.STEP_SEC))
            except Exception as e:
                print(f"[WARN] {U.NET}.{U.STA} scan_record failed for chunk#{idx_chunk}: {e}")
                cur0 = cur1
                continue

            if bool(U.DO_TREMOR):
                for (a_s, b_s, p) in res["tremor_intervals"]:
                    a_abs = dl0 + float(a_s)
                    b_abs = dl0 + float(b_s)
                    center = a_abs + (b_abs - a_abs) * 0.5
                    if (center >= cur0) and (center < cur1):
                        tremor_detections_abs.append((a_abs, b_abs, float(p)))

            if bool(U.DO_EQ):
                for (a_s, b_s, p) in res["eq_intervals"]:
                    a_abs = dl0 + float(a_s)
                    b_abs = dl0 + float(b_s)
                    center = a_abs + (b_abs - a_abs) * 0.5
                    if (center >= cur0) and (center < cur1):
                        eq_detections_abs.append((a_abs, b_abs, float(p)))

            if bool(getattr(U, "SPIKE_GATE", False)) and bool(getattr(U, "DO_UNKNOWN", False)):
                for (a_s, b_s, p) in res.get("unknown_intervals", []):
                    a_abs = dl0 + float(a_s)
                    b_abs = dl0 + float(b_s)
                    center = a_abs + (b_abs - a_abs) * 0.5
                    if (center >= cur0) and (center < cur1):
                        unknown_detections_abs.append((a_abs, b_abs, float(p)))

            if bool(getattr(U, "DO_SHIPTONAL", True)):
                for (a_s, b_s, r) in res.get("shiptonal_intervals", []):
                    a_abs = dl0 + float(a_s)
                    b_abs = dl0 + float(b_s)
                    center = a_abs + (b_abs - a_abs) * 0.5
                    if (center >= cur0) and (center < cur1):
                        shiptonal_detections_abs.append((a_abs, b_abs, float(r)))

            print(f"[SCAN] {U.NET}.{U.STA} chunk#{idx_chunk} kept: "
                  f"tremor={len(tremor_detections_abs)}  eq={len(eq_detections_abs)}  "
                  f"ship={len(shiptonal_detections_abs)}  unk={len(unknown_detections_abs)} (cumulative)")

            cur0 = cur1

        print(f"[INFO] {U.NET}.{U.STA} total detection windows: tremor={len(tremor_detections_abs)}  "
              f"eq={len(eq_detections_abs)}  ship={len(shiptonal_detections_abs)}  unk={len(unknown_detections_abs)}")

        bins_tremor = bucket_event_windows(tremor_detections_abs, scan_start=scan_start, event_window_sec=float(U.EVENT_WINDOW_SEC)) if bool(U.DO_TREMOR) else {}
        bins_eq = bucket_event_windows(eq_detections_abs, scan_start=scan_start, event_window_sec=float(U.EVENT_WINDOW_SEC)) if bool(U.DO_EQ) else {}
        bins_unk = bucket_event_windows(unknown_detections_abs, scan_start=scan_start, event_window_sec=float(U.EVENT_WINDOW_SEC)) if (bool(getattr(U, "SPIKE_GATE", False)) and bool(getattr(U, "DO_UNKNOWN", False))) else {}
        bins_ship = bucket_event_windows(shiptonal_detections_abs, scan_start=scan_start, event_window_sec=float(U.EVENT_WINDOW_SEC)) if bool(getattr(U, "DO_SHIPTONAL", True)) else {}

        events: List[Dict[str, Any]] = []
        for k, info in bins_tremor.items():
            events.append({"event_type": "TREMOR", "k": k, "t0": info["t0"], "t1": info["t1"], "detections": info["detections"], "p_max": info["p_max"]})
        for k, info in bins_eq.items():
            events.append({"event_type": "EARTHQUAKE", "k": k, "t0": info["t0"], "t1": info["t1"], "detections": info["detections"], "p_max": info["p_max"]})
        for k, info in bins_ship.items():
            events.append({"event_type": "SHIPTONAL", "k": k, "t0": info["t0"], "t1": info["t1"], "detections": info["detections"], "p_max": info["p_max"]})
        if bool(getattr(U, "DO_UNKNOWN", False)):
            for k, info in bins_unk.items():
                events.append({"event_type": "UNKNOWN", "k": k, "t0": info["t0"], "t1": info["t1"], "detections": info["detections"], "p_max": info["p_max"]})

        events.sort(key=lambda e: (e["t0"], e["event_type"]))

        out_csv = os.path.join(U.OUT_DIR, "events.csv")
        if not events:
            print(f"[RESULT] {U.NET}.{U.STA} No windows above thresholds. No plots generated.")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([
                    'event_uid','event_type','event_start_iso','event_lat','event_lon',
                    't_abs0_iso','t_abs1_iso','window_s',
                    'network','station','band','chE','chN','chZ','sampling_rate_hz',
                    'pmax','n_detections'
                ])
            print(f"[INFO] Wrote empty {out_csv}")
            return []

        if int(U.MAX_PLOTS) != -1:
            events = events[:int(U.MAX_PLOTS)]

        out_json = os.path.join(U.OUT_DIR, "scan_summary.json")
        json_obj = {
            "user": U.__dict__,
            "cfg": cfg.__dict__,
            "meta": meta,
            "ckpt_meta": {
                "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
                "val_loss": float(ckpt.get("val_loss", float("nan"))) if isinstance(ckpt, dict) else float("nan"),
            },
            "total_detection_windows": {
                "tremor": len(tremor_detections_abs),
                "earthquake": len(eq_detections_abs),
                "unknown": len(unknown_detections_abs),
                "shiptonal": len(shiptonal_detections_abs),
            },
            "total_event_bins": {
                "tremor": len(bins_tremor),
                "earthquake": len(bins_eq),
                "unknown": len(bins_unk),
                "shiptonal": len(bins_ship),
            },
            "exported_event_rows": len(events),
            "used_channels": list(U.CHANNELS),
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(json_obj, f, indent=2)
        print(f"[INFO] Wrote {out_json}")

        csv_rows: List[List[str]] = []

        for j, ev in enumerate(events, start=1):
            event_type = str(ev["event_type"])
            t0 = ev["t0"]
            t1 = ev["t1"]
            dets_abs = ev["detections"]
            pmax = float(ev["p_max"])

            # by your request: DO NOT plot shiptonal
            if event_type == "SHIPTONAL":
                continue

            loc_disp = U.LOC.strip() if U.LOC and U.LOC.strip() else "--"
            event_uid = f"{U.NET}.{U.STA}.{loc_disp}__{event_type}__{t0.strftime('%Y%m%dT%H%M%S')}__W{int(U.EVENT_WINDOW_SEC)}s"
            event_start_iso = t0.isoformat()

            print(f"[EVENT] {U.NET}.{U.STA} {j}/{len(events)}  {event_uid}  pmax={pmax:.3f}  dets_in_bin={len(dets_abs)}")

            try:
                st_ev = download_window(client, t0, t1, attach_response=bool(U.REMOVE_RESPONSE))
            except Exception as e:
                print(f"[WARN] {U.NET}.{U.STA} event download failed {t0} -> {t1}: {e}")
                continue

            if U.REMOVE_RESPONSE and inv is not None:
                try:
                    st_ev.remove_response(inventory=inv, output=U.RESPONSE_OUTPUT, zero_mean=False, taper=True, pre_filt=None)
                except Exception as e:
                    print(f"[WARN] {U.NET}.{U.STA} remove_response failed in event window, continue with raw counts: {e}")

            st_ev = resample_stream(st_ev, float(cfg.SAMPLE_RATE))

            try:
                Xev, chan_names_ev = stream_to_C_T(st_ev, U.CHANNELS, C_target=int(cfg.INPUT_CHANNELS))
            except Exception as e:
                print(f"[WARN] {U.NET}.{U.STA} channel pack failed for event {event_uid}: {e}")
                continue

            ch0_raw = Xev[0].astype(np.float32)

            if event_type == "EARTHQUAKE":
                bp_fmin = float(U.EQ_BP_FMIN)
                bp_fmax = float(U.EQ_BP_FMAX)
            elif event_type == "UNKNOWN":
                bp_fmin = float(getattr(U, "UNKNOWN_BP_FMIN", 2.0))
                bp_fmax = float(getattr(U, "UNKNOWN_BP_FMAX", 15.0))
            else:
                bp_fmin = float(U.BP_FMIN)
                bp_fmax = float(U.BP_FMAX)

            ch0_bp = bandpass_np(ch0_raw, float(cfg.SAMPLE_RATE), bp_fmin, bp_fmax)

            Ttot = Xev.shape[1]
            t_rel = np.arange(Ttot, dtype=np.float32) / float(cfg.SAMPLE_RATE)

            dets_rel = detections_to_rel(dets_abs, win_start=t0, win_sec=float(U.EVENT_WINDOW_SEC))
            merged_rel = merge_intervals(dets_rel, gap=0.0)

            header = (
                f"{U.NET}.{U.STA}.{loc_disp}  Type={event_type}  Chans: {','.join(chan_names_ev)}  Fs={float(cfg.SAMPLE_RATE):.1f} Hz\n"
                f"EventWindow: {t0.isoformat()} — {t1.isoformat()} | pmax={pmax:.3f} | dets={len(dets_abs)} | BP={bp_fmin:.1f}-{bp_fmax:.1f} Hz"
            )

            out_fig = os.path.join(U.OUT_DIR, f"{event_uid}.paper_style_summary.{U.SAVE_FMT}")
            plot_paper_style_summary_event(
                t_rel=t_rel,
                x_raw_ch0=ch0_raw,
                x_bp_ch0=ch0_bp,
                fs=float(cfg.SAMPLE_RATE),
                header=header,
                merged_intervals_rel=merged_rel,
                outpath=out_fig,
                label_prefix="p",
            )
            print(f"[INFO] Saved {out_fig}")

            chE, chN, chZ = map_ENZ_from_chan_names(chan_names_ev)
            csv_rows.append([
                event_uid,
                event_type,
                event_start_iso,
                "" if event_lat is None else f"{float(event_lat):.6f}",
                "" if event_lon is None else f"{float(event_lon):.6f}",
                t0.isoformat(),
                t1.isoformat(),
                f"{float(U.EVENT_WINDOW_SEC):.1f}",
                U.NET,
                U.STA,
                f"{bp_fmin:.1f}-{bp_fmax:.1f}Hz",
                chE,
                chN,
                chZ,
                f"{float(cfg.SAMPLE_RATE):.1f}",
                f"{pmax:.3f}",
                str(len(dets_abs)),
            ])

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                'event_uid','event_type','event_start_iso','event_lat','event_lon',
                't_abs0_iso','t_abs1_iso','window_s',
                'network','station','band','chE','chN','chZ','sampling_rate_hz',
                'pmax','n_detections'
            ])
            w.writerows(csv_rows)

        print(f"[INFO] {U.NET}.{U.STA} Wrote {out_csv}")
        return csv_rows

    finally:
        U.OUT_DIR = prev_out


# ============================================================
# Main
# ============================================================

def main():
    _maybe_override_run_cfg_from_env()
    _maybe_override_threshold_from_env()

    model, cfg, ckpt = load_model(U.CKPT_PATH)

    run_root = make_run_subdir_short(U.OUT_DIR, cfg_sr=float(cfg.SAMPLE_RATE))
    os.makedirs(run_root, exist_ok=True)
    print(f"[INFO] RUN_ROOT = {run_root}")

    # global scan window
    g0 = UTCDateTime(U.START_UTC)
    g1 = UTCDateTime(U.END_UTC)
    if g1 <= g0:
        raise ValueError("END_UTC must be after START_UTC")

    if bool(U.EXCLUDE_EDGE_DAYS):
        g0t, g1t = trim_edges(g0, g1, int(U.EDGE_DAYS))
        if g1t <= g0t:
            raise ValueError("After excluding edge days, global scan window is empty.")
        print(f"[INFO] Global scan trimmed edges: {g0.isoformat()}->{g1.isoformat()}  =>  {g0t.isoformat()}->{g1t.isoformat()}")
        g0, g1 = g0t, g1t
    else:
        print(f"[INFO] Global scan: {g0.isoformat()} -> {g1.isoformat()}")

    print(f"[INFO] step={U.STEP_SEC}s  event_window_sec={U.EVENT_WINDOW_SEC:.1f}  max_plots={U.MAX_PLOTS}")
    print(f"[INFO] DO_TREMOR={U.DO_TREMOR}  TREMOR_P={U.TREMOR_PROB_THRESHOLD:.3f}  agg_mode={U.AGG_MODE}")
    print(f"[INFO] DO_EQ={U.DO_EQ}  EQ_P={U.EQ_PROB_THRESHOLD:.3f}  eq_agg_mode={U.EQ_AGG_MODE}")
    print(f"[INFO] Channel priority (Z): {U.CHANNEL_PRIORITY}  FORCE_CHANNELS={U.FORCE_CHANNELS}")

    client = Client("IRIS")

    # Build station plan
    station_plan: List[Tuple[str, str, UTCDateTime, UTCDateTime]] = []  # (net, sta, t0, t1)

    if bool(U.USE_STATION_FILE):
        rows = read_station_file(U.STATION_FILE)
        # only keep rows for target network if U.NET is set; otherwise run all nets in file
        target_net = U.NET.strip() if U.NET and U.NET.strip() else None

        for r in rows:
            if target_net and r.net != target_net:
                continue

            # per-station deployment window, trimmed edges
            s0, s1 = r.start, r.end
            if bool(U.EXCLUDE_EDGE_DAYS):
                s0, s1 = trim_edges(s0, s1, int(U.EDGE_DAYS))

            if s1 <= s0:
                continue

            inter = intersect(g0, g1, s0, s1)
            if not inter:
                continue
            t0, t1 = inter
            station_plan.append((r.net, r.sta, t0, t1))

        if not station_plan:
            raise RuntimeError("No stations remain after intersecting global scan, deployment windows, and edge trimming.")

        print(f"[INFO] Station plan size after trimming/intersection: {len(station_plan)}")
        if len(station_plan) <= 30:
            for net, sta, t0, t1 in station_plan:
                print(f"  - {net}.{sta}: {t0.isoformat()} -> {t1.isoformat()}")

    else:
        # fallback
        if U.STA_LIST:
            station_plan = [(U.NET, sta, g0, g1) for sta in U.STA_LIST]
        elif U.STA and U.STA.strip():
            station_plan = [(U.NET, U.STA.strip(), g0, g1)]
        elif bool(U.AUTO_DISCOVER):
            # optional: inventory discover by channel priority first entry
            # but you said you have a station list; so keep simple
            raise ValueError("AUTO_DISCOVER not implemented in this trimmed station-plan mode. Use STATION_FILE or STA_LIST.")
        else:
            raise ValueError("No stations specified.")

    network_rows: List[List[str]] = []

    for i, (net, sta, t0, t1) in enumerate(station_plan, start=1):
        U.NET = net
        U.STA = sta

        # choose channel(s)
        if U.FORCE_CHANNELS:
            U.CHANNELS = tuple([c.strip().upper() for c in U.FORCE_CHANNELS])
            chosen = ",".join(U.CHANNELS)
        else:
            best = pick_best_z_channel(
                client=client,
                net=net,
                sta=sta,
                loc=U.LOC,
                t0=t0,
                t1=t1,
                priority=U.CHANNEL_PRIORITY,
            )
            if best is None:
                print(f"[SKIP] {net}.{sta} no usable Z channel found in {t0.isoformat()} -> {t1.isoformat()}")
                continue
            U.CHANNELS = (best,)
            chosen = best

        print(f"\n[NETSCAN] {i}/{len(station_plan)}  Station={net}.{sta}  Window={t0.isoformat()}->{t1.isoformat()}  CH={chosen}")

        try:
            rows = run_one_station(
                client=client,
                model=model,
                cfg=cfg,
                ckpt=ckpt,
                scan_start=t0,
                scan_end=t1,
                station_code=sta,
                base_out_dir=run_root,
            )
            if bool(getattr(U, "WRITE_NETWORK_CSV", True)) and rows:
                loc_disp = U.LOC.strip() if U.LOC and U.LOC.strip() else "--"
                station_dir = os.path.join(run_root, f"{net}.{sta}.{loc_disp}") if bool(getattr(U, "PER_STATION_SUBDIR", True)) else run_root
                for r in rows:
                    network_rows.append(r + [station_dir, chosen, t0.isoformat(), t1.isoformat()])
        except Exception as e:
            print(f"[WARN] Station {net}.{sta} failed: {e}")

    if bool(getattr(U, "WRITE_NETWORK_CSV", True)):
        out_net_csv = os.path.join(run_root, "network_events.csv")
        with open(out_net_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                'event_uid','event_type','event_start_iso','event_lat','event_lon',
                't_abs0_iso','t_abs1_iso','window_s',
                'network','station','band','chE','chN','chZ','sampling_rate_hz',
                'pmax','n_detections',
                'station_out_dir','chosen_channel','scan_start_iso','scan_end_iso'
            ])
            w.writerows(network_rows)
        print(f"[INFO] Wrote {out_net_csv}")

    print("[DONE]")


if __name__ == "__main__":
    main()