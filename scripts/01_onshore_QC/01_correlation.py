#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, Set, Dict
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import pickle
from obspy import UTCDateTime, Trace
from obspy.clients.fdsn import Client
from obspy.signal.filter import envelope, bandpass


#------------------------
# Config
#------------------------

CHANNELS_CSV = "channels_epoch_2009_2025Sep.csv"
EVENTS_CSV = "tremor_events-2009-08-06T00_00_00-2025-09-10T23_59_59.csv"
OUT_DIR = "./tremor_simple_out"

EVENT_TIME_RANGE: Optional[Tuple[str, str]] = ("2009-09-01T00:00:00", "2009-10-01T00:00:00")
MAX_EVENTS: Optional[int] = None

MAX_DIST_KM = 100.0
MAX_STATIONS = None
BAND_PRIORITY = ["HH", "BH", "EH", "EN", "HN", "BN", "SN"]

PAD_BEFORE = 300
PAD_AFTER = 300

BANDPASS_HZ = (1.0, 8.0)
ENVELOPE_LP_HZ = 0.1
DS_RATE_HZ = 1.0

XCORR_MAX_LAG_S = 60.0
HIT_THRESHOLD = 0.7

CWT_SCALES = np.logspace(0, 2, 50)
CWT_WAVELET = "mexh"

FDSN_CLIENTS = ["IRIS"]

SAVE_RAW_WAVEFORMS = False

STFT_Z_WIN_S = 4.0
STFT_Z_STEP_S = 1.0
STFT_Z_FMIN = 1.0
STFT_Z_FMAX = 8.0


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
        rows.append(
            {
                "network": net,
                "station": sta,
                "latitude": lat0,
                "longitude": lon0,
                "chE": info["E"],
                "chN": info["N"],
                "chZ": info["Z"],
                "dist_km": dist0,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("dist_km").head(MAX_STATIONS).reset_index(drop=True)


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


def get_waveforms_for_event(
    sta_rows: pd.DataFrame, t0: datetime, duration_s: float
) -> Dict[Tuple[str, str], Dict[str, Trace]]:
    t_start = UTCDateTime(t0 - timedelta(seconds=PAD_BEFORE))
    t_end = UTCDateTime(t0 + timedelta(seconds=max(duration_s, 1) + PAD_AFTER))

    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]] = {}

    for _, r in sta_rows.iterrows():
        net, sta = str(r["network"]), str(r["station"])
        trio: Dict[str, Trace] = {}

        for comp, col in (("E", "chE"), ("N", "chN"), ("Z", "chZ")):
            ch = r.get(col)
            if isinstance(ch, str) and ch:
                tr = fdsn_fetch(net, sta, ch, t_start, t_end)
                if tr is not None:
                    trio[comp] = tr

        if all(k in trio for k in ("E", "N", "Z")):
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

    mu = np.nanmean(out.data)
    sd = np.nanstd(out.data)
    out.data = (out.data - mu) / (sd + 1e-9)

    return out


def choose_Z_or_EN(trio: Dict[str, Trace]) -> Optional[Trace]:
    return trio.get("Z") or trio.get("E") or trio.get("N")


#------------------------
# STFT and CWT
#------------------------

def _stft_power(x: np.ndarray, fs: float, nperseg: int, noverlap: int):
    win = np.hanning(nperseg)
    step = nperseg - noverlap
    if step <= 0:
        step = 1

    nframes = 1 + max(0, (len(x) - nperseg) // step)
    if nframes <= 0:
        f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
        return np.zeros((len(f), 0), dtype=np.float32), f, np.array([], dtype=np.float32)

    s = np.empty((nperseg // 2 + 1, nframes), dtype=np.float32)
    t = np.empty((nframes,), dtype=np.float32)

    win2 = np.sum(win**2)
    for i in range(nframes):
        start = i * step
        seg = x[start : start + nperseg]
        if len(seg) < nperseg:
            seg = np.pad(seg, (0, nperseg - len(seg)))
        segw = seg * win
        fftv = np.fft.rfft(segw)
        s[:, i] = (np.abs(fftv) ** 2) / (win2 + 1e-12)
        t[i] = (start + 0.5 * nperseg) / fs

    f = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    return s, f, t


def make_spectrogram_Z(
    tr: Trace, fmin: float = 1.0, fmax: float = 8.0, win_s: float = 4.0, step_s: float = 1.0
):
    fs = float(tr.stats.sampling_rate)
    x = bandpass(
        tr.data.astype(np.float32),
        freqmin=BANDPASS_HZ[0],
        freqmax=BANDPASS_HZ[1],
        df=fs,
        corners=4,
        zerophase=True,
    )

    nperseg = int(round(win_s * fs))
    noverlap = int(round(max(0.0, (win_s - step_s)) * fs))

    s, f, t = _stft_power(x, fs, nperseg, noverlap)
    m = (f >= fmin) & (f <= fmax)
    s, f = s[m, :], f[m]

    s = np.log10(1.0 + s)
    mu, sd = np.nanmean(s), np.nanstd(s)
    s = (s - mu) / (sd + 1e-9)
    return s, f, t


#------------------------
# Cross-Correlation
#------------------------

def xcorr_norm_1hz(a: np.ndarray, b: np.ndarray, max_lag_s: float) -> Tuple[np.ndarray, np.ndarray, float]:
    l = min(len(a), len(b))
    if l < 5:
        return np.array([0.0]), np.array([0.0]), 0.0

    a = a[:l].astype(np.float32)
    b = b[:l].astype(np.float32)
    am = np.nanmean(a)
    asd = np.nanstd(a) + 1e-9
    bm = np.nanmean(b)
    bsd = np.nanstd(b) + 1e-9
    a = np.where(np.isfinite(a), (a - am) / asd, 0.0)
    b = np.where(np.isfinite(b), (b - bm) / bsd, 0.0)

    c = np.correlate(a, b, mode="full").astype(np.float32) / max(l - 1, 1)
    lags = np.arange(-l + 1, l, dtype=np.int32)

    m = (lags >= -int(max_lag_s)) & (lags <= int(max_lag_s))
    c = c[m]
    lags = lags[m]

    cmax = float(np.max(c)) if c.size else 0.0
    return lags.astype(np.float32), c, cmax


def compute_all_pairs_xcorr(
    env_traces: Dict[Tuple[str, str], Trace],
    max_lag_s: float,
    t_ev0: Optional[datetime] = None,
    t_ev1: Optional[datetime] = None,
):
    keys = list(env_traces.keys())
    xcorr_results = {}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            ki, kj = keys[i], keys[j]
            ti, tj = env_traces[ki], env_traces[kj]

            fs = float(ti.stats.sampling_rate)
            assert abs(fs - 1.0) < 1e-3, "env traces must be 1 Hz"

            tmin = max(ti.stats.starttime, tj.stats.starttime)
            tmax = min(ti.stats.endtime, tj.stats.endtime)

            if (t_ev0 is not None) and (t_ev1 is not None):
                ev0 = UTCDateTime(t_ev0) if not isinstance(t_ev0, UTCDateTime) else t_ev0
                ev1 = UTCDateTime(t_ev1) if not isinstance(t_ev1, UTCDateTime) else t_ev1
                tmin = max(tmin, ev0)
                tmax = min(tmax, ev1)

            if tmax <= tmin:
                continue

            win_len_s = float(tmax - tmin)
            max_lag_used = max(1.0, min(max_lag_s, 0.5 * win_len_s))

            ai = ti.slice(tmin, tmax, nearest_sample=True).data
            bj = tj.slice(tmin, tmax, nearest_sample=True).data

            lags, corr, cmax = xcorr_norm_1hz(ai, bj, max_lag_used)

            if corr.size:
                max_idx = int(np.argmax(corr))
                lag_at_max = float(lags[max_idx])
            else:
                lag_at_max = 0.0

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

    for (ki, kj), result in xcorr_results.items():
        max_corr = result["max_correlation"]
        if max_corr >= threshold:
            hits[(ki, kj)] = result
            all_hit_stations.add(ki)
            all_hit_stations.add(kj)

    return hits, all_hit_stations


#------------------------
# Plotting
#------------------------

def plot_cwt_vertical(
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]],
    hit_stations: set,
    event_info: str,
    t0: datetime,
    duration_s: float,
    output_dir: Path,
):
    if not hit_stations:
        print("No hit stations for CWT analysis")
        return

    z_traces = {}
    for key in hit_stations:
        if key in sta2tr and "Z" in sta2tr[key]:
            z_traces[key] = sta2tr[key]["Z"]

    if not z_traces:
        print("No Z components available for CWT analysis")
        return

    for key, tr in z_traces.items():
        net, sta = key

        tr_bp = tr.copy()
        try:
            tr_bp.filter("bandpass", freqmin=BANDPASS_HZ[0], freqmax=BANDPASS_HZ[1], corners=4, zerophase=True)
        except Exception:
            continue

        t_start = UTCDateTime(t0 - timedelta(seconds=PAD_BEFORE))
        t_end = UTCDateTime(t0 + timedelta(seconds=duration_s + PAD_AFTER))
        tr_win = tr_bp.slice(t_start, t_end, nearest_sample=True)

        fs = float(tr_win.stats.sampling_rate)
        data = tr_win.data.astype(np.float32, copy=False)
        t_rel = np.arange(len(data)) / fs - PAD_BEFORE

        coeff, freqs = pywt.cwt(data, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / fs)
        mag = np.abs(coeff)

        freqs = np.asarray(freqs, dtype=float)
        m = np.isfinite(freqs) & (freqs > 0)
        freqs = freqs[m]
        mag = mag[m, :]

        order = np.argsort(freqs)
        freqs_sorted = freqs[order]
        mag_sorted = mag[order, :]

        periods = 1.0 / freqs_sorted
        periods = periods[::-1]
        mag_sorted = mag_sorted[::-1, :]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 2]})

        ax1.plot(t_rel, data, "k-", linewidth=0.8)
        ax1.axvspan(0, duration_s, color="red", alpha=0.2, label="Event window")
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"CWT (Y=Period s): {net}.{sta}.Z - {event_info}")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        extent = [t_rel[0], t_rel[-1], periods[0], periods[-1]]
        im = ax2.imshow(
            mag_sorted,
            aspect="auto",
            extent=extent,
            cmap="jet",
            origin="lower",
            interpolation="bilinear",
        )
        ax2.set_xlabel("Time since event start (s)")
        ax2.set_ylabel("Period (s)")
        ax2.set_yscale("log")
        ax2.set_ylim(0.125, 5.0)
        ax2.axvspan(0, duration_s, color="white", alpha=0.3, linestyle="--", linewidth=2)
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.02)
        cbar.set_label("Wavelet Coefficient Magnitude")

        plt.tight_layout()
        cwt_path = output_dir / f"cwt_period_{net}.{sta}.Z.png"
        plt.savefig(cwt_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[OK] CWT (period) plot saved: {cwt_path}")


def plot_stft_cwt_side_by_side(
    sta2tr: dict[tuple[str, str], dict[str, Trace]],
    hit_stations: Set[tuple[str, str]],
    event_info: str,
    t0: datetime,
    duration_s: float,
    output_dir: Path,
) -> None:
    if not hit_stations:
        print("No hit stations for STFT/CWT comparison")
        return

    z_traces = {k: v["Z"] for k, v in sta2tr.items() if k in hit_stations and "Z" in v}
    if not z_traces:
        print("No Z components available for STFT/CWT comparison")
        return

    for key, tr in z_traces.items():
        net, sta = key

        tr_bp = tr.copy()
        try:
            tr_bp.filter("bandpass", freqmin=BANDPASS_HZ[0], freqmax=BANDPASS_HZ[1], corners=4, zerophase=True)
        except Exception:
            print(f"[WARN] Bandpass failed for {net}.{sta}.Z, skipping")
            continue

        t_start = UTCDateTime(t0 - timedelta(seconds=PAD_BEFORE))
        t_end = UTCDateTime(t0 + timedelta(seconds=max(duration_s, 1) + PAD_AFTER))
        tr_win = tr_bp.slice(t_start, t_end, nearest_sample=True)

        fs = float(tr_win.stats.sampling_rate)
        data = tr_win.data.astype(np.float32, copy=False)
        t_rel_full = np.arange(len(data)) / fs - PAD_BEFORE

        s_stft, f_stft, t_stft = make_spectrogram_Z(
            tr_win, fmin=STFT_Z_FMIN, fmax=STFT_Z_FMAX, win_s=STFT_Z_WIN_S, step_s=STFT_Z_STEP_S
        )
        t_stft_rel = t_stft - PAD_BEFORE

        coeff, freqs = pywt.cwt(data, CWT_SCALES, CWT_WAVELET, sampling_period=1.0 / fs)
        mag = np.abs(coeff)
        mag = (mag - mag.mean(axis=1, keepdims=True)) / (mag.std(axis=1, keepdims=True) + 1e-12)

        freqs = np.asarray(freqs, dtype=float)
        m = np.isfinite(freqs) & (freqs > 0)
        freqs = freqs[m]
        mag = mag[m, :]

        order = np.argsort(freqs)
        freqs_sorted = freqs[order]
        mag_sorted = mag[order, :]

        periods = 1.0 / freqs_sorted
        periods = periods[::-1]
        mag_sorted = mag_sorted[::-1, :]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax0 = axes[0]
        im0 = ax0.imshow(
            s_stft,
            origin="lower",
            aspect="auto",
            extent=[
                t_stft_rel[0] if len(t_stft_rel) else 0.0,
                t_stft_rel[-1] if len(t_stft_rel) else 0.0,
                f_stft[0] if len(f_stft) else STFT_Z_FMIN,
                f_stft[-1] if len(f_stft) else STFT_Z_FMAX,
            ],
            cmap="magma",
        )
        ax0.set_title(f"STFT (Z-only): {net}.{sta}\n{event_info}")
        ax0.set_xlabel("Time since event start (s)")
        ax0.set_ylabel("Frequency (Hz)")
        ax0.set_ylim(STFT_Z_FMIN, STFT_Z_FMAX)
        ax0.axvspan(0, duration_s, color="white", alpha=0.18, lw=0)
        cbar0 = fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
        cbar0.set_label("Normalized log-power")

        ax1 = axes[1]
        im1 = ax1.imshow(
            mag_sorted,
            origin="lower",
            aspect="auto",
            extent=[t_rel_full[0], t_rel_full[-1], periods[0], periods[-1]],
            cmap="jet",
            interpolation="bilinear",
        )
        ax1.set_title("CWT")
        ax1.set_xlabel("Time since event start (s)")
        ax1.set_ylabel("Period (s)")
        ax1.set_yscale("log")
        ax1.set_ylim(0.125, 1.0)
        ax1.axvspan(0, duration_s, color="white", alpha=0.25, lw=0)
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Per-frequency z-score")

        plt.tight_layout()
        out_png = output_dir / f"stft_cwt_{net}.{sta}.Z.png"
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] STFT vs CWT (side-by-side) saved: {out_png}")


def save_raw_waveforms(
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]], output_dir: Path, event_time_str: str
):
    waveform_file = output_dir / f"raw_waveforms_{event_time_str}.pkl"

    waveform_data = {}
    for key, trio in sta2tr.items():
        net, sta = key
        waveform_data[f"{net}.{sta}"] = {}
        for comp, tr in trio.items():
            waveform_data[f"{net}.{sta}"][comp] = {
                "data": tr.data,
                "sampling_rate": tr.stats.sampling_rate,
                "starttime": tr.stats.starttime.timestamp,
                "network": tr.stats.network,
                "station": tr.stats.station,
                "channel": tr.stats.channel,
            }

    with open(waveform_file, "wb") as f:
        pickle.dump(waveform_data, f)

    print(f"[OK] Raw waveforms saved: {waveform_file}")


def plot_cross_correlations(xcorr_results: Dict, event_info: str, output_path: Path):
    n_pairs = len(xcorr_results)
    if n_pairs == 0:
        print("No cross-correlations to plot")
        return

    cols = min(3, n_pairs)
    rows = int(np.ceil(n_pairs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_pairs == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()

    for idx, ((ki, kj), result) in enumerate(xcorr_results.items()):
        ax = axes[idx]
        lags = result["lags"]
        corr = result["correlation"]
        cmax = result["max_correlation"]
        pair_name = result["station_pair"]

        is_hit = cmax >= HIT_THRESHOLD
        color = "red" if is_hit else "blue"
        linewidth = 1.5 if is_hit else 1.0

        ax.plot(lags, corr, color=color, linewidth=linewidth)
        ax.axhline(0, color="k", linestyle="--", alpha=0.3)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(HIT_THRESHOLD, color="green", linestyle=":", alpha=0.7, label=f"Hit threshold ({HIT_THRESHOLD})")

        max_idx = np.argmax(corr)
        max_lag = lags[max_idx]
        marker_color = "darkred" if is_hit else "darkblue"
        ax.plot(max_lag, cmax, "o", color=marker_color, markersize=6)

        ax.set_xlabel("Lag (s)")
        ax.set_ylabel("Normalized Correlation")
        hit_str = " [HIT]" if is_hit else ""
        ax.set_title(f"{pair_name}{hit_str}\nMax: {cmax:.3f} @ {max_lag:.0f}s")
        ax.grid(True, alpha=0.3)
        if len(lags) > 0:
            ax.set_xlim(float(lags[0]), float(lags[-1]))
        else:
            ax.set_xlim(-XCORR_MAX_LAG_S, XCORR_MAX_LAG_S)

    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(f"Cross-Correlations: {event_info}", fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Cross-correlation plot saved: {output_path}")


def plot_envelopes(
    env_traces: Dict[Tuple[str, str], Trace],
    event_info: str,
    t0: datetime,
    duration_s: float,
    output_path: Path,
):
    if not env_traces:
        print("No envelope traces to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    keys = sorted(env_traces.keys(), key=lambda x: f"{x[0]}.{x[1]}")

    offset = 0
    for key in keys:
        tr = env_traces[key]
        net, sta = key

        fs = float(tr.stats.sampling_rate)
        t_abs = tr.stats.starttime.timestamp + np.arange(len(tr.data)) / fs
        t_rel = t_abs - UTCDateTime(t0).timestamp

        ax.plot(t_rel, tr.data + offset, label=f"{net}.{sta}", linewidth=0.8)
        offset += 4

    ax.axvspan(0, duration_s, color="red", alpha=0.2, label="Event window")

    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Normalized envelope amplitude (offset)")
    ax.set_title(f"Preprocessed Envelopes: {event_info}")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Envelope plot saved: {output_path}")


def plot_bandpassed_waveforms(
    sta2tr: Dict[Tuple[str, str], Dict[str, Trace]],
    sta_df: pd.DataFrame,
    hit_stations: set,
    ev_lat: float,
    ev_lon: float,
    event_info: str,
    t0: datetime,
    duration_s: float,
    output_path: Path,
):
    if not sta2tr:
        print("No waveform data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 10))

    sta_info = {}
    for _, row in sta_df.iterrows():
        key = (str(row["network"]), str(row["station"]))
        if key in sta2tr and key in hit_stations:
            sta_info[key] = {
                "dist_km": float(row["dist_km"]),
                "lat": float(row["latitude"]),
                "lon": float(row["longitude"]),
            }

    if not sta_info:
        print("No hit stations for waveform plotting")
        return

    keys = sorted(sta_info.keys(), key=lambda x: sta_info[x]["dist_km"])
    colors = plt.cm.tab10(np.linspace(0, 1, len(keys)))

    for i, key in enumerate(keys):
        trio = sta2tr[key]
        net, sta = key
        dist_km = sta_info[key]["dist_km"]

        tr = choose_Z_or_EN(trio)
        if tr is None:
            continue

        tr_bp = tr.copy()
        try:
            tr_bp.filter("bandpass", freqmin=BANDPASS_HZ[0], freqmax=BANDPASS_HZ[1], corners=4, zerophase=True)
        except Exception:
            continue

        fs = float(tr_bp.stats.sampling_rate)
        t_abs = tr_bp.stats.starttime.timestamp + np.arange(len(tr_bp.data)) / fs
        t_rel = t_abs - UTCDateTime(t0).timestamp

        max_amp = np.max(np.abs(tr_bp.data))
        if max_amp > 0:
            data_norm = tr_bp.data / max_amp * 3.0
        else:
            data_norm = tr_bp.data

        component = "Z" if "Z" in trio else ("E" if "E" in trio else "N")
        ax.plot(
            t_rel,
            data_norm + dist_km,
            color=colors[i],
            label=f"{net}.{sta}.{component} ({dist_km:.1f}km)",
            linewidth=0.7,
        )

    ax.axvspan(0, duration_s, color="red", alpha=0.15, label="Event window")
    ax.axvspan(-PAD_BEFORE, 0, color="gray", alpha=0.1, label="Pre-event padding")
    ax.axvspan(duration_s, duration_s + PAD_AFTER, color="gray", alpha=0.1, label="Post-event padding")

    ax.set_xlabel("Time since event start (s)")
    ax.set_ylabel("Distance from event (km) + normalized amplitude")
    ax.set_title(f"Bandpassed Waveforms (1-8 Hz) vs Distance - Hit Stations Only: {event_info}")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.set_xlim(-PAD_BEFORE, duration_s + PAD_AFTER)
    if keys:
        min_dist = min(sta_info[k]["dist_km"] for k in keys)
        max_dist = max(sta_info[k]["dist_km"] for k in keys)
        ax.set_ylim(min_dist - 2, max_dist + 2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Bandpassed waveforms plot saved: {output_path}")


#------------------------
# Event Processing
#------------------------

def process_tremor_event(ev_row: pd.Series, ch_df: pd.DataFrame, outdir: Path) -> None:
    ev_lat = float(pd.to_numeric(ev_row.get("lat"), errors="coerce")) if "lat" in ev_row else np.nan
    ev_lon = float(pd.to_numeric(ev_row.get("lon"), errors="coerce")) if "lon" in ev_row else np.nan
    ev_dur = float(pd.to_numeric(ev_row.get("duration", 120.0), errors="coerce"))
    t0 = to_utc(str(ev_row.get("starttime")))

    if any([pd.isna(ev_lat), pd.isna(ev_lon)]) or t0 is None:
        print("[SKIP] Missing lat/lon/starttime in event")
        return

    event_info = f"{t0.strftime('%Y-%m-%d %H:%M:%S')} ({ev_dur:.0f}s)"
    print(f"\n[PROCESSING] {event_info}")

    sta_df = pick_stations_3c(ch_df, ev_lat, ev_lon)
    if len(sta_df) < 2:
        print(f"[SKIP] Not enough 3C stations ({len(sta_df)})")
        return

    print(f"[INFO] Selected {len(sta_df)} stations")

    print("[INFO] Downloading waveforms...")
    sta2tr = get_waveforms_for_event(sta_df, t0, ev_dur)
    if len(sta2tr) < 2:
        print(f"[SKIP] Not enough stations with 3C data ({len(sta2tr)})")
        return

    print(f"[INFO] Downloaded data from {len(sta2tr)} stations")

    print("[INFO] Preprocessing envelopes...")
    env_traces = {}
    t_abs0 = UTCDateTime(t0 - timedelta(seconds=PAD_BEFORE))
    t_abs1 = UTCDateTime(t0 + timedelta(seconds=ev_dur + PAD_AFTER))

    for key, trio in sta2tr.items():
        tr0 = choose_Z_or_EN(trio)
        if tr0 is None:
            continue

        trw = tr0.copy().slice(t_abs0, t_abs1, nearest_sample=True)
        env = preprocess_env1hz_wech(trw)
        if env is not None:
            env_traces[key] = env

    if len(env_traces) < 2:
        print(f"[SKIP] Not enough envelope traces ({len(env_traces)})")
        return

    print(f"[INFO] Created {len(env_traces)} envelope traces")

    print("[INFO] Computing cross-correlations...")

    t_ev0 = t0
    t_ev1 = t0 + timedelta(seconds=ev_dur)

    xcorr_results = compute_all_pairs_xcorr(env_traces, XCORR_MAX_LAG_S, t_ev0=t_ev0, t_ev1=t_ev1)

    if not xcorr_results:
        print("[SKIP] No cross-correlations computed")
        return

    print(f"[INFO] Computed {len(xcorr_results)} cross-correlations")

    hits, hit_stations = identify_hits(xcorr_results, HIT_THRESHOLD)

    print("[INFO] Applying physical lag filtering...")
    sta_ll = {
        (str(r["network"]), str(r["station"])): (float(r["latitude"]), float(r["longitude"]))
        for _, r in sta_df.iterrows()
    }

    vmin_kmps = 3.0
    lag_cap_max_s = 10.0

    phys_hits = {}
    phys_stations = set()

    for (ki, kj), r in hits.items():
        if "lag_at_max" in r:
            lag_at_max = float(r["lag_at_max"])
        else:
            lags = r["lags"]
            corr = r["correlation"]
            lag_at_max = float(lags[int(np.argmax(corr))]) if corr.size else 0.0

        lat_i, lon_i = sta_ll.get(ki, (None, None))
        lat_j, lon_j = sta_ll.get(kj, (None, None))
        if lat_i is None or lat_j is None:
            continue

        d_km = haversine_km(lat_i, lon_i, lat_j, lon_j)
        lag_cap = min(lag_cap_max_s, d_km / vmin_kmps)

        if abs(lag_at_max) <= lag_cap:
            phys_hits[(ki, kj)] = r
            phys_stations.add(ki)
            phys_stations.add(kj)

    if phys_hits:
        print(f"[FILTER] Physical lag filter kept {len(phys_hits)} / {len(hits)} pairs")
        hits, hit_stations = phys_hits, phys_stations
    else:
        if len(sta_df) >= 2:
            locs = [(float(r["latitude"]), float(r["longitude"])) for _, r in sta_df.iterrows()]
            max_d = 0.0
            for i in range(len(locs)):
                for j in range(i + 1, len(locs)):
                    max_d = max(max_d, haversine_km(locs[i][0], locs[i][1], locs[j][0], locs[j][1]))
            print(
                f"[FILTER] Physical lag filter removed all {len(hits)} pairs; "
                f"consider relaxing vmin_kmps or lag_cap_max_s "
                f"(network max baseline ~{max_d:.1f} km -> d/vmin ~= {max_d / vmin_kmps:.1f}s)"
            )
        else:
            print(f"[FILTER] Physical lag filter removed all {len(hits)} pairs")
        print("[SKIP] No hits found after physical lag filter")
        return

    if not hits:
        print(f"[SKIP] No hits found (no CC > {HIT_THRESHOLD})")
        return

    print(f"[HIT] Found {len(hits)} station pairs with CC > {HIT_THRESHOLD}")
    print(f"[HIT] {len(hit_stations)} stations involved in hits")

    dt_str = t0.strftime("%Y-%m-%dT%H_%M_%S")
    event_outdir = outdir / f"tremor_hit_{dt_str}"
    event_outdir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Saving raw waveforms...")
    if SAVE_RAW_WAVEFORMS:
        save_raw_waveforms(sta2tr, event_outdir, dt_str)
    else:
        print("[SKIP] SAVE_RAW_WAVEFORMS=False, skip saving .pkl")

    print("[INFO] Generating plots for hit event...")

    xcorr_path = event_outdir / f"tremor_{dt_str}_xcorr.png"
    plot_cross_correlations(xcorr_results, event_info, xcorr_path)

    env_path = event_outdir / f"tremor_{dt_str}_envelopes.png"
    plot_envelopes(env_traces, event_info, t0, ev_dur, env_path)

    waveform_path = event_outdir / f"tremor_{dt_str}_waveforms_1-8Hz.png"
    plot_bandpassed_waveforms(sta2tr, sta_df, hit_stations, ev_lat, ev_lon, event_info, t0, ev_dur, waveform_path)

    print("[INFO] Generating CWT plots for hit stations...")
    plot_cwt_vertical(sta2tr, hit_stations, event_info, t0, ev_dur, event_outdir)

    print("[INFO] Generating STFT vs CWT side-by-side plots for hit stations...")
    plot_stft_cwt_side_by_side(sta2tr, hit_stations, event_info, t0, ev_dur, event_outdir)

    max_corrs = [result["max_correlation"] for result in xcorr_results.values()]
    hit_corrs = [result["max_correlation"] for result in hits.values()]
    print(f"[SUMMARY] All correlations: {np.mean(max_corrs):.3f} ± {np.std(max_corrs):.3f}")
    print(f"[SUMMARY] Hit correlations: {np.mean(hit_corrs):.3f} ± {np.std(hit_corrs):.3f}")
    print(f"[SUMMARY] Hit pairs: {list(hits.keys())}")
    print(f"[COMPLETED] Event results saved to: {event_outdir}")


#------------------------
# Main
#------------------------

def main():
    print("=== Simplified Tremor Cross-Correlation Analysis ===")

    outdir = Path(OUT_DIR)

    print("[INFO] Loading channel and event data...")
    ch = pd.read_csv(CHANNELS_CSV)
    ev = pd.read_csv(EVENTS_CSV, low_memory=False)

    ch.columns = ch.columns.str.strip()
    ev.columns = ev.columns.str.strip()
    ch = ch.loc[:, ~ch.columns.duplicated(keep="first")]
    ev = ev.loc[:, ~ev.columns.duplicated(keep="first")]

    for col in ["lat", "lon", "depth", "duration"]:
        if col in ev.columns:
            ev[col] = pd.to_numeric(ev[col], errors="coerce")

    if EVENT_TIME_RANGE is not None:
        tmin = to_utc(EVENT_TIME_RANGE[0])
        tmax = to_utc(EVENT_TIME_RANGE[1])
        ev["t0"] = ev["starttime"].apply(to_utc)
        ev = ev[(ev["t0"] >= tmin) & (ev["t0"] < tmax)].copy()
        print(f"[INFO] Filtered to time range: {tmin} to {tmax}")

    ev.sort_values("starttime", inplace=True)
    if MAX_EVENTS is not None and len(ev) > MAX_EVENTS:
        ev = ev.head(MAX_EVENTS)
        print(f"[INFO] Limited to {MAX_EVENTS} events")

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


if __name__ == "__main__":
    main()