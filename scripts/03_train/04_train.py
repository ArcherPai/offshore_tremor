#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import math
import json
import argparse
import random
import atexit
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


try:
    import pywt
    _HAS_PYWT = True
except Exception:
    pywt = None
    _HAS_PYWT = False

try:
    from sklearn.metrics import (
        accuracy_score, f1_score, recall_score,
        confusion_matrix, classification_report,
        roc_auc_score, average_precision_score,
        precision_recall_curve, roc_curve, auc
    )
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

try:
    from scipy.signal import butter, sosfiltfilt
    _HAS_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    _HAS_SCIPY = False


# -------------------------
# Reproducibility Utilities
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    np.random.seed(base_seed + worker_id)
    random.seed(base_seed + worker_id)


def infer_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Wavelet Transform Helper
# ------------------------
def apply_wavelet_transform(signal_1d: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    if pywt is None:
        raise RuntimeError("pywt is not installed but USE_WAVELET=True. Install PyWavelets or disable wavelet.")
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


# ------------------------
# Config
# ------------------------
@dataclass
class Config:
    RUN_MODE: str = "cpu_smoke" 
    SEED: int = 42

    INPUT_CHANNELS: int = 3
    SAMPLE_RATE: int = 100
    TARGET_LENGTH: int = 15000 
    VAL_FRAC: float = 0.15
    SAMPLE_FRAC: float = 1.0

    TEST_TREMOR_YEARS: tuple = (2025,)
    TEST_EQ_YEARS: tuple = (2021, 2022)
    TEST_NOISE_YEARS: tuple = (2015,)

    USE_WAVELET: bool = True
    WAVELET_TYPE: str = "db4"
    WAVELET_LEVEL: int = 3

    ENABLE_PICK_REGRESSION: bool = False
    RANDOMIZE_PICK_POS: bool = True
    PICK_POS_RANGE: tuple = (0.3, 0.7)

    REAL_NOISE_SCALE_ENABLE: bool = False
    REAL_NOISE_SCALE: float = 1e5

    NOISE_RAND_AMP_ENABLE: bool = True
    NOISE_RAND_AMP_MIN: float = 1e-2
    NOISE_RAND_AMP_MAX: float = 1e2
    NOISE_RAND_AMP_TRAIN_ONLY: bool = True
    NOISE_RAND_AMP_PROB: float = 0.5

    TREMOR_RAND_GAIN_ENABLE: bool = False
    TREMOR_RAND_GAIN_MIN: float = 0.5
    TREMOR_RAND_GAIN_MAX: float = 2.0

    GAUSSIAN_NOISE_ENABLE: bool = False
    GAUSSIAN_NOISE_MODE: str = "add"
    GAUSSIAN_NOISE_STD: float = 1.0
    GAUSSIAN_NOISE_SCALE: float = 10.0
    GAUSSIAN_NOISE_PROB: float = 0.3
    GAUSSIAN_NOISE_TRAIN_ONLY: bool = False

    TREMOR_MULTI_CROP_ENABLE: bool = True
    TREMOR_MULTI_CROP_K: int = 8
    TREMOR_MULTI_CROP_MIN_SEP_S: float = 5.0
    TREMOR_BAND_HZ: tuple = (2.0, 8.0)
    TREMOR_TOTAL_HZ: tuple = (0.5, 20.0)
    TREMOR_PROXY_USE_CORR: bool = True
    TREMOR_PROXY_USE_KURTOSIS_PENALTY: bool = True

    TREMOR_MIN_BAND_RATIO: float = 0.06
    TREMOR_MIN_BAND_RMS_Q: float = 0.50
    TREMOR_MIN_ABS_RMS_Q: float = 0.40
    TREMOR_REQUIRE_POS_CORR: bool = False

    BATCH_SIZE: int = 32
    MAX_EPOCHS: int = 50
    LR: float = 1e-4
    LR_DECAY: float = 0.95
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.3
    GRADIENT_CLIP: float = 1.0
    BALANCE_DATASET: bool = True

    EARLY_STOP_PATIENCE: int = 50
    EARLY_STOP_MIN_DELTA: float = 0.001

    CONV_CHANNELS: tuple = (8, 16, 32)
    LSTM_HIDDEN: int = 32
    LSTM_LAYERS: int = 1
    BIDIRECTIONAL: bool = True

    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    PERSISTENT_WORKERS: bool = True
    PREFETCH_FACTOR: int = 4

    USE_AMP: bool = True
    USE_COMPILE: bool = True

    ENABLE_VIZ: bool = True
    SAVE_DIR: str = "/home/bxd240002/scratch/Archer/offshore_tremor/results/results_three_class"
    N_EXAMPLES_OK: int = 10
    N_EXAMPLES_BAD: int = 10

    FORCE_CPU: bool = False
    DEVICE: str = "cpu"

    H5_MAX_OPEN: int = 4
    H5_RDCC_NBYTES: int = 64 * 1024 * 1024
    H5_RDCC_NSLOTS: int = 1_000_003
    H5_RDCC_W0: float = 0.75

    FAST_TEST: bool = False

    USE_SPECTROGRAM: bool = True
    STFT_N_FFT: int = 256
    STFT_HOP_LENGTH: int = 128

    TREMOR_PRE_STFT_FILTER_ENABLE: bool = True
    TREMOR_PRE_STFT_FILTER_HZ: tuple = (0.0, 20.0)  # (f_lo, f_hi) ; 0-20 ~= lowpass 20
    TREMOR_PRE_STFT_FILTER_ORDER: int = 4


    def apply_profile(self):
        if self.RUN_MODE == "cpu_smoke":
            self.FORCE_CPU = True
            self.USE_WAVELET = False
            self.USE_SPECTROGRAM = False
            self.TARGET_LENGTH = 6000
            self.BATCH_SIZE = 8
            self.MAX_EPOCHS = 20
            self.EARLY_STOP_PATIENCE = 5
            self.SAMPLE_FRAC = 0.05

            self.REAL_NOISE_SCALE_ENABLE = False
            self.NOISE_RAND_AMP_ENABLE = True
            self.NOISE_RAND_AMP_MIN = 1e-2
            self.NOISE_RAND_AMP_MAX = 1e3
            self.NOISE_RAND_AMP_PROB = 0.5

            self.GAUSSIAN_NOISE_ENABLE = False
            self.BALANCE_DATASET = False

            self.NUM_WORKERS = 0
            self.PIN_MEMORY = False
            self.PERSISTENT_WORKERS = False
            self.PREFETCH_FACTOR = 2

            self.TREMOR_MULTI_CROP_K = 3
            self.H5_MAX_OPEN = 2

            self.USE_AMP = False
            self.USE_COMPILE = False
            self.FAST_TEST = True

        elif self.RUN_MODE == "gpu_train":
            self.FORCE_CPU = False
            self.NUM_WORKERS = 4
            self.PIN_MEMORY = True
            self.PERSISTENT_WORKERS = True
            self.PREFETCH_FACTOR = 4

            self.H5_MAX_OPEN = 8
            self.USE_AMP = True
            self.USE_COMPILE = True
            self.FAST_TEST = False
        else:
            raise ValueError(f"Unknown RUN_MODE={self.RUN_MODE}")

        if self.USE_WAVELET and (not _HAS_PYWT):
            print("[WARN] pywt not found; disabling wavelet features (USE_WAVELET=False).")
            self.USE_WAVELET = False

        if not _HAS_SKLEARN:
            raise RuntimeError(
                "scikit-learn (sklearn) is required.\n"
                "conda install -c conda-forge scikit-learn -y\n"
            )

        self.DEVICE = str(infer_device(force_cpu=self.FORCE_CPU))


def to_config_dict(cfg: Config) -> dict:
    d = asdict(cfg)
    d["CONV_CHANNELS"] = list(cfg.CONV_CHANNELS)
    d["PICK_POS_RANGE"] = list(cfg.PICK_POS_RANGE)
    d["TREMOR_BAND_HZ"] = list(cfg.TREMOR_BAND_HZ)
    d["TREMOR_TOTAL_HZ"] = list(cfg.TREMOR_TOTAL_HZ)
    d["TEST_TREMOR_YEARS"] = list(cfg.TEST_TREMOR_YEARS)
    d["TEST_EQ_YEARS"] = list(cfg.TEST_EQ_YEARS)
    d["TEST_NOISE_YEARS"] = list(cfg.TEST_NOISE_YEARS)
    return d


# ------------------------
# HDF5 access helpers
# ------------------------
def _try_get_data_root(h5: h5py.File):
    return h5["data"] if "data" in h5 else h5


class H5HandleCache:
    def __init__(self, max_open: int = 4, rdcc_nbytes: int = 0, rdcc_nslots: int = 0, rdcc_w0: float = 0.75):
        self.max_open = int(max(1, max_open))
        self._pid = os.getpid()
        self._lru: "OrderedDict[str, h5py.File]" = OrderedDict()
        self._open_kwargs: Dict[str, Any] = {"libver": "latest"}
        if int(rdcc_nbytes) > 0:
            self._open_kwargs["rdcc_nbytes"] = int(rdcc_nbytes)
        if int(rdcc_nslots) > 0:
            self._open_kwargs["rdcc_nslots"] = int(rdcc_nslots)
        self._open_kwargs["rdcc_w0"] = float(rdcc_w0)
        atexit.register(self.close_all)

    def _reset_if_forked(self):
        pid = os.getpid()
        if pid != self._pid:
            try:
                self.close_all()
            except Exception:
                pass
            self._pid = pid
            self._lru = OrderedDict()

    def get(self, filepath: str) -> h5py.File:
        self._reset_if_forked()
        fp = str(filepath)
        if fp in self._lru:
            f = self._lru.pop(fp)
            self._lru[fp] = f
            return f
        f = h5py.File(fp, "r", **self._open_kwargs)
        self._lru[fp] = f
        while len(self._lru) > self.max_open:
            _, old_f = self._lru.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass
        return f

    def close_all(self):
        for _, f in list(self._lru.items()):
            try:
                f.close()
            except Exception:
                pass
        self._lru.clear()

    def __del__(self):
        try:
            self.close_all()
        except Exception:
            pass


# ------------------------
# Year parsing
# ------------------------
_YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")


def _extract_year_from_any(val) -> int:
    if val is None:
        return -1
    try:
        if isinstance(val, (np.integer, int)) and 1900 <= int(val) <= 2100:
            return int(val)
    except Exception:
        pass

    s = str(val)
    if not s or s.lower() in ("nan", "none"):
        return -1

    dt = pd.to_datetime(s, errors="coerce", utc=True)
    if not pd.isna(dt):
        try:
            return int(dt.year)
        except Exception:
            pass

    m = _YEAR_RE.search(s)
    if m:
        try:
            y = int(m.group(1))
            if 1900 <= y <= 2100:
                return y
        except Exception:
            return -1
    return -1


def add_year_column(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    df = df.copy()
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(-1).astype(int)
        return df

    if kind == "comcat":
        candidates = ["time", "event_time", "origin_time", "trace_start_time", "datetime", "timestamp"]
    elif kind == "tremor":
        candidates = ["event_time", "time", "origin_time", "t0", "start_time", "datetime", "timestamp", "per_event_h5"]
    elif kind == "noise":
        candidates = ["trace_start_time", "start_time", "time", "datetime", "timestamp", "file", "waveform_file"]
    else:
        candidates = list(df.columns)

    candidates = [c for c in candidates if c in df.columns]
    years = np.full(len(df), -1, dtype=int)

    for c in candidates:
        if (years >= 0).all():
            break
        vals = df[c].astype(str).values
        for i, v in enumerate(vals):
            if years[i] >= 0:
                continue
            y = _extract_year_from_any(v)
            if y >= 0:
                years[i] = y

    if np.any(years < 0):
        fallback_cols = [c for c in ["per_event_h5", "trace_name", "h5_key", "station", "network"] if c in df.columns]
        for i in np.where(years < 0)[0]:
            for c in fallback_cols:
                y = _extract_year_from_any(df.iloc[i].get(c, None))
                if y >= 0:
                    years[i] = y
                    break

    df["year"] = years
    return df


def split_by_class_year_sets(df_all: pd.DataFrame, cfg: Config):
    trem_y = set(int(x) for x in cfg.TEST_TREMOR_YEARS)
    eq_y = set(int(x) for x in cfg.TEST_EQ_YEARS)
    noi_y = set(int(x) for x in cfg.TEST_NOISE_YEARS)

    df = df_all.copy()
    year = df["year"].astype(int).values
    lab = df["label"].astype(int).values

    is_known = year >= 0
    is_test = np.zeros(len(df), dtype=bool)
    is_test |= (is_known & (lab == 2) & np.isin(year, list(trem_y)))
    is_test |= (is_known & (lab == 1) & np.isin(year, list(eq_y)))
    is_test |= (is_known & (lab == 0) & np.isin(year, list(noi_y)))

    test_df = df[is_test].copy()
    train_pool = df[~is_test].copy()

    train_pool = train_pool.sample(frac=1.0, random_state=cfg.SEED).reset_index(drop=True)
    n_val = int(len(train_pool) * float(cfg.VAL_FRAC))
    val_df = train_pool.iloc[:n_val].copy()
    train_df = train_pool.iloc[n_val:].copy()
    return train_df, val_df, test_df


# ------------------------
# Signal helpers
# ------------------------
def resample_trace_np(x: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    src_sr = float(src_sr)
    dst_sr = float(dst_sr)
    if src_sr <= 0 or dst_sr <= 0:
        return x
    if abs(src_sr - dst_sr) < 1e-6:
        return x

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"resample_trace_np expects 2D (C,L), got shape={x.shape}")

    C, L = x.shape
    dur = (L - 1) / src_sr if L > 1 else 0.0
    L_new = int(round(dur * dst_sr)) + 1 if dur > 0 else 1
    if L_new < 2:
        L_new = max(1, int(round(L * dst_sr / src_sr)))

    t_old = np.linspace(0.0, max(dur, 0.0), num=L, dtype=np.float64)
    t_new = np.linspace(0.0, max(dur, 0.0), num=L_new, dtype=np.float64)

    y = np.zeros((C, L_new), dtype=np.float32)
    for c in range(C):
        y[c] = np.interp(t_new, t_old, x[c].astype(np.float64, copy=False)).astype(np.float32)

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def lowpass_or_bandpass_np(x: np.ndarray, sr: float, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """
    x: (C, L)
    If f_lo<=0 and f_hi>0 => lowpass
    If 0<f_lo<f_hi<nyq => bandpass
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"filter expects (C,L), got {x.shape}")

    nyq = 0.5 * float(sr)
    f_lo = float(max(0.0, f_lo))
    f_hi = float(min(max(0.0, f_hi), nyq))

    if f_hi <= 0.0:
        return x
    if f_lo <= 0.0 and f_hi >= nyq:
        return x 

    if _HAS_SCIPY:
        if f_lo <= 0.0:
            Wn = f_hi / nyq
            sos = butter(int(order), Wn, btype="lowpass", output="sos")
        else:
            Wn = [f_lo / nyq, f_hi / nyq]
            sos = butter(int(order), Wn, btype="bandpass", output="sos")

        y = np.zeros_like(x, dtype=np.float32)
        for c in range(x.shape[0]):
            if x.shape[1] < 3 * (2 * order + 1):
                y[c] = x[c]
            else:
                y[c] = sosfiltfilt(sos, x[c]).astype(np.float32)
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    C, L = x.shape
    freqs = np.fft.rfftfreq(L, d=1.0 / float(sr))
    X = np.fft.rfft(x.astype(np.float64), axis=1)

    mask = (freqs <= f_hi) if f_lo <= 0.0 else ((freqs >= f_lo) & (freqs <= f_hi))
    X *= mask[None, :]
    y = np.fft.irfft(X, n=L, axis=1).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def _safe_kurtosis(x1d: np.ndarray) -> float:
    x = x1d.astype(np.float64, copy=False)
    mu = x.mean()
    v = x - mu
    m2 = np.mean(v * v) + 1e-12
    m4 = np.mean((v * v) * (v * v))
    return float(m4 / (m2 * m2))


def tremor_proxy_score(
    x_win: np.ndarray,
    sr: float,
    band_hz=(2.0, 8.0),
    total_hz=(0.5, 20.0),
    use_corr=True,
    use_kurtosis_penalty=True
) -> float:
    x_win = np.asarray(x_win)
    if x_win.ndim != 2:
        x_win = np.squeeze(x_win)
        if x_win.ndim == 1:
            x_win = x_win[np.newaxis, :]
        elif x_win.ndim != 2:
            return 0.0

    C, L = x_win.shape
    if L < 8:
        return 0.0

    x = np.nan_to_num(x_win.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)

    freqs = np.fft.rfftfreq(L, d=1.0 / float(sr))
    X = np.fft.rfft(x, axis=1)
    P = (X.real * X.real + X.imag * X.imag)

    def band_power(Pch, f0, f1):
        m = (freqs >= f0) & (freqs <= f1)
        if not np.any(m):
            return 0.0
        return float(Pch[m].sum())

    b0, b1 = float(band_hz[0]), float(band_hz[1])
    t0, t1 = float(total_hz[0]), float(total_hz[1])

    ratios = []
    for ch in range(C):
        p_band = band_power(P[ch], b0, b1)
        p_tot = band_power(P[ch], t0, t1) + 1e-12
        ratios.append(p_band / p_tot)
    score = float(np.mean(ratios))

    if use_corr and C >= 2:
        xc = x - x.mean(axis=1, keepdims=True)
        std = x.std(axis=1)
        good = std > 1e-8
        if np.sum(good) >= 2:
            xn = xc[good] / (std[good][:, None] + 1e-6)
            corrs = []
            for i in range(xn.shape[0]):
                for j in range(i + 1, xn.shape[0]):
                    c = float(np.mean(xn[i] * xn[j]))
                    if np.isfinite(c):
                        corrs.append(c)
            if corrs:
                pos = [c for c in corrs if c > 0.0]
                corr_bonus = float(np.mean(pos)) if pos else 0.0
                score += 0.15 * corr_bonus

    if use_kurtosis_penalty:
        ks = [_safe_kurtosis(x[ch]) for ch in range(C)]
        k = float(np.mean(ks))
        penalty = max(0.0, (k - 8.0) / 20.0)
        score -= 0.2 * penalty

    if not np.isfinite(score):
        return 0.0
    return float(score)


def tremor_band_metrics(
    x_win: np.ndarray,
    sr: float,
    band_hz: Tuple[float, float],
    total_hz: Tuple[float, float],
) -> Tuple[float, float]:
    x_win = np.asarray(x_win)
    if x_win.ndim != 2:
        x_win = np.squeeze(x_win)
        if x_win.ndim == 1:
            x_win = x_win[np.newaxis, :]
        else:
            return 0.0, 0.0

    C, L = x_win.shape
    if L < 8:
        return 0.0, 0.0

    x = np.nan_to_num(x_win.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    freqs = np.fft.rfftfreq(L, d=1.0 / float(sr))
    X = np.fft.rfft(x, axis=1)
    P = (X.real * X.real + X.imag * X.imag)

    b0, b1 = float(band_hz[0]), float(band_hz[1])
    t0, t1 = float(total_hz[0]), float(total_hz[1])
    mb = (freqs >= b0) & (freqs <= b1)
    mt = (freqs >= t0) & (freqs <= t1)
    if not np.any(mb) or not np.any(mt):
        return 0.0, 0.0

    ratios = []
    rmses = []
    for ch in range(C):
        p_band = float(P[ch][mb].sum())
        p_tot = float(P[ch][mt].sum()) + 1e-12
        ratios.append(p_band / p_tot)
        rmses.append(math.sqrt(p_band / max(1, L)))
    return float(np.mean(ratios)), float(np.mean(rmses))


# =========================
# Dataset utilities
# =========================
def _first_dataset_under(node, max_depth=3, _depth=0):
    import h5py as _h5py
    if isinstance(node, _h5py.Dataset):
        return node
    if not isinstance(node, _h5py.Group):
        return None
    if _depth >= max_depth:
        return None
    for k in node.keys():
        child = node[k]
        ds = _first_dataset_under(child, max_depth=max_depth, _depth=_depth + 1)
        if ds is not None:
            return ds
    return None


def _ensure_C(x: np.ndarray, C_target: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        x = np.squeeze(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        else:
            raise ValueError(f"_ensure_C expects 2D, got {x.shape}")

    C, L = x.shape
    if C >= C_target:
        return x[:C_target, :]
    out = np.zeros((C_target, L), dtype=x.dtype)
    out[:C, :] = x
    return out


def _dataset_to_C_L(x: np.ndarray, input_channels: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    elif x.ndim == 2:
        if x.shape[1] in (1, 2, 3, 4) and x.shape[0] > x.shape[1]:
            x = x.T
    elif x.ndim == 3:
        x = x[0]
        if x.ndim == 2 and x.shape[1] in (1, 2, 3, 4) and x.shape[0] > x.shape[1]:
            x = x.T
    else:
        x = np.squeeze(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]

    if x.ndim != 2:
        raise ValueError(f"cannot coerce shape={x.shape} to 2D")
    return x[:input_channels, :]


def _log_uniform(a: float, b: float) -> float:
    """Sample log-uniform in [a,b], a>0."""
    a = float(a); b = float(b)
    a = max(a, 1e-20)
    b = max(b, a * 1.0000001)
    lo = math.log10(a)
    hi = math.log10(b)
    u = np.random.uniform(lo, hi)
    return float(10.0 ** u)


# =========================
# Dataset (CLASS)
# =========================
class SeismicDataset(Dataset):
    def __init__(self, df: pd.DataFrame, h5_paths: dict, config: Config,
                 is_training: bool = False, return_meta: bool = False, return_raw_window: bool = False):
        self.meta = df.reset_index(drop=True)
        self.h5_paths = h5_paths
        self.cfg = config
        self.is_training = is_training
        self.return_meta = return_meta
        self.return_raw_window = return_raw_window

        self._h5cache = H5HandleCache(
            max_open=int(self.cfg.H5_MAX_OPEN),
            rdcc_nbytes=int(self.cfg.H5_RDCC_NBYTES),
            rdcc_nslots=int(self.cfg.H5_RDCC_NSLOTS),
            rdcc_w0=float(self.cfg.H5_RDCC_W0),
        )

    def __len__(self):
        return len(self.meta)

    @staticmethod
    def _normalize(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean = np.mean(x, axis=1, keepdims=True)
        std = np.std(x, axis=1, keepdims=True) + 1e-6
        y = (x - mean) / std
        return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    def _apply_wavelet_to_channels(self, x_win_normed: np.ndarray) -> np.ndarray:
        if not self.cfg.USE_WAVELET:
            return np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)
        if pywt is None:
            raise RuntimeError("pywt is not installed but USE_WAVELET=True.")
        feats = []
        for c in range(x_win_normed.shape[0]):
            recs = apply_wavelet_transform(
                x_win_normed[c].astype(np.float32, copy=False),
                wavelet=self.cfg.WAVELET_TYPE,
                level=int(self.cfg.WAVELET_LEVEL)
            )
            feats.append(recs.astype(np.float32, copy=False))
        x_wav = np.concatenate(feats, axis=0)
        Lw = x_wav.shape[1]
        x_wav = np.nan_to_num(x_wav[:, :Lw], nan=0.0, posinf=0.0, neginf=0.0)
        return x_wav

    def _apply_stft_to_channels(self, x_win_normed: np.ndarray) -> np.ndarray:
        """
        Returns stacked spectrogram magnitudes with shape:
          (C * freq_bins, n_frames)
        """
        if not self.cfg.USE_SPECTROGRAM:
            return np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)

        n_fft = int(self.cfg.STFT_N_FFT)
        hop = int(self.cfg.STFT_HOP_LENGTH if self.cfg.STFT_HOP_LENGTH and self.cfg.STFT_HOP_LENGTH > 0 else n_fft // 2)
        window = torch.hann_window(n_fft)

        feats = []
        for c in range(x_win_normed.shape[0]):
            sig = torch.tensor(x_win_normed[c], dtype=torch.float32)
            spec = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True)
            spec_mag = spec.abs().numpy().astype(np.float32)  
            feats.append(spec_mag)

        if not feats:
            return np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)

        x_spec = np.vstack(feats)  
        x_spec = np.nan_to_num(x_spec, nan=0.0, posinf=0.0, neginf=0.0)
        return x_spec

    def _load_trace_from_combined(self, file_path: str, trace_name: str) -> np.ndarray:
        h5 = self._h5cache.get(file_path)
        root = _try_get_data_root(h5)

        if "$" in trace_name:
            group_name, rest = trace_name.split("$", 1)
            idx_str = rest.split(",", 1)[0]
            trace_index = int(idx_str)
        else:
            group_name = trace_name
            trace_index = None

        if group_name not in root:
            raise KeyError(f"[H5] group '{group_name}' not found in {file_path}")

        ds = root[group_name]
        x = np.asarray(ds[:] if trace_index is None else ds[trace_index])

        if x.ndim == 1:
            x = x[np.newaxis, :]
        elif x.ndim == 3 and trace_index is None:
            raise ValueError(f"[H5] dataset '{group_name}' is 3D (N,C,L) but trace_index missing.")

        x = _dataset_to_C_L(x, self.cfg.INPUT_CHANNELS)
        x = _ensure_C(x, self.cfg.INPUT_CHANNELS)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _load_trace_from_tremor_event_file(self, per_event_h5: str, net: str, sta: str) -> np.ndarray:
        h5 = self._h5cache.get(per_event_h5)
        root = h5["raw_waveforms"] if "raw_waveforms" in h5 else (h5["data"] if "data" in h5 else h5)

        name1 = f"{net}.{sta}"
        name2 = f"{sta}"

        if name1 in root:
            target = root[name1]
        elif name2 in root:
            target = root[name2]
        else:
            keys = list(root.keys())
            hit = None
            for k in keys:
                if k == name1 or k.endswith(f".{sta}") or k == name2:
                    hit = k
                    break
            if hit is None:
                raise KeyError(f"[TREMOR] cannot find '{name1}' in {per_event_h5}")
            target = root[hit]

        ds = _first_dataset_under(target, max_depth=4)
        if ds is None:
            raise KeyError(f"[TREMOR] found group but no dataset in {per_event_h5}")

        x = np.asarray(ds[...])
        x = _dataset_to_C_L(x, self.cfg.INPUT_CHANNELS)
        x = _ensure_C(x, self.cfg.INPUT_CHANNELS)
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _stable_int_hash(*items) -> int:
        """
        Stable hash for deterministic RNG across runs.
        Avoid Python's built-in hash() because it's randomized per process.
        """
        s = "|".join("" if x is None else str(x) for x in items)
        h = 2166136261
        for ch in s.encode("utf-8", errors="ignore"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return int(h)

    def _rng_for_row(self, idx: int, row: pd.Series, tag: str) -> np.random.RandomState:
        """
        Deterministic RNG for val/test windowing; training continues to use global np.random.
        """
        seed = self._stable_int_hash(
            tag,
            idx,
            row.get("h5_key", ""),
            row.get("trace_name", ""),
            row.get("network", ""),
            row.get("station", ""),
            row.get("year", -1),
        )
        return np.random.RandomState(seed)

    def _select_tremor_window_scheme_b(self, x: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
        if rng is None:
            rng = np.random  

        C, L = x.shape
        T = int(self.cfg.TARGET_LENGTH)

        if L <= T:
            out = np.zeros((C, T), dtype=x.dtype)
            out[:, :L] = x
            return out

        K = int(max(1, self.cfg.TREMOR_MULTI_CROP_K))
        min_sep = int(max(1, round(float(self.cfg.TREMOR_MULTI_CROP_MIN_SEP_S) * float(self.cfg.SAMPLE_RATE))))
        max_start = L - T
        if max_start <= 0:
            return x[:, :T]

        starts = []
        tries = 0
        while (len(starts) < K) and (tries < 50 * K):
            s = int(rng.randint(0, max_start + 1))
            if all(abs(s - s0) >= min_sep for s0 in starts):
                starts.append(s)
            tries += 1
        if not starts:
            starts = [int(rng.randint(0, max_start + 1))]

        best_s = starts[0]
        best_score = -1e9
        for s in starts:
            w = x[:, s:s + T]
            sc = tremor_proxy_score(
                w,
                sr=float(self.cfg.SAMPLE_RATE),
                band_hz=self.cfg.TREMOR_BAND_HZ,
                total_hz=self.cfg.TREMOR_TOTAL_HZ,
                use_corr=bool(self.cfg.TREMOR_PROXY_USE_CORR),
                use_kurtosis_penalty=bool(self.cfg.TREMOR_PROXY_USE_KURTOSIS_PENALTY),
            )
            if np.isfinite(sc) and sc > best_score:
                best_score = sc
                best_s = s
        return np.nan_to_num(x[:, best_s:best_s + T], nan=0.0, posinf=0.0, neginf=0.0)

    def __getitem__(self, idx: int):
        row = self.meta.iloc[idx]
        label = int(row["label"])
        h5_key = row.get("h5_key", None)
        if h5_key is None:
            raise KeyError("Missing 'h5_key' column")

        if h5_key in self.h5_paths:
            file_path = self.h5_paths[h5_key]
            trace_name = row.get("trace_name", None)
            if trace_name is None:
                raise KeyError(f"h5_key='{h5_key}' but missing trace_name")
            x = self._load_trace_from_combined(file_path, str(trace_name))
        else:
            per_event_h5 = str(h5_key)
            net = str(row.get("network", "")).strip()
            sta = str(row.get("station", "")).strip()
            if not net or not sta:
                raise KeyError("tremor per-event requires network/station")
            x = self._load_trace_from_tremor_event_file(per_event_h5, net, sta)

        src_sr = row.get("sampling_rate_hz", None)
        if (src_sr is None) and ("trace_sampling_rate_hz" in row.index):
            src_sr = row.get("trace_sampling_rate_hz", None)
        if (src_sr is None) and (label == 0):
            src_sr = 200.0  
        if src_sr is None or (isinstance(src_sr, float) and np.isnan(src_sr)):
            src_sr = float(self.cfg.SAMPLE_RATE)
        src_sr = float(src_sr)
        dst_sr = float(self.cfg.SAMPLE_RATE)
        pick_idx = int(row.get("pick_index", -1)) if not pd.isna(row.get("pick_index", -1)) else -1
        if abs(src_sr - dst_sr) > 1e-6:
            if (label == 1) and (pick_idx >= 0):
                pick_idx = int(round(pick_idx * (dst_sr / src_sr)))
            x = resample_trace_np(x, src_sr=src_sr, dst_sr=dst_sr)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        C, L = x.shape
        T = int(self.cfg.TARGET_LENGTH)

        if label == 1 and pick_idx >= 0:
            if self.is_training and self.cfg.RANDOMIZE_PICK_POS:
                a, b = self.cfg.PICK_POS_RANGE
                pick_pos_in_win = int(np.random.uniform(a, b) * T)
                start = pick_idx - pick_pos_in_win
            else:
                start = pick_idx - T // 2
            start = max(0, min(start, max(0, L - T)))
            end = start + T
            if end <= L:
                x_win = x[:, start:end]
            else:
                x_win = np.zeros((C, T), dtype=x.dtype)
                avail = max(0, L - start)
                if avail > 0:
                    x_win[:, :avail] = x[:, start:]
            pick_in_win = pick_idx - start
            pick_norm = (pick_in_win / float(T)) if (0 <= pick_in_win < T and self.cfg.ENABLE_PICK_REGRESSION) else -1.0
        else:
            pick_norm = -1.0
            if (label == 2) and bool(self.cfg.TREMOR_MULTI_CROP_ENABLE):
                if "best_start" in row.index and not pd.isna(row.get("best_start", np.nan)):
                    s = int(row["best_start"])
                    s = max(0, min(s, max(0, L - T)))
                    w = x[:, s:s + T]
                    if w.shape[1] < T:
                        pad = np.zeros((C, T), dtype=x.dtype)
                        pad[:, :w.shape[1]] = w
                        w = pad
                    x_win = w
                else:
                    rng = None if self.is_training else self._rng_for_row(idx, row, tag="tremor_scheme_b")
                    x_win = self._select_tremor_window_scheme_b(x, rng=rng)
            else:
                if L <= T:
                    x_win = np.zeros((C, T), dtype=x.dtype)
                    x_win[:, :L] = x
                else:
                    if self.is_training:
                        start = int(np.random.randint(0, L - T + 1))
                    else:
                        rng = self._rng_for_row(idx, row, tag="eval_crop")
                        start = int(rng.randint(0, L - T + 1))
                    x_win = x[:, start:start + T]

            if (label == 2) and self.is_training and self.cfg.TREMOR_RAND_GAIN_ENABLE:
                g = float(np.random.uniform(self.cfg.TREMOR_RAND_GAIN_MIN, self.cfg.TREMOR_RAND_GAIN_MAX))
                x_win = x_win * g
            if label == 0 and self.cfg.REAL_NOISE_SCALE_ENABLE:
                x_win = x_win * float(self.cfg.REAL_NOISE_SCALE)
            if label == 0 and self.cfg.NOISE_RAND_AMP_ENABLE:
                do_aug = self.is_training or (not self.cfg.NOISE_RAND_AMP_TRAIN_ONLY)
                if do_aug and (np.random.rand() < float(self.cfg.NOISE_RAND_AMP_PROB)):
                    k = _log_uniform(self.cfg.NOISE_RAND_AMP_MIN, self.cfg.NOISE_RAND_AMP_MAX)
                    x_win = x_win * float(k)
            if label == 0 and self.cfg.GAUSSIAN_NOISE_ENABLE:
                inject_allowed = self.is_training or (not self.cfg.GAUSSIAN_NOISE_TRAIN_ONLY)
                if inject_allowed and (np.random.rand() < float(self.cfg.GAUSSIAN_NOISE_PROB)):
                    noise = (self.cfg.GAUSSIAN_NOISE_STD * self.cfg.GAUSSIAN_NOISE_SCALE) * \
                            np.random.randn(*x_win.shape).astype(x_win.dtype)
                    x_win = noise if self.cfg.GAUSSIAN_NOISE_MODE == "replace" else (x_win + noise)

        x_win = _ensure_C(x_win, self.cfg.INPUT_CHANNELS)
        x_win = np.nan_to_num(x_win, nan=0.0, posinf=0.0, neginf=0.0)

        if (
            label == 2
            and self.is_training
            and bool(self.cfg.TREMOR_PRE_STFT_FILTER_ENABLE)
        ):
            f0, f1 = self.cfg.TREMOR_PRE_STFT_FILTER_HZ
            x_win = lowpass_or_bandpass_np(
                x_win,
                sr=float(self.cfg.SAMPLE_RATE),
                f_lo=float(f0),
                f_hi=float(f1),
                order=int(self.cfg.TREMOR_PRE_STFT_FILTER_ORDER),
            )

        x_win_normed = self._normalize(x_win)

        x_combined = None

        x_wav = None
        x_spec = None
        if self.cfg.USE_WAVELET:
            x_wav = self._apply_wavelet_to_channels(x_win_normed)
        if self.cfg.USE_SPECTROGRAM:
            x_spec = self._apply_stft_to_channels(x_win_normed)

        if self.cfg.USE_WAVELET and self.cfg.USE_SPECTROGRAM:
            Lw = x_wav.shape[1]
            Ls = x_spec.shape[1]
            L_min = min(Lw, Ls)
            x_wav = x_wav[:, :L_min]
            x_spec = x_spec[:, :L_min]
            x_combined = np.vstack([x_spec, x_wav])
        elif self.cfg.USE_SPECTROGRAM:
            x_combined = x_spec  
        elif self.cfg.USE_WAVELET:
            Lw = x_wav.shape[1]
            if x_win_normed.shape[1] != Lw:
                x_win_normed = x_win_normed[:, :Lw]
            x_combined = np.vstack([x_win_normed, x_wav])
        else:
            x_combined = x_win_normed

        X = torch.as_tensor(np.nan_to_num(x_combined, nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
        y = torch.as_tensor(label, dtype=torch.long)
        pick = torch.as_tensor(float(pick_norm), dtype=torch.float32)

        if not (self.return_meta or self.return_raw_window):
            return X, y, pick

        meta = None
        if self.return_meta:
            meta = {
                "ds_index": int(idx),
                "label": int(label),
                "year": int(row.get("year", -1)) if not pd.isna(row.get("year", -1)) else -1,
                "h5_key": str(h5_key),
                "trace_name": str(row.get("trace_name", "")) if ("trace_name" in row.index) else "",
                "network": str(row.get("network", "")) if ("network" in row.index) else "",
                "station": str(row.get("station", "")) if ("station" in row.index) else "",
                "pick_index": int(row.get("pick_index", -1)) if ("pick_index" in row.index and not pd.isna(row.get("pick_index", -1))) else -1,
                "best_start": int(row.get("best_start", -1)) if ("best_start" in row.index and not pd.isna(row.get("best_start", -1))) else -1,
            }

        raw_win = x_win_normed.astype(np.float32, copy=False) if self.return_raw_window else None
        return X, y, pick, meta, raw_win

    def close(self):
        self._h5cache.close_all()


def collate_with_meta(batch):
    """
    Robust collate:
    - If raw_win is None for any sample, return raw_wins=None (do NOT stack).
    """
    Xs, ys, picks, metas, raws = zip(*batch)
    metas = [m if m is not None else {} for m in metas]

    X = torch.stack([x if torch.is_tensor(x) else torch.as_tensor(x) for x in Xs], dim=0)
    y = torch.as_tensor(ys, dtype=torch.long)
    pick = torch.as_tensor(picks, dtype=torch.float32)

    if any(r is None for r in raws):
        raw_wins = None
    else:
        raw_wins = torch.stack([r if torch.is_tensor(r) else torch.as_tensor(r) for r in raws], dim=0)

    return X, y, pick, metas, raw_wins


# ------------------------
# OFFLINE: Precompute best_start for tremor with constraints
# ------------------------
def _load_single_tremor_trace_for_best_start(
    row: pd.Series,
    cfg: Config,
    h5_paths: Dict[str, str],
    h5cache: H5HandleCache,
) -> Tuple[np.ndarray, float]:
    h5_key = row.get("h5_key", None)
    if h5_key is None:
        raise KeyError("tremor row missing h5_key")

    if h5_key in h5_paths:
        file_path = h5_paths[h5_key]
        trace_name = row.get("trace_name", None)
        if trace_name is None:
            raise KeyError("tremor row missing trace_name for combined h5")
        h5 = h5cache.get(file_path)
        root = _try_get_data_root(h5)

        tn = str(trace_name)
        if "$" in tn:
            group_name, rest = tn.split("$", 1)
            idx_str = rest.split(",", 1)[0]
            trace_index = int(idx_str)
        else:
            group_name = tn
            trace_index = None

        if group_name not in root:
            raise KeyError(f"[H5] group '{group_name}' not found in {file_path}")
        ds = root[group_name]
        x = np.asarray(ds[:] if trace_index is None else ds[trace_index])

        if x.ndim == 1:
            x = x[np.newaxis, :]
        elif x.ndim == 3 and trace_index is None:
            raise ValueError("3D dataset but missing trace_index")
        x = _dataset_to_C_L(x, cfg.INPUT_CHANNELS)
        x = _ensure_C(x, cfg.INPUT_CHANNELS)
    else:
        per_event_h5 = str(h5_key)
        net = str(row.get("network", "")).strip()
        sta = str(row.get("station", "")).strip()
        if not net or not sta:
            raise KeyError("tremor per-event requires network/station")
        h5 = h5cache.get(per_event_h5)
        root = h5["raw_waveforms"] if "raw_waveforms" in h5 else (h5["data"] if "data" in h5 else h5)

        name1 = f"{net}.{sta}"
        name2 = f"{sta}"
        if name1 in root:
            target = root[name1]
        elif name2 in root:
            target = root[name2]
        else:
            keys = list(root.keys())
            hit = None
            for k in keys:
                if k == name1 or k.endswith(f".{sta}") or k == name2:
                    hit = k
                    break
            if hit is None:
                raise KeyError(f"[TREMOR] cannot find {name1} in {per_event_h5}")
            target = root[hit]
        ds = _first_dataset_under(target, max_depth=4)
        if ds is None:
            raise KeyError(f"[TREMOR] no dataset in {per_event_h5}")
        x = np.asarray(ds[...])
        x = _dataset_to_C_L(x, cfg.INPUT_CHANNELS)
        x = _ensure_C(x, cfg.INPUT_CHANNELS)

    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    src_sr = row.get("sampling_rate_hz", None)
    if (src_sr is None) and ("trace_sampling_rate_hz" in row.index):
        src_sr = row.get("trace_sampling_rate_hz", None)
    if src_sr is None or (isinstance(src_sr, float) and np.isnan(src_sr)):
        src_sr = float(cfg.SAMPLE_RATE)

    src_sr = float(src_sr)
    dst_sr = float(cfg.SAMPLE_RATE)
    if abs(src_sr - dst_sr) > 1e-6:
        x = resample_trace_np(x, src_sr=src_sr, dst_sr=dst_sr)

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), float(cfg.SAMPLE_RATE)


def compute_best_start_with_constraints(
    x: np.ndarray,
    sr: float,
    cfg: Config,
) -> Tuple[int, float, Dict[str, float]]:
    x = np.asarray(x)
    C, L = x.shape
    T = int(cfg.TARGET_LENGTH)
    if L <= T:
        return 0, 0.0, {"mode": 0.0}

    max_start = L - T
    K = int(max(32, cfg.TREMOR_MULTI_CROP_K * 8))
    min_sep = int(max(1, round(float(cfg.TREMOR_MULTI_CROP_MIN_SEP_S) * float(cfg.SAMPLE_RATE))))

    starts = []
    tries = 0
    while (len(starts) < K) and (tries < 200 * K):
        s = int(np.random.randint(0, max_start + 1))
        if all(abs(s - s0) >= min_sep for s0 in starts):
            starts.append(s)
        tries += 1
    if not starts:
        starts = [int(np.random.randint(0, max_start + 1))]

    proxy_scores = []
    band_ratios = []
    band_rmses = []
    abs_rmses = []
    corr_means = []

    for s in starts:
        w = x[:, s:s + T]
        sc = tremor_proxy_score(
            w, sr=sr,
            band_hz=cfg.TREMOR_BAND_HZ,
            total_hz=cfg.TREMOR_TOTAL_HZ,
            use_corr=cfg.TREMOR_PROXY_USE_CORR,
            use_kurtosis_penalty=cfg.TREMOR_PROXY_USE_KURTOSIS_PENALTY,
        )
        br, brms = tremor_band_metrics(w, sr=sr, band_hz=cfg.TREMOR_BAND_HZ, total_hz=cfg.TREMOR_TOTAL_HZ)
        ar = float(np.sqrt(np.mean(w.astype(np.float64) ** 2)))
        cm = 0.0
        if C >= 2:
            ww = w.astype(np.float64)
            ww = ww - ww.mean(axis=1, keepdims=True)
            std = ww.std(axis=1) + 1e-12
            good = std > 1e-8
            if np.sum(good) >= 2:
                z = ww[good] / std[good][:, None]
                cvals = []
                for i in range(z.shape[0]):
                    for j in range(i + 1, z.shape[0]):
                        cvals.append(float(np.mean(z[i] * z[j])))
                if cvals:
                    cm = float(np.mean(cvals))

        proxy_scores.append(float(sc))
        band_ratios.append(float(br))
        band_rmses.append(float(brms))
        abs_rmses.append(float(ar))
        corr_means.append(float(cm))

    proxy_scores = np.asarray(proxy_scores, dtype=float)
    band_ratios = np.asarray(band_ratios, dtype=float)
    band_rmses = np.asarray(band_rmses, dtype=float)
    abs_rmses = np.asarray(abs_rmses, dtype=float)
    corr_means = np.asarray(corr_means, dtype=float)

    thr_band_rms = float(np.quantile(band_rmses, cfg.TREMOR_MIN_BAND_RMS_Q)) if len(band_rmses) else 0.0
    thr_abs_rms = float(np.quantile(abs_rmses, cfg.TREMOR_MIN_ABS_RMS_Q)) if len(abs_rmses) else 0.0

    valid = (band_ratios >= float(cfg.TREMOR_MIN_BAND_RATIO)) & (band_rmses >= thr_band_rms) & (abs_rmses >= thr_abs_rms)
    if cfg.TREMOR_REQUIRE_POS_CORR:
        valid = valid & (corr_means > 0.0)

    if np.any(valid):
        idx_best = int(np.argmax(proxy_scores[valid]))
        valid_indices = np.where(valid)[0]
        k_best = int(valid_indices[idx_best])
        best_start = int(starts[k_best])
        best_score = float(proxy_scores[k_best])
        used_valid = 1.0
    else:
        k_best = int(np.argmax(proxy_scores))
        best_start = int(starts[k_best])
        best_score = float(proxy_scores[k_best])
        used_valid = 0.0

    diag = {
        "used_valid": float(used_valid),
        "proxy_score": float(best_score),
        "band_ratio": float(band_ratios[k_best]),
        "band_rms": float(band_rmses[k_best]),
        "abs_rms": float(abs_rmses[k_best]),
        "corr_mean": float(corr_means[k_best]),
        "thr_band_rms": float(thr_band_rms),
        "thr_abs_rms": float(thr_abs_rms),
        "n_candidates": float(len(starts)),
        "n_valid": float(np.sum(valid)),
    }
    return best_start, best_score, diag


def precompute_best_windows_for_tremor_csv(
    tremor_csv_in: str,
    tremor_csv_out: str,
    cfg: Config,
    h5_paths: Dict[str, str],
    overwrite: bool = False,
    max_rows: int = -1,
):
    df = pd.read_csv(tremor_csv_in)
    if max_rows > 0:
        df = df.head(int(max_rows)).copy()

    if "per_event_h5" in df.columns:
        df["h5_key"] = df["per_event_h5"].astype(str)
    elif "h5_key" not in df.columns:
        df["h5_key"] = "tremor_master"

    for col in ["best_start", "best_score", "best_used_valid", "best_band_ratio", "best_band_rms", "best_abs_rms",
                "best_thr_band_rms", "best_thr_abs_rms", "best_n_valid"]:
        if (col not in df.columns) or overwrite:
            df[col] = np.nan

    h5cache = H5HandleCache(
        max_open=int(max(1, cfg.H5_MAX_OPEN)),
        rdcc_nbytes=int(cfg.H5_RDCC_NBYTES),
        rdcc_nslots=int(cfg.H5_RDCC_NSLOTS),
        rdcc_w0=float(cfg.H5_RDCC_W0),
    )

    n = len(df)
    ok = 0
    fail = 0
    print(f"[BESTWIN] Input rows: {n} | overwrite={overwrite} | out={tremor_csv_out}")
    print(f"[BESTWIN] Constraints: min_ratio={cfg.TREMOR_MIN_BAND_RATIO} band_rms_q={cfg.TREMOR_MIN_BAND_RMS_Q} abs_rms_q={cfg.TREMOR_MIN_ABS_RMS_Q} require_pos_corr={cfg.TREMOR_REQUIRE_POS_CORR}")

    for i in range(n):
        if (not overwrite) and (not pd.isna(df.loc[i, "best_start"])):
            ok += 1
            if (i + 1) % 200 == 0:
                print(f"[BESTWIN] progress {i+1}/{n} (kept) ok={ok} fail={fail}")
            continue

        row = df.iloc[i]
        try:
            x, sr = _load_single_tremor_trace_for_best_start(row, cfg, h5_paths, h5cache)
            best_start, best_score, diag = compute_best_start_with_constraints(x, sr, cfg)

            df.loc[i, "best_start"] = int(best_start)
            df.loc[i, "best_score"] = float(best_score)
            df.loc[i, "best_used_valid"] = float(diag.get("used_valid", 0.0))
            df.loc[i, "best_band_ratio"] = float(diag.get("band_ratio", 0.0))
            df.loc[i, "best_band_rms"] = float(diag.get("band_rms", 0.0))
            df.loc[i, "best_abs_rms"] = float(diag.get("abs_rms", 0.0))
            df.loc[i, "best_thr_band_rms"] = float(diag.get("thr_band_rms", 0.0))
            df.loc[i, "best_thr_abs_rms"] = float(diag.get("thr_abs_rms", 0.0))
            df.loc[i, "best_n_valid"] = float(diag.get("n_valid", 0.0))

            ok += 1
        except Exception as e:
            fail += 1
            df.loc[i, "best_start"] = np.nan
            df.loc[i, "best_score"] = np.nan
            if fail <= 10:
                print(f"[BESTWIN][WARN] row {i} failed: {e}")

        if (i + 1) % 200 == 0:
            print(f"[BESTWIN] progress {i+1}/{n} ok={ok} fail={fail}")

    h5cache.close_all()
    df.to_csv(tremor_csv_out, index=False)
    print(f"[BESTWIN] Done. ok={ok} fail={fail} -> {tremor_csv_out}")


# ------------------------
# Model
# ------------------------
class ConvLSTMClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # Determine input channels based on configuration
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


# ------------------------
# Eval
# ------------------------
@torch.no_grad()
def evaluate_with_examples(model, loader, cfg: Config):
    model.eval()
    y_true_all, y_pred_all, y_prob_all, records = [], [], [], []

    for batch in loader:
        X, y, pick, metas, raw_wins = batch
        X = X.to(cfg.DEVICE, non_blocking=True)
        y = y.to(cfg.DEVICE, non_blocking=True)

        logits, _, _ = model(X)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

        y_true_np = y.cpu().numpy().astype(int)
        y_pred_np = pred.cpu().numpy().astype(int)
        y_probs_np = probs.cpu().numpy().astype(float)

        raw_np = None
        if raw_wins is not None:
            raw_np = raw_wins.cpu().numpy().astype(np.float32)

        for i in range(len(y_true_np)):
            m = metas[i] if metas is not None else {}
            rec = {
                **dict(m),
                "y_true": int(y_true_np[i]),
                "y_pred": int(y_pred_np[i]),
                "p_noise": float(y_probs_np[i, 0]),
                "p_eq": float(y_probs_np[i, 1]),
                "p_tremor": float(y_probs_np[i, 2]),
            }
            if raw_np is not None:
                rec["raw_win"] = raw_np[i]
            records.append(rec)

        y_true_all.extend(y_true_np.tolist())
        y_pred_all.extend(y_pred_np.tolist())
        y_prob_all.extend(y_probs_np.tolist())

    y_true_all = np.asarray(y_true_all)
    y_pred_all = np.asarray(y_pred_all)
    y_prob_all = np.asarray(y_prob_all)

    acc = accuracy_score(y_true_all, y_pred_all) if len(y_true_all) else 0.0
    f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0) if len(y_true_all) else 0.0

    auroc = 0.0
    auprc = 0.0
    if len(np.unique(y_true_all)) == 3 and len(y_true_all):
        true_onehot = np.zeros((len(y_true_all), 3), dtype=int)
        true_onehot[np.arange(len(y_true_all)), y_true_all] = 1
        try:
            auroc = roc_auc_score(true_onehot, y_prob_all, multi_class="ovr", average="macro")
        except Exception:
            pass
        try:
            auprc = average_precision_score(true_onehot, y_prob_all, average="macro")
        except Exception:
            pass

    metrics = {"accuracy": float(acc), "f1_macro": float(f1), "auroc_macro": float(auroc), "auprc_macro": float(auprc)}
    return metrics, y_true_all, y_prob_all, y_pred_all, records


# ------------------------
# Training (AMP + compile)
# ------------------------
def train_model(model, train_loader, val_loader, cfg: Config, out_dir: str, device: torch.device):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.LR_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    use_amp = bool(cfg.USE_AMP and device.type == "cuda")
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_macro": [],
        "rec_noise": [],
        "rec_eq": [],
        "rec_tremor": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience = 0
    ckpt_path = os.path.join(out_dir, "checkpoints", "best_model.pt")

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        t_epoch0 = time.perf_counter()
        t_train0 = time.perf_counter()
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            X, y = batch[0], batch[1]
            X = X.to(cfg.DEVICE, non_blocking=True)
            y = y.to(cfg.DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                logits, _, _ = model(X)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if cfg.GRADIENT_CLIP > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item()) * X.size(0)

        t_train1 = time.perf_counter()
        train_loss = total_loss / len(train_loader.dataset) if len(train_loader.dataset) else float("inf")

        t_val0 = time.perf_counter()
        model.eval()
        vloss = 0.0
        y_true_all = []
        y_pred_all = []

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch[0], batch[1]
                X = X.to(cfg.DEVICE, non_blocking=True)
                y = y.to(cfg.DEVICE, non_blocking=True)
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    logits, _, _ = model(X)
                    loss = criterion(logits, y)
                if torch.isfinite(loss):
                    vloss += float(loss.item()) * X.size(0)
                preds = torch.argmax(logits, dim=1)
                y_true_all.append(y.cpu().numpy())
                y_pred_all.append(preds.cpu().numpy())

        t_val1 = time.perf_counter()
        val_loss = vloss / len(val_loader.dataset) if len(val_loader.dataset) else float("inf")
        t_epoch1 = time.perf_counter()

        y_true_all = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
        y_pred_all = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([])

        if len(y_true_all):
            val_acc = accuracy_score(y_true_all, y_pred_all)
            val_f1 = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)
            rec_noise, rec_eq, rec_tr = recall_score(
                y_true_all, y_pred_all, labels=[0, 1, 2], average=None, zero_division=0
            )
        else:
            val_acc = val_f1 = rec_noise = rec_eq = rec_tr = 0.0

        history["epoch"].append(int(epoch))
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_f1_macro"].append(float(val_f1))
        history["rec_noise"].append(float(rec_noise))
        history["rec_eq"].append(float(rec_eq))
        history["rec_tremor"].append(float(rec_tr))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch:03d} | "
            f"Train loss={train_loss:.4f} | Val loss={val_loss:.4f} | "
            f"Acc={val_acc:.3f} F1m={val_f1:.3f} | "
            f"Rec: N={rec_noise:.3f} EQ={rec_eq:.3f} TR={rec_tr:.3f} | "
            f"time: train={(t_train1-t_train0)/60:.2f}m, "
            f"val={(t_val1-t_val0)/60:.2f}m, "
            f"total={(t_epoch1-t_epoch0)/60:.2f}m"
        )

        if val_loss + cfg.EARLY_STOP_MIN_DELTA < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience = 0
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "val_loss": best_val_loss, "config": to_config_dict(cfg)},
                ckpt_path
            )
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch={best_epoch}, best val_loss={best_val_loss:.4f})")
                break

        scheduler.step()

    if not os.path.exists(ckpt_path):
        torch.save(
            {"epoch": cfg.MAX_EPOCHS, "model_state_dict": model.state_dict(), "val_loss": float("inf"), "config": to_config_dict(cfg)},
            ckpt_path
        )

    return history, ckpt_path



# =========================
# VIZ + Example Export
# =========================
def save_history_csv(history: dict, out_csv: str):
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)

def save_test_summary_csv(
    out_csv: str,
    cfg: Config,
    ckpt: dict,
    metrics: dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_name: str,
):
    rep = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["Noise", "Earthquake", "Tremor"],
        output_dict=True,
        zero_division=0,
    )

    row = {
        "run_name": run_name,
        "best_epoch": int(ckpt.get("epoch", -1)),
        "best_val_loss": float(ckpt.get("val_loss", np.nan)),

        "test_accuracy": float(metrics.get("accuracy", np.nan)),
        "test_f1_macro": float(metrics.get("f1_macro", np.nan)),
        "test_auroc_macro": float(metrics.get("auroc_macro", np.nan)),
        "test_auprc_macro": float(metrics.get("auprc_macro", np.nan)),

        "precision_noise": float(rep["Noise"]["precision"]),
        "recall_noise": float(rep["Noise"]["recall"]),
        "f1_noise": float(rep["Noise"]["f1-score"]),

        "precision_eq": float(rep["Earthquake"]["precision"]),
        "recall_eq": float(rep["Earthquake"]["recall"]),
        "f1_eq": float(rep["Earthquake"]["f1-score"]),

        "precision_tremor": float(rep["Tremor"]["precision"]),
        "recall_tremor": float(rep["Tremor"]["recall"]),
        "f1_tremor": float(rep["Tremor"]["f1-score"]),

        "macro_precision": float(rep["macro avg"]["precision"]),
        "macro_recall": float(rep["macro avg"]["recall"]),
        "macro_f1": float(rep["macro avg"]["f1-score"]),

        "weighted_precision": float(rep["weighted avg"]["precision"]),
        "weighted_recall": float(rep["weighted avg"]["recall"]),
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),

        "run_mode": cfg.RUN_MODE,
        "use_wavelet": bool(cfg.USE_WAVELET),
        "use_spectrogram": bool(cfg.USE_SPECTROGRAM),
        "target_length": int(cfg.TARGET_LENGTH),
        "sample_rate": int(cfg.SAMPLE_RATE),
        "batch_size": int(cfg.BATCH_SIZE),
        "lr": float(cfg.LR),
        "weight_decay": float(cfg.WEIGHT_DECAY),
        "dropout": float(cfg.DROPOUT),
        "conv_channels": "-".join(map(str, cfg.CONV_CHANNELS)),
        "lstm_hidden": int(cfg.LSTM_HIDDEN),
        "lstm_layers": int(cfg.LSTM_LAYERS),
        "bidirectional": bool(cfg.BIDIRECTIONAL),
        "stft_n_fft": int(cfg.STFT_N_FFT),
        "stft_hop_length": int(cfg.STFT_HOP_LENGTH),
        "seed": int(cfg.SEED),
    }

    pd.DataFrame([row]).to_csv(out_csv, index=False)

def append_test_summary_csv(master_csv: str, row_csv: str):
    df_row = pd.read_csv(row_csv)
    if os.path.exists(master_csv):
        df_master = pd.read_csv(master_csv)
        df_master = pd.concat([df_master, df_row], ignore_index=True)
    else:
        df_master = df_row.copy()
    df_master.to_csv(master_csv, index=False)

def plot_loss(history: dict, out_png: str):
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    if len(tr) == 0 and len(va) == 0:
        return
    plt.figure()
    if len(tr) > 0:
        plt.plot(tr, label="train_loss")
    if len(va) > 0:
        plt.plot(va, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, out_png: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    if _HAS_SNS:
        sns.heatmap(
            cm, annot=True, fmt="d", cbar=False,
            xticklabels=["Noise", "Earthquake", "Tremor"],
            yticklabels=["Noise", "Earthquake", "Tremor"],
        )
    else:
        plt.imshow(cm, aspect="auto")
        plt.colorbar()
        plt.xticks([0, 1, 2], ["Noise", "Earthquake", "Tremor"])
        plt.yticks([0, 1, 2], ["Noise", "Earthquake", "Tremor"])
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_pr_roc_multiclass(y_true: np.ndarray, y_probs: np.ndarray, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    if y_true is None or y_probs is None or len(y_true) == 0:
        return

    classes = [(0, "Noise"), (1, "Earthquake"), (2, "Tremor")]
    for cid, cname in classes:
        y_bin = (y_true == cid).astype(int)
        p = y_probs[:, cid].astype(float)

        try:
            prec, rec, _ = precision_recall_curve(y_bin, p)
            ap = average_precision_score(y_bin, p)
            plt.figure()
            plt.plot(rec, prec)
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR Curve: {cname} (AP={ap:.4f})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"pr_{cname.lower()}.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] PR plot failed for {cname}: {e}")

        try:
            fpr, tpr, _ = roc_curve(y_bin, p)
            rocA = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve: {cname} (AUC={rocA:.4f})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"roc_{cname.lower()}.png"), dpi=200)
            plt.close()
        except Exception as e:
            print(f"[WARN] ROC plot failed for {cname}: {e}")


def _safe_tag(s: str, max_len: int = 90) -> str:
    s = str(s) if s is not None else ""
    s = re.sub(r"[^A-Za-z0-9_.\-]+", "_", s)
    return s[:max_len] if len(s) > max_len else s


def _class_name(cid: int) -> str:
    return {0: "Noise", 1: "Earthquake", 2: "Tremor"}.get(int(cid), f"Class{cid}")


def _get_true_conf(rec: dict) -> float:
    yt = int(rec.get("y_true", -1))
    if yt == 0:
        return float(rec.get("p_noise", 0.0))
    if yt == 1:
        return float(rec.get("p_eq", 0.0))
    if yt == 2:
        return float(rec.get("p_tremor", 0.0))
    return 0.0


def _get_pred_conf(rec: dict) -> float:
    yp = int(rec.get("y_pred", -1))
    if yp == 0:
        return float(rec.get("p_noise", 0.0))
    if yp == 1:
        return float(rec.get("p_eq", 0.0))
    if yp == 2:
        return float(rec.get("p_tremor", 0.0))
    return 0.0


def _plot_raw_window_png(raw: np.ndarray, cfg: Config, title: str, out_png: str):
    raw = np.asarray(raw)
    if raw.ndim != 2:
        return
    C, L = raw.shape
    t = np.arange(L) / float(cfg.SAMPLE_RATE)

    plt.figure(figsize=(10, 4))
    for ch in range(C):
        plt.plot(t, raw[ch], label=f"ch{ch}")
    plt.title(title[:180])
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized amplitude (window)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def plot_waveform_and_spectrogram(raw: np.ndarray, cfg: Config, title: str, out_png: str):
    raw = np.asarray(raw)
    if raw.ndim != 2:
        return
    C, L = raw.shape
    t = np.arange(L) / float(cfg.SAMPLE_RATE)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    ax_wave, ax_spec = axes[0], axes[1]

    for ch in range(C):
        ax_wave.plot(t, raw[ch], label=f"ch{ch}")
    ax_wave.set_title(title[:180])
    ax_wave.set_ylabel("Normalized Amplitude")
    ax_wave.legend(loc="upper right", fontsize=8)

    n_fft = int(getattr(cfg, "STFT_N_FFT", 256))
    hop = int(getattr(cfg, "STFT_HOP_LENGTH", None) or (n_fft // 2))
    sig = torch.tensor(raw[0], dtype=torch.float32)
    window = torch.hann_window(n_fft)
    spec = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=window, center=False, return_complex=True)
    spec_mag = spec.abs().numpy()
    spec_db = 20 * np.log10(spec_mag + 1e-6)
    F, Tt = spec_db.shape[0], spec_db.shape[1]
    freqs = np.linspace(0, cfg.SAMPLE_RATE / 2, F)
    times = np.linspace(0, t[-1], Tt)
    vmax = spec_db.max()
    vmin = vmax - 80.0
    ax_spec.imshow(
        spec_db, origin="lower", aspect="auto",
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        cmap="viridis", vmin=vmin, vmax=vmax
    )
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_title("Spectrogram (Ch0)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def save_records_csv(records: list, out_csv: str):
    if not records:
        return
    rows = []
    for r in records:
        rr = dict(r)
        rr.pop("raw_win", None)
        rows.append(rr)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def save_examples_ok_per_class(records: list, out_dir: str, cfg: Config, n_per_class: int = 10):
    if not records:
        print("[WARN] No records for OK examples.")
        return

    fig_root = os.path.join(out_dir, "figures", "examples_ok")
    os.makedirs(fig_root, exist_ok=True)

    ok = [
        r for r in records
        if int(r.get("y_true", -9)) == int(r.get("y_pred", -8))
        and (r.get("raw_win", None) is not None)
    ]
    ok = sorted(ok, key=_get_true_conf, reverse=True)

    picked = {0: 0, 1: 0, 2: 0}
    for r in ok:
        yt = int(r["y_true"])
        if yt not in picked:
            continue
        if picked[yt] >= int(n_per_class):
            continue

        cname = _class_name(yt)
        cdir = os.path.join(fig_root, f"class_{yt}_{cname.lower()}")
        os.makedirs(cdir, exist_ok=True)

        net = r.get("network", "")
        sta = r.get("station", "")
        year = r.get("year", "")
        trace_name = r.get("trace_name", "")
        h5_key = r.get("h5_key", "")
        best_start = r.get("best_start", "")
        conf = _get_true_conf(r)

        fn = (
            f"ok_{picked[yt]+1:02d}"
            f"_y{yt}"
            f"_conf{conf:.3f}"
            f"_{_safe_tag(net)}.{_safe_tag(sta)}"
            f"_{_safe_tag(year)}"
            f"_{_safe_tag(trace_name)}.png"
        )
        out_png = os.path.join(cdir, fn)

        title = f"OK {cname} | conf(true)={conf:.3f} | net={net} sta={sta} year={year} best_start={best_start} h5_key={h5_key}"
        if cfg.USE_SPECTROGRAM:
            plot_waveform_and_spectrogram(r["raw_win"], cfg, title, out_png)
        else:
            _plot_raw_window_png(r["raw_win"], cfg, title, out_png)
        picked[yt] += 1

        if sum(picked.values()) >= 3 * int(n_per_class):
            break

    for k in (0, 1, 2):
        cname = _class_name(k)
        print(f"[INFO] OK examples saved for {cname}: {picked[k]}/{n_per_class} -> {os.path.join(fig_root, f'class_{k}_{cname.lower()}')}")


def save_examples_bad_per_class(records: list, out_dir: str, cfg: Config, n_per_class: int = 10):
    if not records:
        print("[WARN] No records for BAD examples.")
        return

    fig_root = os.path.join(out_dir, "figures", "examples_bad")
    os.makedirs(fig_root, exist_ok=True)

    bad = [
        r for r in records
        if int(r.get("y_true", -9)) != int(r.get("y_pred", -8))
        and (r.get("raw_win", None) is not None)
    ]
    bad = sorted(bad, key=_get_pred_conf, reverse=True)

    picked = {0: 0, 1: 0, 2: 0}
    for r in bad:
        yt = int(r["y_true"])
        yp = int(r["y_pred"])
        if yt not in picked:
            continue
        if picked[yt] >= int(n_per_class):
            continue

        tname = _class_name(yt)
        pname = _class_name(yp)
        cdir = os.path.join(fig_root, f"true_{yt}_{tname.lower()}")
        os.makedirs(cdir, exist_ok=True)

        net = r.get("network", "")
        sta = r.get("station", "")
        year = r.get("year", "")
        trace_name = r.get("trace_name", "")
        h5_key = r.get("h5_key", "")
        best_start = r.get("best_start", "")

        conf_true = _get_true_conf(r)
        conf_pred = _get_pred_conf(r)

        fn = (
            f"bad_{picked[yt]+1:02d}"
            f"_true{yt}_pred{yp}"
            f"_pPred{conf_pred:.3f}_pTrue{conf_true:.3f}"
            f"_{_safe_tag(net)}.{_safe_tag(sta)}"
            f"_{_safe_tag(year)}"
            f"_{_safe_tag(trace_name)}.png"
        )
        out_png = os.path.join(cdir, fn)

        title = (
            f"BAD true={tname} pred={pname} | p(pred)={conf_pred:.3f} p(true)={conf_true:.3f} | "
            f"net={net} sta={sta} year={year} best_start={best_start} h5_key={h5_key}"
        )
        if cfg.USE_SPECTROGRAM:
            plot_waveform_and_spectrogram(r["raw_win"], cfg, title, out_png)
        else:
            _plot_raw_window_png(r["raw_win"], cfg, title, out_png)
        picked[yt] += 1

        if sum(picked.values()) >= 3 * int(n_per_class):
            break

    for k in (0, 1, 2):
        cname = _class_name(k)
        print(f"[INFO] BAD examples saved for true={cname}: {picked[k]}/{n_per_class} -> {os.path.join(fig_root, f'true_{k}_{cname.lower()}')}")


def save_tremor_fp_fn(records: list, out_dir: str, cfg: Config, n_each: int = 10):
    if not records:
        return

    fig_root = os.path.join(out_dir, "figures", "tremor_fp_fn")
    os.makedirs(fig_root, exist_ok=True)

    fn_list = [
        r for r in records
        if int(r.get("y_true", -1)) == 2 and int(r.get("y_pred", -1)) != 2 and (r.get("raw_win", None) is not None)
    ]
    fn_list = sorted(fn_list, key=lambda r: float(r.get("p_tremor", 0.0)), reverse=True)[: int(n_each)]

    fp_list = [
        r for r in records
        if int(r.get("y_true", -1)) != 2 and int(r.get("y_pred", -1)) == 2 and (r.get("raw_win", None) is not None)
    ]
    fp_list = sorted(fp_list, key=lambda r: float(r.get("p_tremor", 0.0)), reverse=True)[: int(n_each)]

    def _save_list(lst, sub, tag):
        d = os.path.join(fig_root, sub)
        os.makedirs(d, exist_ok=True)
        for i, r in enumerate(lst, 1):
            yt = int(r.get("y_true", -1))
            yp = int(r.get("y_pred", -1))
            net = r.get("network", "")
            sta = r.get("station", "")
            year = r.get("year", "")
            trace_name = r.get("trace_name", "")
            pt = float(r.get("p_tremor", 0.0))
            ptrue = _get_true_conf(r)
            ppred = _get_pred_conf(r)
            fnm = (
                f"{tag}_{i:02d}"
                f"_true{yt}_pred{yp}"
                f"_pT{pt:.3f}_pPred{ppred:.3f}_pTrue{ptrue:.3f}"
                f"_{_safe_tag(net)}.{_safe_tag(sta)}_{_safe_tag(year)}_{_safe_tag(trace_name)}.png"
            )
            out_path = os.path.join(d, fnm)
            title = f"{tag} | true={_class_name(yt)} pred={_class_name(yp)} | p_tremor={pt:.3f} p_pred={ppred:.3f} p_true={ptrue:.3f}"
            if cfg.USE_SPECTROGRAM:
                plot_waveform_and_spectrogram(r["raw_win"], cfg, title, out_path)
            else:
                _plot_raw_window_png(r["raw_win"], cfg, title, out_path)

    _save_list(fn_list, "FN_true_tremor_pred_not", "TREMOR_FN")
    _save_list(fp_list, "FP_pred_tremor_true_not", "TREMOR_FP")

    print(f"[INFO] Tremor FN saved: {len(fn_list)}/{n_each} -> {os.path.join(fig_root, 'FN_true_tremor_pred_not')}")
    print(f"[INFO] Tremor FP saved: {len(fp_list)}/{n_each} -> {os.path.join(fig_root, 'FP_pred_tremor_true_not')}")


# ------------------------
# misc
# ------------------------
def balance_three_way(df0, df1, df2, seed=42):
    n = min(len(df0), len(df1), len(df2))
    return df0.sample(n=n, random_state=seed), df1.sample(n=n, random_state=seed), df2.sample(n=n, random_state=seed)


def safe_mkdirs(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    for sub in ["checkpoints", "saved_models", "figures"]:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)


def run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_year_list(s: str):
    if s is None:
        return tuple()
    s = str(s).strip()
    if not s:
        return tuple()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        try:
            out.append(int(p))
        except Exception:
            pass
    return tuple(out)


# ------------------------
# Main
# ------------------------
def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_mode", default="cpu_smoke", choices=["cpu_smoke", "gpu_train"])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--test_tremor_years", type=str, default="2025")
    ap.add_argument("--test_eq_years", type=str, default="2021,2022")
    ap.add_argument("--test_noise_years", type=str, default="2015")

    ap.add_argument("--tremor_csv", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/tremor_channels_master_2017-2025.csv")
    ap.add_argument("--tremor_h5_master", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/tremor_raw_master_2017-2025.hdf5")
    ap.add_argument("--noise_csv", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/metadata000001.csv")
    ap.add_argument("--noise_h5", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/waveforms000001.hdf5")
    ap.add_argument("--comcat_csv", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/comcat_metadata.csv")
    ap.add_argument("--comcat_h5", default="/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/comcat_waveforms.hdf5")

    ap.add_argument("--save_dir", default="/home/bxd240002/scratch/Archer/offshore_tremor/results/results_three_class")
    ap.add_argument("--no_viz", action="store_true")
    ap.add_argument("--fast_test", action="store_true")
    ap.add_argument("--torch_threads", type=int, default=1)

    ap.add_argument("--tremor_k", type=int, default=None)
    ap.add_argument("--tremor_band", type=str, default=None)
    ap.add_argument("--tremor_total", type=str, default=None)

    ap.add_argument("--precompute_best_windows", action="store_true")
    ap.add_argument("--best_out_csv", type=str, default=None)
    ap.add_argument("--best_overwrite", action="store_true")
    ap.add_argument("--best_max_rows", type=int, default=-1)

    ap.add_argument("--best_min_ratio", type=float, default=None)
    ap.add_argument("--best_band_rms_q", type=float, default=None)
    ap.add_argument("--best_abs_rms_q", type=float, default=None)
    ap.add_argument("--best_require_pos_corr", action="store_true")

    ap.add_argument("--noise_rand_amp_enable", action="store_true")
    ap.add_argument("--noise_rand_amp_disable", action="store_true")
    ap.add_argument("--noise_amp_min", type=float, default=None)
    ap.add_argument("--noise_amp_max", type=float, default=None)
    ap.add_argument("--noise_amp_apply_eval", action="store_true")

    ap.add_argument("--use_spectrogram", action="store_true")
    ap.add_argument("--spectrogram_nfft", type=int, default=None)
    ap.add_argument("--spectrogram_hop", type=int, default=None)

    args = ap.parse_args()

    cfg = Config(RUN_MODE=args.run_mode, SEED=args.seed, SAVE_DIR=args.save_dir)
    cfg.TEST_TREMOR_YEARS = _parse_year_list(args.test_tremor_years)
    cfg.TEST_EQ_YEARS = _parse_year_list(args.test_eq_years)
    cfg.TEST_NOISE_YEARS = _parse_year_list(args.test_noise_years)
    cfg.apply_profile()

    if args.tremor_k is not None:
        cfg.TREMOR_MULTI_CROP_K = int(args.tremor_k)
    if args.tremor_band is not None:
        a, b = args.tremor_band.split(",")
        cfg.TREMOR_BAND_HZ = (float(a), float(b))
    if args.tremor_total is not None:
        a, b = args.tremor_total.split(",")
        cfg.TREMOR_TOTAL_HZ = (float(a), float(b))

    if args.no_viz:
        cfg.ENABLE_VIZ = False
    if args.fast_test:
        cfg.FAST_TEST = True

    if args.best_min_ratio is not None:
        cfg.TREMOR_MIN_BAND_RATIO = float(args.best_min_ratio)
    if args.best_band_rms_q is not None:
        cfg.TREMOR_MIN_BAND_RMS_Q = float(args.best_band_rms_q)
    if args.best_abs_rms_q is not None:
        cfg.TREMOR_MIN_ABS_RMS_Q = float(args.best_abs_rms_q)
    if args.best_require_pos_corr:
        cfg.TREMOR_REQUIRE_POS_CORR = True

    if args.noise_rand_amp_enable:
        cfg.NOISE_RAND_AMP_ENABLE = True
    if args.noise_rand_amp_disable:
        cfg.NOISE_RAND_AMP_ENABLE = False
    if args.noise_amp_min is not None:
        cfg.NOISE_RAND_AMP_MIN = float(args.noise_amp_min)
    if args.noise_amp_max is not None:
        cfg.NOISE_RAND_AMP_MAX = float(args.noise_amp_max)
    cfg.NOISE_RAND_AMP_TRAIN_ONLY = (not args.noise_amp_apply_eval)

    if args.use_spectrogram:
        cfg.USE_SPECTROGRAM = True
    if args.spectrogram_nfft is not None:
        cfg.STFT_N_FFT = int(args.spectrogram_nfft)
    if args.spectrogram_hop is not None:
        cfg.STFT_HOP_LENGTH = int(args.spectrogram_hop)

    set_seed(cfg.SEED)
    torch.set_num_threads(int(args.torch_threads))

    device = infer_device(force_cpu=cfg.FORCE_CPU)
    cfg.DEVICE = str(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    h5_paths = {"comcat": args.comcat_h5, "noise": args.noise_h5}
    if os.path.exists(args.tremor_h5_master):
        h5_paths["tremor_master"] = args.tremor_h5_master

    if args.precompute_best_windows:
        out_csv = args.best_out_csv or (args.tremor_csv + ".best.csv")
        precompute_best_windows_for_tremor_csv(
            tremor_csv_in=args.tremor_csv,
            tremor_csv_out=out_csv,
            cfg=cfg,
            h5_paths=h5_paths,
            overwrite=bool(args.best_overwrite),
            max_rows=int(args.best_max_rows),
        )
        return

    rid = run_id()
    out_dir = os.path.join(cfg.SAVE_DIR, f"{cfg.RUN_MODE}_{rid}")
    safe_mkdirs(out_dir)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(to_config_dict(cfg), f, indent=2)

    print(f"[INFO] RUN_MODE={cfg.RUN_MODE} device={cfg.DEVICE}")
    print(f"[INFO] USE_WAVELET={cfg.USE_WAVELET} USE_SPECTROGRAM={cfg.USE_SPECTROGRAM} TARGET_LENGTH={cfg.TARGET_LENGTH} BATCH_SIZE={cfg.BATCH_SIZE}")
    if cfg.USE_SPECTROGRAM:
        print(f"[INFO] STFT: n_fft={cfg.STFT_N_FFT} hop={cfg.STFT_HOP_LENGTH}")
    print(f"[INFO] TREMOR best-window: min_ratio={cfg.TREMOR_MIN_BAND_RATIO} band_rms_q={cfg.TREMOR_MIN_BAND_RMS_Q} abs_rms_q={cfg.TREMOR_MIN_ABS_RMS_Q}")
    print(f"[INFO] NOISE rand-amp: enable={cfg.NOISE_RAND_AMP_ENABLE} range=[{cfg.NOISE_RAND_AMP_MIN}, {cfg.NOISE_RAND_AMP_MAX}] train_only={cfg.NOISE_RAND_AMP_TRAIN_ONLY}")
    print(f"[INFO] OUT_DIR={out_dir}")

    df_quake = pd.read_csv(args.comcat_csv)
    df_tremor = pd.read_csv(args.tremor_csv)
    df_noise = pd.read_csv(args.noise_csv)

    df_quake["label"] = 1
    df_noise["label"] = 0
    df_tremor["label"] = 2

    df_quake["h5_key"] = "comcat"
    df_noise["h5_key"] = "noise"
    if "per_event_h5" in df_tremor.columns:
        df_tremor["h5_key"] = df_tremor["per_event_h5"].astype(str)
    else:
        df_tremor["h5_key"] = "tremor_master"

    if "trace_P_arrival_sample" in df_quake.columns:
        df_quake["pick_index"] = pd.to_numeric(df_quake["trace_P_arrival_sample"], errors="coerce").fillna(-1).astype(int)
    else:
        df_quake["pick_index"] = -1

    df_noise["pick_index"] = -1
    df_tremor["pick_index"] = -1

    valid_quakes = df_quake["pick_index"] >= 0
    if int((~valid_quakes).sum()) > 0:
        df_quake = df_quake[valid_quakes].copy()

    if ("batch_key" in df_noise.columns) and ("batch_index" in df_noise.columns):
        df_noise["trace_name"] = df_noise.apply(lambda r: f"{r['batch_key']}${int(r['batch_index'])}", axis=1)
    else:
        raise KeyError("[NOISE] metadata requires batch_key and batch_index")

    if "trace_name" not in df_quake.columns:
        raise KeyError("[COMCAT] comcat_metadata.csv must contain trace_name")

    df_quake = add_year_column(df_quake, kind="comcat")
    df_tremor = add_year_column(df_tremor, kind="tremor")
    df_noise = add_year_column(df_noise, kind="noise")

    if 0 < cfg.SAMPLE_FRAC < 1.0:
        df_quake = df_quake.sample(frac=cfg.SAMPLE_FRAC, random_state=cfg.SEED)
        df_tremor = df_tremor.sample(frac=cfg.SAMPLE_FRAC, random_state=cfg.SEED)
        df_noise = df_noise.sample(frac=cfg.SAMPLE_FRAC, random_state=cfg.SEED)

    if cfg.BALANCE_DATASET:
        df_noise, df_quake, df_tremor = balance_three_way(df_noise, df_quake, df_tremor, seed=cfg.SEED)

    df_all = pd.concat([df_quake, df_tremor, df_noise], ignore_index=True)
    train_df, val_df, test_df = split_by_class_year_sets(df_all, cfg)

    train_ds = SeismicDataset(train_df, h5_paths, cfg, is_training=True, return_meta=False, return_raw_window=False)
    val_ds = SeismicDataset(val_df, h5_paths, cfg, is_training=False, return_meta=False, return_raw_window=False)
    test_ds = SeismicDataset(test_df, h5_paths, cfg, is_training=False, return_meta=True, return_raw_window=(not cfg.FAST_TEST))

    use_cuda = (device.type == "cuda")
    pin_memory = bool(cfg.PIN_MEMORY and use_cuda)
    dl_kwargs = {}
    if cfg.NUM_WORKERS > 0:
        dl_kwargs["persistent_workers"] = bool(cfg.PERSISTENT_WORKERS)
        dl_kwargs["prefetch_factor"] = int(cfg.PREFETCH_FACTOR)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn if cfg.NUM_WORKERS > 0 else None,
        **dl_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory,
        worker_init_fn=worker_init_fn if cfg.NUM_WORKERS > 0 else None,
        **dl_kwargs
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=pin_memory,
        collate_fn=collate_with_meta,
        worker_init_fn=worker_init_fn if cfg.NUM_WORKERS > 0 else None,
        **dl_kwargs
    )

    model = ConvLSTMClassifier(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] trainable params = {n_params:,}")

    if cfg.USE_COMPILE and device.type == "cuda":
        try:
            model = torch.compile(model)
            print("[INFO] torch.compile enabled.")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")

    history, ckpt_path = train_model(model, train_loader, val_loader, cfg, out_dir, device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    save_history_csv(history, os.path.join(out_dir, "training_history.csv"))

    final_path = os.path.join(out_dir, "saved_models", "final_model.pt")
    torch.save(
        {
            "epoch": int(ckpt.get("epoch", -1)),
            "val_loss": float(ckpt.get("val_loss", float("nan"))),
            "model_state_dict": model.state_dict(),
            "config": to_config_dict(cfg),
        },
        final_path
    )
    print(f"[INFO] Saved final(best) model to: {final_path}")

    metrics, y_true, y_probs, y_pred, records = evaluate_with_examples(model, test_loader, cfg)
    print("\n[TEST] Metrics:", metrics)
    save_test_summary_csv(
        out_csv=os.path.join(out_dir, "test_summary.csv"),
        cfg=cfg,
        ckpt=ckpt,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        run_name=os.path.basename(out_dir),
    )
    append_test_summary_csv(
        master_csv=os.path.join(cfg.SAVE_DIR, "experiment_summary.csv"),
        row_csv=os.path.join(out_dir, "test_summary.csv"),
    )

    print("\n[TEST] Classification report:")
    print(classification_report(y_true, y_pred, labels=[0, 1, 2], target_names=["Noise", "Earthquake", "Tremor"], zero_division=0))

    if cfg.ENABLE_VIZ:
        fig_dir = os.path.join(out_dir, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        plot_loss(history, os.path.join(fig_dir, "loss_curve.png"))
        plot_confusion_matrix(y_true, y_pred, os.path.join(fig_dir, "confusion_matrix.png"))
        plot_pr_roc_multiclass(y_true, y_probs, os.path.join(fig_dir, "curves"))

        save_records_csv(records, os.path.join(out_dir, "test_records.csv"))

        has_raw = any(("raw_win" in r) and (r["raw_win"] is not None) for r in records)
        if not has_raw:
            print("[WARN] No raw_win in records (FAST_TEST or return_raw_window=False). Skipping waveform example exports.")
        else:
            save_examples_ok_per_class(records, out_dir, cfg, n_per_class=int(cfg.N_EXAMPLES_OK))
            save_examples_bad_per_class(records, out_dir, cfg, n_per_class=int(cfg.N_EXAMPLES_BAD))
            save_tremor_fp_fn(records, out_dir, cfg, n_each=int(cfg.N_EXAMPLES_BAD))

    train_ds.close(); val_ds.close(); test_ds.close()

    t1 = time.perf_counter()
    print(f"[TIME] Total runtime: {(t1 - t0)/60:.2f} min ({t1 - t0:.1f} s)")
    print("[DONE]")


if __name__ == "__main__":
    main()
