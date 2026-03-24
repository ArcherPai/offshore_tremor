#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AUTO-RUN Robustness Test (TRUE 3-CLASS) for Conv+LSTM Seismic Detector
Label mapping: 0=Noise, 1=Earthquake, 2=Tremor

This robustness script:
- Runs with NO CLI args (edit USER EDIT section only)
- Uses TRUE 3-class batches (Noise + EQ + Tremor simultaneously)
- Uses Tremor Scheme-B style multi-crop + proxy selection
- Uses bounded per-process HDF5 LRU handle cache to avoid Errno 24

Two robustness scenarios (sweeping k = 10^log10_k):
1) REAL_NOISE_AMP:
   Noise:    k * real_noise
   EQ:       eq_window + k * real_noise
   Tremor:   tremor_window(selected by Scheme-B) + k * real_noise

2) REAL_PLUS_WHITE_AMP:
   Base:     real_noise + k * white_noise
   Noise:    base
   EQ:       eq_window + base
   Tremor:   tremor_window + base

NEW (requested):
- Apply the SAME "tremor-only pre-STFT filter" in robustness:
  * Filter tremor window BEFORE mixing noise and BEFORE z-score/STFT/wavelet.
  * Default: bandpass/lowpass (0–20 Hz) (implemented as lowpass if f_lo<=0).

Outputs (timestamped run dir):
- CSV with metrics per scenario/k
- JSON summary
- Figures:
    * overall accuracy vs log10(k)
    * macro-F1 vs log10(k)
    * per-class recall vs log10(k)
    * confusion matrices at selected k values
"""

import os
import json
import math
import random
from datetime import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import h5py

import torch
import torch.nn as nn

from sklearn.metrics import f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# optional wavelet
try:
    import pywt
    _HAS_PYWT = True
except Exception:
    pywt = None
    _HAS_PYWT = False

# optional scipy filter (preferred)
try:
    from scipy.signal import butter, sosfiltfilt
    _HAS_SCIPY = True
except Exception:
    butter = None
    sosfiltfilt = None
    _HAS_SCIPY = False


# =========================
# USER EDIT (paths + knobs)
# =========================
MODEL_PT = "/home/bxd240002/scratch/Archer/offshore_tremor/results/results_three_class_stft/gpu_train_20260317_205228/checkpoints/best_model.pt"
CONFIG_JSON = None  # If ckpt has ckpt["config"] (yours does), keep None.

COMCAT_CSV = "/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/comcat_metadata.csv"
COMCAT_H5  = "/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/comcat_waveforms.hdf5"

NOISE_CSV = "/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/metadata000001.csv"
NOISE_H5  = "/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/waveforms000001.hdf5"

TREMOR_CSV = "/home/bxd240002/scratch/Archer/offshore_tremor/data/labels/tremor_channels_master_2017-2025_best_modified.csv"
TREMOR_H5_MASTER = "/home/bxd240002/scratch/Archer/offshore_tremor/data/hdf5/tremor_raw_master_2017-2025.hdf5"

OUT_DIR = "/home/bxd240002/scratch/Archer/offshore_tremor/results/robustness_three_class_true3_autorun"

SEED = 42
FORCE_CPU = True
H5_MAX_OPEN = 4

# sweep k=10^log10_k
LOG10_KS = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
STEPS_PER_K = 30
BATCH_SIZE = 60
SAMPLES_PER_CLASS_POOL = 5000

# white noise
WHITE_STD = 1.0

# confusion matrices to save (by log10_k)
CONFUSION_PLOT_LOG10K = [0, 3, 6]


# =========================
# Utilities
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if FORCE_CPU:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _try_get_data_root(h5: h5py.File):
    return h5["data"] if "data" in h5 else h5


class H5HandleCache:
    """Per-process LRU HDF5 handle cache (prevents too many open files)."""
    def __init__(self, max_open=4):
        self.max_open = int(max(1, max_open))
        self._pid = os.getpid()
        self._lru: "OrderedDict[str, h5py.File]" = OrderedDict()

    def _reset_if_forked(self):
        pid = os.getpid()
        if pid != self._pid:
            self.close_all()
            self._pid = pid
            self._lru = OrderedDict()

    def get(self, filepath: str) -> h5py.File:
        self._reset_if_forked()
        fp = str(filepath)
        if fp in self._lru:
            f = self._lru.pop(fp)
            self._lru[fp] = f
            return f

        f = h5py.File(fp, "r")
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


def _dataset_to_C_L(x: np.ndarray, input_channels: int) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 1:
        x = x[np.newaxis, :]

    elif x.ndim == 2:
        # prefer (C,L)
        if x.shape[1] in (1, 2, 3, 4) and x.shape[0] > x.shape[1]:
            x = x.T

    elif x.ndim == 3:
        # take first sample
        x = x[0]
        if x.ndim == 2 and x.shape[1] in (1, 2, 3, 4) and x.shape[0] > x.shape[1]:
            x = x.T

    x = np.squeeze(x)
    if x.ndim == 1:
        x = x[np.newaxis, :]

    if x.ndim != 2:
        raise ValueError(f"Cannot coerce to 2D, shape={x.shape}")

    if x.shape[0] > input_channels:
        x = x[:input_channels, :]
    return x


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
    if C == C_target:
        return x
    if C > C_target:
        return x[:C_target, :]
    out = np.zeros((C_target, L), dtype=x.dtype)
    out[:C, :] = x
    return out


def _zscore_channels(x: np.ndarray, eps=1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + eps
    y = (x - m) / s
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def extract_center_window(x: np.ndarray, target_length: int) -> np.ndarray:
    C, L = x.shape
    T = int(target_length)
    if L <= T:
        out = np.zeros((C, T), dtype=x.dtype)
        out[:, :L] = x
        return out
    s = max(0, (L - T) // 2)
    return x[:, s:s + T]


def extract_random_window(x: np.ndarray, target_length: int) -> np.ndarray:
    C, L = x.shape
    T = int(target_length)
    if L <= T:
        out = np.zeros((C, T), dtype=x.dtype)
        out[:, :L] = x
        return out
    s = np.random.randint(0, L - T + 1)
    return x[:, s:s + T]


def gen_white_noise(C: int, L: int, std: float = 1.0) -> np.ndarray:
    return (std * np.random.randn(C, L)).astype(np.float32)


# -------------------------
# Resampling (matches your training approach)
# -------------------------
def resample_trace_np(x: np.ndarray, src_sr: float, dst_sr: float) -> np.ndarray:
    """
    Robust 1D linear resample for (C,L) array.
    Handles edge cases where L==0 or L==1 to avoid np.interp errors.
    """
    src_sr = float(src_sr)
    dst_sr = float(dst_sr)

    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"resample_trace_np expects 2D (C,L), got shape={x.shape}")

    C, L = x.shape

    # ---- critical edge cases ----
    if L <= 0:
        return np.zeros((C, 1), dtype=np.float32)

    if L == 1:
        if src_sr <= 0 or dst_sr <= 0 or abs(src_sr - dst_sr) < 1e-6:
            return x.astype(np.float32, copy=False)
        L_new = int(max(1, round(L * (dst_sr / src_sr))))
        return np.repeat(x.astype(np.float32, copy=False), repeats=L_new, axis=1)

    # ---- normal cases ----
    if src_sr <= 0 or dst_sr <= 0 or abs(src_sr - dst_sr) < 1e-6:
        return x.astype(np.float32, copy=False)

    dur = (L - 1) / src_sr
    if dur <= 0:
        return x.astype(np.float32, copy=False)

    L_new = int(round(dur * dst_sr)) + 1
    if L_new < 2:
        L_new = 2

    t_old = np.linspace(0.0, dur, num=L, dtype=np.float64)
    t_new = np.linspace(0.0, dur, num=L_new, dtype=np.float64)

    y = np.zeros((C, L_new), dtype=np.float32)
    for c in range(C):
        fp = x[c].astype(np.float64, copy=False)
        y[c] = np.interp(t_new, t_old, fp).astype(np.float32)

    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


# -------------------------
# Tremor-only pre-STFT filter (NEW)
# -------------------------
def lowpass_or_bandpass_np(x: np.ndarray, sr: float, f_lo: float, f_hi: float, order: int = 4) -> np.ndarray:
    """
    Apply (C,L) zero-phase SOS filter if scipy is available, otherwise FFT mask fallback.

    Behavior:
    - if f_lo <= 0 and f_hi > 0: lowpass at f_hi
    - if f_lo > 0 and f_hi >= nyq: highpass at f_lo
    - if f_lo > 0 and f_hi > 0 and f_hi < nyq: bandpass [f_lo, f_hi]
    - if invalid (no effective filtering): return x unchanged
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[np.newaxis, :]
    if x.ndim != 2:
        raise ValueError(f"filter expects 2D (C,L), got {x.shape}")

    sr = float(sr)
    if sr <= 0:
        return x

    nyq = 0.5 * sr
    f_lo = float(f_lo)
    f_hi = float(f_hi)
    order = int(max(1, order))

    # clamp to valid range
    f_lo_c = max(0.0, min(f_lo, nyq * 0.999))
    f_hi_c = max(0.0, min(f_hi, nyq * 0.999))

    # decide mode
    mode = None
    wn = None
    if f_lo_c <= 0.0 and f_hi_c > 0.0:
        mode = "lowpass"
        wn = f_hi_c / nyq
    elif f_lo_c > 0.0 and f_hi_c >= nyq * 0.999:
        mode = "highpass"
        wn = f_lo_c / nyq
    elif f_lo_c > 0.0 and f_hi_c > f_lo_c:
        mode = "bandpass"
        wn = [f_lo_c / nyq, f_hi_c / nyq]
    else:
        return x

    # scipy path (preferred)
    if _HAS_SCIPY:
        try:
            sos = butter(order, wn, btype=mode, output="sos")
            y = np.zeros_like(x, dtype=np.float32)
            for c in range(x.shape[0]):
                y[c] = sosfiltfilt(sos, x[c]).astype(np.float32, copy=False)
            return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception:
            pass  # fall back to FFT mask below

    # FFT fallback (brick-wall)
    C, L = x.shape
    if L < 8:
        return x
    freqs = np.fft.rfftfreq(L, d=1.0 / sr)
    X = np.fft.rfft(x, axis=1)

    if mode == "lowpass":
        mask = freqs <= f_hi_c
    elif mode == "highpass":
        mask = freqs >= f_lo_c
    else:  # bandpass
        mask = (freqs >= f_lo_c) & (freqs <= f_hi_c)

    X *= mask[np.newaxis, :]
    y = np.fft.irfft(X, n=L, axis=1).astype(np.float32)
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# Wavelet
# =========================
def apply_wavelet_transform(signal_1d: np.ndarray, wavelet="db4", level=3) -> np.ndarray:
    if not _HAS_PYWT:
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


def _wavelet_only(x_win_normed: np.ndarray, cfg: dict) -> np.ndarray:
    if not bool(cfg.get("USE_WAVELET", False)):
        return np.zeros((0, x_win_normed.shape[1]), dtype=np.float32)
    feats = []
    for c in range(x_win_normed.shape[0]):
        recs = apply_wavelet_transform(
            x_win_normed[c],
            wavelet=cfg.get("WAVELET_TYPE", "db4"),
            level=int(cfg.get("WAVELET_LEVEL", 3)),
        ).astype(np.float32, copy=False)
        feats.append(recs)
    x_wav = np.concatenate(feats, axis=0)  # (C*(level+1), Lw)
    return np.nan_to_num(x_wav, nan=0.0, posinf=0.0, neginf=0.0)


def _stft_stack(x_win_normed: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Returns (C*freq_bins, n_frames) spectrogram magnitude, matching training.
    """
    n_fft = int(cfg.get("STFT_N_FFT", 256))
    hop = int(cfg.get("STFT_HOP_LENGTH", 0)) or (n_fft // 2)

    window = torch.hann_window(n_fft)

    feats = []
    for c in range(x_win_normed.shape[0]):
        sig = torch.tensor(x_win_normed[c], dtype=torch.float32)
        spec = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=window,
                          center=False, return_complex=True)
        spec_mag = spec.abs().numpy().astype(np.float32)  # (freq_bins, n_frames)
        feats.append(spec_mag)

    if not feats:
        return np.zeros((0, 1), dtype=np.float32)

    x_spec = np.vstack(feats)  # (C*freq_bins, n_frames)
    return np.nan_to_num(x_spec, nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# Tremor Scheme-B proxy score
# =========================
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
        else:
            return 0.0

    C, L = x_win.shape
    if L < 8:
        return 0.0

    x = x_win.astype(np.float64, copy=False)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

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
            if len(corrs) > 0:
                pos = [c for c in corrs if c > 0.0]
                corr_bonus = float(np.mean(pos)) if len(pos) else 0.0
                if np.isfinite(corr_bonus):
                    score += 0.15 * corr_bonus

    if use_kurtosis_penalty:
        ks = [_safe_kurtosis(x[ch]) for ch in range(C)]
        k = float(np.mean(ks))
        penalty = max(0.0, (k - 8.0) / 20.0)
        score -= 0.2 * penalty

    if not np.isfinite(score):
        return 0.0
    return float(score)


def select_tremor_window_scheme_b(
    x: np.ndarray,
    target_length: int,
    sr: float,
    K: int = 8,
    min_sep_s: float = 5.0,
    band_hz=(2.0, 8.0),
    total_hz=(0.5, 20.0),
    use_corr=True,
    use_kurtosis_penalty=True,
) -> np.ndarray:
    """
    Multi-crop K windows and select best by tremor_proxy_score.
    """
    x = np.asarray(x)
    if x.ndim != 2:
        x = np.squeeze(x)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        else:
            return extract_random_window(x, target_length)

    C, L = x.shape
    T = int(target_length)

    if L <= T:
        out = np.zeros((C, T), dtype=x.dtype)
        out[:, :L] = x
        return out

    K = int(max(1, K))
    min_sep = int(max(1, round(float(min_sep_s) * float(sr))))
    max_start = L - T
    if max_start <= 0:
        return x[:, :T]

    starts = []
    tries = 0
    max_tries = 50 * K
    while (len(starts) < K) and (tries < max_tries):
        s = int(np.random.randint(0, max_start + 1))
        if all(abs(s - s0) >= min_sep for s0 in starts):
            starts.append(s)
        tries += 1
    if len(starts) == 0:
        starts = [int(np.random.randint(0, max_start + 1))]

    best_s = starts[0]
    best_score = -1e9
    for s in starts:
        w = x[:, s:s + T]
        sc = tremor_proxy_score(
            w,
            sr=float(sr),
            band_hz=band_hz,
            total_hz=total_hz,
            use_corr=bool(use_corr),
            use_kurtosis_penalty=bool(use_kurtosis_penalty),
        )
        if np.isfinite(sc) and (sc > best_score):
            best_score = sc
            best_s = s

    x_best = x[:, best_s:best_s + T]
    return np.nan_to_num(x_best, nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# Model (compat + same channels logic)
# =========================
class ConvLSTMClassifier(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

        use_spec = bool(cfg.get("USE_SPECTROGRAM", False))
        use_wav  = bool(cfg.get("USE_WAVELET", False))

        if use_spec and use_wav:
            n_fft = int(cfg.get("STFT_N_FFT", 256))
            freq_bins = n_fft // 2 + 1
            in_channels = int(cfg["INPUT_CHANNELS"]) * ((int(cfg.get("WAVELET_LEVEL", 3)) + 1) + freq_bins)
        elif use_spec:
            n_fft = int(cfg.get("STFT_N_FFT", 256))
            freq_bins = n_fft // 2 + 1
            in_channels = int(cfg["INPUT_CHANNELS"]) * freq_bins
        elif use_wav:
            # training uses raw + wav => C + C*(level+1) = C*(level+2)
            in_channels = int(cfg["INPUT_CHANNELS"]) * (int(cfg.get("WAVELET_LEVEL", 3)) + 2)
        else:
            in_channels = int(cfg["INPUT_CHANNELS"])

        conv_layers = []
        in_ch = in_channels
        conv_channels = cfg.get("CONV_CHANNELS", [32, 64, 128])
        for out_ch in conv_channels:
            conv_layers += [
                nn.Conv1d(in_ch, int(out_ch), kernel_size=5, padding=2),
                nn.BatchNorm1d(int(out_ch)),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),
                nn.Dropout(float(cfg["DROPOUT"])),
            ]
            in_ch = int(out_ch)

        self.conv = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(
            input_size=int(conv_channels[-1]),
            hidden_size=int(cfg["LSTM_HIDDEN"]),
            num_layers=int(cfg["LSTM_LAYERS"]),
            batch_first=True,
            bidirectional=bool(cfg["BIDIRECTIONAL"]),
            dropout=float(cfg["DROPOUT"]) if int(cfg["LSTM_LAYERS"]) > 1 else 0.0,
        )

        lstm_out_dim = int(cfg["LSTM_HIDDEN"]) * (2 if bool(cfg["BIDIRECTIONAL"]) else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, int(cfg["LSTM_HIDDEN"])),
            nn.ReLU(),
            nn.Dropout(float(cfg["DROPOUT"])),
            nn.Linear(int(cfg["LSTM_HIDDEN"]), 3),
        )

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.transpose(1, 2)
        _, (h_n, _) = self.lstm(conv_out)
        if bool(self.cfg["BIDIRECTIONAL"]):
            h = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h = h_n[-1]
        logits = self.classifier(h)
        return logits


def load_cfg_and_model(model_pt: str, config_json, dev: torch.device):
    """
    Priority:
      1) cfg from checkpoint dict: ckpt["config"] / ckpt["cfg"] / ...
      2) fallback to config_json on disk (if provided)
    """
    ckpt = torch.load(model_pt, map_location=dev, weights_only=False)

    cfg = None
    if isinstance(ckpt, dict):
        for k in ["config", "cfg", "train_cfg", "hparams", "args"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                cfg = ckpt[k]
                print(f"[INFO] Config source: checkpoint['{k}']")
                break

    if cfg is None:
        if (config_json is None) or (str(config_json).strip() == ""):
            raise FileNotFoundError(
                "No config found inside checkpoint and CONFIG_JSON is None/empty. "
                "Provide CONFIG_JSON or save config into checkpoint."
            )
        with open(config_json, "r") as f:
            cfg = json.load(f)
        print("[INFO] Config source: CONFIG_JSON")

    # ---- defaults for robustness compatibility ----
    cfg.setdefault("USE_WAVELET", False)
    cfg.setdefault("WAVELET_TYPE", "db4")
    cfg.setdefault("WAVELET_LEVEL", 3)
    cfg.setdefault("INPUT_CHANNELS", 3)
    cfg.setdefault("TARGET_LENGTH", 15000)
    cfg.setdefault("SAMPLE_RATE", 100)
    cfg.setdefault("CONV_CHANNELS", [32, 64, 128])
    cfg.setdefault("LSTM_HIDDEN", 128)
    cfg.setdefault("LSTM_LAYERS", 2)
    cfg.setdefault("BIDIRECTIONAL", True)
    cfg.setdefault("DROPOUT", 0.3)

    cfg.setdefault("USE_SPECTROGRAM", True)
    cfg.setdefault("STFT_N_FFT", 256)
    cfg.setdefault("STFT_HOP_LENGTH", 128)

    # tremor scheme-B defaults
    cfg.setdefault("TREMOR_MULTI_CROP_K", 8)
    cfg.setdefault("TREMOR_MULTI_CROP_MIN_SEP_S", 5.0)
    cfg.setdefault("TREMOR_BAND_HZ", [2.0, 8.0])
    cfg.setdefault("TREMOR_TOTAL_HZ", [0.5, 20.0])
    cfg.setdefault("TREMOR_PROXY_USE_CORR", True)
    cfg.setdefault("TREMOR_PROXY_USE_KURTOSIS_PENALTY", True)

    # --- NEW: tremor-only pre-STFT filter defaults (for robustness alignment) ---
    # If your checkpoint config already contains these, they will override setdefault.
    cfg.setdefault("TREMOR_PRE_STFT_FILTER_ENABLE", True)
    cfg.setdefault("TREMOR_PRE_STFT_FILTER_F_LO", 0.0)
    cfg.setdefault("TREMOR_PRE_STFT_FILTER_F_HI", 20.0)
    cfg.setdefault("TREMOR_PRE_STFT_FILTER_ORDER", 4)

    if "TREMOR_MULTI_CROP_ENABLE" in cfg and (not bool(cfg.get("TREMOR_MULTI_CROP_ENABLE", True))):
        cfg["TREMOR_MULTI_CROP_K"] = 1

    if cfg["USE_WAVELET"] and (not _HAS_PYWT):
        print("[WARN] pywt missing; force USE_WAVELET=False for robustness test.")
        cfg["USE_WAVELET"] = False

    # ---- state dict ----
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    model = ConvLSTMClassifier(cfg).to(dev)

    if isinstance(state, dict):
        has_orig = any(k.startswith("_orig_mod.") for k in state.keys())
        if has_orig:
            new_state = {}
            for k, v in state.items():
                nk = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
                new_state[nk] = v
            state = new_state
            print("[INFO] Stripped '_orig_mod.' prefix from checkpoint state_dict (torch.compile artifact).")

    model.load_state_dict(state, strict=True)
    model.eval()
    return cfg, model


# =========================
# Pools (load+sample windows)
# =========================
class Pools:
    def __init__(self, cfg: dict, h5max: int):
        self.cfg = cfg
        self.cache = H5HandleCache(max_open=h5max)

        self.df_eq = None
        self.df_tr = None
        self.df_noise = None

    def close(self):
        self.cache.close_all()

    def _load_combined_trace(self, h5_path: str, trace_name: str) -> np.ndarray:
        h5 = self.cache.get(h5_path)
        root = _try_get_data_root(h5)
        if "$" not in trace_name:
            raise ValueError("trace_name must be group$index for combined H5 layout.")
        group, rest = trace_name.split("$", 1)
        idx = int(rest.split(",", 1)[0])
        if group not in root:
            raise KeyError(f"group '{group}' not in {h5_path}")
        x = np.asarray(root[group][idx])
        x = _dataset_to_C_L(x, int(self.cfg["INPUT_CHANNELS"]))
        x = _ensure_C(x, int(self.cfg["INPUT_CHANNELS"]))
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _first_dataset_under(self, node, max_depth=4, _depth=0):
        import h5py as _h5py
        if isinstance(node, _h5py.Dataset):
            return node
        if not isinstance(node, _h5py.Group) or _depth >= max_depth:
            return None
        for k in node.keys():
            ds = self._first_dataset_under(node[k], max_depth=max_depth, _depth=_depth + 1)
            if ds is not None:
                return ds
        return None

    def _load_tremor_per_event(self, per_event_h5: str, net: str, sta: str) -> np.ndarray:
        h5 = self.cache.get(per_event_h5)
        if "raw_waveforms" in h5:
            root = h5["raw_waveforms"]
        elif "data" in h5:
            root = h5["data"]
        else:
            root = h5

        name1 = f"{net}.{sta}"
        name2 = f"{sta}"

        if name1 in root:
            target = root[name1]
        elif name2 in root:
            target = root[name2]
        else:
            hit = None
            for k in list(root.keys()):
                if k == name1 or k.endswith(f".{sta}") or k == name2:
                    hit = k
                    break
            if hit is None:
                raise KeyError(f"Cannot find tremor key for {net}.{sta} in {per_event_h5}")
            target = root[hit]

        ds = self._first_dataset_under(target, max_depth=4)
        if ds is None:
            raise KeyError(f"Found tremor group but no dataset under it: {target.name}")

        x = np.asarray(ds[...])
        x = _dataset_to_C_L(x, int(self.cfg["INPUT_CHANNELS"]))
        x = _ensure_C(x, int(self.cfg["INPUT_CHANNELS"]))
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def build(self):
        # EQ
        dfq = pd.read_csv(COMCAT_CSV)
        if "trace_name" not in dfq.columns:
            raise KeyError("comcat_metadata.csv must have 'trace_name'")

        if "trace_P_arrival_sample" in dfq.columns:
            dfq["pick_index"] = pd.to_numeric(dfq["trace_P_arrival_sample"], errors="coerce").fillna(-1).astype(int)
        else:
            dfq["pick_index"] = -1
        dfq = dfq[dfq["pick_index"] >= 0].reset_index(drop=True)

        # Tremor
        dft = pd.read_csv(TREMOR_CSV)
        if "per_event_h5" not in dft.columns:
            raise KeyError("tremor CSV must have 'per_event_h5'.")
        if "network" not in dft.columns or "station" not in dft.columns:
            raise KeyError("tremor CSV must have 'network' and 'station'.")

        # Noise
        dfn = pd.read_csv(NOISE_CSV)
        if "trace_name" not in dfn.columns:
            if "batch_key" in dfn.columns and "batch_index" in dfn.columns:
                dfn["trace_name"] = dfn.apply(lambda r: f"{r['batch_key']}${int(r['batch_index'])}", axis=1)
            else:
                raise KeyError("noise CSV must have 'trace_name' or (batch_key,batch_index).")

        # Downsample pools
        if SAMPLES_PER_CLASS_POOL is not None and SAMPLES_PER_CLASS_POOL > 0:
            dfq = dfq.sample(n=min(len(dfq), SAMPLES_PER_CLASS_POOL), random_state=SEED).reset_index(drop=True)
            dft = dft.sample(n=min(len(dft), SAMPLES_PER_CLASS_POOL), random_state=SEED).reset_index(drop=True)
            dfn = dfn.sample(n=min(len(dfn), SAMPLES_PER_CLASS_POOL), random_state=SEED).reset_index(drop=True)

        self.df_eq = dfq
        self.df_tr = dft
        self.df_noise = dfn

        print(f"[INFO] Pools: EQ={len(self.df_eq)} Tremor={len(self.df_tr)} Noise={len(self.df_noise)}")

    def _infer_src_sr(self, row: pd.Series, default_sr: float, label: int) -> float:
        """
        Mirrors your training heuristics:
        - try sampling_rate_hz, trace_sampling_rate_hz
        - for noise, if missing, default 200
        - else fallback to cfg sample rate
        """
        src_sr = row.get("sampling_rate_hz", None)
        if (src_sr is None) and ("trace_sampling_rate_hz" in row.index):
            src_sr = row.get("trace_sampling_rate_hz", None)
        if (src_sr is None) and (label == 0):
            src_sr = 200.0
        if src_sr is None or (isinstance(src_sr, float) and np.isnan(src_sr)):
            src_sr = float(default_sr)
        return float(src_sr)

    def sample_eq_window(self) -> np.ndarray:
        row = self.df_eq.sample(n=1).iloc[0]
        x = self._load_combined_trace(COMCAT_H5, str(row["trace_name"]))

        dst_sr = float(self.cfg["SAMPLE_RATE"])
        src_sr = self._infer_src_sr(row, default_sr=dst_sr, label=1)
        pick = int(row["pick_index"])

        if abs(src_sr - dst_sr) > 1e-6:
            pick = int(round(pick * (dst_sr / src_sr)))
            x = resample_trace_np(x, src_sr=src_sr, dst_sr=dst_sr)

        T = int(self.cfg["TARGET_LENGTH"])
        C, L = x.shape
        start = max(0, min(pick - T // 2, max(0, L - T)))
        if (start + T) <= L:
            win = x[:, start:start + T]
        else:
            win = extract_center_window(x, T)
        return np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_tremor_window(self) -> np.ndarray:
        row = self.df_tr.sample(n=1).iloc[0]
        per_event_h5 = str(row["per_event_h5"])
        net = str(row["network"]).strip()
        sta = str(row["station"]).strip()
        x = self._load_tremor_per_event(per_event_h5, net, sta)

        dst_sr = float(self.cfg["SAMPLE_RATE"])
        src_sr = row.get("sampling_rate_hz", None)
        if src_sr is None or (isinstance(src_sr, float) and np.isnan(src_sr)):
            src_sr = dst_sr
        src_sr = float(src_sr)

        if abs(src_sr - dst_sr) > 1e-6:
            x = resample_trace_np(x, src_sr=src_sr, dst_sr=dst_sr)

        T = int(self.cfg["TARGET_LENGTH"])
        C, L = x.shape

        best_start = row.get("best_start", None)
        has_best = (best_start is not None) and (not (isinstance(best_start, float) and np.isnan(best_start)))
        if has_best:
            s = int(best_start)
            s = max(0, min(s, max(0, L - T)))
            win = x[:, s:s + T]
            if win.shape[1] < T:
                out = np.zeros((C, T), dtype=x.dtype)
                out[:, :win.shape[1]] = win
                win = out
        else:
            win = select_tremor_window_scheme_b(
                x,
                target_length=T,
                sr=dst_sr,
                K=int(self.cfg.get("TREMOR_MULTI_CROP_K", 8)),
                min_sep_s=float(self.cfg.get("TREMOR_MULTI_CROP_MIN_SEP_S", 5.0)),
                band_hz=tuple(self.cfg.get("TREMOR_BAND_HZ", [2.0, 8.0])),
                total_hz=tuple(self.cfg.get("TREMOR_TOTAL_HZ", [0.5, 20.0])),
                use_corr=bool(self.cfg.get("TREMOR_PROXY_USE_CORR", True)),
                use_kurtosis_penalty=bool(self.cfg.get("TREMOR_PROXY_USE_KURTOSIS_PENALTY", True)),
            )

        return np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)

    def sample_noise_window(self) -> np.ndarray:
        row = self.df_noise.sample(n=1).iloc[0]
        x = self._load_combined_trace(NOISE_H5, str(row["trace_name"]))

        dst_sr = float(self.cfg["SAMPLE_RATE"])
        src_sr = self._infer_src_sr(row, default_sr=dst_sr, label=0)

        if abs(src_sr - dst_sr) > 1e-6:
            x = resample_trace_np(x, src_sr=src_sr, dst_sr=dst_sr)

        T = int(self.cfg["TARGET_LENGTH"])
        win = extract_random_window(x, T)
        return np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)


# =========================
# Robustness evaluation
# =========================
@torch.no_grad()
def run_model_probs(model, dev, X_batch_np: np.ndarray) -> np.ndarray:
    X = torch.tensor(X_batch_np, dtype=torch.float32).to(dev)
    out = model(X)
    logits = out[0] if isinstance(out, (tuple, list)) else out
    probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
    return probs  # (B,3)


def build_model_input_batch(cfg: dict, windows: list) -> np.ndarray:
    """
    Match training feature construction:
      - Z-score waveform per channel
      - Optional STFT mag (stacked over channels)
      - Optional wavelet bands (stacked over channels)
      - Combine exactly like training:
          * spec + wav -> vstack([spec, wav]) with length aligned by min
          * spec only  -> spec
          * wav only   -> vstack([raw, wav]) with length aligned
          * none       -> raw
    """
    Xs = []
    use_wav = bool(cfg.get("USE_WAVELET", False))
    use_spec = bool(cfg.get("USE_SPECTROGRAM", False))

    for w in windows:
        w = _ensure_C(w, int(cfg["INPUT_CHANNELS"]))
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)

        raw = _zscore_channels(w).astype(np.float32, copy=False)

        x_wav = _wavelet_only(raw, cfg) if use_wav else None
        x_spec = _stft_stack(raw, cfg) if use_spec else None

        if use_spec and use_wav:
            Lmin = min(x_spec.shape[1], x_wav.shape[1])
            x = np.vstack([x_spec[:, :Lmin], x_wav[:, :Lmin]])
        elif use_spec:
            x = x_spec
        elif use_wav:
            Lmin = min(raw.shape[1], x_wav.shape[1])
            x = np.vstack([raw[:, :Lmin], x_wav[:, :Lmin]])
        else:
            x = raw

        Xs.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False))

    return np.stack(Xs, axis=0)  # (B, Cin, Lfeat)


def metrics_threeclass(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0)) if len(y_true) else 0.0

    rec = {}
    for c, name in [(0, "noise"), (1, "eq"), (2, "tremor")]:
        m = (y_true == c)
        rec[f"recall_{name}"] = float((y_pred[m] == c).mean()) if m.any() else float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return acc, f1m, rec, cm


def plot_curve(xs, ys, title, xlabel, ylabel, out_png):
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o")
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, out_png, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    labels = ["Noise", "EQ", "TR"]
    plt.xticks([0, 1, 2], labels, rotation=30)
    plt.yticks([0, 1, 2], labels)
    for i in range(3):
        for j in range(3):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250, bbox_inches="tight")
    plt.close()


def make_threeclass_batch(pools: Pools, cfg: dict, scenario: str, k: float):
    """
    Returns:
      y_true: (B,)
      windows: list length B, each (C,T)

    B is split equally across classes (Noise/EQ/TR).

    NEW: tremor-only pre-STFT filter is applied to the tremor window BEFORE mixing noise.
    """
    assert scenario in ("REAL_NOISE_AMP", "REAL_PLUS_WHITE_AMP")

    if BATCH_SIZE % 3 != 0:
        raise ValueError(f"BATCH_SIZE must be divisible by 3 for equal class sampling. Got {BATCH_SIZE}")

    n_each = BATCH_SIZE // 3
    C = int(cfg["INPUT_CHANNELS"])
    T = int(cfg["TARGET_LENGTH"])

    windows = []
    y_true = []

    # Noise class (0)
    for _ in range(n_each):
        rn = pools.sample_noise_window()
        if scenario == "REAL_NOISE_AMP":
            x = (k * rn)
        else:
            wn = gen_white_noise(C, T, std=WHITE_STD)
            x = rn + (k * wn)
        windows.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        y_true.append(0)

    # EQ class (1)
    for _ in range(n_each):
        eq = pools.sample_eq_window()
        rn = pools.sample_noise_window()
        if scenario == "REAL_NOISE_AMP":
            x = eq + (k * rn)
        else:
            wn = gen_white_noise(C, T, std=WHITE_STD)
            x = eq + rn + (k * wn)
        windows.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        y_true.append(1)

    # Tremor class (2)
    for _ in range(n_each):
        tr = pools.sample_tremor_window()

        # --- NEW: tremor-only pre-STFT filter (align training distribution) ---
        if bool(cfg.get("TREMOR_PRE_STFT_FILTER_ENABLE", False)):
            sr = float(cfg.get("SAMPLE_RATE", 100.0))
            f_lo = float(cfg.get("TREMOR_PRE_STFT_FILTER_F_LO", 0.0))
            f_hi = float(cfg.get("TREMOR_PRE_STFT_FILTER_F_HI", 20.0))
            order = int(cfg.get("TREMOR_PRE_STFT_FILTER_ORDER", 4))
            tr = lowpass_or_bandpass_np(tr, sr=sr, f_lo=f_lo, f_hi=f_hi, order=order)

        rn = pools.sample_noise_window()
        if scenario == "REAL_NOISE_AMP":
            x = tr + (k * rn)
        else:
            wn = gen_white_noise(C, T, std=WHITE_STD)
            x = tr + rn + (k * wn)

        windows.append(np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        y_true.append(2)

    # shuffle
    idx = np.random.permutation(len(y_true))
    windows = [windows[i] for i in idx]
    y_true = np.asarray([y_true[i] for i in idx], dtype=np.int64)
    return y_true, windows


# =========================
# Main
# =========================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUT_DIR, f"robust_{ts}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    set_seed(SEED)
    dev = get_device()

    print(f"[INFO] Device: {dev}")
    cfg, model = load_cfg_and_model(MODEL_PT, CONFIG_JSON, dev)
    print(f"[INFO] Loaded model: {MODEL_PT}")
    print(
        f"[INFO] USE_WAVELET={cfg['USE_WAVELET']} "
        f"WAVELET={cfg.get('WAVELET_TYPE', 'db4')} L={cfg.get('WAVELET_LEVEL', 3)} "
        f"USE_SPECTROGRAM={cfg.get('USE_SPECTROGRAM', False)} "
        f"STFT_N_FFT={cfg.get('STFT_N_FFT', 256)} HOP={cfg.get('STFT_HOP_LENGTH', 128)} "
        f"TARGET_LENGTH={cfg['TARGET_LENGTH']} SR={cfg.get('SAMPLE_RATE', 100)}"
    )
    print(
        f"[INFO] Tremor Scheme-B: K={cfg.get('TREMOR_MULTI_CROP_K', 8)} "
        f"min_sep_s={cfg.get('TREMOR_MULTI_CROP_MIN_SEP_S', 5.0)} "
        f"band={cfg.get('TREMOR_BAND_HZ', [2.0, 8.0])} total={cfg.get('TREMOR_TOTAL_HZ', [0.5, 20.0])}"
    )
    print(
        f"[INFO] Tremor pre-STFT filter: enable={cfg.get('TREMOR_PRE_STFT_FILTER_ENABLE', False)} "
        f"f_lo={cfg.get('TREMOR_PRE_STFT_FILTER_F_LO', 0.0)} "
        f"f_hi={cfg.get('TREMOR_PRE_STFT_FILTER_F_HI', 20.0)} "
        f"order={cfg.get('TREMOR_PRE_STFT_FILTER_ORDER', 4)} "
        f"(scipy={'yes' if _HAS_SCIPY else 'no'} fallback=FFT)"
    )

    pools = Pools(cfg, h5max=H5_MAX_OPEN)
    pools.build()

    # quick sanity: k=1, REAL_NOISE_AMP
    y_san, windows_san = make_threeclass_batch(pools, cfg, "REAL_NOISE_AMP", k=1.0)
    X_san = build_model_input_batch(cfg, windows_san)
    probs_san = run_model_probs(model, dev, X_san)
    pred_san = np.argmax(probs_san, axis=1)
    acc0, f1m0, rec0, cm0 = metrics_threeclass(y_san, pred_san)
    print(f"[SANITY] REAL_NOISE_AMP k=1 | acc={acc0:.4f} f1_macro={f1m0:.4f} {rec0}")
    plot_confusion(
        cm0,
        os.path.join(fig_dir, "confusion_sanity_REAL_NOISE_AMP_k1.png"),
        title="Sanity Confusion (REAL_NOISE_AMP, k=1)",
    )

    results_rows = []
    saved_cms = {}

    def sweep(scenario: str):
        xs = []
        accs = []
        f1ms = []
        r_noise = []
        r_eq = []
        r_tr = []

        for lg in LOG10_KS:
            k = 10.0 ** float(lg)

            y_all = []
            p_all = []

            for _ in range(STEPS_PER_K):
                y_true, windows = make_threeclass_batch(pools, cfg, scenario, k=k)
                X = build_model_input_batch(cfg, windows)
                probs = run_model_probs(model, dev, X)
                pred = np.argmax(probs, axis=1).astype(np.int64)

                y_all.append(y_true)
                p_all.append(pred)

            y_all = np.concatenate(y_all)
            p_all = np.concatenate(p_all)

            acc, f1m, rec, cm = metrics_threeclass(y_all, p_all)

            xs.append(int(lg))
            accs.append(acc)
            f1ms.append(f1m)
            r_noise.append(rec["recall_noise"])
            r_eq.append(rec["recall_eq"])
            r_tr.append(rec["recall_tremor"])

            results_rows.append({
                "timestamp": ts,
                "scenario": scenario,
                "log10_k": int(lg),
                "k": float(k),
                "accuracy": float(acc),
                "f1_macro": float(f1m),
                "recall_noise": float(rec["recall_noise"]),
                "recall_eq": float(rec["recall_eq"]),
                "recall_tremor": float(rec["recall_tremor"]),
                "n_samples": int(len(y_all)),
            })

            print(
                f"[{scenario}] log10_k={lg:>2} k={k:.1e} "
                f"acc={acc:.4f} f1m={f1m:.4f} "
                f"rec_noise={rec['recall_noise']:.3f} rec_eq={rec['recall_eq']:.3f} rec_tr={rec['recall_tremor']:.3f}"
            )

            if int(lg) in CONFUSION_PLOT_LOG10K:
                saved_cms[f"{scenario}_log10k{int(lg)}"] = cm

        prefix = scenario
        plot_curve(xs, accs,
                   title=f"Overall Accuracy vs log10(k) [{scenario}]",
                   xlabel="log10(k)", ylabel="Accuracy",
                   out_png=os.path.join(fig_dir, f"{prefix}_accuracy_overall.png"))

        plot_curve(xs, f1ms,
                   title=f"Macro-F1 vs log10(k) [{scenario}]",
                   xlabel="log10(k)", ylabel="Macro-F1",
                   out_png=os.path.join(fig_dir, f"{prefix}_f1_macro.png"))

        plot_curve(xs, r_noise,
                   title=f"Recall(Noise) vs log10(k) [{scenario}]",
                   xlabel="log10(k)", ylabel="Recall(Noise)",
                   out_png=os.path.join(fig_dir, f"{prefix}_recall_noise.png"))

        plot_curve(xs, r_eq,
                   title=f"Recall(EQ) vs log10(k) [{scenario}]",
                   xlabel="log10(k)", ylabel="Recall(EQ)",
                   out_png=os.path.join(fig_dir, f"{prefix}_recall_eq.png"))

        plot_curve(xs, r_tr,
                   title=f"Recall(Tremor) vs log10(k) [{scenario}]",
                   xlabel="log10(k)", ylabel="Recall(Tremor)",
                   out_png=os.path.join(fig_dir, f"{prefix}_recall_tremor.png"))

    print("[RUN] Scenario 1: REAL_NOISE_AMP")
    sweep("REAL_NOISE_AMP")

    print("[RUN] Scenario 2: REAL_PLUS_WHITE_AMP")
    sweep("REAL_PLUS_WHITE_AMP")

    # save confusion matrices
    for key, cm in saved_cms.items():
        plot_confusion(cm, os.path.join(fig_dir, f"confusion_{key}.png"), title=f"Confusion: {key}")

    # save CSV/JSON
    df = pd.DataFrame(results_rows)
    out_csv = os.path.join(run_dir, f"robust_three_class_true3_{ts}.csv")
    df.to_csv(out_csv, index=False)

    summary = {
        "timestamp": ts,
        "model_pt": MODEL_PT,
        "config_json": CONFIG_JSON,
        "paths": {
            "comcat_csv": COMCAT_CSV, "comcat_h5": COMCAT_H5,
            "noise_csv": NOISE_CSV, "noise_h5": NOISE_H5,
            "tremor_csv": TREMOR_CSV,
            "tremor_h5_master": TREMOR_H5_MASTER
        },
        "settings": {
            "seed": SEED,
            "force_cpu": FORCE_CPU,
            "h5_max_open": H5_MAX_OPEN,
            "log10_ks": LOG10_KS,
            "steps_per_k": STEPS_PER_K,
            "batch_size": BATCH_SIZE,
            "samples_per_class_pool": SAMPLES_PER_CLASS_POOL,
            "white_std": WHITE_STD,
            "confusion_plot_log10k": CONFUSION_PLOT_LOG10K
        },
        "filter": {
            "has_scipy": bool(_HAS_SCIPY),
            "tremor_pre_stft_filter_enable": bool(cfg.get("TREMOR_PRE_STFT_FILTER_ENABLE", False)),
            "tremor_pre_stft_filter_f_lo": float(cfg.get("TREMOR_PRE_STFT_FILTER_F_LO", 0.0)),
            "tremor_pre_stft_filter_f_hi": float(cfg.get("TREMOR_PRE_STFT_FILTER_F_HI", 20.0)),
            "tremor_pre_stft_filter_order": int(cfg.get("TREMOR_PRE_STFT_FILTER_ORDER", 4)),
        },
        "outputs": {
            "run_dir": run_dir,
            "csv": out_csv,
            "fig_dir": fig_dir
        }
    }
    out_json = os.path.join(run_dir, f"robust_three_class_true3_{ts}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    pools.close()

    print(f"[SAVE] CSV:  {out_csv}")
    print(f"[SAVE] JSON: {out_json}")
    print(f"[SAVE] FIGS: {fig_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()
