"""Fast synthetic ECG-like data generator for mathematical verification.

The synthetic data are intentionally lightweight. They are not used as clinical
evidence. Their purpose is to exercise the complete reference-estimation,
scoring, invariance, and exact-decomposition pipeline quickly.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def gaussian_pulse(t, center, width, amplitude):
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def synthetic_ecg_lead(
    n_samples: int,
    fs: int,
    heart_rate: float = 70.0,
    lead_gain: float = 1.0,
    lead_phase: float = 0.0,
    noise_std: float = 0.025,
    abnormal: bool = False,
    abnormal_strength: float = 1.0,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(n_samples) / fs
    period = 60.0 / heart_rate
    y = np.zeros_like(t)
    beat_times = np.arange(0.4 + lead_phase, t[-1] + period, period)

    for bt in beat_times:
        p_amp = 0.10 * lead_gain
        q_amp = -0.12 * lead_gain
        r_amp = 1.00 * lead_gain
        s_amp = -0.25 * lead_gain
        tw_amp = 0.30 * lead_gain

        p_w = 0.035
        q_w = 0.012
        r_w = 0.014
        s_w = 0.016
        tw_w = 0.080

        if abnormal:
            r_amp *= 0.75 + 0.10 * rng.normal()
            s_amp *= 1.40 + 0.10 * rng.normal()
            tw_amp *= -0.60 + 0.10 * rng.normal()
            r_w *= 1.60
            tw_w *= 1.30

        y += gaussian_pulse(t, bt - 0.20, p_w, p_amp)
        y += gaussian_pulse(t, bt - 0.04, q_w, q_amp)
        y += gaussian_pulse(t, bt, r_w, r_amp)
        y += gaussian_pulse(t, bt + 0.04, s_w, s_amp)
        y += gaussian_pulse(t, bt + 0.28, tw_w, tw_amp)

        if abnormal:
            mask = (t >= bt + 0.07) & (t <= bt + 0.22)
            y[mask] += -0.10 * abnormal_strength * lead_gain

    baseline = 0.05 * np.sin(2 * np.pi * 0.25 * t + 2 * np.pi * rng.random())
    noise = noise_std * rng.normal(size=n_samples)
    y = y + baseline + noise
    y = y - np.median(y)
    y = y / (np.percentile(np.abs(y), 95) + 1e-12)
    return y


def synthetic_record(record_id: int, abnormal: bool, seed: int, n_samples: int, fs: int, n_leads: int):
    local_rng = np.random.default_rng(seed + 1000 * record_id + int(abnormal))
    hr = local_rng.uniform(58, 88)
    signals = []
    for ell in range(n_leads):
        gain = local_rng.uniform(0.75, 1.25) * (1.0 + 0.06 * ell)
        phase = local_rng.uniform(-0.015, 0.015)
        noise_std = local_rng.uniform(0.018, 0.035)
        strength = local_rng.uniform(0.8, 1.3)
        lead = synthetic_ecg_lead(
            n_samples,
            fs,
            heart_rate=hr + local_rng.normal(0, 2.0),
            lead_gain=gain,
            lead_phase=phase,
            noise_std=noise_std,
            abnormal=abnormal,
            abnormal_strength=strength,
            rng=local_rng,
        )
        signals.append(lead)
    return np.stack(signals, axis=1)


def bandpass_filter(x, fs=100, low=0.5, high=40.0, order=3):
    nyq = 0.5 * fs
    high_eff = min(high, 0.95 * nyq)
    b, a = butter(order, [low / nyq, high_eff / nyq], btype="bandpass")
    if x.ndim == 1:
        return filtfilt(b, a, x)
    return np.stack([filtfilt(b, a, x[:, j]) for j in range(x.shape[1])], axis=1)


def extract_multilead_windows(record, N=128, hop=64):
    n_samples, n_leads = record.shape
    starts = np.arange(0, n_samples - N + 1, hop)
    out = np.zeros((len(starts), n_leads, N), dtype=float)
    for i, s in enumerate(starts):
        out[i] = record[s : s + N, :].T
    return out, starts
