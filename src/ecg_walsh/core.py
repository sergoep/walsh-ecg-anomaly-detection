"""Core mathematics for the SaSiDa ECG Walsh-Hadamard reproducibility package.

The implementation follows the compact journal manuscript:
"Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal
Invariance and Exact Time-Domain Score Decomposition".

The central identities verified by this code are:
  1. Full-covariance Walsh-domain scoring is equal to the corresponding time-domain
     regularized Mahalanobis score up to numerical error.
  2. Symmetric whitening and inverse Walsh-Hadamard back-projection give an exact
     detector-relative decomposition: d_delta^2(c) = ||z||_2^2 = ||r||_2^2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.linalg import hadamard


@dataclass
class ReferenceModel:
    """Reference template and precomputed matrices for anomaly scoring."""

    W: np.ndarray
    mu_c: np.ndarray
    sigma_delta: np.ndarray
    sigma_inv: np.ndarray
    sigma_inv_sqrt: np.ndarray
    mu_x: np.ndarray
    sigma_x_delta: np.ndarray
    sigma_x_inv: np.ndarray
    trace_scale: float
    condition_number: float


def _check_power_of_two(n: int) -> None:
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"N must be a positive power of two; got N={n}.")


def _sign_changes(row: np.ndarray) -> int:
    return int(np.sum(row[:-1] * row[1:] < 0))


def sequency_ordered_walsh(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the normalized sequency-ordered Walsh-Hadamard matrix.

    Rows are sorted by increasing number of sign changes; ties are resolved by
    increasing Sylvester natural-order index.
    """

    _check_power_of_two(N)
    H = hadamard(N).astype(float)
    W_nat = H / np.sqrt(N)
    changes = np.array([_sign_changes(H[i]) for i in range(N)])
    natural_idx = np.arange(N)
    order = np.lexsort((natural_idx, changes))
    return W_nat[order, :], order, changes[order]


def _regularized_covariance(X: np.ndarray, delta: float) -> Tuple[np.ndarray, float]:
    """Unbiased covariance plus trace-scaled ridge."""

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array of windows.")
    if len(X) < 2:
        raise ValueError("At least two reference windows are required.")
    sigma_hat = np.cov(X, rowvar=False, bias=False)
    trace_scale = float(np.trace(sigma_hat) / sigma_hat.shape[0])
    if trace_scale <= 0:
        raise ValueError(
            "The reference covariance has zero trace. Add a small absolute ridge "
            "after signal normalization, or provide nonconstant reference windows."
        )
    return sigma_hat + delta * trace_scale * np.eye(sigma_hat.shape[0]), trace_scale


def fit_reference(X_train_ref: np.ndarray, W: np.ndarray, delta: float = 0.01) -> ReferenceModel:
    """Fit the pooled healthy reference template.

    Parameters
    ----------
    X_train_ref:
        Array of healthy reference windows, shape (L, N), pooled over retained leads.
    W:
        Sequency-ordered Walsh-Hadamard matrix, shape (N, N).
    delta:
        Trace-scaled Tikhonov regularization parameter.
    """

    if X_train_ref.ndim != 2:
        raise ValueError("X_train_ref must have shape (L, N).")
    if W.shape != (X_train_ref.shape[1], X_train_ref.shape[1]):
        raise ValueError("W must have shape (N, N) matching the window length.")

    C_train_ref = X_train_ref @ W.T
    mu_c = C_train_ref.mean(axis=0)
    mu_x = X_train_ref.mean(axis=0)

    sigma_delta, trace_scale = _regularized_covariance(C_train_ref - mu_c, delta)
    sigma_x_delta, _ = _regularized_covariance(X_train_ref - mu_x, delta)

    eigvals, evecs = np.linalg.eigh(sigma_delta)
    eigvals = np.clip(eigvals, 1e-12, None)
    sigma_inv_sqrt = (evecs * (1.0 / np.sqrt(eigvals))) @ evecs.T
    sigma_inv = np.linalg.inv(sigma_delta)
    sigma_x_inv = np.linalg.inv(sigma_x_delta)
    condition_number = float(eigvals.max() / eigvals.min())

    return ReferenceModel(
        W=W,
        mu_c=mu_c,
        sigma_delta=sigma_delta,
        sigma_inv=sigma_inv,
        sigma_inv_sqrt=sigma_inv_sqrt,
        mu_x=mu_x,
        sigma_x_delta=sigma_x_delta,
        sigma_x_inv=sigma_x_inv,
        trace_scale=trace_scale,
        condition_number=condition_number,
    )


def score_window_walsh(x_window: np.ndarray, model: ReferenceModel) -> float:
    """Regularized full-covariance Walsh-domain Mahalanobis score."""

    c = model.W @ x_window
    v = c - model.mu_c
    return float(v.T @ model.sigma_inv @ v)


def score_window_time_domain(x_window: np.ndarray, model: ReferenceModel) -> float:
    """Corresponding time-domain regularized Mahalanobis score."""

    v = x_window - model.mu_x
    return float(v.T @ model.sigma_x_inv @ v)


def decompose_window_walsh(x_window: np.ndarray, model: ReferenceModel):
    """Return score, whitened residual z, back-projected residual r, and checks."""

    c = model.W @ x_window
    v = c - model.mu_c
    z = model.sigma_inv_sqrt @ v
    r = model.W.T @ z
    score = float(v.T @ model.sigma_inv @ v)
    score_from_z = float(z.T @ z)
    score_from_r = float(np.sum(r ** 2))
    return score, z, r, score_from_z, score_from_r


def score_multilead_windows(windows: np.ndarray, model: ReferenceModel):
    """Score multilead windows and apply sensitivity-oriented max-fusion.

    Parameters
    ----------
    windows:
        Array with shape (num_windows, num_leads, N).
    model:
        Fitted reference model.
    """

    if windows.ndim != 3:
        raise ValueError("windows must have shape (num_windows, num_leads, N).")
    T, M, _ = windows.shape
    lead_scores = np.zeros((T, M), dtype=float)
    for t in range(T):
        for ell in range(M):
            lead_scores[t, ell] = score_window_walsh(windows[t, ell, :], model)
    fused_scores = np.max(lead_scores, axis=1)
    argmax_leads = np.argmax(lead_scores, axis=1)
    return lead_scores, fused_scores, argmax_leads


def aggregate_contributions(record: np.ndarray, starts: np.ndarray, windows: np.ndarray, model: ReferenceModel):
    """Aggregate detector-relative contribution profiles over overlapping windows."""

    lead_scores, fused_scores, argmax_leads = score_multilead_windows(windows, model)
    n_samples = record.shape[0]
    N = windows.shape[2]
    A_abs_sum = np.zeros(n_samples, dtype=float)
    A_abs_count = np.zeros(n_samples, dtype=float)
    A_energy = np.zeros(n_samples, dtype=float)

    for t, start in enumerate(starts):
        ell = argmax_leads[t]
        score, _, r, _, score_r = decompose_window_walsh(windows[t, ell, :], model)
        if not np.allclose(score, score_r, rtol=1e-8, atol=1e-8):
            raise RuntimeError("Exact decomposition failed beyond numerical tolerance.")
        idx = np.arange(start, start + N)
        A_abs_sum[idx] += np.abs(r)
        A_abs_count[idx] += 1.0
        A_energy[idx] += r ** 2

    A_abs = A_abs_sum / np.maximum(A_abs_count, 1.0)
    return A_abs, A_energy, fused_scores, lead_scores
