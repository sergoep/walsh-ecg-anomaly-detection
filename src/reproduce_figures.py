import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Settings
# ==========================================================
SEED = 7
np.random.seed(SEED)

OUTDIR = "figures"
os.makedirs(OUTDIR, exist_ok=True)

# Windowing
FS = 250                    # sampling rate (Hz) for synthetic demonstration
N = 256                     # must be power of 2
HOP = 64                    # overlap step
LAMBDA = 0.10               # shrinkage intensity
ALPHA = 0.01                # false alarm target (empirical threshold)

# ==========================================================
# Walsh-Hadamard (fast) and Sequency ordering (correct)
# ==========================================================
def hadamard_matrix(n: int) -> np.ndarray:
    """Construct Hadamard matrix H_n with Sylvester recursion."""
    assert (n & (n - 1)) == 0 and n > 0
    H = np.array([[1.0]])
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H

def wht_matrix(n: int) -> np.ndarray:
    """Normalized Walsh-Hadamard transform matrix W_n = H_n / sqrt(n)."""
    H = hadamard_matrix(n)
    return H / np.sqrt(n)

def sequency_of_row(w: np.ndarray) -> int:
    """Number of sign changes in a +-1 row vector."""
    return int(np.sum(np.abs(np.diff(w)) > 0))

def sequency_permutation(W: np.ndarray) -> np.ndarray:
    """
    Return permutation indices that sort rows by sequency (ascending),
    breaking ties by original index.
    """
    H = np.sign(W * np.sqrt(W.shape[0]))  # recover +-1 pattern
    seq = np.array([sequency_of_row(H[i, :]) for i in range(H.shape[0])])
    perm = np.lexsort((np.arange(W.shape[0]), seq))  # stable deterministic
    return perm

def wht_sequency(x: np.ndarray, W: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Sequency-ordered Walsh coefficients."""
    c_nat = W @ x
    return c_nat[perm]

def iwht_sequency(c_seq: np.ndarray, W: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Inverse transform from sequency ordered coefficients."""
    # invert permutation
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    c_nat = c_seq[inv]
    return W.T @ c_nat

# Precompute
W = wht_matrix(N)
perm = sequency_permutation(W)

# ==========================================================
# Synthetic ECG generator (open, reproducible demonstration)
# ==========================================================
def synthetic_ecg(t: np.ndarray, hr_bpm: float = 70.0, noise: float = 0.01) -> np.ndarray:
    """
    Simple synthetic ECG-like waveform as sum of Gaussian pulses per beat.
    This is not a physiological model; it is a reproducible signal generator.
    """
    rr = 60.0 / hr_bpm
    x = np.zeros_like(t)
    beat_times = np.arange(t[0] + 0.2, t[-1] - 0.2, rr)

    # P, Q, R, S, T waves (rough)
    for bt in beat_times:
        x += 0.08 * np.exp(-0.5 * ((t - (bt - 0.20)) / 0.03) ** 2)   # P
        x += -0.12 * np.exp(-0.5 * ((t - (bt - 0.03)) / 0.01) ** 2) # Q
        x += 1.00 * np.exp(-0.5 * ((t - bt) / 0.012) ** 2)          # R
        x += -0.25 * np.exp(-0.5 * ((t - (bt + 0.02)) / 0.012) ** 2)# S
        x += 0.30 * np.exp(-0.5 * ((t - (bt + 0.25)) / 0.05) ** 2)  # T

    x += noise * np.random.randn(len(t))
    return x

def inject_anomaly(x: np.ndarray, fs: int, start_s: float, end_s: float, kind: str = "st_shift") -> np.ndarray:
    """
    Inject an interpretable anomaly into ECG segment.
    - st_shift: baseline shift in a local interval
    - spike: impulsive artifact
    - widen_qrs: local convolution widening
    """
    y = x.copy()
    a = int(start_s * fs)
    b = int(end_s * fs)
    a = max(a, 0); b = min(b, len(y))

    if kind == "st_shift":
        y[a:b] += 0.15
    elif kind == "spike":
        idx = (a + b) // 2
        y[idx:idx+3] += 0.8
    elif kind == "widen_qrs":
        # mild smoothing in the segment
        k = np.array([0.2, 0.6, 0.2])
        seg = y[a:b]
        if len(seg) > 3:
            y[a:b] = np.convolve(seg, k, mode="same")
    else:
        raise ValueError("Unknown anomaly kind.")
    return y

# ==========================================================
# Windowing utilities
# ==========================================================
def frame_signal(x: np.ndarray, n: int, hop: int) -> np.ndarray:
    """Return frames shape (T, n)."""
    T = 1 + (len(x) - n) // hop
    frames = np.stack([x[i*hop:i*hop+n] for i in range(T)], axis=0)
    return frames

def overlap_add(values_per_frame: np.ndarray, n: int, hop: int, out_len: int, agg: str = "max") -> np.ndarray:
    """
    Aggregate frame-level per-sample quantities back to full length.
    values_per_frame: shape (T, n)
    """
    T = values_per_frame.shape[0]
    accum = np.zeros(out_len)
    count = np.zeros(out_len)

    if agg == "mean":
        for t in range(T):
            start = t * hop
            end = start + n
            accum[start:end] += values_per_frame[t]
            count[start:end] += 1.0
        return accum / np.maximum(count, 1.0)

    if agg == "max":
        accum[:] = 0.0
        for t in range(T):
            start = t * hop
            end = start + n
            accum[start:end] = np.maximum(accum[start:end], values_per_frame[t])
        return accum

    raise ValueError("agg must be 'mean' or 'max'.")

# ==========================================================
# Reference estimation and scoring
# ==========================================================
def shrink_cov(S: np.ndarray, lam: float) -> np.ndarray:
    tr = np.trace(S)
    n = S.shape[0]
    return (1.0 - lam) * S + lam * (tr / n) * np.eye(n)

def fit_reference(C: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    C: (L, N) sequency-ordered coeff vectors.
    Returns mu, Sigma_lam, inv(Sigma_lam) via Cholesky solve.
    """
    mu = C.mean(axis=0)
    X = C - mu
    S = (X.T @ X) / max(C.shape[0] - 1, 1)
    Sig = shrink_cov(S, lam)
    # robust inverse via cholesky
    Lc = np.linalg.cholesky(Sig)
    # function handle: inv(Sig) @ v = solve(Lc.T, solve(Lc, v))
    return mu, Sig, Lc

def mahalanobis_sq(c: np.ndarray, mu: np.ndarray, Lc: np.ndarray) -> float:
    v = c - mu
    y = np.linalg.solve(Lc, v)
    return float(y @ y)

def standardized_deviation(c: np.ndarray, mu: np.ndarray, Lc: np.ndarray) -> np.ndarray:
    v = c - mu
    return np.linalg.solve(Lc, v)  # z = Sigma^{-1/2} (c-mu)

# ==========================================================
# Visualization helpers
# ==========================================================
def plot_color_coded_ecg(t, x, score, title, path_png):
    """
    Color-coded line: we draw segments colored by score intensity.
    """
    # normalize score to [0,1]
    s = score.copy()
    s = s - np.min(s)
    if np.max(s) > 0:
        s = s / np.max(s)

    fig, ax = plt.subplots(figsize=(10, 3))
    for i in range(len(x) - 1):
        ax.plot(t[i:i+2], x[i:i+2], color=plt.cm.inferno(s[i]), linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)

def plot_heatmap(M, title, path_png, xlabel="Time frame", ylabel="Sequency index"):
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(M.T, aspect="auto", origin="lower")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)

def plot_score_curve(scores, thr, title, path_png):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(scores, linewidth=1.5)
    ax.axhline(thr, linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time frame")
    ax.set_ylabel(r"$d^2(c_t)$")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)

# ==========================================================
# Main demo
# ==========================================================
def main():
    # --- create synthetic healthy and test signals ---
    dur = 10.0
    t = np.arange(0, dur, 1 / FS)
    x_healthy = synthetic_ecg(t, hr_bpm=70, noise=0.01)

    x_test = synthetic_ecg(t, hr_bpm=70, noise=0.01)
    x_test = inject_anomaly(x_test, FS, start_s=4.0, end_s=5.0, kind="st_shift")
    x_test = inject_anomaly(x_test, FS, start_s=7.0, end_s=7.2, kind="spike")

    # --- frame signals ---
    Fh = frame_signal(x_healthy, N, HOP)
    Ft = frame_signal(x_test, N, HOP)

    # --- compute sequency-ordered Walsh coeffs ---
    Ch = np.stack([wht_sequency(fr, W, perm) for fr in Fh], axis=0)
    Ct = np.stack([wht_sequency(fr, W, perm) for fr in Ft], axis=0)

    # --- fit reference from healthy coefficients ---
    mu, Sig, Lc = fit_reference(Ch, LAMBDA)

    # --- compute scores + standardized deviations + time contributions ---
    scores = np.array([mahalanobis_sq(Ct[i], mu, Lc) for i in range(Ct.shape[0])])

    # empirical threshold from held-out healthy frames (here: second half)
    mid = Ch.shape[0] // 2
    scores_h = np.array([mahalanobis_sq(Ch[i], mu, Lc) for i in range(mid, Ch.shape[0])])
    thr = float(np.quantile(scores_h, 1.0 - ALPHA))

    # time contributions per frame: r_t = W^T z_t (but must invert sequency perm back to natural)
    Rt = []
    for i in range(Ct.shape[0]):
        z = standardized_deviation(Ct[i], mu, Lc)  # sequency domain standardized
        # map z back through inverse sequency -> natural -> W^T
        # We want r = W^T z_nat, where z_nat corresponds to natural ordering coefficients.
        # Our z is in sequency-ordered coordinates; convert to natural first.
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(perm))
        z_nat = z[inv]
        r = W.T @ z_nat
        Rt.append(np.abs(r))  # magnitude contribution
    Rt = np.stack(Rt, axis=0)  # (T, N)

    # aggregate |r| back to time axis for color coding
    agg_r = overlap_add(Rt, N, HOP, out_len=len(x_test), agg="max")

    # --- FIG 1: healthy vs test ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, x_healthy, linewidth=1.2, label="Healthy (synthetic)")
    ax.plot(t, x_test, linewidth=1.2, label="Test (synthetic with anomalies)", alpha=0.85)
    ax.set_title("Fig. 1 — Synthetic ECG: Healthy vs. Test with Injected Anomalies")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig1_healthy_vs_test.png"), dpi=200)
    plt.close(fig)

    # --- FIG 2: time–sequency heatmap of |coeff| for test ---
    heat = np.abs(Ct)
    plot_heatmap(heat, "Fig. 2 — Time–Sequency Map: |Walsh coefficients| (test)", os.path.join(OUTDIR, "fig2_heatmap_coeffs_test.png"))

    # --- FIG 3: score curve with threshold ---
    plot_score_curve(scores, thr, "Fig. 3 — Global anomaly score with empirical threshold", os.path.join(OUTDIR, "fig3_score_threshold.png"))

    # --- FIG 4: color-coded ECG highlighting (time localization) ---
    plot_color_coded_ecg(t, x_test, agg_r, "Fig. 4 — Time-localized anomaly highlighting (back-projection)", os.path.join(OUTDIR, "fig4_color_coded_ecg.png"))

    # --- FIG 5: multi-lead fusion demonstration (synthetic 3 leads) ---
    # Create 3 leads as mildly perturbed versions; inject anomaly only in lead 2.
    x1 = x_test + 0.005 * np.random.randn(len(x_test))
    x2 = inject_anomaly(x_test, FS, 4.0, 5.0, kind="st_shift")  # stronger in lead 2
    x3 = x_test + 0.005 * np.random.randn(len(x_test))

    def score_signal(xsig):
        F = frame_signal(xsig, N, HOP)
        C = np.stack([wht_sequency(fr, W, perm) for fr in F], axis=0)
        return np.array([mahalanobis_sq(C[i], mu, Lc) for i in range(C.shape[0])])

    s1 = score_signal(x1)
    s2 = score_signal(x2)
    s3 = score_signal(x3)
    smax = np.maximum(np.maximum(s1, s2), s3)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(s1, linewidth=1.1, label="Lead 1")
    ax.plot(s2, linewidth=1.1, label="Lead 2 (focal anomaly)")
    ax.plot(s3, linewidth=1.1, label="Lead 3")
    ax.plot(smax, linewidth=1.6, label="Max-fusion", alpha=0.9)
    ax.axhline(thr, linestyle="--", linewidth=1.1, label="Threshold")
    ax.set_title("Fig. 5 — Multi-lead fusion: max emphasizes focal deviations")
    ax.set_xlabel("Time frame")
    ax.set_ylabel(r"$d^2$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "fig5_multilead_fusion.png"), dpi=200)
    plt.close(fig)

    print("Done. Figures saved to:", OUTDIR)

if __name__ == "__main__":
    main()
