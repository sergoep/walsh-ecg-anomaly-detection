"""
Real-data PTB-XL evaluation for the Walsh-Hadamard ECG anomaly-scoring article.

This script reproduces the PTB-XL pilot reported in the Signal Processing submission
package. It reads PTB-XL v1.0.3 directly from PhysioNet using WFDB, builds a NORM
reference model, evaluates non-NORM out-of-reference records, verifies the operator
identities, and writes article-ready tables/figures.

Run in Colab or locally:
    pip install -r requirements.txt
    python scripts/ptbxl_realdata_walsh_evaluation.py
"""

from __future__ import annotations

import ast
import json
import os
import time
import zipfile
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from scipy.linalg import hadamard
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

RANDOM_SEED = 20260705
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

PN_DIR = "ptb-xl/1.0.3"
PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3/"

OUT_DIR = Path("walsh_ecg_realdata_results")
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
CACHE_DIR = OUT_DIR / "cache_windows"
for d in [OUT_DIR, FIG_DIR, TAB_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FS = 100
LOWCUT = 0.5
HIGHCUT = 40.0
FILTER_ORDER = 3
N = 128
HOP = 64
DELTA = 0.01
ALPHA = 0.05

PTBXL_LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
SELECTED_LEADS = ["I", "II", "V2"]

# Pilot values reported in the manuscript package. Increase for a larger run.
MAX_TRAIN_NORM_RECORDS = 300
MAX_VAL_NORM_RECORDS = 100
MAX_TEST_NORM_RECORDS = 200
MAX_TEST_ABN_RECORDS = 200

ENABLE_ISOLATION_FOREST = True
MAX_IFOREST_TRAIN_WINDOWS = 50000
IFOREST_N_ESTIMATORS = 200
BLOCKS = [(0, 8), (8, 32), (32, 64), (64, 128)]

lead_indices = [PTBXL_LEADS.index(lead) for lead in SELECTED_LEADS]
M = len(SELECTED_LEADS)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def latex_float(x, digits: int = 4) -> str:
    if pd.isna(x):
        return "--"
    try:
        x = float(x)
    except Exception:
        return str(x)
    if abs(x) >= 1000 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def make_latex_table(df: pd.DataFrame, caption: str, label: str, path: Path) -> None:
    cols = list(df.columns)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{" + "l" * len(cols) + "}",
        "\\hline",
        " & ".join(cols) + " \\\\",
        "\\hline",
    ]
    for _, row in df.iterrows():
        vals = [latex_float(v) if isinstance(v, (float, np.floating)) else str(v) for v in row.values]
        lines.append(" & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    write_text(path, "\n".join(lines) + "\n")


def normalized_hadamard(n: int) -> np.ndarray:
    if n & (n - 1) != 0:
        raise ValueError("n must be a power of two")
    return hadamard(n).astype(np.float64) / np.sqrt(n)


def sequency_order_from_matrix(w_nat: np.ndarray) -> np.ndarray:
    sign_changes = np.sum(np.diff(w_nat, axis=1) != 0, axis=1)
    return np.lexsort((np.arange(w_nat.shape[0]), sign_changes))


def sequency_ordered_walsh(n: int) -> Tuple[np.ndarray, np.ndarray]:
    w_nat = normalized_hadamard(n)
    order = sequency_order_from_matrix(w_nat)
    return w_nat[order, :], order


W, SEQUENCY_ORDER = sequency_ordered_walsh(N)
W_ORTH_ERR_LEFT = float(np.linalg.norm(W @ W.T - np.eye(N), ord=2))
W_ORTH_ERR_RIGHT = float(np.linalg.norm(W.T @ W - np.eye(N), ord=2))


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load_ptbxl_metadata() -> Tuple[pd.DataFrame, pd.DataFrame]:
    db = pd.read_csv(PTBXL_BASE_URL + "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(PTBXL_BASE_URL + "scp_statements.csv", index_col=0)
    db["scp_codes_dict"] = db["scp_codes"].apply(ast.literal_eval)

    def diagnostic_superclasses(code_dict):
        classes = set()
        for code in code_dict.keys():
            if code in scp.index:
                row = scp.loc[code]
                if safe_float(row.get("diagnostic", 0)) == 1:
                    diagnostic_class = row.get("diagnostic_class", np.nan)
                    if isinstance(diagnostic_class, str) and diagnostic_class != "nan":
                        classes.add(diagnostic_class)
        return sorted(classes)

    db["diagnostic_superclasses"] = db["scp_codes_dict"].apply(diagnostic_superclasses)
    db["superclass_string"] = db["diagnostic_superclasses"].apply(lambda x: "|".join(x))
    db["is_norm_reference"] = db["diagnostic_superclasses"].apply(lambda x: set(x) == {"NORM"})
    db["is_non_norm"] = db["diagnostic_superclasses"].apply(lambda x: len(set(x) - {"NORM"}) > 0)
    return db[db["filename_lr"].notna()].copy(), scp


def subsample_df(df: pd.DataFrame, max_n: Optional[int], seed: int) -> pd.DataFrame:
    if max_n is None or len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=seed).copy()


def build_splits(db: pd.DataFrame):
    train_norm = db[(db["strat_fold"].between(1, 8)) & (db["is_norm_reference"])].copy()
    val_norm = db[(db["strat_fold"] == 9) & (db["is_norm_reference"])].copy()
    test_norm = db[(db["strat_fold"] == 10) & (db["is_norm_reference"])].copy()
    test_abn = db[(db["strat_fold"] == 10) & (db["is_non_norm"])].copy()
    return (
        subsample_df(train_norm, MAX_TRAIN_NORM_RECORDS, RANDOM_SEED + 1),
        subsample_df(val_norm, MAX_VAL_NORM_RECORDS, RANDOM_SEED + 2),
        subsample_df(test_norm, MAX_TEST_NORM_RECORDS, RANDOM_SEED + 3),
        subsample_df(test_abn, MAX_TEST_ABN_RECORDS, RANDOM_SEED + 4),
    )


def read_ptbxl_record(filename_lr: str) -> Tuple[np.ndarray, int]:
    """Correct PTB-XL remote reading for wfdb 4.x."""
    filename_lr = str(filename_lr).strip().replace("\\", "/")
    parts = filename_lr.split("/")
    if len(parts) < 2:
        raise RuntimeError(f"Unexpected PTB-XL filename_lr format: {filename_lr}")
    record_name = parts[-1]
    remote_subdir = "/".join(parts[:-1])
    remote_pn_dir = f"{PN_DIR}/{remote_subdir}"
    try:
        sig, fields = wfdb.rdsamp(record_name, pn_dir=remote_pn_dir)
    except Exception as e1:
        try:
            sig, fields = wfdb.rdsamp(filename_lr, pn_dir=PN_DIR)
        except Exception as e2:
            raise RuntimeError(
                f"Cannot read PTB-XL record: {filename_lr}\n"
                f"Attempt 1: wfdb.rdsamp({record_name!r}, pn_dir={remote_pn_dir!r}): {repr(e1)}\n"
                f"Attempt 2: wfdb.rdsamp({filename_lr!r}, pn_dir={PN_DIR!r}): {repr(e2)}"
            )
    fs = int(fields.get("fs", FS))
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[1] < max(lead_indices) + 1 or sig.shape[0] < N:
        raise RuntimeError(f"Unexpected signal shape for {filename_lr}: {sig.shape}")
    return sig[:, lead_indices], fs


def bandpass_filter(sig: np.ndarray, fs: int) -> np.ndarray:
    high = min(HIGHCUT, fs / 2.0 - 1e-3)
    if high <= LOWCUT:
        return sig.copy()
    sos = butter(FILTER_ORDER, [LOWCUT, high], btype="bandpass", fs=fs, output="sos")
    try:
        return sosfiltfilt(sos, sig, axis=0)
    except Exception:
        return sig.copy()


def robust_normalize_per_record(sig: np.ndarray) -> np.ndarray:
    x = np.asarray(sig, dtype=np.float64).copy()
    med = np.nanmedian(x, axis=0, keepdims=True)
    mad = np.nanmedian(np.abs(x - med), axis=0, keepdims=True)
    std = np.nanstd(x, axis=0, keepdims=True)
    scale = np.where(1.4826 * mad < 1e-8, std, 1.4826 * mad)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return np.nan_to_num((x - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def extract_windows(sig: np.ndarray, n: int = N, hop: int = HOP) -> np.ndarray:
    t, l = sig.shape
    if t < n:
        return np.zeros((0, l, n), dtype=np.float64)
    starts = np.arange(0, t - n + 1, hop)
    windows = np.empty((len(starts), l, n), dtype=np.float64)
    for i, s in enumerate(starts):
        windows[i] = sig[s : s + n, :].T
    return windows


def cache_path_for_record(ecg_id: int) -> Path:
    leads_tag = "_".join(SELECTED_LEADS).replace("/", "")
    return CACHE_DIR / f"ecg_{int(ecg_id)}_N{N}_hop{HOP}_{leads_tag}.npz"


def load_preprocessed_windows(row: pd.Series, use_cache: bool = True) -> Tuple[np.ndarray, int]:
    cp = cache_path_for_record(int(row.name))
    if use_cache and cp.exists():
        data = np.load(cp)
        return data["windows"], int(data["fs"])
    sig, fs = read_ptbxl_record(row["filename_lr"])
    sig = robust_normalize_per_record(bandpass_filter(sig, fs))
    windows = extract_windows(sig, N, HOP)
    if use_cache:
        np.savez_compressed(cp, windows=windows, fs=fs)
    return windows, fs


def windows_to_time_features(windows: np.ndarray) -> List[np.ndarray]:
    return [windows[:, ell, :].astype(np.float64) for ell in range(windows.shape[1])]


def windows_to_walsh_features(windows: np.ndarray) -> List[np.ndarray]:
    return [(windows[:, ell, :] @ W.T).astype(np.float64) for ell in range(windows.shape[1])]


def windows_to_walsh_concat_features(windows: np.ndarray) -> np.ndarray:
    return np.concatenate([windows[:, ell, :] @ W.T for ell in range(windows.shape[1])], axis=1).astype(np.float32)


def regularize_covariance(sigma: np.ndarray, delta: float = DELTA) -> np.ndarray:
    n = sigma.shape[0]
    tr = float(np.trace(sigma))
    return sigma + (1e-8 if tr <= 1e-14 else delta * tr / n) * np.eye(n)


def inv_and_invsqrt_spd(a: np.ndarray):
    a = 0.5 * (a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, 1e-12)
    return (vecs * (1.0 / vals)) @ vecs.T, (vecs * (1.0 / np.sqrt(vals))) @ vecs.T, vals


def fit_full_cov_model(x: np.ndarray, delta: float = DELTA) -> Dict:
    mu = np.mean(x, axis=0)
    xc = x - mu
    sigma = np.eye(x.shape[1]) if x.shape[0] <= 1 else np.cov(xc, rowvar=False, bias=False)
    sigma_delta = regularize_covariance(sigma, delta)
    inv_a, invsqrt_a, eigvals = inv_and_invsqrt_spd(sigma_delta)
    return {"mu": mu, "Sigma": sigma, "Sigma_delta": sigma_delta, "inv": inv_a, "invsqrt": invsqrt_a,
            "eigvals": eigvals, "condition": float(np.max(eigvals) / np.min(eigvals)),
            "trace_scaled_s": float(np.trace(sigma) / x.shape[1])}


def fit_diag_cov_model(x: np.ndarray, delta: float = DELTA) -> Dict:
    mu = np.mean(x, axis=0)
    var = np.var(x - mu, axis=0, ddof=1)
    s = max(float(np.mean(var)), 1e-8)
    var_delta = np.maximum(var + delta * s, 1e-12)
    return {"mu": mu, "var": var, "var_delta": var_delta, "inv_diag": 1.0 / var_delta,
            "invsqrt_diag": 1.0 / np.sqrt(var_delta), "condition": float(np.max(var_delta) / np.min(var_delta)),
            "trace_scaled_s": s}


def block_project_covariance(sigma: np.ndarray, blocks: List[Tuple[int, int]]) -> np.ndarray:
    out = np.zeros_like(sigma)
    for a, b in blocks:
        out[a:b, a:b] = sigma[a:b, a:b]
    return out


def fit_block_cov_model(x: np.ndarray, blocks: List[Tuple[int, int]], delta: float = DELTA) -> Dict:
    mu = np.mean(x, axis=0)
    xc = x - mu
    sigma = np.eye(x.shape[1]) if x.shape[0] <= 1 else np.cov(xc, rowvar=False, bias=False)
    sigma_b = block_project_covariance(sigma, blocks)
    sigma_delta = regularize_covariance(sigma_b, delta)
    inv_a, invsqrt_a, eigvals = inv_and_invsqrt_spd(sigma_delta)
    cross_energy = float(np.linalg.norm(sigma - sigma_b, ord="fro") ** 2)
    retained_energy = float(np.linalg.norm(sigma_b, ord="fro") ** 2)
    return {"mu": mu, "Sigma": sigma, "SigmaB": sigma_b, "Sigma_delta": sigma_delta, "inv": inv_a,
            "invsqrt": invsqrt_a, "eigvals": eigvals, "condition": float(np.max(eigvals) / np.min(eigvals)),
            "trace_scaled_s": float(np.trace(sigma_b) / x.shape[1]), "cross_block_energy": cross_energy,
            "retained_block_energy": retained_energy, "cross_to_retained_ratio": float(cross_energy / max(retained_energy, 1e-12))}


def quad_scores_full(x: np.ndarray, model: Dict) -> np.ndarray:
    d = x - model["mu"]
    return np.einsum("ij,jk,ik->i", d, model["inv"], d)


def quad_scores_diag(x: np.ndarray, model: Dict) -> np.ndarray:
    d = x - model["mu"]
    return np.sum((d ** 2) * model["inv_diag"], axis=1)


def exact_decomposition_profile_walsh(c: np.ndarray, model: Dict):
    d = c - model["mu"]
    z = model["invsqrt"] @ d
    r = W.T @ z
    score = float(d.T @ model["inv"] @ d)
    return r, score, abs(score - float(np.sum(r ** 2)))


def collect_reference_features(df: pd.DataFrame):
    time_lists = [[] for _ in range(M)]
    walsh_lists = [[] for _ in range(M)]
    iforest_list = []
    total_windows = 0
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Collecting training reference windows"):
        try:
            windows, _ = load_preprocessed_windows(row, use_cache=True)
            if windows.shape[0] == 0:
                failed += 1
                continue
            time_feats = windows_to_time_features(windows)
            walsh_feats = windows_to_walsh_features(windows)
            for ell in range(M):
                time_lists[ell].append(time_feats[ell])
                walsh_lists[ell].append(walsh_feats[ell])
            if ENABLE_ISOLATION_FOREST:
                iforest_list.append(windows_to_walsh_concat_features(windows))
            total_windows += windows.shape[0]
        except Exception as exc:
            failed += 1
            if failed <= 5:
                print("Training read error:", row.name, row.get("filename_lr"), repr(exc))
    if total_windows == 0:
        raise RuntimeError("No training windows collected")
    time_by_lead = [np.vstack(time_lists[ell]) for ell in range(M)]
    walsh_by_lead = [np.vstack(walsh_lists[ell]) for ell in range(M)]
    iforest_x = np.vstack(iforest_list) if ENABLE_ISOLATION_FOREST and iforest_list else None
    if iforest_x is not None and iforest_x.shape[0] > MAX_IFOREST_TRAIN_WINDOWS:
        idx = rng.choice(iforest_x.shape[0], size=MAX_IFOREST_TRAIN_WINDOWS, replace=False)
        iforest_x = iforest_x[idx]
    info = {"total_training_records": int(len(df)), "total_training_windows": int(total_windows),
            "failed_training_records": int(failed), "selected_leads": SELECTED_LEADS,
            "windows_per_lead": {SELECTED_LEADS[ell]: int(time_by_lead[ell].shape[0]) for ell in range(M)}}
    return time_by_lead, walsh_by_lead, iforest_x, info


def main() -> None:
    print("Package versions:", "numpy", np.__version__, "pandas", pd.__version__, "wfdb", wfdb.__version__)
    print("Walsh errors:", W_ORTH_ERR_LEFT, W_ORTH_ERR_RIGHT)
    metadata, _ = load_ptbxl_metadata()
    train_norm_df, val_norm_df, test_norm_df, test_abn_df = build_splits(metadata)
    split_summary = pd.DataFrame([
        {"split": "train_norm", "records": len(train_norm_df), "patients": train_norm_df["patient_id"].nunique()},
        {"split": "val_norm", "records": len(val_norm_df), "patients": val_norm_df["patient_id"].nunique()},
        {"split": "test_norm", "records": len(test_norm_df), "patients": test_norm_df["patient_id"].nunique()},
        {"split": "test_non_norm", "records": len(test_abn_df), "patients": test_abn_df["patient_id"].nunique()},
    ])
    print(split_summary)
    save_dataframe(split_summary, TAB_DIR / "ptbxl_split_summary.csv")

    # Diagnostic
    row0 = train_norm_df.iloc[0]
    sig0, fs0 = read_ptbxl_record(row0["filename_lr"])
    win0 = extract_windows(robust_normalize_per_record(bandpass_filter(sig0, fs0)), N, HOP)
    print("Diagnostic:", row0.name, row0["filename_lr"], sig0.shape, fs0, win0.shape)

    start = time.time()
    time_train_by_lead, walsh_train_by_lead, iforest_train_x, train_info = collect_reference_features(train_norm_df)
    save_json(TAB_DIR / "training_collection_info.json", train_info)

    global models, iforest_model, MODEL_NAMES, thresholds
    models = {"time_full": [], "walsh_full": [], "walsh_diag": [], "walsh_block": []}
    fit_rows = []
    for ell, lead in enumerate(SELECTED_LEADS):
        print("Fitting", lead)
        m_time = fit_full_cov_model(time_train_by_lead[ell], DELTA)
        m_wfull = fit_full_cov_model(walsh_train_by_lead[ell], DELTA)
        m_wdiag = fit_diag_cov_model(walsh_train_by_lead[ell], DELTA)
        m_wblock = fit_block_cov_model(walsh_train_by_lead[ell], BLOCKS, DELTA)
        models["time_full"].append(m_time); models["walsh_full"].append(m_wfull)
        models["walsh_diag"].append(m_wdiag); models["walsh_block"].append(m_wblock)
        fit_rows += [
            {"lead": lead, "model": "time_full", "trace_scaled_s": m_time["trace_scaled_s"], "condition": m_time["condition"], "cross_to_retained_ratio": np.nan},
            {"lead": lead, "model": "walsh_full", "trace_scaled_s": m_wfull["trace_scaled_s"], "condition": m_wfull["condition"], "cross_to_retained_ratio": np.nan},
            {"lead": lead, "model": "walsh_diag", "trace_scaled_s": m_wdiag["trace_scaled_s"], "condition": m_wdiag["condition"], "cross_to_retained_ratio": np.nan},
            {"lead": lead, "model": "walsh_block", "trace_scaled_s": m_wblock["trace_scaled_s"], "condition": m_wblock["condition"], "cross_to_retained_ratio": m_wblock["cross_to_retained_ratio"]},
        ]
    save_dataframe(pd.DataFrame(fit_rows), TAB_DIR / "reference_model_fit_summary.csv")

    iforest_model = None
    if ENABLE_ISOLATION_FOREST and iforest_train_x is not None:
        print("Fitting Isolation Forest")
        iforest_model = IsolationForest(n_estimators=IFOREST_N_ESTIMATORS, contamination="auto", random_state=RANDOM_SEED, n_jobs=-1)
        iforest_model.fit(iforest_train_x)
    MODEL_NAMES = ["time_full", "walsh_full", "walsh_diag", "walsh_block"] + (["isolation_forest_walsh"] if iforest_model is not None else [])

    def score_windows_by_model(windows: np.ndarray, model_name: str) -> np.ndarray:
        if windows.shape[0] == 0:
            return np.array([], dtype=np.float64)
        if model_name == "time_full":
            feats = windows_to_time_features(windows)
            return np.max(np.vstack([quad_scores_full(feats[ell], models["time_full"][ell]) for ell in range(M)]), axis=0)
        if model_name == "walsh_full":
            feats = windows_to_walsh_features(windows)
            return np.max(np.vstack([quad_scores_full(feats[ell], models["walsh_full"][ell]) for ell in range(M)]), axis=0)
        if model_name == "walsh_diag":
            feats = windows_to_walsh_features(windows)
            return np.max(np.vstack([quad_scores_diag(feats[ell], models["walsh_diag"][ell]) for ell in range(M)]), axis=0)
        if model_name == "walsh_block":
            feats = windows_to_walsh_features(windows)
            return np.max(np.vstack([quad_scores_full(feats[ell], models["walsh_block"][ell]) for ell in range(M)]), axis=0)
        if model_name == "isolation_forest_walsh":
            return -iforest_model.score_samples(windows_to_walsh_concat_features(windows))
        raise ValueError(model_name)

    all_scores = {m: [] for m in MODEL_NAMES}
    for _, row in tqdm(val_norm_df.iterrows(), total=len(val_norm_df), desc="Calibrating thresholds"):
        try:
            windows, _ = load_preprocessed_windows(row, use_cache=True)
            for m in MODEL_NAMES:
                s = score_windows_by_model(windows, m)
                if len(s): all_scores[m].append(s)
        except Exception as exc:
            print("Validation error", row.name, repr(exc))
    thresholds = {}
    val_stats = []
    for m in MODEL_NAMES:
        v = np.concatenate(all_scores[m])
        thresholds[m] = float(np.quantile(v, 1 - ALPHA))
        val_stats.append({"model": m, "num_validation_window_scores": int(len(v)), "threshold_alpha": ALPHA,
                          "threshold": thresholds[m], "validation_mean": float(np.mean(v)),
                          "validation_std": float(np.std(v)), "validation_q95": float(np.quantile(v, .95))})
    save_dataframe(pd.DataFrame(val_stats), TAB_DIR / "validation_thresholds.csv")

    def summarize_record_scores(scores):
        if len(scores) == 0:
            return {"window_max": np.nan, "window_mean": np.nan, "window_q95": np.nan}
        return {"window_max": float(np.max(scores)), "window_mean": float(np.mean(scores)), "window_q95": float(np.quantile(scores, .95))}

    rows = []
    failed_eval = 0
    test_eval_df = pd.concat([test_norm_df.assign(_label=0, _label_name="NORM"), test_abn_df.assign(_label=1, _label_name="non_NORM")])
    for _, row in tqdm(test_eval_df.iterrows(), total=len(test_eval_df), desc="Evaluating test records"):
        try:
            windows, _ = load_preprocessed_windows(row, use_cache=True)
            for m in MODEL_NAMES:
                ws = score_windows_by_model(windows, m)
                summ = summarize_record_scores(ws)
                rows.append({"ecg_id": int(row.name), "patient_id": int(row["patient_id"]), "filename_lr": row["filename_lr"],
                             "label": int(row["_label"]), "label_name": row["_label_name"], "superclass_string": row["superclass_string"],
                             "model": m, "num_windows": int(len(ws)), **summ, "threshold": thresholds[m],
                             "any_window_above_threshold": int(np.any(ws > thresholds[m])) if len(ws) else 0})
        except Exception as exc:
            failed_eval += 1
            if failed_eval <= 5: print("Evaluation error", row.name, repr(exc))
    record_scores_df = pd.DataFrame(rows)
    save_dataframe(record_scores_df, TAB_DIR / "record_scores_all_models.csv")

    def tpr_at_fpr(y_true, score, target_fpr=.05):
        fpr, tpr, _ = roc_curve(y_true, score)
        valid = np.where(fpr <= target_fpr)[0]
        return float(np.max(tpr[valid])) if len(valid) else 0.0

    def thresholded_metrics(y_true, score, threshold):
        pred = (score > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        return {"threshold_sensitivity": float(tp / max(tp + fn, 1)), "threshold_specificity": float(tn / max(tn + fp, 1)),
                "threshold_precision": float(tp / max(tp + fp, 1)), "threshold_accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}

    metric_rows = []
    aggregations = {"max": "window_max", "mean": "window_mean", "q95": "window_q95"}
    for m in MODEL_NAMES:
        dfm = record_scores_df[record_scores_df["model"] == m].copy()
        y = dfm["label"].values.astype(int)
        for agg_name, col in aggregations.items():
            score = dfm[col].values.astype(float)
            mask = np.isfinite(score)
            yy, ss = y[mask], score[mask]
            auc = roc_auc_score(yy, ss) if len(np.unique(yy)) >= 2 else np.nan
            ap = average_precision_score(yy, ss) if len(np.unique(yy)) >= 2 else np.nan
            tpr5 = tpr_at_fpr(yy, ss) if len(np.unique(yy)) >= 2 else np.nan
            metric_rows.append({"model": m, "aggregation": agg_name, "n_records": int(len(yy)),
                                "n_positive_non_norm": int(np.sum(yy == 1)), "n_negative_norm": int(np.sum(yy == 0)),
                                "roc_auc": float(auc), "average_precision": float(ap), "tpr_at_fpr_5pct": float(tpr5),
                                **thresholded_metrics(yy, ss, thresholds[m])})
    metrics_df = pd.DataFrame(metric_rows)
    save_dataframe(metrics_df, TAB_DIR / "record_level_metrics.csv")
    article_metrics = metrics_df[metrics_df["aggregation"] == "q95"][["model", "aggregation", "n_records", "n_positive_non_norm", "n_negative_norm", "roc_auc", "average_precision", "tpr_at_fpr_5pct", "threshold_sensitivity", "threshold_specificity"]].copy()
    save_dataframe(article_metrics, TAB_DIR / "article_ablation_table_q95.csv")
    make_latex_table(article_metrics.rename(columns={"model": "Model", "aggregation": "Aggregation", "n_records": "Records", "n_positive_non_norm": "Non-NORM", "n_negative_norm": "NORM", "roc_auc": "AUC", "average_precision": "AUPRC", "tpr_at_fpr_5pct": "TPR@5\\%FPR", "threshold_sensitivity": "Sensitivity", "threshold_specificity": "Specificity"}),
                     "Real-data PTB-XL record-level ablation under patient-wise stratified folds.", "tab:ptbxl-real-ablation", TAB_DIR / "article_ablation_table_q95.tex")

    max_walsh_time, max_full_dec, max_block_dec, count_windows = 0.0, 0.0, 0.0, 0
    for _, row in tqdm(pd.concat([val_norm_df, test_norm_df, test_abn_df]).head(50).iterrows(), total=50, desc="Verifying identities"):
        try:
            windows, _ = load_preprocessed_windows(row, use_cache=True)
            time_feats = windows_to_time_features(windows)
            walsh_feats = windows_to_walsh_features(windows)
            for ell in range(M):
                st = quad_scores_full(time_feats[ell], models["time_full"][ell])
                sw = quad_scores_full(walsh_feats[ell], models["walsh_full"][ell])
                max_walsh_time = max(max_walsh_time, float(np.max(np.abs(st - sw))))
                for k in range(min(5, walsh_feats[ell].shape[0])):
                    _, _, ef = exact_decomposition_profile_walsh(walsh_feats[ell][k], models["walsh_full"][ell])
                    _, _, eb = exact_decomposition_profile_walsh(walsh_feats[ell][k], models["walsh_block"][ell])
                    max_full_dec, max_block_dec = max(max_full_dec, ef), max(max_block_dec, eb)
            count_windows += windows.shape[0]
        except Exception:
            pass
    verification = {"random_seed": RANDOM_SEED, "N": N, "hop": HOP, "delta": DELTA, "alpha": ALPHA,
                    "selected_leads": SELECTED_LEADS, "walsh_orthogonality_error_left_2norm": W_ORTH_ERR_LEFT,
                    "walsh_orthogonality_error_right_2norm": W_ORTH_ERR_RIGHT,
                    "max_walsh_time_score_discrepancy": max_walsh_time,
                    "max_full_exact_decomposition_error": max_full_dec,
                    "max_block_exact_decomposition_error": max_block_dec,
                    "verified_records": 50, "verified_windows": int(count_windows)}
    save_json(TAB_DIR / "verification_report.json", verification)
    verification_df = pd.DataFrame([
        {"Quantity": "||W W^T - I||_2", "Observed value": verification["walsh_orthogonality_error_left_2norm"]},
        {"Quantity": "||W^T W - I||_2", "Observed value": verification["walsh_orthogonality_error_right_2norm"]},
        {"Quantity": "Max Walsh/time-domain score discrepancy", "Observed value": verification["max_walsh_time_score_discrepancy"]},
        {"Quantity": "Max full-covariance decomposition error", "Observed value": verification["max_full_exact_decomposition_error"]},
        {"Quantity": "Max block-covariance decomposition error", "Observed value": verification["max_block_exact_decomposition_error"]},
    ])
    save_dataframe(verification_df, TAB_DIR / "article_verification_table.csv")
    make_latex_table(verification_df, "Numerical verification of orthogonal-invariance and exact-decomposition identities on real PTB-XL windows.", "tab:real-verification", TAB_DIR / "article_verification_table.tex")

    # Figures
    def plot_roc_curves():
        plt.figure(figsize=(7.5, 5.5))
        for m in MODEL_NAMES:
            dfm = record_scores_df[record_scores_df["model"] == m].copy()
            y = dfm["label"].values.astype(int); s = dfm["window_q95"].values.astype(float)
            mask = np.isfinite(s); y, s = y[mask], s[mask]
            if len(np.unique(y)) < 2: continue
            fpr, tpr, _ = roc_curve(y, s); auc = roc_auc_score(y, s)
            plt.plot(fpr, tpr, label=f"{m} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
        plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("PTB-XL record-level ROC curves")
        plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG_DIR / "fig_realdata_roc_q95.png", dpi=300); plt.close()

    def plot_pr_curves():
        plt.figure(figsize=(7.5, 5.5))
        for m in MODEL_NAMES:
            dfm = record_scores_df[record_scores_df["model"] == m].copy()
            y = dfm["label"].values.astype(int); s = dfm["window_q95"].values.astype(float)
            mask = np.isfinite(s); y, s = y[mask], s[mask]
            if len(np.unique(y)) < 2: continue
            p, r, _ = precision_recall_curve(y, s); ap = average_precision_score(y, s)
            plt.plot(r, p, label=f"{m} (AP={ap:.3f})")
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PTB-XL record-level precision-recall curves")
        plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(FIG_DIR / "fig_realdata_pr_q95.png", dpi=300); plt.close()

    def plot_ablation_bar():
        df = article_metrics.sort_values("roc_auc", ascending=False)
        plt.figure(figsize=(8, 5)); plt.bar(df["model"], df["roc_auc"]); plt.xticks(rotation=30, ha="right")
        plt.ylabel("Record-level ROC-AUC"); plt.title("PTB-XL real-data ablation, q95 aggregation")
        plt.tight_layout(); plt.savefig(FIG_DIR / "fig_realdata_ablation_auc.png", dpi=300); plt.close()

    plot_roc_curves(); plot_pr_curves(); plot_ablation_bar()

    best = article_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    methods_text = f"""Real-data PTB-XL protocol.\n\nThe evaluation used PTB-XL v1.0.3 low-resolution records sampled at 100 Hz. Folds 1--8 were used for NORM reference estimation, fold 9 for NORM threshold calibration, and fold 10 for final NORM versus non-NORM testing. Leads: {', '.join(SELECTED_LEADS)}. Window length N={N}, hop={HOP}, delta={DELTA}, alpha={ALPHA}.\n"""
    results_text = f"""Real-data PTB-XL results.\n\nThe evaluation used {len(train_norm_df)} NORM training records, {len(val_norm_df)} NORM validation records, {len(test_norm_df)} NORM test records, and {len(test_abn_df)} non-NORM test records. The maximum Walsh/time-domain discrepancy was {max_walsh_time:.3e}. The full and block decomposition errors were {max_full_dec:.3e} and {max_block_dec:.3e}. Best q95 model: {best['model']}, AUC={best['roc_auc']:.4f}, AUPRC={best['average_precision']:.4f}.\n"""
    write_text(OUT_DIR / "article_methods_realdata.txt", methods_text)
    write_text(OUT_DIR / "article_results_realdata.txt", results_text)
    manifest = {"title": "Real-data PTB-XL evaluation for Walsh-Hadamard ECG anomaly scoring", "random_seed": RANDOM_SEED,
                "physionet_database": "PTB-XL", "physionet_version": "1.0.3", "pn_dir": PN_DIR,
                "sampling_rate_hz": FS, "bandpass_hz": [LOWCUT, HIGHCUT], "window_length": N, "hop": HOP,
                "delta": DELTA, "alpha": ALPHA, "selected_leads": SELECTED_LEADS, "blocks": BLOCKS,
                "models": MODEL_NAMES, "elapsed_seconds": time.time() - start, "failed_eval_records": failed_eval,
                "verification": verification}
    save_json(OUT_DIR / "run_manifest.json", manifest)

    zip_path = Path("walsh_ecg_realdata_results.zip")
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_DIR):
            for file in files:
                p = Path(root) / file
                zf.write(p, arcname=str(p.relative_to(OUT_DIR.parent)))
    print("DONE", OUT_DIR.resolve(), zip_path.resolve())
    print(article_metrics)
    print(verification_df)


if __name__ == "__main__":
    main()
