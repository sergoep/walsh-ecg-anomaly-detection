#!/usr/bin/env python3
"""Run the fast ECG Walsh-Hadamard verification demo.

This script creates a small synthetic ECG-like dataset, fits the pooled healthy
reference model, verifies orthogonal invariance and exact score decomposition,
produces figures, and writes CSV/JSON outputs under results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from src.ecg_walsh.core import (
    aggregate_contributions,
    decompose_window_walsh,
    fit_reference,
    score_multilead_windows,
    score_window_time_domain,
    score_window_walsh,
    sequency_ordered_walsh,
)
from src.ecg_walsh.synthetic import bandpass_filter, extract_multilead_windows, synthetic_record


def collect_pooled_healthy_windows(records, meta_df, split_name, N, hop):
    pooled = []
    for i, row in meta_df.iterrows():
        if row["split"] == split_name and row["label"] == 0:
            windows, _ = extract_multilead_windows(records[i], N=N, hop=hop)
            for t in range(windows.shape[0]):
                for ell in range(windows.shape[1]):
                    pooled.append(windows[t, ell, :])
    return np.asarray(pooled)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default="results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--fs", type=int, default=100)
    parser.add_argument("--N", type=int, default=128)
    parser.add_argument("--hop", type=int, default=64)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--n-leads", type=int, default=3)
    parser.add_argument("--record-seconds", type=int, default=12)
    parser.add_argument("--train-healthy", type=int, default=36)
    parser.add_argument("--val-healthy", type=int, default=12)
    parser.add_argument("--test-healthy", type=int, default=20)
    parser.add_argument("--test-abnormal", type=int, default=20)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    n_samples = args.fs * args.record_seconds
    W, order, sequencies = sequency_ordered_walsh(args.N)
    orth_err_1 = float(np.linalg.norm(W @ W.T - np.eye(args.N), ord=2))
    orth_err_2 = float(np.linalg.norm(W.T @ W - np.eye(args.N), ord=2))

    records, meta = [], []
    rid = 0
    for split, label, count in [
        ("train", 0, args.train_healthy),
        ("val", 0, args.val_healthy),
        ("test", 0, args.test_healthy),
        ("test", 1, args.test_abnormal),
    ]:
        for _ in range(count):
            x = synthetic_record(rid, bool(label), args.seed, n_samples, args.fs, args.n_leads)
            records.append(bandpass_filter(x, fs=args.fs))
            meta.append({"record_id": rid, "split": split, "label": label})
            rid += 1
    meta_df = pd.DataFrame(meta)

    X_train_ref = collect_pooled_healthy_windows(records, meta_df, "train", args.N, args.hop)
    model = fit_reference(X_train_ref, W, delta=args.delta)

    val_scores = []
    for i, row in meta_df.iterrows():
        if row["split"] == "val" and row["label"] == 0:
            windows, _ = extract_multilead_windows(records[i], N=args.N, hop=args.hop)
            _, fused, _ = score_multilead_windows(windows, model)
            val_scores.extend(fused.tolist())
    val_scores = np.asarray(val_scores)
    tau = float(np.quantile(val_scores, 1.0 - args.alpha))

    test_window_scores, test_window_labels = [], []
    test_record_scores_max, test_record_scores_mean, test_record_scores_q95 = [], [], []
    test_record_labels, test_record_ids = [], []

    for i, row in meta_df.iterrows():
        if row["split"] == "test":
            windows, _ = extract_multilead_windows(records[i], N=args.N, hop=args.hop)
            _, fused, _ = score_multilead_windows(windows, model)
            label = int(row["label"])
            test_window_scores.extend(fused.tolist())
            test_window_labels.extend([label] * len(fused))
            test_record_scores_max.append(float(np.max(fused)))
            test_record_scores_mean.append(float(np.mean(fused)))
            test_record_scores_q95.append(float(np.quantile(fused, 0.95)))
            test_record_labels.append(label)
            test_record_ids.append(int(row["record_id"]))

    test_window_scores = np.asarray(test_window_scores)
    test_window_labels = np.asarray(test_window_labels)
    test_record_labels = np.asarray(test_record_labels)
    test_record_scores_max = np.asarray(test_record_scores_max)
    test_record_scores_mean = np.asarray(test_record_scores_mean)
    test_record_scores_q95 = np.asarray(test_record_scores_q95)

    window_auc = float(roc_auc_score(test_window_labels, test_window_scores))
    record_auc_max = float(roc_auc_score(test_record_labels, test_record_scores_max))
    record_auc_mean = float(roc_auc_score(test_record_labels, test_record_scores_mean))
    record_auc_q95 = float(roc_auc_score(test_record_labels, test_record_scores_q95))
    cm = confusion_matrix(test_window_labels, (test_window_scores > tau).astype(int))

    # Orthogonal invariance check.
    diffs = []
    sample_count = 0
    for i, row in meta_df.iterrows():
        if row["split"] == "test":
            windows, _ = extract_multilead_windows(records[i], N=args.N, hop=args.hop)
            for t in range(min(3, windows.shape[0])):
                for ell in range(args.n_leads):
                    xw = windows[t, ell, :]
                    diffs.append(abs(score_window_walsh(xw, model) - score_window_time_domain(xw, model)))
                    sample_count += 1
            if sample_count > 80:
                break
    diffs = np.asarray(diffs)

    # Exact decomposition check and figures.
    abnormal_indices = meta_df[(meta_df["split"] == "test") & (meta_df["label"] == 1)].index.tolist()
    chosen_i = abnormal_indices[0]
    chosen_record = records[chosen_i]
    chosen_windows, chosen_starts = extract_multilead_windows(chosen_record, N=args.N, hop=args.hop)
    chosen_lead_scores, chosen_fused, chosen_argmax = score_multilead_windows(chosen_windows, model)
    t_star = int(np.argmax(chosen_fused))
    ell_star = int(chosen_argmax[t_star])
    score, z, r, score_z, score_r = decompose_window_walsh(chosen_windows[t_star, ell_star, :], model)

    # Optional lightweight comparator.
    iso_auc = None
    try:
        iso = IsolationForest(n_estimators=100, contamination=args.alpha, random_state=args.seed, n_jobs=-1)
        iso.fit(X_train_ref)
        iso_scores, iso_labels = [], []
        for i, row in meta_df.iterrows():
            if row["split"] == "test":
                windows, _ = extract_multilead_windows(records[i], N=args.N, hop=args.hop)
                flat = windows.reshape(-1, args.N)
                s = -iso.score_samples(flat)
                s = s.reshape(windows.shape[0], windows.shape[1]).max(axis=1)
                iso_scores.extend(s.tolist())
                iso_labels.extend([int(row["label"])] * len(s))
        iso_auc = float(roc_auc_score(np.asarray(iso_labels), np.asarray(iso_scores)))
    except Exception:
        iso_auc = None

    # Contribution profile.
    A_abs, A_energy, fused_scores, _ = aggregate_contributions(
        chosen_record, chosen_starts, chosen_windows, model
    )
    time_axis = np.arange(chosen_record.shape[0]) / args.fs
    plt.figure(figsize=(12, 4))
    plt.plot(time_axis, chosen_record[:, ell_star], linewidth=1.0, label=f"lead {ell_star}")
    A_scaled = A_abs / (np.max(A_abs) + 1e-12)
    y_min, y_max = plt.ylim()
    plt.fill_between(time_axis, y_min, y_min + A_scaled * (y_max - y_min) * 0.35, alpha=0.35, label="contribution magnitude")
    plt.title("Synthetic ECG-like waveform with detector-relative contribution profile")
    plt.xlabel("Time, seconds")
    plt.ylabel("Amplitude / overlay")
    plt.legend(loc="upper right")
    plt.tight_layout()
    fig_contrib = figdir / "fig_contribution_profile.png"
    plt.savefig(fig_contrib, dpi=160)
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.plot(fused_scores, marker="o", linewidth=1.0, label="max-fusion score")
    plt.axhline(tau, linestyle="--", label=f"validation threshold, alpha={args.alpha}")
    plt.title("Window-level fused anomaly score trace")
    plt.xlabel("Window index")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    fig_score = figdir / "fig_score_trace.png"
    plt.savefig(fig_score, dpi=160)
    plt.close()

    windows_lead = np.array([chosen_windows[t, ell_star, :] for t in range(chosen_windows.shape[0])])
    coeff = windows_lead @ W.T
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coeff).T, aspect="auto", origin="lower")
    plt.colorbar(label="|Walsh-Hadamard coefficient|")
    plt.title("Time-sequency representation")
    plt.xlabel("Window index")
    plt.ylabel("Sequency-ordered index")
    plt.tight_layout()
    fig_seq = figdir / "fig_time_sequency.png"
    plt.savefig(fig_seq, dpi=160)
    plt.close()

    fpr, tpr, _ = roc_curve(test_window_labels, test_window_scores)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"score, AUC={window_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("Window-level ROC")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    fig_roc = figdir / "fig_roc.png"
    plt.savefig(fig_roc, dpi=160)
    plt.close()

    summary = {
        "N": args.N,
        "hop": args.hop,
        "fs": args.fs,
        "delta": args.delta,
        "alpha": args.alpha,
        "L_train_reference_windows": int(len(X_train_ref)),
        "trace_scale": float(model.trace_scale),
        "condition_number": float(model.condition_number),
        "threshold_tau": tau,
        "window_auc": window_auc,
        "record_auc_max": record_auc_max,
        "record_auc_mean": record_auc_mean,
        "record_auc_q95": record_auc_q95,
        "max_invariance_error": float(diffs.max()),
        "exact_decomposition_error": float(abs(score - score_r)),
        "orthogonality_error_WWT": orth_err_1,
        "orthogonality_error_WTW": orth_err_2,
        "isolation_forest_window_auc": iso_auc,
    }
    pd.DataFrame([summary]).to_csv(outdir / "summary_metrics.csv", index=False)
    pd.DataFrame({
        "record_id": test_record_ids,
        "label": test_record_labels,
        "score_max": test_record_scores_max,
        "score_mean": test_record_scores_mean,
        "score_q95": test_record_scores_q95,
    }).to_csv(outdir / "record_level_scores.csv", index=False)
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(outdir / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("ECG Walsh-Hadamard verification summary\n")
        f.write("=" * 45 + "\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
        f.write("\nWindow confusion matrix at calibrated threshold:\n")
        f.write(str(cm))
        f.write("\n")

    print("Run complete. Main checks:")
    print(f"  max invariance error:       {summary['max_invariance_error']:.3e}")
    print(f"  exact decomposition error:  {summary['exact_decomposition_error']:.3e}")
    print(f"  orthogonality error WWT:    {orth_err_1:.3e}")
    print(f"Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
