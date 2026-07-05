# Walsh-Hadamard ECG real-data PTB-XL evaluation

This folder contains real-data outputs for the manuscript:

Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates:
Orthogonal Invariance and Exact Time-Domain Score Decomposition

## Dataset

PTB-XL version 1.0.3, low-resolution 100 Hz WFDB records.

## Protocol

- Train reference: PTB-XL folds 1--8, diagnostic superclass exactly NORM.
- Validation threshold: fold 9, NORM.
- Test negatives: fold 10, NORM.
- Test positives: fold 10, at least one non-NORM diagnostic superclass.
- Window length: N=128
- Hop: 64
- Regularization delta: 0.01
- Calibration alpha: 0.05
- Leads: I, II, V2

## Main files

- tables/article_ablation_table_q95.csv
- tables/article_ablation_table_q95.tex
- tables/article_verification_table.csv
- tables/article_verification_table.tex
- figures/fig_realdata_roc_q95.png
- figures/fig_realdata_pr_q95.png
- figures/fig_realdata_ablation_auc.png
- figures/fig_time_sequency_real_ptbxl.png
- figures/fig_contribution_overlay_real_walsh_block.png
- article_methods_realdata.txt
- article_results_realdata.txt
- run_manifest.json

## Interpretation

The full-covariance Walsh-Hadamard model is mathematically equivalent to
the corresponding time-domain full-covariance model under orthonormal
coordinate changes. Therefore, performance differences should not be claimed
between these two full-covariance versions. Walsh-specific inductive bias
enters through diagonal or sequency-block covariance structure.
