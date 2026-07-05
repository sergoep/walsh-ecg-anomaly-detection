Walsh-Hadamard ECG anomaly scoring
Reproducibility package for the manuscript:
Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition
This repository contains two reproducibility layers.
1. Fast synthetic verification layer
The synthetic layer verifies the algebraic and numerical identities used by the paper: Walsh-Hadamard orthogonality, equality between time-domain and Walsh full-covariance scores, exact time-domain score decomposition, and figure-generation workflow.
2. Real-data PTB-XL pilot layer
The real-data pilot evaluates the same operator family on PTB-XL v1.0.3 low-resolution 100 Hz WFDB records under a patient/fold-wise protocol:
folds 1--8: NORM reference estimation;
fold 9: NORM threshold calibration;
fold 10: NORM and non-NORM testing;
selected leads: I, II, V2;
window length: N = 128;
hop: 64;
regularization: delta = 0.01;
threshold calibration: alpha = 0.05.
The PTB-XL pilot is reported as methodological real-data verification and stress testing of the Walsh-Hadamard quadratic reference operator. It is not claimed as clinical diagnostic validation.
Main folders
```text
paper/
  main_signal_processing_realdata.tex
  main_signal_processing_realdata.pdf

scripts/
  ptbxl_realdata_walsh_evaluation.py

notebooks/
  ptbxl_realdata_walsh_evaluation_COLAB.ipynb

results/realdata_ptbxl/
  tables/
  figures/
  article_methods_realdata.txt
  article_results_realdata.txt
  run_manifest.json
  README_realdata_outputs.md

docs/
  Cover_letter_Signal_Processing.txt
  Highlights.txt
  Author_Declaration.txt
  Graphical_Abstract.png
  Graphical_Abstract.pdf
```
Run in Colab
Open:
```text
notebooks/ptbxl_realdata_walsh_evaluation_COLAB.ipynb
```
Run all cells. The notebook writes:
```text
walsh_ecg_realdata_results/
walsh_ecg_realdata_results.zip
```
Run locally
```bash
pip install -r requirements.txt
python scripts/ptbxl_realdata_walsh_evaluation.py
```
Key result files
```text
results/realdata_ptbxl/tables/article_ablation_table_q95.csv
results/realdata_ptbxl/tables/article_verification_table.csv
results/realdata_ptbxl/figures/fig_realdata_roc_q95.png
results/realdata_ptbxl/figures/fig_realdata_pr_q95.png
results/realdata_ptbxl/figures/fig_realdata_ablation_auc.png
```
DOI information
Latest all-versions Zenodo DOI:
```text
https://doi.org/10.5281/zenodo.18135574
```
Previous Version 3 DOI:
```text
https://doi.org/10.5281/zenodo.21179510
```
Current PTB-XL real-data version DOI:
```text
TO_BE_REPLACED_AFTER_ZENODO_NEW_VERSION
```
After creating a new Zenodo version, replace the placeholder above and update `CITATION.cff`.
Interpretation warning
The full-covariance Walsh-Hadamard model is mathematically equivalent to the corresponding time-domain full-covariance model under orthonormal coordinate changes. Therefore, performance differences should not be claimed between these two full-covariance versions. Walsh-specific inductive bias enters through diagonal or sequency-block covariance structure.
