# Walsh-Hadamard ECG anomaly scoring

Reproducibility package for the manuscript:

**Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition**

This repository contains the public reproducibility materials for the Walsh-Hadamard ECG anomaly-scoring study. The manuscript itself is not included in this public repository/release because it is under journal submission.

This repository contains two reproducibility layers.

## 1. Fast synthetic verification layer

The synthetic layer verifies the algebraic and numerical identities used by the paper:

- Walsh-Hadamard orthogonality;
- equality between time-domain and Walsh full-covariance scores;
- exact time-domain score decomposition;
- multilead fusion workflow;
- figure-generation workflow.

This layer is intended for fast numerical verification of the operator identities. It is not intended as clinical diagnostic validation.

## 2. Real-data PTB-XL pilot layer

The real-data pilot evaluates the same operator family on PTB-XL v1.0.3 low-resolution 100 Hz WFDB records under a patient/fold-wise protocol:

- folds 1--8: NORM reference estimation;
- fold 9: NORM threshold calibration;
- fold 10: NORM and non-NORM testing;
- selected leads: I, II, V2;
- window length: N = 128;
- hop: 64;
- regularization: delta = 0.01;
- threshold calibration: alpha = 0.05.

The PTB-XL pilot is reported as methodological real-data verification and stress testing of the Walsh-Hadamard quadratic reference operator. It is not claimed as clinical diagnostic validation.

## Main folders

```text
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
  DOI_and_reference_check.txt
  README_Submission.txt

zenodo/
  ZENODO_NEW_VERSION_METADATA.md
