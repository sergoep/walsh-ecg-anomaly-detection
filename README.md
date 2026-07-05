# Walsh-Hadamard ECG anomaly scoring

Reproducibility package for the manuscript:

**Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition**

This repository contains the public reproducibility materials for the Walsh-Hadamard ECG anomaly-scoring study. The manuscript itself is not included in this public repository or release because it is under journal submission.

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
```

## Run in Colab

Open:

```text
notebooks/ptbxl_realdata_walsh_evaluation_COLAB.ipynb
```

Run all cells. The notebook writes:

```text
walsh_ecg_realdata_results/
walsh_ecg_realdata_results.zip
```

The Colab workflow uses only the notebook, scripts, and requirements file. The manuscript files are not required for Colab execution.

## Run locally

```bash
pip install -r requirements.txt
python scripts/ptbxl_realdata_walsh_evaluation.py
```

## Key result files

```text
results/realdata_ptbxl/tables/article_ablation_table_q95.csv
results/realdata_ptbxl/tables/article_verification_table.csv
results/realdata_ptbxl/figures/fig_realdata_roc_q95.png
results/realdata_ptbxl/figures/fig_realdata_pr_q95.png
results/realdata_ptbxl/figures/fig_realdata_ablation_auc.png
results/realdata_ptbxl/run_manifest.json
```

## Real-data PTB-XL pilot summary

The real-data PTB-XL pilot uses a NORM reference class and a non-NORM out-of-reference test class. The pilot is intended to verify the behavior of the quadratic reference operator on real ECG waveforms and to stress test the operator family under patient/fold-wise separation.

The pilot should be interpreted as follows:

- it verifies that the implementation works on real PTB-XL WFDB records;
- it verifies the expected equality between time-domain and Walsh full-covariance scores up to floating-point error;
- it verifies the exact score-decomposition identities on real ECG windows;
- it compares full-covariance, diagonal, sequency-block, and generic anomaly-detection baselines;
- it does not claim clinical diagnostic performance.

## DOI and archive information

Latest all-versions Zenodo DOI:

```text
https://doi.org/10.5281/zenodo.18135574
```

Previous Version 3 DOI:

```text
https://doi.org/10.5281/zenodo.21179510
```

Current PTB-XL real-data reproducibility version:

```text
https://zenodo.org/records/21209846
```

Current PTB-XL real-data version DOI:

```text
https://doi.org/10.5281/zenodo.21209846
```

GitHub release for the current reproducibility version:

```text
https://github.com/sergoep/walsh-ecg-anomaly-detection/releases/tag/v1.1.0-ptbxl-realdata
```

## Citation

If you use this repository, please cite the current Zenodo version:

```text
Episkoposian, S. A., & Chaltikyan, G. (2026).
Walsh-Hadamard ECG anomaly scoring: reproducibility package with PTB-XL real-data pilot.
Zenodo. https://doi.org/10.5281/zenodo.21209846
```

For the previous synthetic verification package, cite:

```text
Episkoposian, S. A., & Chaltikyan, G. (2026).
Walsh-domain reference-based ECG anomaly detection: reproducible code and verification package.
Zenodo. https://doi.org/10.5281/zenodo.21179510
```

## Interpretation warning

The full-covariance Walsh-Hadamard model is mathematically equivalent to the corresponding time-domain full-covariance model under orthonormal coordinate changes. Therefore, performance differences should not be claimed between these two full-covariance versions.

Walsh-specific inductive bias enters through diagonal covariance or sequency-block covariance structure.

The PTB-XL pilot is a real-data methodological stress test and reproducibility layer. It is not a clinical diagnostic validation study.

## License

This repository is released under the MIT License.

## Repository

```text
https://github.com/sergoep/walsh-ecg-anomaly-detection
```
