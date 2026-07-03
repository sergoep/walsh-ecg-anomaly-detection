# ECG Walsh-Hadamard anomaly scoring

Reference implementation for the manuscript:

**Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition**

Authors: **Sergo A. Episkoposian** and **Georgi Chaltikyan**

Latest archived software DOI: https://doi.org/10.5281/zenodo.18135574

Version used for the SaSiDa manuscript: https://doi.org/10.5281/zenodo.21179510

This repository contains a compact, reproducible implementation of the mathematical pipeline used in the SaSiDa submission version of the article. The default script uses a fast synthetic ECG-like dataset so that the code can be run quickly in Google Colab or locally.

## What this repository verifies

The repository verifies the algebraic core of the paper:

1. **Orthogonal invariance.** The full-covariance Walsh-Hadamard Mahalanobis score equals the corresponding time-domain full-covariance Mahalanobis score up to numerical precision.

2. **Exact detector-relative decomposition.** Symmetric whitening followed by inverse Walsh-Hadamard back-projection satisfies

   ```text
   d_delta^2(c) = ||z||_2^2 = ||r||_2^2.
   ```

3. **Multilead max-fusion.** Lead-wise quadratic scores are fused by a transparent maximum rule, and the threshold is calibrated on held-out healthy validation windows.

## Important scope note

The default synthetic run is a **verification run**, not clinical validation. Its purpose is to exercise the reference-estimation, scoring, invariance, decomposition, plotting, and packaging workflow quickly. Journal-grade ECG performance results must be produced from a patient-wise PTB-XL or external-dataset protocol and archived separately.

The current archived verification output confirms numerical identities at machine precision. The synthetic AUC is not used as evidence of detector superiority.

## Repository structure

```text
.
├── run_demo.py                         # end-to-end fast verification script
├── requirements.txt                    # Python dependencies
├── CITATION.cff                        # citation metadata for GitHub/Zenodo
├── LICENSE                             # MIT license
├── src/ecg_walsh/core.py               # Walsh transform, reference model, scoring, decomposition
├── src/ecg_walsh/synthetic.py          # fast ECG-like synthetic generator
├── src/ecg_walsh/ptbxl_hook.py         # optional PTB-XL WFDB loading hook
├── results/                            # verification outputs from the current run
│   ├── summary_metrics.csv
│   ├── record_level_scores.csv
│   └── figures/
├── paper/                              # manuscript LaTeX source for reference
└── docs/                               # GitHub and Zenodo instructions
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_demo.py
```

The script writes results to `results/`.

## Google Colab

In Colab, upload this repository folder or clone it from GitHub, then run:

```bash
!pip install -r requirements.txt
!python run_demo.py --outdir results
```

## Expected checks from the current verification run

The included current run produced:

```text
orthogonality_error_WWT: 2.674969644124714e-16
orthogonality_error_WTW: 2.674969644124714e-16
max_invariance_error:    6.252776074688882e-13
exact_decomposition_error: 6.394884621840902e-14
```

These values confirm the mathematical identities of the manuscript up to numerical precision.

## Outputs

Running `python run_demo.py` creates:

- `results/summary_metrics.csv`
- `results/record_level_scores.csv`
- `results/run_config.json`
- `results/summary_report.txt`
- `results/figures/fig_contribution_profile.png`
- `results/figures/fig_score_trace.png`
- `results/figures/fig_time_sequency.png`
- `results/figures/fig_roc.png`

## Optional PTB-XL hook

The file `src/ecg_walsh/ptbxl_hook.py` contains a minimal WFDB example for reading one PTB-XL record. Full PTB-XL benchmarking should be implemented with:

- patient-wise train/validation/test splits;
- NORM-only healthy reference estimation;
- non-NORM out-of-reference evaluation;
- fixed preprocessing, window length, overlap, and lead handling;
- patient-level clustered uncertainty quantification;
- archived output tables corresponding exactly to the manuscript.

## Zenodo

A Zenodo record already exists for this project.

Latest all-versions DOI:

https://doi.org/10.5281/zenodo.18135574

Version 3 DOI, corresponding to the current SaSiDa manuscript package:

https://doi.org/10.5281/zenodo.21179510

For updates, do **not** create a separate new Zenodo project. Use the existing Zenodo record and create a **New version** if you need to archive an updated GitHub package. The manuscript should cite the archived version that exactly matches the reported code and generated results.

## Citation

Please cite the archived Zenodo record and the associated manuscript:

```bibtex
@software{episkoposian_chaltikyan_2026_ecg_walsh,
  author  = {Episkoposian, Sergo A. and Chaltikyan, Georgi},
  title   = {Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition},
  year    = {2026},
  publisher = {Zenodo},
  doi     = {10.5281/zenodo.18135575},
  url     = {https://doi.org/10.5281/zenodo.18135575}
}
```

## License

MIT License.
