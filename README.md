# Walsh-domain ECG Anomaly Detection (Interpretable Reference Framework)

This repository provides a fully reproducible, transparent Python implementation of an interpretable ECG anomaly detection framework based on a Walsh–Hadamard (sequency-domain) representation and a statistical healthy-reference template (mean/covariance with shrinkage).

The code reproduces all figures used in the accompanying manuscript using a self-contained synthetic ECG generator (methodological demonstration). No clinical claims are made.

## Repository structure

- `src/reproduce_figures.py` — main script (reproduces all figures deterministically)
- `notebooks/walsh_ecg_demo.ipynb` — Colab / interactive demo notebook
- `figures/` — output directory for generated figures (tracked via `.gitkeep`)
- `requirements.txt` — minimal Python dependencies

## System requirements

- Python >= 3.9 (recommended: 3.10+)
- OS: Windows / Linux / macOS
- Packages: NumPy, Matplotlib (see `requirements.txt`)

## Quick start (local)

1. Clone the repository:
   ```bash
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git
   cd <REPO_NAME>
