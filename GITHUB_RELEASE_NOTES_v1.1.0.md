# v1.1.0 – PTB-XL real-data pilot and Signal Processing package

This release extends the previous synthetic verification package by adding a real-data PTB-XL pilot evaluation under a patient/fold-wise protocol.

## Added

- Real-data PTB-XL evaluation script.
- Colab notebook for reproducing the PTB-XL pilot.
- Patient/fold-wise protocol:
  - folds 1--8: NORM reference estimation;
  - fold 9: NORM threshold calibration;
  - fold 10: NORM and non-NORM testing.
- Record-level ablation metrics.
- Operator-identity verification on real PTB-XL windows.
- ROC, PR, ablation, score distribution, and contribution-overlay figures.
- Signal Processing submission manuscript package.

## Important interpretation

The PTB-XL pilot is methodological real-data verification and stress testing. It is not clinical diagnostic validation. The full-covariance Walsh-Hadamard score is equal to the time-domain full-covariance score up to floating-point error, as predicted by the orthogonal-invariance theorem.

## DOI

After publishing the Zenodo new version, replace this placeholder:

```text
TO_BE_REPLACED_AFTER_ZENODO_NEW_VERSION
```
