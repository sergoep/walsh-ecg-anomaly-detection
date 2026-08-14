# Final 12-lead PTB-XL study

Final frozen reproducibility materials for the manuscript:

**From Orthogonal Invariance to Walsh Spectral Multipliers: Lead-Conditioned Reference Geometry and Exact Attribution for 12-Lead ECG**

Authors: **S. A. Episkoposyan and G. Chaltikyan**

This directory contains the locked confirmatory and comparator results used in the final 12-lead PTB-XL study.

## 1. Purpose

The study investigates a mathematically structured reference-based ECG scoring framework in Walsh-Hadamard coordinates.

The central distinction is between:

1. **full-covariance quadratic scoring**, which is invariant under an orthogonal change of coordinates and therefore is mathematically equivalent in the time and Walsh-Hadamard domains; and

2. **Walsh-structured spectral models**, where diagonal or block/sequence-conditioned covariance assumptions introduce a genuinely basis-dependent inductive bias.

The empirical study is intended to test these constructions on real 12-lead ECG data under a strictly separated patient/fold protocol. It is not presented as a clinical diagnostic validation study.

## 2. Dataset and locked protocol

Dataset: **PTB-XL v1.0.3**

Sampling frequency: **100 Hz**

Window length: **128 samples**

Hop length: **64 samples**

Bandpass preprocessing: **0.5--40 Hz**

Regularization parameter: **delta = 0.01**

Patient/fold separation:

- folds 1--8: NORM reference estimation;
- fold 9: validation/model selection;
- fold 10: locked confirmatory test;
- no patient overlap between training/reference, validation, and test partitions.

The final test fold contains:

- **2198 ECG records**
- **1904 patients**
- **912 NORM records**
- **1286 non-NORM records**

The primary confirmatory task was fixed as:

**NORM vs HYP**

with the locked primary score:

**all12__max__mean**

Secondary confirmatory tasks were:

- NORM vs CD
- NORM vs STTC
- NORM vs MI

The global NORM vs all-non-NORM comparison was treated as exploratory.

## 3. Primary confirmatory result

For the locked primary task **NORM vs HYP**, the full-covariance 12-lead reference model obtained:

- AUROC: **0.841850**
- 95% CI: **0.811266--0.870301**
- AUPRC: **0.667356**
- 95% CI: **0.603465--0.732388**
- records: **1174**
- patients: **1076**
- HYP-positive records: **262**
- NORM-negative records: **912**

Confidence intervals were obtained using **5000 patient-level bootstrap repetitions** with the locked bootstrap seed.

## 4. Secondary confirmatory results

Using the same locked primary model:

| Task | AUROC | 95% CI | AUPRC | 95% CI |
|---|---:|---:|---:|---:|
| NORM vs CD | 0.677824 | 0.643656--0.710727 | 0.607977 | 0.558971--0.657360 |
| NORM vs STTC | 0.675977 | 0.644136--0.707170 | 0.588917 | 0.543279--0.638015 |
| NORM vs MI | 0.639567 | 0.606622--0.672399 | 0.560118 | 0.514762--0.607867 |
| NORM vs all non-NORM | 0.661945 | 0.637165--0.685405 | 0.756532 | 0.729505--0.782730 |

These results should not be interpreted as optimized disease-specific classifiers. They quantify the behavior of a reference-deviation operator under fixed out-of-reference pathology groups.

## 5. Independent comparator study

The frozen comparator package evaluates four scoring constructions on the same final test cohort:

1. `full_covariance_LCRG`
2. `diag_mahalanobis`
3. `pca_reconstruction`
4. `isolation_forest`

For the primary NORM-vs-HYP task, the audited AUROC values are:

| Method | AUROC |
|---|---:|
| Full-covariance LCRG | **0.841850** |
| Diagonal Mahalanobis | **0.860415** |
| PCA reconstruction | **0.678791** |
| Isolation Forest | **0.866488** |

These comparator results are retained without post-hoc modification.

In particular, the full-covariance construction is **not claimed to outperform every generic anomaly-detection baseline**. The role of the full-covariance model is primarily mathematical: it provides the coordinate-invariant reference geometry from which the Walsh-structured constructions and exact attribution identities are derived.

## 6. Mathematical interpretation

Let \(W\) denote the normalized Walsh-Hadamard matrix and let

\[
z = W(x-\mu).
\]

For a full covariance matrix \(\Sigma\), orthogonality of \(W\) gives

\[
(W\Sigma W^\top)^{-1}
=
W\Sigma^{-1}W^\top.
\]

Consequently,

\[
z^\top(W\Sigma W^\top)^{-1}z
=
(x-\mu)^\top\Sigma^{-1}(x-\mu).
\]

Thus the full-covariance quadratic score is exactly invariant under the Walsh-Hadamard coordinate transformation.

This identity is a mathematical constraint on interpretation: any empirical advantage attributed specifically to Walsh coordinates must arise from additional structure imposed in those coordinates, such as diagonal covariance, sequence-block conditioning, spectral multipliers, regularization, truncation, or attribution structure.

The numerical implementation independently verifies Walsh orthogonality to machine precision.

## 7. Exact attribution

For a quadratic Walsh-domain score

\[
Q(z)=z^\top A z,
\]

the contribution associated with coordinate \(k\) is defined by

\[
c_k=z_k(Az)_k,
\]

so that the exact conservation identity

\[
Q(z)=\sum_k c_k
\]

holds.

This is an algebraic decomposition of the score itself. It is not a post-hoc surrogate explanation.

The final study uses this identity as the mathematical basis for Walsh-coordinate attribution and its aggregation across leads and spectral/sequence groups.

## 8. Publication freeze

The archive

`LCRG_PUBLICATION_FREEZE.zip`

contains the frozen outputs used for the final manuscript audit.

The automated publication audit verifies:

- confirmatory record count: **2198**
- baseline record count: **2198**
- presence of the locked primary NORM-vs-HYP result
- locked primary AUROC
- presence of all four comparator methods
- comparator AUROC locks
- paired comparator results
- finite scores for all methods
- consistency of the final test cohort

The final automated audit terminated with:

**FINAL PUBLICATION AUDIT: ALL PASS**

No model retraining or reopening of test fold 10 for model selection is performed by the publication-freeze procedure.

## 9. Reproducibility status

The repository distinguishes three layers:

- mathematical/operator verification;
- real-data PTB-XL experimentation;
- final locked 12-lead publication freeze.

Earlier repository materials are retained for provenance. The contents of this directory constitute the frozen final-study layer.

The PTB-XL data themselves are not redistributed. They must be obtained from the original PTB-XL/PhysioNet source under the corresponding data terms.

## 10. Interpretation limits

The reported results support methodological and mathematical validation of the proposed framework on a large public ECG dataset.

They do **not** establish:

- clinical diagnostic validity;
- prospective clinical performance;
- superiority to modern supervised ECG diagnostic networks;
- disease-specific optimality;
- invariance of Walsh-structured models under arbitrary orthogonal transformations.

The principal contribution is the connection between orthogonal invariance, Walsh-domain structured reference geometry, exact quadratic attribution, and reproducible 12-lead ECG experimentation.

## 11. Archive

Final frozen result archive:

`LCRG_PUBLICATION_FREEZE.zip`

Repository:

`sergoep/walsh-ecg-anomaly-detection`

The permanent Zenodo record and DOI for this final frozen study will be added after the corresponding archive version is deposited.
