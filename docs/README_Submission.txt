Signal Processing / Elsevier submission package

Manuscript title:
Interpretable ECG Anomaly Scoring in Walsh-Hadamard Coordinates: Orthogonal Invariance and Exact Time-Domain Score Decomposition

Target journal:
Signal Processing (Elsevier), ISSN 0165-1684

Package contents:
1. main_signal_processing.tex
   Final LaTeX source in Elsevier elsarticle format.

2. main_signal_processing.pdf
   Locally compiled verification PDF. Elsevier still requires editable source files at submission.

3. figures/fig1_time_sequency.png
4. figures/fig2_score_trace.png
5. figures/fig3_contribution_overlay.png
   Figure files used by the manuscript.

6. Highlights.txt
   Required Elsevier highlights file. Contains 5 highlights, each within the 85-character limit.

7. Cover_letter_Signal_Processing.txt
   Cover letter focused on Signal Processing scope: statistical signal processing, detection/estimation, transform-domain scoring, structured covariance modeling, and biomedical signal processing.

8. Author_Declaration.txt
   Declarations for funding, competing interests, ethics, consent, author contributions, and AI-assisted language/editorial use.

9. Graphical_Abstract.png
10. Graphical_Abstract.pdf
   Separate graphical abstract files. Use the PDF version if the system accepts it; otherwise use the PNG.

Important consistency notes:
- Zenodo latest DOI: https://doi.org/10.5281/zenodo.18135574
- Manuscript-specific Zenodo Version 3 DOI: https://doi.org/10.5281/zenodo.21179510
- GitHub repository: https://github.com/sergoep/walsh-ecg-anomaly-detection
- The repository/software record title does not need to match the manuscript title word-for-word, provided that the release and README identify the manuscript version.

Final technical checks performed:
- LaTeX compiles successfully with pdflatex.
- The abstract is below the 250-word limit stated in the Signal Processing author guide.
- The keywords list contains 6 keywords, within the journal's 1-7 keyword range.
- Highlights are supplied as a separate editable text file.
- A graphical abstract is supplied as a separate file.
- All internal LaTeX cross-references are resolved.
- All bibliography entries are cited in the manuscript.
- All cited keys are present in the bibliography.
- DOI formatting was checked for all references where DOI information is available.

Submission recommendation:
Upload main_signal_processing.tex, all figure files, Highlights.txt, Graphical_Abstract.pdf or Graphical_Abstract.png, Cover_letter_Signal_Processing.txt, and Author_Declaration.txt. Upload the PDF only if the Elsevier system requests a reviewer PDF or compiled proof.


REAL-DATA PTB-XL ADDITION
-------------------------
This revised package includes a real-data PTB-XL pilot in addition to the fast synthetic verification run. The main manuscript file is:
  main_signal_processing_realdata.tex

The compiled check PDF is:
  main_signal_processing_realdata.pdf

Real-data outputs are included under:
  tables/
  figures/
  article_methods_realdata.txt
  article_results_realdata.txt
  run_manifest.json

The real-data pilot verifies the algebraic identities on public ECG windows. The observed AUC values are reported transparently as a methodological stress test and should not be described as clinical diagnostic validation.
