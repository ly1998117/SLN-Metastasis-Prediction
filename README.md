# MRI-based Intratumoral Heterogeneity and Transcriptome Profiling for Sentinel Lymph Node Metastasis Prediction in cN0 Breast Cancer

This project investigates the integration of MRI-derived intratumoral heterogeneity (ITH) features and transcriptomic profiling to predict sentinel lymph node (SLN) metastasis in clinically node-negative (cN0) breast cancer. It leverages radiomics pipelines, subregion-based ITH analysis, and machine learning models including logistic regression and SHAP-based explainability.

## Highlights

- Subregion-based **Intratumoral Heterogeneity (ITH)** feature extraction.
- Multimodal integration: radiomics + clinical + RNA-seq data.
- Robust feature selection: `ttest`, `Spearman`, `MRMR`, `LASSO`.
- Multiple predictor variants: clinical-only, radiomics-only, combined models.
- SHAP explainability and statistical evaluation (DeLong test, ROC, calibration).
- Internal-external validation framework with reproducibility ensured.

## Project Structure

```plaintext
ith_assessment.py        # Main module for ITH analysis
radiomics_analysis/      # Feature selection, model training, visualization
dataset/
  └── <CROPPed Data>     # Expected path for MRI datasets, annotations, and CSVs
  └── subregion.py       # Subregion-based processing (ITH Features)
```

## Setup
•	Python 3.10+
•	Dependencies:
```python
pip install -r requirements.txt
```

## Data Organization
```plaintext
SLN_internal/
    cropped/
        CSV/
            data.csv
            SVmask.csv
            train_index.csv
            test_index.csv
SLN_external/
    cropped/
        CSV/
            ...
```

## Quick Start
```python
python ith_assessment.py
```