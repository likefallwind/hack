# Sporotrichosis SCI Analysis Pipeline

This folder provides a reusable, publication-oriented analysis script for retrospective cutaneous sporotrichosis cohorts.

## What this script does

`sporotrichosis_analysis.py` performs:

1. **Data cleaning with study-aligned variable definitions**
   - Outcome (`diagnostic_type`) mapped to 3 non-ordinal categories:
     - Fixed cutaneous form
     - Lymphocutaneous form
     - Skin disseminated form
   - Excludes cases with unknown/empty onset site.
   - Standardizes season to Spring/Summer/Autumn/Winter.

2. **Multivariable multinomial logistic regression**
   - Uses `statsmodels.MNLogit`.
   - Exports OR, 95% CI, and p-values to CSV.
   - Generates a forest plot.

3. **Machine-learning model comparison**
   - Models: CatBoost (if installed), Random Forest, Logistic Regression.
   - Stratified cross-validation.
   - Metrics: Accuracy, Macro-F1, ROC-AUC (OvR).
   - Exports a comparison table.

4. **SCI-ready visualizations (English)**
   - Ordered stacked bar chart: diagnostic composition by season.
   - Trend line chart: seasonal trend by diagnostic type.
   - Forest plot: OR results from multinomial logistic regression.

## Output formats

Every figure is exported as:

- **PDF** (vector, white background, 600 dpi)
- **PNG** (transparent background, 600 dpi)
- **TIFF** (300 dpi, transparent background)

## Usage

```bash
python analysis/sporotrichosis_analysis.py \
  --input /path/to/your_data.csv \
  --output-dir analysis_outputs \
  --diagnosis-col diagnostic_type \
  --site-col onset_site \
  --season-col season \
  --covariates age sex immunosuppression
```

## Required input columns

- `diagnostic_type` (or `--diagnosis-col`)
- `onset_site` (or `--site-col`)
- `season` (or `--season-col`)
- Optional covariates listed in `--covariates`

## Dependency note

Install analysis dependencies if not already available:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels catboost
```

CatBoost is optional; if unavailable, the script still runs Random Forest and Logistic Regression comparisons.
