# SalaryScope AI: Developer Compensation & AI Adoption Analytics
### Predictive Analysis Platform — Stack Overflow Annual Developer Survey 2024

A complete, reproducible predictive analytics platform: data-quality audit → EDA → statistical analysis of 6 primary objectives →
machine-learning models for **salary forecasting (regression)** and **AI-tool adoption (classification)** → explainability and error audit.

## Dataset
Stack Overflow Annual Developer Survey 2024 (65,437 respondents), TidyTuesday 2024-week-36 extract.
Provenance, licence, checksums and an honest note on how the file was obtained: `data/DATASET_DOCUMENTATION.md`.

## Project structure
```
.
├── README.md  requirements.txt  build_notebook.py
├── data/raw/            original CSVs (checksums in DATASET_DOCUMENTATION.md)
├── data/processed/      cleaned cohorts written by the notebook
├── notebooks/CA2_Predictive_Analysis.ipynb   executed notebook (Parts A-G)
├── nb_src/              cell source files the notebook is assembled from
├── src/                 reusable package: config, data_prep, features, modeling, stats_utils, plotting, predict
├── reports/figures/     36 figures, 300 dpi PNG
├── reports/results/     results.json (all numbers) + tables/*.csv
├── models/              best_salary_model.joblib, best_ai_adoption_model.joblib (+ all tuned models)
├── report/              Predictive_Analysis_CA2_Report.docx (LPU format)
├── report_build/        scripts that generate the DOCX from results.json
└── docs/                SCREENSHOT_CHECKLIST.md, VIVA_PREP.md
```

## Set-up
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace notebooks/CA2_Predictive_Analysis.ipynb   # ~20-30 min, 1 CPU
# or interactively: jupyter lab notebooks/CA2_Predictive_Analysis.ipynb
```
Everything is seeded (`RANDOM_STATE = 42`) so results reproduce exactly on the same library versions.

## Use the trained models
```python
from src.predict import predict_salary, predict_ai_adoption
predict_salary({"country": "India", "years_code": 9, "years_code_pro": 5, "dev_type": "Full-stack dev", "age": "25-34"})
predict_ai_adoption({"country": "India", "years_code": 9, "years_code_pro": 5, "age": "25-34"})
```

## Methodology in one paragraph
Cohorts are built with documented rules (professional developers with USD salary; country-aware outlier fences; impossible
experience values blanked). Leakage is prevented by excluding `comp_total`, `currency` and the conditionally-asked AI
follow-up questions, and by fitting all imputers/encoders inside CV pipelines. Models are tuned on the training split only
(RandomizedSearchCV), compared by 5-fold CV and evaluated once on a 20 % hold-out.

## Results
Computed by the executed notebook (full numbers: `reports/results/results.json`; narrative: report Chapter 4).

| Objective | Headline result |
|---|---|
| O1 | Country alone explains R² = 0.51 of log-pay variance; India's median USD 17,945 = 0.12x the USA median, lowest of 28 large markets |
| O2 | Spearman rho = 0.48; pay gain 9.2 %/yr at entry, 5.3 %/yr at 10 yrs, plateau after about 24 yrs; steeper early slope in India |
| O3 | Role, company size, work mode and education add only about 3 points of R² beyond country + experience (final R² = 0.574); remote work +29 % adjusted |
| O4 | 61.8 % use AI tools; adoption falls with age/experience; India 67.9 % vs 61.4 % elsewhere |
| O5 | 74 % favourable, only 43 % trust accuracy, 12 % see AI as job threat; career-stage effects negligible |
| O6 | LightGBM: test R² = 0.624, MAE USD 28,585, MdAPE 27.4 %; RMSE cut 39.7 % vs global baseline (**40 % target narrowly missed**), 15.3 % vs country-median baseline |
| O7 | LightGBM: ROC-AUC = 0.669 (criterion >= 0.65 met); F1 at 0.5 is below the trivial all-yes baseline (criterion not met at 0.5); leaky variables would inflate AUC to 1.000 |
| O8 | Country dominates salary importance; total years coded, role and country lead for adoption; salary error highest for Ukraine, Brazil, India |

## Uploading to GitHub (you must do this with your own account)
```bash
git init
git add .
git commit -m "Predictive Analysis CA2: Stack Overflow 2024 developer survey"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git   # create the empty repo on github.com first
git push -u origin main
```
Note: `models/all_*.joblib` (all tuned models, 9-24 MB) are re-created when you run the notebook; they are not included in the delivered zip to keep it small. Only the two `best_*` models are shipped.
Add a licence note: the data are ODbL (Stack Overflow); code is yours to license.

## Honest limitations
Self-selected survey; nominal USD pay; observational (no causal claims); salary R² about 0.6; adoption AUC about 0.67.
