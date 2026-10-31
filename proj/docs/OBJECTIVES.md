# Research objectives and objective-to-analysis mapping

| # | Research question | Relevance | Variables & methods | Visualisations | Success measure | Where answered |
|---|---|---|---|---|---|---|
| O1 | How much of pay variation does country explain; where is India? | Salary benchmarking, offshoring, career choice | country, USD pay; Kruskal-Wallis, eta², Mann-Whitney+Holm, Cliff's delta, bootstrap CI | Box plot by country; median + CI bars | eta²/R² of country; India:USA median ratio with CI | Report §4.1, notebook Part D-O1 |
| O2 | How strongly and how long does experience raise pay? | Career planning | years_code_pro, log pay; Spearman, quadratic OLS (HC3), delta-method CI | Hexbin+LOWESS; band medians; marginal-return curves | rho with CI; % per year at 0/5/10/20 yrs; India-vs-USA slope test | §4.2, Part D-O2 |
| O3 | Net effects of role, work mode, company size, education | Fair-pay and hiring decisions | OLS with controls, nested delta R², partial eta² | Forest plots; variance decomposition | Adjusted % differences with CI; delta R² per block | §4.3, Part D-O3 |
| O4 | Who uses AI tools? | Tool strategy, training | chi², Cramer's V, Wilson CI, z-test | Stacked bars; dot plots; heat map | Adoption % with CI; V ranking; India gap | §4.4, Part D-O4 |
| O5 | Sentiment, trust, job-threat by career stage | Change management | Spearman, Kruskal, chi², Mann-Whitney | Distributions; stacked bars; correlation heat map | rho (experience-trust); threat gaps | §4.5, Part D-O5 |
| O6 | Predict annual pay (>= 4 models) | Benchmarking tool | Baselines, Ridge, RF, HistGB, LightGBM; CV; tuning | Model comparison; diagnostics; learning curve | RMSE reduction >= 40 %; beat country baseline | §4.6, Part E-O6 |
| O7 | Predict AI adoption (>= 4 models) | Targeting AI enablement | LogReg, RF, HistGB, LightGBM; CV; thresholds | ROC/PR; confusion; calibration | ROC-AUC >= 0.65; F1 above baseline | §4.7, Part E-O7 |
| O8 | Explain models; audit errors | Trust, fairness | Permutation importance, SHAP, PDP, segment errors | Importance bars; SHAP; PDP; error by segment | Ranked drivers; error gaps incl. India | §4.8, Part F-O8 |
