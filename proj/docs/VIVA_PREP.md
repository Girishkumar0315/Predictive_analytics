# Viva preparation — likely questions with model answers
(Numbers quoted are from the executed notebook; re-check them against `reports/results/results.json` before your viva.)

## A. Project and data
**1. Why did you choose this dataset?** It is real, large (65,437 rows), current (AI tools in software work), has two natural targets (a continuous salary and a binary AI-adoption flag) plus ordinal attitude questions, includes India, and has an official source with an open licence. I profiled four other candidates; they were too small or had no meaningful target.

**2. Why is the data not from Kaggle / data.gov.in?** Those sites were not reachable from the environment where the project was built. I used the TidyTuesday GitHub copy of the official Stack Overflow release and documented commit hash and checksums. (Be ready to say you can re-download from survey.stackoverflow.co.)

**3. What are the main limitations of the data?** Self-selected respondents, self-reported nominal USD pay, 64 % missing salary, conditional AI questions, single-response extract only, one survey year.

**4. What is "structural missingness"?** Missing because of survey routing, not randomly: the AI follow-ups were only shown to AI users/planners. So the missingness itself carries information and must not be imputed or used as a predictor of the screening question.

## B. Cleaning and EDA
**5. How did you treat outliers and why that way?** Absolute bounds (USD 1k-1M) plus country-specific 3×IQR fences on log-salary. A global fence would delete low-income countries' salaries and bias the study. A sensitivity check (HistGradientBoosting re-fitted under three policies) showed absolute RMSE depends on the policy (0.58-0.67; stricter = lower), so error levels are conditional on my choice; I did not repeat the full model comparison under each policy.

**6. Why log-transform salary?** Raw skewness ≈ 53; log makes the target far less skewed, turns multiplicative differences into additive ones, and makes errors relative (percent-like).

**7. Why Spearman / Kruskal–Wallis instead of Pearson / ANOVA?** Log-salary is heavy-tailed and variances differ across countries (Levene p < 0.001). Rank-based tests do not assume normality. For OLS I used HC3 robust standard errors.

**8. What is Cramér's V?** An effect size for association between two categorical variables, V = √(χ²/n / min(k−1, r−1)); 0 = independent, 1 = perfect. I used the bias-corrected version.

**9. Statistical vs practical significance?** With n ≈ 30,000, even ρ = −0.05 is significant. For attitudes versus experience, ε² and ρ show the effect is negligible, so I reported effect sizes, not just p-values.

## C. Key findings (O1–O5)
**10. How much does country explain?** About 51 % of variance in log-pay (R² of a country-dummy regression); η² ≈ 0.49 among the 28 large countries.

**11. India vs USA?** India's median USD ≈ 17,900 is ≈ 0.12× the USA median (CI ≈ 0.11–0.13); India ranks last of 28 large markets. Nominal, not PPP-adjusted.

**12. How did you estimate the experience premium?** Quadratic log-linear OLS with country fixed effects and HC3 errors; marginal return = 100(exp(β₁+2β₂x)−1) % per year with delta-method CI. About 9 % per year at entry, ≈ 1 % at 20 years, turning point ≈ 24 years. The downward bend is a quadratic artefact; LOWESS shows a plateau.

**13. Why is India's early-career slope steeper?** Low base pay grows faster in percentage terms; interaction test p < 0.001. It does not imply catching up in dollars.

**14. Is the remote-work premium causal?** No. After adjusting for country, role, experience, size and education remote workers earn ≈ 29 % more, but employer type, seniority and foreign employers may explain it.

**15. What drives AI adoption?** Age, experience, role and country (V ≈ 0.1-0.2); not Stack Overflow visit frequency. Overall 61.8 % use AI; India 67.9 % vs 61.4 % elsewhere.

## D. Machine learning (O6–O7)
**16. What is data leakage and where did you prevent it?** Using information at training time that would not be available at prediction time or that reveals the target. I excluded `comp_total` (target twin), `currency`, and `ai_sent/ai_acc/ai_complex/ai_threat` (asked only to AI users). Demonstration: AUC 0.67 honestly vs 1.00 with the leaky columns. Also, all imputers/scalers/encoders are fitted inside CV pipelines.

**17. What is target encoding and why cross-fitted?** Replaces a category (e.g. country) by a smoothed mean of the target. If computed on the same rows it leaks the target; scikit-learn's `TargetEncoder` uses internal cross-fitting so each row is encoded with statistics from other folds.

**18. Why these models?** Baselines (global median, country median) give a floor; Ridge is a regularised linear model; Random Forest averages bootstrapped trees (variance reduction); HistGradientBoosting and LightGBM add trees sequentially to fit residuals (bias reduction) and capture interactions such as country × experience.

**19. Explain gradient boosting.** F_m(x) = F_{m−1}(x) + η·h_m(x), where each small tree h_m is fitted to the current residuals (negative gradient of squared error). Learning rate η and tree size control over-fitting.

**20. How did you avoid overfitting the hold-out set?** Tuning used only training data (3-fold randomized search); comparison used 5-fold CV on training data; the test set was evaluated once.

**21. Which metrics and why?** Regression: RMSE/MAE/R² on log scale, MAE and MdAPE in USD for interpretability. Classification: ROC-AUC (threshold-free), PR-AUC, balanced accuracy, F1, Brier (calibration).

**22. Your RMSE-reduction target was 40 % but you got 39.7 %. What do you say?** I fixed the criterion before modelling and report that it was narrowly missed rather than changing it. The more informative comparison is against the country-median baseline (≈ 15 % lower RMSE). The data's ceiling is R² ≈ 0.6.

**23. Why is F1 misleading in O7?** 61.8 % are users, so "predict yes for everyone" already has F1 = 0.764. I report ROC-AUC (0.669), calibration and balanced accuracy with a threshold chosen on out-of-fold training predictions.

**24. Are the four learners really different?** No: paired bootstrap and CV standard deviations show differences are tiny (AUC differences < 0.01); simpler models are almost as good, so the choice is marginal.

**25. What does the learning curve say?** Validation RMSE is still falling slowly and there is a train/validation gap, so more data (or better features) helps more than more complexity.

## E. Explainability (O8)
**26. Permutation importance vs SHAP?** Permutation importance measures the loss in performance when a feature is shuffled (global, model-agnostic, can be misleading with correlated features). SHAP assigns each feature a Shapley-value contribution per prediction (local, additive, consistent) and aggregates to global importance.

**27. Where does the salary model fail?** Highly dispersed markets (Ukraine, Brazil, India: MdAPE ≈ 37-55 % vs ≈ 20-25 % for Germany/USA) and respondents with unusually low reported pay.

## F. Wrap-up
**28. What would you do with more time?** Multi-year surveys with temporal validation, PPP adjustment, multi-select tech-stack features, conformal prediction intervals, multilevel models for country, causal methods for remote work.

**29. What are you least sure about?** Whether self-selected Stack Overflow respondents represent Indian developers in general, and whether pay is comparable across currencies and reporting practices.

**30. Can you explain the whole pipeline in 60 seconds?** Real survey → decode and audit quality → build documented cohorts → EDA and test-selection → eight pre-defined objectives (3 on pay, 2 on AI attitudes, 2 prediction tasks, 1 explainability) → leakage-safe pipelines, tuning on train, CV comparison, one-shot test → SHAP/permutation and error audit → honest limits.
