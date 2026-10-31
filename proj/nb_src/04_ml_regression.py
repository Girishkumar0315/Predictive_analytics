# %% [markdown]
# # Part E — Machine learning
# ## O6 — Predicting annual compensation (regression)
# **Prediction task.** Given a professional developer's profile (country, role, experience, company size, education,
# work mode, Stack Overflow engagement, AI use) predict annual compensation in USD.
# **Target.** $y=\ln(\text{USD})$ — the log transform removes the extreme skew; $\exp(\hat y)$ estimates the *conditional median* pay.
#
# **Metrics.** With residuals $e_i=y_i-\hat y_i$:
# $$\text{RMSE}=\sqrt{\tfrac1n\sum e_i^2},\quad \text{MAE}=\tfrac1n\sum|e_i|,\quad R^2=1-\frac{\sum e_i^2}{\sum (y_i-\bar y)^2},\quad
# \text{MdAPE}=\operatorname{median}\Big|\tfrac{\hat{\text{USD}}_i-\text{USD}_i}{\text{USD}_i}\Big|$$
# **Leakage prevention.** (i) `comp_total` (the local-currency twin of the target) and `currency` are excluded;
# (ii) imputers, scalers, one-hot vocabularies and **target encoders are fitted inside a `Pipeline`**, i.e. on the training
# part of each CV fold only; (iii) the 20 % hold-out test set is used **once**, after all tuning.
# **Success criterion (set in advance):** the best model must cut the baseline RMSE(log) by **≥ 40 %** and beat the
# domain-aware *country-median* baseline.

# %%
lk = pd.DataFrame([(k, v) for k, v in LEAKAGE_REGISTER.items()], columns=["column", "reason excluded / restricted"])
save_table(lk, "tbl_leakage_register", "Columns excluded from predictors and the reason (leakage register)", "O6")
lk

# %%
X_all = model_frame(cohort, SALARY_SPEC); y_all = cohort.log_comp
strat = cohort.country.where(cohort.country.isin(cohort.country.value_counts().head(8).index), "Other")
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=strat)
print("train", X_tr.shape, "test", X_te.shape, "| predictors:", X_all.shape[1])
R["O6_setup"] = dict(n_train=len(X_tr), n_test=len(X_te), n_features_raw=int(X_all.shape[1]), n_cv_folds=CV_FOLDS)

# %%
t0 = time.time()
zoo_r = regression_zoo(SALARY_SPEC)
fit_r, cmp_r, bp_r, folds_r = tune_and_compare(zoo_r, X_tr, y_tr, "regression", n_iter=8, tune_cv=3, eval_cv=CV_FOLDS)
print(f"total {time.time()-t0:.0f}s")
cmp_r

# %%
# Hold-out evaluation (test set touched ONCE) --------------------------------------------------
test_rows = []; preds_te = {}
for name, m in fit_r.items():
    p_ = m.predict(X_te); preds_te[name] = p_
    test_rows.append({"model": name, **reg_metrics(y_te, p_)})
test_r = pd.DataFrame(test_rows)
comp_r = cmp_r.merge(test_r, on="model").rename(columns={"R2": "test_R2", "RMSE_log": "test_RMSE_log", "MAE_log": "test_MAE_log", "MAE_USD": "test_MAE_USD", "MdAPE_pct": "test_MdAPE_pct", "Within25_pct": "test_Within25_pct"})
comp_r["tuned_params"] = comp_r.model.map(lambda k: json.dumps(bp_r[k]) if bp_r[k] else "-")
save_table(comp_r.round(4), "tbl_o6_model_comparison", "Regression models: 5-fold CV (train) and hold-out test metrics", "O6")
best_name = comp_r.sort_values("cv_rmse_mean").iloc[0].model
second_name = comp_r[~comp_r.model.str.startswith("Baseline")].sort_values("cv_rmse_mean").iloc[1].model
base_rmse = comp_r.set_index("model").loc["Baseline: global median", "test_RMSE_log"]; best_rmse = comp_r.set_index("model").loc[best_name, "test_RMSE_log"]
cm_rmse = comp_r.set_index("model").loc["Baseline: country median", "test_RMSE_log"]
print("BEST (lowest CV RMSE):", best_name, "| runner-up:", second_name)
print(f"RMSE reduction vs global-median baseline: {100*(1-best_rmse/base_rmse):.1f}%  (target >= 40%) | vs country-median: {100*(1-best_rmse/cm_rmse):.1f}%")
comp_r[["model", "cv_rmse_mean", "cv_rmse_std", "cv_r2_mean", "test_R2", "test_RMSE_log", "test_MAE_USD", "test_MdAPE_pct", "test_Within25_pct"]].round(3)

# %%
# Uncertainty: bootstrap CI of test metrics for the best model + paired bootstrap vs runner-up --------------
rng = np.random.default_rng(RANDOM_STATE); yt = y_te.values
e_b = yt - preds_te[best_name]; e_s = yt - preds_te[second_name]
idx = rng.integers(0, len(yt), (2000, len(yt)))
rmse_bs = np.sqrt((e_b[idx] ** 2).mean(1)); r2_bs = 1 - (e_b[idx] ** 2).sum(1) / ((yt[idx] - yt[idx].mean(1, keepdims=True)) ** 2).sum(1)
diff_bs = (e_s[idx] ** 2).mean(1) - (e_b[idx] ** 2).mean(1)             # >0  => best model has lower MSE
R["O6"] = dict(best=best_name, runner_up=second_name, comparison=comp_r.round(4).to_dict("records"), best_params=bp_r,
               rmse_reduction_vs_global_pct=float(100 * (1 - best_rmse / base_rmse)), rmse_reduction_vs_country_pct=float(100 * (1 - best_rmse / cm_rmse)),
               test_rmse_ci=[float(np.percentile(rmse_bs, 2.5)), float(np.percentile(rmse_bs, 97.5))], test_r2_ci=[float(np.percentile(r2_bs, 2.5)), float(np.percentile(r2_bs, 97.5))],
               paired_mse_diff_ci=[float(np.percentile(diff_bs, 2.5)), float(np.percentile(diff_bs, 97.5))], paired_diff_prob_best_better=float((diff_bs > 0).mean()),
               fold_scores=folds_r, success_rmse_cut_40=bool(100 * (1 - best_rmse / base_rmse) >= 40), success_beats_country_median=bool(best_rmse < cm_rmse))
print({k: v for k, v in R["O6"].items() if k not in ("comparison", "best_params", "fold_scores")})

# %%
# Model comparison figure ---------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
cr = comp_r.copy(); cr["short"] = cr.model.str.replace("Baseline: ", "Base: ", regex=False).str.replace("HistGradientBoosting", "HistGB").str.replace("Random Forest", "RF").str.replace("Ridge regression", "Ridge")
colr = [PALETTE["grey"] if m.startswith("Base") else (PALETTE["orange"] if m == best_name else PALETTE["blue"]) for m in cr.model]
axes[0].barh(cr.short, cr.cv_rmse_mean, xerr=cr.cv_rmse_std, color=colr, error_kw=dict(capsize=2, lw=1)); axes[0].set_xlabel("5-fold CV RMSE (log scale) ↓"); axes[0].set_title("(a) Cross-validated RMSE")
axes[1].barh(cr.short, cr.test_R2, color=colr); axes[1].set_xlabel("Hold-out R² ↑"); axes[1].set_title("(b) Test R²"); axes[1].set_yticklabels([])
for i, v in enumerate(cr.test_R2): axes[1].text(max(v, 0) + .01, i, f"{v:.3f}", va="center", fontsize=8)
axes[2].barh(cr.short, cr.test_MAE_USD, color=colr); axes[2].set_xlabel("Hold-out MAE (USD) ↓"); axes[2].set_title("(c) Test MAE in dollars"); axes[2].set_yticklabels([])
axes[2].xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
for ax in axes: ax.invert_yaxis()
savefig(fig, "fig24_regression_model_comparison", "Regression model comparison: cross-validated RMSE with ±1 SD (a), hold-out R² (b) and hold-out MAE in USD (c). Grey = baselines, orange = selected model.", "O6")
plt.show()

# %%
best_r = fit_r[best_name]; p_best = preds_te[best_name]
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
hb = axes[0].hexbin(np.exp(p_best), np.exp(y_te), gridsize=45, xscale="log", yscale="log", cmap="Blues", mincnt=1, bins="log")
lims = [3e3, 1e6]; axes[0].plot(lims, lims, color=PALETTE["red"], lw=1.2, ls="--", label="perfect prediction"); axes[0].set_xlim(lims); axes[0].set_ylim(lims)
axes[0].set_xlabel("Predicted compensation (USD)"); axes[0].set_ylabel("Actual compensation (USD)"); axes[0].set_title(f"(a) Predicted vs actual — {best_name}"); axes[0].legend(loc="upper left")
resid = y_te.values - p_best
axes[1].hist(resid, bins=60, color=PALETTE["green"], density=True); xs_ = np.linspace(-3, 3, 200); axes[1].plot(xs_, stats.norm.pdf(xs_, resid.mean(), resid.std()), color=PALETTE["red"], lw=1.4, label="normal fit")
axes[1].set_xlabel("Residual (log scale)"); axes[1].set_title("(b) Residual distribution"); axes[1].legend()
axes[2].scatter(p_best, resid, s=3, alpha=.25, color=PALETTE["blue"]); axes[2].axhline(0, color=PALETTE["red"], lw=1)
sm_ = lw.lowess(resid, p_best, frac=.3); axes[2].plot(sm_[:, 0], sm_[:, 1], color=PALETTE["orange"], lw=2)
axes[2].set_xlabel("Predicted log-compensation"); axes[2].set_ylabel("Residual"); axes[2].set_title("(c) Residuals vs fitted")
savefig(fig, "fig25_regression_diagnostics", "Diagnostics of the selected regression model on the hold-out set: predicted vs actual pay (a), residual distribution (b) and residuals vs fitted values with LOWESS (c).", "O6")
plt.show()
R["O6"]["residual"] = dict(mean=float(resid.mean()), sd=float(resid.std()), skew=float(stats.skew(resid)), kurt=float(stats.kurtosis(resid)), shapiro_p=float(stats.shapiro(resid[:5000]).pvalue))
print(R["O6"]["residual"])

# %%
# Outlier-policy sensitivity (design-decision check) and learning curve ---------------------------------------------
from sklearn.model_selection import learning_curve
hgb_params = bp_r["HistGradientBoosting"]; sens = []
for label, kw_ in [("Absolute bounds only (USD 1k–1M)", dict(policy="bounds")), ("Country 3×IQR (chosen)", dict(policy="country_iqr", k=3.0)), ("Country 1.5×IQR (strict)", dict(policy="country_iqr", k=1.5))]:
    c_, _ = build_salary_cohort(add_features(decode(df_raw, crosswalk)), **kw_)
    Xs_, ys_ = model_frame(c_, SALARY_SPEC), c_.log_comp
    mdl = clone(zoo_r["HistGradientBoosting"][0]).set_params(**hgb_params)
    cvr = cross_validate(mdl, Xs_, ys_, cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2", "mae": "neg_mean_absolute_error"})
    sens.append(dict(policy=label, n=len(c_), cv_rmse_log=-cvr["test_rmse"].mean(), cv_r2=cvr["test_r2"].mean(), cv_mae_log=-cvr["test_mae"].mean()))
sens = pd.DataFrame(sens); save_table(sens.round(4), "tbl_o6_outlier_sensitivity", "Sensitivity of HistGradientBoosting CV performance to the outlier policy", "O6"); R["O6"]["sensitivity"] = sens.round(4).to_dict("records")
mdl = clone(zoo_r["HistGradientBoosting"][0]).set_params(**hgb_params)
ts, tr_s, va_s = learning_curve(mdl, X_tr, y_tr, train_sizes=[.1, .2, .4, .7, 1.0], cv=KFold(5, shuffle=True, random_state=RANDOM_STATE), scoring="neg_root_mean_squared_error")
R["O6"]["learning_curve"] = dict(sizes=ts.tolist(), train=(-tr_s.mean(1)).tolist(), val=(-va_s.mean(1)).tolist(), val_sd=va_s.std(1).tolist())
fig, ax = plt.subplots(figsize=(6.2, 3.8))
ax.plot(ts, -tr_s.mean(1), "o-", color=PALETTE["blue"], label="Training RMSE"); ax.plot(ts, -va_s.mean(1), "o-", color=PALETTE["orange"], label="Validation RMSE (5-fold)")
ax.fill_between(ts, -va_s.mean(1) - va_s.std(1), -va_s.mean(1) + va_s.std(1), color=PALETTE["orange"], alpha=.2)
ax.set_xlabel("Training-set size (rows)"); ax.set_ylabel("RMSE (log scale)"); ax.set_title("Learning curve — HistGradientBoosting"); ax.legend()
savefig(fig, "fig26_learning_curve", "Learning curve of the tuned gradient-boosting model: the validation error is still slowly falling, so more data would help slightly.", "O6")
plt.show(); sens

# %%
joblib.dump(best_r, MODEL_DIR / "best_salary_model.joblib")
joblib.dump(fit_r, MODEL_DIR / "all_salary_models.joblib")
X_te.assign(actual_log=y_te.values, pred_log=p_best).to_csv(RES_DIR / "salary_test_predictions.csv")
save_results(); print("saved:", best_name)
