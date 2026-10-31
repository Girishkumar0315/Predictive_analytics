# %% [markdown]
# ## O7 — Predicting AI-tool adoption (binary classification)
# **Prediction task.** Will a developer currently use AI tools in their workflow? (`uses_ai` = 1 if `ai_select = "Yes"`,
# else 0). Population: all 60 907 respondents with a known answer.
#
# **Metrics.** With TP/FP/TN/FN: $\text{Precision}=\frac{TP}{TP+FP}$, $\text{Recall}=\frac{TP}{TP+FN}$,
# $F_1=\frac{2PR}{P+R}$, ROC-AUC = $P(\hat p_{pos}>\hat p_{neg})$, PR-AUC = area under precision–recall curve,
# Brier = $\frac1n\sum(\hat p_i-y_i)^2$ (calibration). **Baseline:** always predict the majority class.
# **Success criterion (set in advance):** ROC-AUC ≥ 0.65 on the hold-out set, and F1 above the majority baseline.
#
# ### Leakage demonstration (why `ai_sent`, `ai_acc`, `ai_complex`, `ai_threat` are forbidden here)
# Their *presence* reveals the target (see O5 conditionality table). We train one quick model **with** and **without**
# them to measure how misleading a leaky evaluation would be.

# %%
X_a = model_frame(ai_cohort, ADOPT_SPEC); y_a = ai_cohort.uses_ai.astype(int)
X_atr, X_ate, y_atr, y_ate = train_test_split(X_a, y_a, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_a)
print("train", X_atr.shape, "test", X_ate.shape, "| positive rate train %.3f / test %.3f" % (y_atr.mean(), y_ate.mean()))
leak_cols = ["ai_sent", "ai_acc", "ai_complex", "ai_threat"]
leaky_spec = dict(ADOPT_SPEC, low=ADOPT_SPEC["low"] + leak_cols)
Xl = model_frame(ai_cohort, leaky_spec)
Xl_tr, Xl_te = Xl.loc[X_atr.index], Xl.loc[X_ate.index]
def quick(spec, Xtr_, Xte_):
    m = Pipeline([("pre", make_preprocessor("tree", spec, "binary")), ("m", HistGradientBoostingClassifier(random_state=RANDOM_STATE))]).fit(Xtr_, y_atr)
    return roc_auc_score(y_ate, m.predict_proba(Xte_)[:, 1])
auc_honest, auc_leaky = quick(ADOPT_SPEC, X_atr, X_ate), quick(leaky_spec, Xl_tr, Xl_te)
R["O7_leakage_demo"] = dict(auc_honest=float(auc_honest), auc_leaky=float(auc_leaky))
print(f"Test ROC-AUC without leaky columns = {auc_honest:.3f}  |  WITH conditional AI-question columns = {auc_leaky:.3f}  (inflated by {auc_leaky-auc_honest:+.3f})")

# %%
t0 = time.time()
zoo_c = classification_zoo(ADOPT_SPEC)
fit_c, cmp_c, bp_c, folds_c = tune_and_compare(zoo_c, X_atr, y_atr, "classification", n_iter=8, tune_cv=3, eval_cv=CV_FOLDS)
print(f"total {time.time()-t0:.0f}s"); cmp_c

# %%
test_rows, proba_te = [], {}
for name, m in fit_c.items():
    pr = m.predict_proba(X_ate)[:, 1]; proba_te[name] = pr
    test_rows.append({"model": name, **clf_metrics(y_ate, pr, 0.5)})
test_c = pd.DataFrame(test_rows)
comp_c = cmp_c.merge(test_c, on="model").rename(columns={c: f"test_{c}" for c in test_c.columns if c != "model"})
comp_c["tuned_params"] = comp_c.model.map(lambda k: json.dumps(bp_c[k]) if bp_c[k] else "-")
save_table(comp_c.round(4), "tbl_o7_model_comparison", "Classification models: 5-fold CV (train) and hold-out test metrics", "O7")
best_c_name = comp_c.sort_values("cv_roc_auc_mean", ascending=False).iloc[0].model
best_c = fit_c[best_c_name]
# F1-optimal threshold chosen on OUT-OF-FOLD train predictions only (no test peeking)
oof = cross_val_predict(best_c, X_atr, y_atr, cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE), method="predict_proba")[:, 1]
from sklearn.metrics import balanced_accuracy_score
ths = np.linspace(.2, .8, 61); f1s = [f1_score(y_atr, oof >= t) for t in ths]; thr_opt = float(ths[int(np.argmax(f1s))])
bals = [balanced_accuracy_score(y_atr, oof >= t) for t in ths]; thr_bal = float(ths[int(np.argmax(bals))])   # balanced-accuracy (Youden-type) operating point
m05 = clf_metrics(y_ate, proba_te[best_c_name], 0.5); mopt = clf_metrics(y_ate, proba_te[best_c_name], thr_opt); mbal = clf_metrics(y_ate, proba_te[best_c_name], thr_bal)
rng = np.random.default_rng(RANDOM_STATE); pt = proba_te[best_c_name]; yt_ = y_ate.values; bs = []
for _ in range(1000):
    i = rng.integers(0, len(yt_), len(yt_)); bs.append(roc_auc_score(yt_[i], pt[i]))
base_f1 = comp_c.set_index("model").loc["Baseline: majority class", "test_F1"]
R["O7"] = dict(best=best_c_name, comparison=comp_c.round(4).to_dict("records"), best_params=bp_c, threshold_opt=thr_opt, threshold_bal=thr_bal, metrics_at_05=m05, metrics_at_opt=mopt, metrics_at_bal=mbal,
               auc_ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], fold_scores=folds_c, positive_rate=float(y_a.mean()),
               n_train=len(X_atr), n_test=len(X_ate), success_auc_065=bool(m05["ROC_AUC"] >= 0.65), success_f1_at_05=bool(m05["F1"] > base_f1), success_f1_at_opt=bool(mopt["F1"] > base_f1), baseline_f1=float(base_f1))
print("BEST:", best_c_name, "| threshold (F1-opt on OOF) =", thr_opt)
print("balanced thr =", thr_bal, {k: round(v, 3) for k, v in mbal.items()}); print("default 0.5 :", {k: round(v, 3) for k, v in m05.items()}); print("tuned thr   :", {k: round(v, 3) for k, v in mopt.items()})
comp_c[["model", "cv_roc_auc_mean", "cv_roc_auc_std", "test_Accuracy", "test_Precision", "test_Recall", "test_F1", "test_ROC_AUC", "test_PR_AUC", "test_Brier"]].round(3)

# %%
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
cc_ = comp_c.copy(); cc_["short"] = cc_.model.str.replace("Baseline: majority class", "Base: majority").str.replace("HistGradientBoosting", "HistGB").str.replace("Random Forest", "RF").str.replace("Logistic Regression", "LogReg")
colc = [PALETTE["grey"] if m.startswith("Base") else (PALETTE["orange"] if m == best_c_name else PALETTE["blue"]) for m in cc_.model]
axes[0].barh(cc_.short, cc_.cv_roc_auc_mean, xerr=cc_.cv_roc_auc_std, color=colc, error_kw=dict(capsize=2)); axes[0].set_xlim(.45, .75); axes[0].set_xlabel("5-fold CV ROC-AUC ↑"); axes[0].set_title("(a) Cross-validated ROC-AUC")
axes[1].barh(cc_.short, cc_.test_F1, color=colc); axes[1].set_xlabel("Hold-out F1 ↑"); axes[1].set_title("(b) Test F1 (threshold 0.5)"); axes[1].set_yticklabels([])
axes[2].barh(cc_.short, cc_.test_ROC_AUC, color=colc); axes[2].set_xlim(.45, .75); axes[2].set_xlabel("Hold-out ROC-AUC ↑"); axes[2].set_title("(c) Test ROC-AUC"); axes[2].set_yticklabels([])
for a_, col in [(axes[1], "test_F1"), (axes[2], "test_ROC_AUC")]:
    for i, v in enumerate(cc_[col]): a_.text(v + .004, i, f"{v:.3f}", va="center", fontsize=8)
for ax in axes: ax.invert_yaxis()
savefig(fig, "fig27_classification_model_comparison", "Classification model comparison: cross-validated ROC-AUC (a), hold-out F1 (b) and hold-out ROC-AUC (c). Grey = baseline, orange = selected model.", "O7")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
for (name, pr), col in zip(proba_te.items(), [PALETTE["grey"], PALETTE["green"], PALETTE["purple"], PALETTE["blue"], PALETTE["orange"]]):
    fpr, tpr, _ = roc_curve(y_ate, pr); axes[0].plot(fpr, tpr, color=col, lw=2.2 if name == best_c_name else 1.3, label=f"{name} ({roc_auc_score(y_ate, pr):.3f})")
    p_, r_, _ = precision_recall_curve(y_ate, pr); axes[1].plot(r_, p_, color=col, lw=2.2 if name == best_c_name else 1.3)
axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_xlabel("False-positive rate"); axes[0].set_ylabel("True-positive rate"); axes[0].set_title("(a) ROC curves (test)"); axes[0].legend(fontsize=7.5, loc="lower right")
axes[1].axhline(y_ate.mean(), color="k", ls="--", lw=.8); axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision"); axes[1].set_title("(b) Precision–recall curves (test)")
savefig(fig, "fig28_roc_pr_curves", "ROC curves with AUC in the legend (a) and precision–recall curves (b) for all classifiers on the hold-out set.", "O7")
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
cmx = confusion_matrix(y_ate, pt >= thr_bal); cmn = cmx / cmx.sum(1, keepdims=True)
sns.heatmap(cmn, annot=np.array([[f"{cmn[i, j]:.1%}\n({cmx[i, j]:,})" for j in range(2)] for i in range(2)]), fmt="", cmap="Blues", cbar=False, ax=axes[0], xticklabels=["Predicted: No", "Predicted: Yes"], yticklabels=["Actual: No", "Actual: Yes"], vmin=0, vmax=1)
axes[0].set_title(f"(a) Confusion matrix (thr = {thr_bal:.2f})")
fr, mp = calibration_curve(y_ate, pt, n_bins=10, strategy="quantile"); axes[1].plot(mp, fr, "o-", color=PALETTE["orange"], label=best_c_name); axes[1].plot([0, 1], [0, 1], "k--", lw=.8, label="perfect")
axes[1].set_xlabel("Mean predicted probability"); axes[1].set_ylabel("Observed fraction using AI"); axes[1].set_title("(b) Calibration curve"); axes[1].legend()
P_, Rr_, F_ = [], [], []
for t in ths: pr_ = pt >= t; P_.append(precision_score(y_ate, pr_)); Rr_.append(recall_score(y_ate, pr_)); F_.append(f1_score(y_ate, pr_))
axes[2].plot(ths, P_, label="Precision", color=PALETTE["blue"]); axes[2].plot(ths, Rr_, label="Recall", color=PALETTE["green"]); axes[2].plot(ths, F_, label="F1", color=PALETTE["orange"], lw=2)
axes[2].axvline(thr_opt, color=PALETTE["red"], ls="--", lw=1); axes[2].axvline(thr_bal, color=PALETTE["purple"], ls=":", lw=1.5, label="balanced-acc. optimum"); axes[2].set_xlabel("Decision threshold"); axes[2].set_title("(c) Threshold trade-off (test)"); axes[2].legend()
savefig(fig, "fig29_confusion_calibration_threshold", "Selected classifier on the hold-out set: normalised confusion matrix at the balanced-accuracy threshold (a), calibration curve (b) and precision/recall/F1 versus decision threshold (c).", "O7")
plt.show()
R["O7"]["confusion"] = cmx.tolist()
joblib.dump(best_c, MODEL_DIR / "best_ai_adoption_model.joblib"); joblib.dump(fit_c, MODEL_DIR / "all_ai_models.joblib")
json.dump({"threshold_balanced": thr_bal, "threshold_f1": thr_opt, "model": best_c_name}, open(MODEL_DIR / "ai_model_threshold.json", "w"))
save_results()
