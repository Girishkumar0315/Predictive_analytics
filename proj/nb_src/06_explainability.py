# %% [markdown]
# ## O8 — Explainability and error analysis
# **Research question.** Which inputs drive the two models, in which direction, and where do the models fail?
# **Methods.** (1) *Permutation importance* on the hold-out set: the increase in error when one raw column is randomly
# shuffled, $I_j=\text{Err}(\text{shuffled}_j)-\text{Err}(\text{original})$; (2) *SHAP values* (Shapley-value attribution,
# $\phi_j=\sum_{S\subseteq F\setminus\{j\}}\frac{|S|!(|F|-|S|-1)!}{|F|!}[f(S\cup j)-f(S)]$) on the LightGBM model, a
# gradient-boosting model with performance close to the selected one; (3) *partial dependence* computed by hand so that
# derived features are recomputed consistently; (4) *segment-wise error audit*.
# **Success:** a ranked list of drivers with uncertainty, and quantified error gaps across countries / segments.

# %%
from sklearn.inspection import permutation_importance
import shap
best_r = joblib.load(MODEL_DIR / "best_salary_model.joblib"); best_c = joblib.load(MODEL_DIR / "best_ai_adoption_model.joblib")
all_r = joblib.load(MODEL_DIR / "all_salary_models.joblib"); all_c = joblib.load(MODEL_DIR / "all_ai_models.joblib")
NICE = {"years_code_pro": "Professional years", "years_code": "Total years coded", "years_pro_sq": "Prof. years²", "country": "Country", "dev_type": "Job role",
        "org_log": "Org. size (log)", "org_short": "Org. size (category)", "ed_rank": "Education rank", "ed_short": "Education", "age_mid": "Age", "remote_work": "Work arrangement",
        "so_visit_rank": "SO visit frequency", "so_part_freq": "SO participation", "so_comm": "SO community", "so_account": "SO account", "exp_gap": "Years coded before 1st job",
        "pro_share": "Professional share of career", "purchase_influence": "Purchase influence", "buildvs_buy": "Build-vs-buy", "r_used": "Used R", "ai_select": "Uses AI tools",
        "main_branch": "Coding relationship"}
def perm_plot(model, X, y, scoring, title, fname, caption, obj, sign=1, n_repeats=5):
    pi = permutation_importance(model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=1)
    d = pd.DataFrame({"feature": X.columns, "imp": sign * pi.importances_mean, "sd": pi.importances_std}).sort_values("imp", ascending=False)
    d["share_pct"] = d.imp.clip(lower=0) / d.imp.clip(lower=0).sum() * 100
    fig, ax = plt.subplots(figsize=(7.0, 5.0)); t = d.head(14).iloc[::-1]
    ax.barh(t.feature.map(lambda c: NICE.get(c, c)), t.imp, xerr=t.sd, color=PALETTE["blue"], error_kw=dict(capsize=2, lw=1))
    ax.set_xlabel(title); ax.set_title("Permutation importance (hold-out set)")
    savefig(fig, fname, caption, obj); plt.show(); return d
imp_r = perm_plot(best_r, X_te, y_te, "neg_root_mean_squared_error", "Increase in RMSE (log) when feature is shuffled", "fig30_perm_importance_salary",
                  f"Permutation importance for the selected salary model ({R['O6']['best']}).", "O8")
imp_c = perm_plot(best_c, X_ate, y_ate, "roc_auc", "Decrease in ROC-AUC when feature is shuffled", "fig31_perm_importance_ai",
                  f"Permutation importance for the selected AI-adoption model ({R['O7']['best']}).", "O8", n_repeats=5)
save_table(imp_r.round(4), "tbl_o8_perm_importance_salary", "Permutation importance of raw predictors - salary model", "O8"); save_table(imp_c.round(4), "tbl_o8_perm_importance_ai", "Permutation importance of raw predictors - AI-adoption model", "O8")
R["O8"] = dict(salary_top=imp_r.head(8).round(4).to_dict("records"), ai_top=imp_c.head(8).round(4).to_dict("records"),
               salary_top3_share=float(imp_r.head(3).share_pct.sum()), ai_top3_share=float(imp_c.head(3).share_pct.sum()))
imp_r.head(8).round(4)

# %%
# SHAP on LightGBM (tree SHAP is exact and fast for tree ensembles) --------------------------------------------------
def shap_plot(pipe, X, fname, caption, title, n=1500):
    Xs = X.sample(n, random_state=RANDOM_STATE); pre, mdl = pipe[:-1], pipe[-1]
    Xt = pd.DataFrame(pre.transform(Xs), columns=pre.get_feature_names_out())
    Xt.columns = [NICE.get(c, c.replace("missingindicator_", "missing: ")) for c in Xt.columns]
    sv = shap.TreeExplainer(mdl).shap_values(Xt)
    sv = sv[1] if isinstance(sv, list) else sv
    plt.figure(figsize=(7.4, 5.4), layout="none"); shap.summary_plot(sv, Xt, max_display=14, show=False, plot_size=None)
    fig = plt.gcf(); fig.axes[0].set_title(title); fig.axes[0].set_xlabel("SHAP value (impact on model output)")
    savefig(fig, fname, caption, "O8"); plt.show()
    imp = pd.DataFrame({"feature": Xt.columns, "mean_abs_shap": np.abs(sv).mean(0)}).sort_values("mean_abs_shap", ascending=False)
    return imp, sv, Xt
shap_r, sv_r, Xt_r = shap_plot(all_r["LightGBM"], X_te, "fig32_shap_salary", "SHAP summary of the LightGBM salary model: each dot is a respondent; colour = feature value; x = effect on predicted log-compensation.", "SHAP — salary (LightGBM, log-USD)")
shap_c, sv_c, Xt_c = shap_plot(all_c["LightGBM"], X_ate, "fig33_shap_ai", "SHAP summary of the LightGBM AI-adoption model (effect on log-odds of using AI).", "SHAP — AI adoption (LightGBM, log-odds)")
R["O8"]["shap_salary_top"] = shap_r.head(8).round(4).to_dict("records"); R["O8"]["shap_ai_top"] = shap_c.head(8).round(4).to_dict("records")
shap_r.head(6).round(4)

# %%
# Partial dependence (computed manually so derived experience features stay consistent) ------------------------------
def pdp_experience(model, Xbase, grid, mask=None):
    out = []
    Xb = Xbase if mask is None else Xbase[mask]
    for g in grid:
        Xg = Xb.copy(); Xg["years_code_pro"] = float(g); Xg["years_pro_sq"] = float(g) ** 2
        Xg["exp_gap"] = Xg["years_code"] - g; Xg["pro_share"] = g / Xg["years_code"].where(Xg["years_code"] > 0)
        out.append(np.exp(model.predict(Xg)).mean())          # average predicted USD
    return np.array(out)
grid = np.arange(0, 31, 2)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9))
for c, col in [("All", PALETTE["grey"]), ("USA", PALETTE["blue"]), ("Germany", PALETTE["green"]), ("India", PALETTE["orange"])]:
    mask = None if c == "All" else (X_te.country == c).values
    axes[0].plot(grid, pdp_experience(best_r, X_te, grid, mask), "o-", ms=3, color=col, label=c, lw=2 if c == "India" else 1.5)
axes[0].set_yscale("log"); axes[0].set_xlabel("Years coded professionally (set for everyone)"); axes[0].set_ylabel("Average predicted pay (USD, log axis)")
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")); axes[0].set_title("(a) Partial dependence: experience → pay"); axes[0].legend()
ages_g = np.array([21, 29.5, 39.5, 49.5, 59.5])
pd_age = []
for g in ages_g:
    Xg = X_ate.copy(); Xg["age_mid"] = g; pd_age.append(best_c.predict_proba(Xg)[:, 1].mean() * 100)
axes[1].plot(ages_g, pd_age, "o-", color=PALETTE["purple"]); axes[1].set_xlabel("Age (bracket midpoint set for everyone)"); axes[1].set_ylabel("Average predicted AI-adoption probability (%)"); axes[1].set_title("(b) Partial dependence: age → AI adoption")
savefig(fig, "fig34_partial_dependence", "Partial dependence: average predicted pay as professional experience is varied, by country (a); average predicted AI-adoption probability as age is varied (b).", "O8")
plt.show()
R["O8"]["pdp_exp"] = {c: pdp_experience(best_r, X_te, np.array([0, 5, 10, 20, 30]), None if c == "All" else (X_te.country == c).values).round(0).tolist() for c in ["All", "USA", "India", "Germany"]}
R["O8"]["pdp_age"] = dict(zip(ages_g.tolist(), np.round(pd_age, 2).tolist()))

# %%
# ---- Error audit: salary model -----------------------------------------------------------------------------------------
te = cohort.loc[X_te.index].copy(); te["pred_log"] = best_r.predict(X_te); te["resid"] = te.log_comp - te.pred_log
te["pred_usd"] = np.exp(te.pred_log); te["ape"] = (te.pred_usd - te.converted_comp_yearly).abs() / te.converted_comp_yearly; te["ae_usd"] = (te.pred_usd - te.converted_comp_yearly).abs()
def seg(col, keep=None, minn=40):
    g = te.groupby(col).agg(n=("ape", "size"), MdAPE_pct=("ape", lambda s: s.median() * 100), MAE_USD=("ae_usd", "mean"), bias_log=("resid", "mean"), within25=("ape", lambda s: (s <= .25).mean() * 100))
    g = g[g.n >= minn]; return g.sort_values("n", ascending=False)
seg_c = seg("country"); seg_e = seg("exp_band", minn=40).reindex(bands); seg_r = seg("dev_type", minn=60).head(10)
save_table(seg_c.round(3).reset_index().head(12), "tbl_o8_error_by_country", "Salary-model error by country (hold-out, n >= 40)", "O8")
ind_te, rest_te = te[te.country == "India"], te[te.country != "India"]
R["O8"]["error"] = dict(overall_mdape=float(te.ape.median() * 100), overall_mae=float(te.ae_usd.mean()), india_mdape=float(ind_te.ape.median() * 100), india_mae=float(ind_te.ae_usd.mean()), india_n=len(ind_te),
                        rest_mdape=float(rest_te.ape.median() * 100), usa_mdape=float(te[te.country == "USA"].ape.median() * 100), india_bias=float(ind_te.resid.mean()),
                        by_country=seg_c.round(3).reset_index().head(10).to_dict("records"), by_exp=seg_e.round(3).reset_index().to_dict("records"), by_role=seg_r.round(3).reset_index().to_dict("records"))
fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
t = seg_c.head(10).sort_values("MdAPE_pct"); axes[0].barh(t.index, t.MdAPE_pct, color=[PALETTE["orange"] if c == "India" else PALETTE["blue"] for c in t.index]); axes[0].set_xlabel("Median absolute % error"); axes[0].set_title("(a) By country (10 largest)")
t = seg_e; axes[1].bar(t.index, t.MdAPE_pct, color=PALETTE["green"]); axes[1].set_xlabel("Professional experience (years)"); axes[1].set_ylabel("Median absolute % error"); axes[1].set_title("(b) By experience band")
t = seg_r.sort_values("MdAPE_pct"); axes[2].barh(t.index, t.MdAPE_pct, color=PALETTE["purple"]); axes[2].set_xlabel("Median absolute % error"); axes[2].set_title("(c) By job role (10 largest)")
savefig(fig, "fig35_salary_error_by_segment", "Hold-out error of the salary model by country (a), experience band (b) and job role (c); India highlighted in orange.", "O8")
plt.show()
worst = te.assign(err_usd=te.pred_usd - te.converted_comp_yearly).sort_values("ae_usd", ascending=False).head(8)[["country", "dev_type", "years_code_pro", "org_short", "converted_comp_yearly", "pred_usd", "err_usd"]].round(0)
save_table(worst, "tbl_o8_worst_predictions", "The eight largest salary-prediction errors on the hold-out set", "O8"); R["O8"]["worst"] = worst.to_dict("records")
seg_c.head(8).round(2)

# %%
# ---- Error audit: AI-adoption model ------------------------------------------------------------------------------------------
ta = ai_cohort.loc[X_ate.index].copy(); ta["p"] = best_c.predict_proba(X_ate)[:, 1]; ta["pred"] = (ta.p >= R["O7"]["threshold_bal"]).astype(int); ta["ok"] = (ta.pred == ta.uses_ai)
def cseg(col, minn=300, keep=None):
    rows = []
    for k, g in ta.groupby(col):
        if len(g) >= minn and g.uses_ai.nunique() == 2: rows.append(dict(group=k, n=len(g), accuracy=g.ok.mean() * 100, auc=roc_auc_score(g.uses_ai, g.p), pos_rate=g.uses_ai.mean() * 100, pred_rate=g.pred.mean() * 100))
    return pd.DataFrame(rows)
cs_age = cseg("age"); cs_age = cs_age[cs_age.group != "Prefer not to say"].set_index("group").reindex([a for a in AGE_ORDER if a in set(cs_age.group)]).reset_index()
cs_c = cseg("country", 150).sort_values("n", ascending=False).head(10); cs_r = cseg("dev_type", 150).sort_values("n", ascending=False).head(10)
R["O8"]["clf_error"] = dict(by_age=cs_age.round(2).to_dict("records"), by_country=cs_c.round(2).to_dict("records"), by_role=cs_r.round(2).to_dict("records"),
                            india=cseg("country", 100).query("group == 'India'").round(2).to_dict("records"))
save_table(cs_c.round(2), "tbl_o8_clf_error_country", "AI-adoption model: hold-out accuracy and AUC by country", "O8")
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9))
for ax, d_, title in [(axes[0], cs_age, "(a) By age bracket"), (axes[1], cs_c.sort_values("auc"), "(b) By country (10 largest)"), (axes[2], cs_r.sort_values("auc"), "(c) By job role (10 largest)")]:
    cols = [PALETTE["orange"] if g == "India" else PALETTE["blue"] for g in d_.group]
    ax.barh(d_.group, d_.auc, color=cols); ax.axvline(R["O7"]["metrics_at_05"]["ROC_AUC"], color=PALETTE["red"], ls="--", lw=1); ax.set_xlim(.45, .8); ax.set_xlabel("Hold-out ROC-AUC within group"); ax.set_title(title)
    if ax is axes[0]: ax.invert_yaxis()
savefig(fig, "fig36_ai_error_by_segment", "Hold-out ROC-AUC of the AI-adoption classifier within age brackets (a), countries (b) and roles (c); dashed line = overall AUC.", "O8")
plt.show(); cs_c.round(2)

# %%
# Demonstration of the saved models on two contrasting developer profiles -----------------------------------------------
from src.predict import predict_salary, predict_ai_adoption
profiles = {"Mid-level full-stack dev, India": dict(country="India", years_code=9, years_code_pro=5, age="25-34", dev_type="Full-stack dev"),
            "Mid-level full-stack dev, USA": dict(country="USA", years_code=9, years_code_pro=5, age="25-34", dev_type="Full-stack dev"),
            "Senior engineering manager, Germany": dict(country="Germany", years_code=22, years_code_pro=18, age="35-44", dev_type="Engineering manager", org_short="1,000 to 4,999", ed_short="Master's"),
            "Student-age Python learner, Brazil": dict(country="Brazil", years_code=3, years_code_pro=0, age="18-24", dev_type="Student", main_branch="I am learning to code", ed_short="Some college")}
demo = pd.DataFrame([dict(profile=k, predicted_salary_usd=round(predict_salary(v, best_r), -2), p_uses_ai=round(predict_ai_adoption(v, best_c), 3)) for k, v in profiles.items()])
save_table(demo, "tbl_demo_predictions", "Example predictions of the saved models for illustrative profiles", "O8"); R["demo"] = demo.to_dict("records"); save_results(); demo
