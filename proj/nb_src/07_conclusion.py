# %% [markdown]
# # Part G — Conclusions: objective-to-evidence map

# %%
o = R
rows = [
 ("O1", "Country explains pay", f"Country alone explains R² = {o['O1']['R2_country_all']:.2f} of log-pay variance (η² = {o['O1']['eta2_top']:.2f} among the 28 largest countries); India's median USD {o['O1']['india_median']:,.0f} is {o['O1']['india_usa_ratio']:.2f}× the USA median (95 % CI {o['O1']['india_usa_ratio_ci'][0]:.2f}–{o['O1']['india_usa_ratio_ci'][1]:.2f})", "fig09, fig10"),
 ("O2", "Experience premium", f"Spearman ρ = {o['O2']['spearman_all']['rho']:.2f}; ≈ {o['O2']['marginal'][0]['pct_per_year']:.1f} % per extra year at entry, flattening at ≈ {o['O2']['peak_years']:.0f} yrs; India's early-career slope is steeper (interaction p {fmt_p(o['O2']['interaction_linear']['p'])})", "fig11, fig12"),
 ("O3", "Net effects", f"Adding role, company size, work mode and education lifts R² from {o['O3']['nested'][1]['R2']:.3f} to {o['O3']['nested'][-1]['R2']:.3f}; remote work +{[a for a in o['O3']['adj'] if a['term']=='Remote'][0]['pct']:.0f} % vs in-person after adjustment", "fig13–fig15"),
 ("O4", "AI adoption landscape", f"{o['O4']['overall_rate']:.1f} % use AI; adoption falls with age/experience; India {o['O4']['india']['rate']:.1f} % vs rest {o['O4']['india']['rest_rate']:.1f} %", "fig16–fig19"),
 ("O5", "AI attitudes", f"{o['O5']['fav_pct']:.0f} % favourable but only {o['O5']['trust_positive_pct']:.0f} % trust accuracy; {o['O5']['threat_yes_pct']:.0f} % see AI as a job threat; experience effects are tiny (|ρ| < 0.06)", "fig20–fig23"),
 ("O6", "Salary prediction", f"Best: {o['O6']['best']}; test R² = {[c for c in o['O6']['comparison'] if c['model']==o['O6']['best']][0]['test_R2']:.3f}; RMSE(log) cut by {o['O6']['rmse_reduction_vs_global_pct']:.1f} % vs baseline", "fig24–fig26"),
 ("O7", "AI-adoption prediction", f"Best: {o['O7']['best']}; test ROC-AUC = {o['O7']['metrics_at_05']['ROC_AUC']:.3f}, balanced accuracy = {o['O7']['metrics_at_bal']['BalAcc']:.3f} at threshold {o['O7']['threshold_bal']:.2f}", "fig27–fig29"),
 ("O8", "Explainability & errors", f"Top salary drivers: {', '.join(NICE.get(d['feature'], d['feature']) for d in o['O8']['salary_top'][:3])}; India MdAPE {o['O8']['error']['india_mdape']:.0f} % vs USA {o['O8']['error']['usa_mdape']:.0f} %", "fig30–fig36"),
]
summary = pd.DataFrame(rows, columns=["objective", "topic", "headline result (computed from this notebook's own outputs)", "figures"])
save_table(summary, "tbl_objective_mapping", "Objective-to-evidence mapping with headline results", "All")
pd.set_option("display.max_colwidth", 220); print(summary.to_string(index=False))
R["success"] = {k: R["O6"][k] for k in ("success_rmse_cut_40", "success_beats_country_median")} | {k: R["O7"][k] for k in ("success_auc_065", "success_f1_at_05", "success_f1_at_opt")}
save_results()
print("\nPre-registered success criteria:", json.dumps(R["success"], indent=1))
print("Figures:", len(list(FIG_DIR.glob("*.png"))), "| tables:", len(list((RES_DIR / "tables").glob("*.csv"))))
