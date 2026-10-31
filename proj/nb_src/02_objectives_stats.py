# %% [markdown]
# # Part D — Statistical analysis for objectives O1–O5
# Reminder of the test choices justified by EDA: log-salary is heavy-tailed with unequal spread across countries,
# so we use **rank-based** tests and **heteroscedasticity-robust (HC3)** regression inference.
#
# $$H=\frac{12}{N(N+1)}\sum_{i=1}^{k}\frac{R_i^{2}}{n_i}-3(N+1),\qquad \varepsilon^{2}=\frac{H}{N-1}\quad\text{(Kruskal–Wallis and its effect size)}$$
# $$\eta^{2}=\frac{SS_{between}}{SS_{total}},\qquad V=\sqrt{\frac{\tilde\chi^{2}/n}{\min(k-1,r-1)}}\quad\text{(Cramér's V)},\qquad
# \hat p\pm z\sqrt{\tfrac{\hat p(1-\hat p)}{n}+\tfrac{z^{2}}{4n^{2}}}\Big/\Big(1+\tfrac{z^{2}}{n}\Big)\ \text{(Wilson)}$$

# %%
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
import re

def fmt_p(p): return "< 0.001" if p < 0.001 else f"{p:.3f}"

def boot_median_ci(x, n=1000, seed=RANDOM_STATE):
    x = np.asarray(x); rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), (n, len(x)))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

cohort = pd.read_csv(DATA_PROC / "salary_cohort_clean.csv")
ai_cohort = pd.read_csv(DATA_PROC / "ai_cohort_clean.csv")
print(cohort.shape, ai_cohort.shape)

# %% [markdown]
# ## O1 — Geographic compensation inequality and India's position
# **Research question.** How much of the variation in developer pay is explained by country alone, and how does India
# compare with the largest developer markets? **Relevance.** Offshoring, migration decisions and salary benchmarking.
# **Variables:** `country`, `converted_comp_yearly` (USD, **nominal — not purchasing-power adjusted**).
# **Output & success:** η² / R² of country; India : USA median ratio with a 95 % bootstrap CI; pairwise effect sizes.

# %%
cc = cohort.country.value_counts()
top_c = cc[cc >= 150].index.tolist()
sub1 = cohort[cohort.country.isin(top_c)]
groups = [g.log_comp.values for _, g in sub1.groupby("country")]
H, pH = stats.kruskal(*groups); eps2 = epsilon_squared_kruskal(H, len(sub1))
eta2_top = eta_squared_from_groups(groups)
m_all = smf.ols("log_comp ~ C(country)", data=cohort).fit()
rows = []
usa_med = cohort.loc[cohort.country == "USA", "converted_comp_yearly"].median()
for c in top_c:
    x = cohort.loc[cohort.country == c, "converted_comp_yearly"].values
    lo, hi = boot_median_ci(x)
    rows.append(dict(country=c, n=len(x), median_usd=np.median(x), ci_low=lo, ci_high=hi,
                     q1=np.percentile(x, 25), q3=np.percentile(x, 75), ratio_to_usa=np.median(x) / usa_med))
ctab = pd.DataFrame(rows).sort_values("median_usd", ascending=False).reset_index(drop=True)
ctab["rank"] = np.arange(1, len(ctab) + 1)
# India vs each other top country: Mann-Whitney + Holm + Cliff's delta (positive = India higher)
ind = cohort.loc[cohort.country == "India", "log_comp"].values
pw = []
for c in top_c:
    if c == "India": continue
    o = cohort.loc[cohort.country == c, "log_comp"].values
    pw.append(dict(country=c, mwu_p=stats.mannwhitneyu(ind, o).pvalue, cliffs_delta_india_vs=cliffs_delta(ind, o)))
pw = pd.DataFrame(pw); pw["p_holm"] = multipletests(pw.mwu_p, method="holm")[1]
ctab = ctab.merge(pw, on="country", how="left")
save_table(ctab.round(3), "tbl_o1_country_stats", "Median annual compensation (USD) by country with 95 % bootstrap CI and India-vs-country effect sizes", "O1")
ind_row = ctab[ctab.country == "India"].iloc[0]
# bootstrap CI of the India:USA median ratio
rng = np.random.default_rng(RANDOM_STATE)
xi = cohort.loc[cohort.country == "India", "converted_comp_yearly"].values; xu = cohort.loc[cohort.country == "USA", "converted_comp_yearly"].values
ratios = [np.median(rng.choice(xi, len(xi))) / np.median(rng.choice(xu, len(xu))) for _ in range(2000)]
R["O1"] = dict(top_countries=top_c, n_top=len(sub1), H=float(H), p=float(pH), epsilon2=eps2, eta2_top=eta2_top,
               R2_country_all=float(m_all.rsquared), adjR2_country_all=float(m_all.rsquared_adj), n_all_countries=int(cohort.country.nunique()),
               india_median=float(ind_row.median_usd), india_n=int(ind_row.n), india_rank=int(ind_row["rank"]), n_ranked=len(ctab),
               usa_median=float(usa_med), india_usa_ratio=float(np.median(xi) / usa_med),
               india_usa_ratio_ci=[float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
               max_country=ctab.iloc[0][["country", "median_usd"]].to_dict(), min_country=ctab.iloc[-1][["country", "median_usd"]].to_dict(),
               india_sig_lower_than=int((ctab.country.ne("India") & (ctab.p_holm < 0.05) & (ctab.cliffs_delta_india_vs < 0)).sum()),
               india_sig_higher_than=int((ctab.country.ne("India") & (ctab.p_holm < 0.05) & (ctab.cliffs_delta_india_vs > 0)).sum()),
               top_over_bottom=float(ctab.median_usd.max() / ctab.median_usd.min()))
print(json.dumps({k: v for k, v in R["O1"].items() if k not in ("top_countries",)}, indent=1, default=str))
ctab.round(2)

# %%
fig, ax = plt.subplots(figsize=(7.4, 6.2))
ordc = ctab.country.tolist()
sns.boxplot(data=sub1, y="country", x="converted_comp_yearly", order=ordc, ax=ax, fliersize=1.5, linewidth=.8,
            palette={c: (PALETTE["orange"] if c == "India" else PALETTE["blue"]) for c in ordc}, hue="country", legend=False)
ax.set_xscale("log"); ax.set_xlabel("Annual compensation (USD, log scale)"); ax.set_ylabel("")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
ax.set_title(f"Developer compensation by country (n ≥ 150 salaries; {len(top_c)} countries)")
savefig(fig, "fig09_country_boxplot", "Distribution of annual compensation (USD, log axis) by country, ordered by median. India is highlighted in orange.", "O1")
plt.show()

fig, ax = plt.subplots(figsize=(7.4, 6.2))
d_ = ctab.iloc[::-1]
cols = [PALETTE["orange"] if c == "India" else PALETTE["blue"] for c in d_.country]
ax.barh(d_.country, d_.median_usd, color=cols, xerr=[d_.median_usd - d_.ci_low, d_.ci_high - d_.median_usd], error_kw=dict(lw=1, capsize=2))
for i, (v, r_) in enumerate(zip(d_.median_usd, d_.ratio_to_usa)): ax.text(d_.ci_high.iloc[i] + 2500, i, f"{r_:.2f}× USA", va="center", fontsize=8)
ax.set_xlabel("Median annual compensation (USD) with 95 % bootstrap CI"); ax.set_xlim(0, ctab.ci_high.max() * 1.25)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k")); ax.set_title("Median pay and ratio to the USA median")
savefig(fig, "fig10_country_median_ci", "Median annual compensation with 95 % bootstrap confidence intervals; labels give the ratio to the USA median.", "O1")
plt.show()

# %% [markdown]
# ## O2 — Experience premium and diminishing returns
# **Research question.** How strongly does professional experience raise pay, and when do the returns flatten?
# **Variables:** `years_code_pro`, `log_comp`, `country`. **Methods:** Spearman ρ (overall and within large countries);
# log-linear quadratic OLS with country fixed effects and HC3 errors; delta-method CIs for the marginal effect.
# $$\ln(\text{Pay}_i)=\beta_0+\beta_1 x_i+\beta_2 x_i^{2}+\gamma_{country(i)}+\varepsilon_i,\qquad
# \frac{\partial \ln \text{Pay}}{\partial x}=\beta_1+2\beta_2 x \;\Rightarrow\; \%\Delta\text{Pay per extra year}\approx 100\,(e^{\beta_1+2\beta_2 x}-1),\qquad x^{*}=-\frac{\beta_1}{2\beta_2}$$
# **Success:** ρ with 95 % CI; % gain per year at 0/5/10/20 years; test of India-vs-USA difference in slope.

# %%
e = cohort.dropna(subset=["years_code_pro"]).copy()
e["is_india"] = (e.country == "India").astype(int)
sp_all = spearman(e.years_code_pro, e.log_comp)
sp_c = {c: spearman(e.loc[e.country == c, "years_code_pro"], e.loc[e.country == c, "log_comp"]) for c in ["USA", "India", "Germany", "UK", "Ukraine", "France", "Canada"]}
m2 = smf.ols("log_comp ~ years_code_pro + years_pro_sq + C(country)", data=e).fit(cov_type="HC3")
b1, b2 = m2.params["years_code_pro"], m2.params["years_pro_sq"]
names = ["years_code_pro", "years_pro_sq"]; V2 = m2.cov_params().loc[names, names].values
def marg(x, b1=b1, b2=b2, V=V2):
    g = np.array([1, 2 * x]); me = g @ np.array([b1, b2]); se = np.sqrt(g @ V @ g)
    return [100 * (np.exp(v) - 1) for v in (me, me - 1.96 * se, me + 1.96 * se)]
me_tab = pd.DataFrame([dict(years=x, pct_per_year=marg(x)[0], ci_low=marg(x)[1], ci_high=marg(x)[2]) for x in [0, 2, 5, 10, 15, 20, 30]])
peak = -b1 / (2 * b2) if b2 < 0 else np.nan
# India-vs-USA slope difference
iu = e[e.country.isin(["India", "USA"])]
m_int = smf.ols("log_comp ~ years_code_pro + years_pro_sq + is_india + years_code_pro:is_india + years_pro_sq:is_india", data=iu).fit(cov_type="HC3")
# separate fits
sep = {}
for c in ["USA", "India", "Germany", "UK"]:
    mc = smf.ols("log_comp ~ years_code_pro + years_pro_sq", data=e[e.country == c]).fit(cov_type="HC3")
    sep[c] = dict(b1=float(mc.params["years_code_pro"]), b2=float(mc.params["years_pro_sq"]), V=mc.cov_params().values.tolist()[1:], r2=float(mc.rsquared),
                  pct_at_5=float(100 * (np.exp(mc.params["years_code_pro"] + 10 * mc.params["years_pro_sq"]) - 1)),
                  pct_at_0=float(100 * (np.exp(mc.params["years_code_pro"]) - 1)))
# bootstrap CI of Spearman (overall) as a robustness check
bl = bootstrap_ci(lambda d: stats.spearmanr(d.years_code_pro, d.log_comp)[0], e[["years_code_pro", "log_comp"]], n_boot=300)
R["O2"] = dict(n=len(e), spearman_all=sp_all, spearman_by_country=sp_c, spearman_boot_ci=list(bl),
               beta1=float(b1), beta2=float(b2), peak_years=float(peak), r2_quad=float(m2.rsquared),
               p_beta1=float(m2.pvalues["years_code_pro"]), p_beta2=float(m2.pvalues["years_pro_sq"]), marginal=me_tab.round(2).to_dict("records"),
               interaction_linear=dict(coef=float(m_int.params["years_code_pro:is_india"]), p=float(m_int.pvalues["years_code_pro:is_india"]),
                                       coef_sq=float(m_int.params["years_pro_sq:is_india"]), p_sq=float(m_int.pvalues["years_pro_sq:is_india"])),
               separate_fits={k: {kk: vv for kk, vv in v.items() if kk != "V"} for k, v in sep.items()})
save_table(me_tab.round(2), "tbl_o2_marginal", "Estimated % change in pay per additional year of professional experience (country fixed effects)", "O2")
print(json.dumps({k: v for k, v in R["O2"].items() if k != "marginal"}, indent=1, default=str)); me_tab.round(2)

# %%
bands = ["0-2", "3-5", "6-10", "11-20", "21+"]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
hb = axes[0].hexbin(e.years_code_pro, np.log10(e.converted_comp_yearly), gridsize=40, cmap="Blues", mincnt=1, bins="log")
import statsmodels.nonparametric.smoothers_lowess as lw
lo_ = lw.lowess(np.log10(e.converted_comp_yearly), e.years_code_pro, frac=0.3, return_sorted=True)
axes[0].plot(lo_[:, 0], lo_[:, 1], color=PALETTE["orange"], lw=2.2, label="LOWESS trend")
axes[0].set_xlabel("Years coded professionally"); axes[0].set_ylabel("log₁₀(annual compensation, USD)"); axes[0].legend(loc="lower right")
axes[0].set_title("(a) Experience vs pay (all countries)"); fig.colorbar(hb, ax=axes[0], label="Respondents (log colour scale)", shrink=.85, pad=0.02)
for c, col in zip(["All", "USA", "Germany", "UK", "India"], [PALETTE["grey"], PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["orange"]]):
    dd = e if c == "All" else e[e.country == c]
    g = dd.groupby("exp_band").converted_comp_yearly
    med = g.median().reindex(bands); n_ = g.size().reindex(bands)
    ci = [boot_median_ci(dd.loc[dd.exp_band == b, "converted_comp_yearly"].values, n=300) if n_[b] >= 20 else (np.nan, np.nan) for b in bands]
    axes[1].errorbar(range(len(bands)), med, yerr=[med - [x[0] for x in ci], [x[1] for x in ci] - med], marker="o", ms=4, capsize=2, label=c, color=col, lw=1.6)
axes[1].set_xticks(range(len(bands))); axes[1].set_xticklabels(bands); axes[1].set_yscale("log"); axes[1].set_xlabel("Professional experience band (years)")
axes[1].set_ylabel("Median annual compensation (USD, log scale)"); axes[1].set_title("(b) Median pay by experience band"); axes[1].legend(ncol=2)
axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
savefig(fig, "fig11_experience_vs_pay", "Relationship between professional experience and pay: hexbin density with LOWESS trend (a) and median pay by experience band for selected countries with 95 % bootstrap CIs (b).", "O2")
plt.show()

# %%
xs = np.linspace(0, 40, 81)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
prem = 100 * (np.exp(b1 * xs + b2 * xs ** 2) - 1)
Vf = [(np.array([x, x * x]) @ V2 @ np.array([x, x * x])) ** .5 for x in xs]
lo_c = 100 * (np.exp(b1 * xs + b2 * xs ** 2 - 1.96 * np.array(Vf)) - 1); hi_c = 100 * (np.exp(b1 * xs + b2 * xs ** 2 + 1.96 * np.array(Vf)) - 1)
axes[0].plot(xs, prem, color=PALETTE["blue"], lw=2); axes[0].fill_between(xs, lo_c, hi_c, color=PALETTE["blue"], alpha=.2)
if not np.isnan(peak): axes[0].axvline(peak, color=PALETTE["red"], ls="--", lw=1); axes[0].text(peak + .5, prem.min() + 5, f"peak ≈ {peak:.0f} yrs", color=PALETTE["red"], fontsize=9)
axes[0].set_xlabel("Years coded professionally"); axes[0].set_ylabel("Pay premium vs 0 years (%)"); axes[0].set_title("(a) Cumulative premium (country-adjusted)")
xs2 = np.linspace(0, 30, 61)      # display range limited to where every country has data
for c, col in [("Pooled", PALETTE["grey"]), ("USA", PALETTE["blue"]), ("India", PALETTE["orange"]), ("Germany", PALETTE["green"]), ("UK", PALETTE["purple"])]:
    if c == "Pooled": bb1, bb2 = b1, b2
    else: bb1, bb2 = sep[c]["b1"], sep[c]["b2"]
    axes[1].plot(xs2, 100 * (np.exp(bb1 + 2 * bb2 * xs2) - 1), label=c, color=col, lw=2 if c in ("Pooled", "India") else 1.4)
axes[1].axhline(0, color="black", lw=.8); axes[1].set_xlabel("Years coded professionally"); axes[1].set_ylabel("Pay change per extra year (%)")
axes[1].set_title("(b) Marginal return to one more year"); axes[1].legend()
savefig(fig, "fig12_experience_marginal", "Quadratic log-linear model: cumulative premium with 95 % CI (a) and marginal percentage return per additional year, pooled and by country (b).", "O2")
plt.show()

# %% [markdown]
# ## O3 — Net effects of role, work mode, organisation size and education
# **Research question.** After holding country and experience constant, how much more or less do different roles,
# work arrangements, company sizes and education levels earn? **Method:** OLS on log-pay with HC3 errors; adjusted
# effect = $100(e^{\beta}-1)\%$ relative to a reference category; block importance by nested-model $\Delta R^{2}$ and
# partial $\eta^2_p=SS_{effect}/(SS_{effect}+SS_{resid})$. **Reference categories** (largest/most typical): Full-stack
# dev, In-person, 100 to 499 employees, Bachelor's.

# %%
o3 = cohort.dropna(subset=["years_code_pro", "dev_type", "remote_work", "org_short", "ed_short"]).copy()
o3 = o3[o3.org_short != "Don't know"]; o3 = o3[o3.ed_short != "Other"]
role_n = o3.dev_type.value_counts(); o3["dev_g"] = np.where(o3.dev_type.map(role_n) >= 100, o3.dev_type, "Other (rare roles)")
cn = o3.country.value_counts(); o3["country_g"] = np.where(o3.country.map(cn) >= 100, o3.country, "Other (small countries)")
REF = dict(dev='Full-stack dev', rem='In-person', org='100 to 499', ed="Bachelor's")
f_c = "C(country_g)"; f_x = "years_code_pro + years_pro_sq"
f_d = f'C(dev_g, Treatment(reference="{REF["dev"]}"))'; f_r = f'C(remote_work, Treatment(reference="{REF["rem"]}"))'
f_o = f'C(org_short, Treatment(reference="{REF["org"]}"))'; f_e = f'C(ed_short, Treatment(reference="{REF["ed"]}"))'
blocks = [("M0: country only", f_c), ("M1: + experience", f"{f_c} + {f_x}"), ("M2: + job role", f"{f_c} + {f_x} + {f_d}"),
          ("M3: + organisation size", f"{f_c} + {f_x} + {f_d} + {f_o}"), ("M4: + work arrangement", f"{f_c} + {f_x} + {f_d} + {f_o} + {f_r}"),
          ("M5: + education", f"{f_c} + {f_x} + {f_d} + {f_o} + {f_r} + {f_e}")]
fits, nest = [], []
for name, rhs in blocks:
    mm = smf.ols(f"log_comp ~ {rhs}", data=o3).fit(); fits.append(mm)
    prev = fits[-2] if len(fits) > 1 else None
    pF = float(mm.compare_f_test(prev)[1]) if prev is not None else float(mm.f_pvalue)
    nest.append(dict(model=name, R2=mm.rsquared, adj_R2=mm.rsquared_adj, delta_R2=mm.rsquared - (prev.rsquared if prev is not None else 0), p_F_block=pF, n=int(mm.nobs)))
nest = pd.DataFrame(nest); save_table(nest.round(4), "tbl_o3_nested_models", "Nested OLS models: incremental explanatory power of each predictor block", "O3")
full = smf.ols(f"log_comp ~ {blocks[-1][1]}", data=o3).fit(cov_type="HC3")
aov = anova_lm(smf.ols(f"log_comp ~ {blocks[-1][1]}", data=o3).fit(), typ=2)
ss_res = aov.loc["Residual", "sum_sq"]; aov["partial_eta2"] = aov["sum_sq"] / (aov["sum_sq"] + ss_res); aov = aov.drop(index="Residual")
aov.index = [re.sub(r"C\((\w+)(,.*)?\)", r"\1", i) for i in aov.index]
coef = pd.DataFrame({"coef": full.params, "se": full.bse, "p": full.pvalues, "lo": full.conf_int()[0], "hi": full.conf_int()[1]})
coef["pct"] = 100 * (np.exp(coef.coef) - 1); coef["pct_lo"] = 100 * (np.exp(coef.lo) - 1); coef["pct_hi"] = 100 * (np.exp(coef.hi) - 1)
coef["term"] = coef.index.map(lambda s: re.sub(r".*\[T\.(.*)\]", r"\1", s)); coef["var"] = coef.index.map(lambda s: re.sub(r"C\((\w+)[,)].*", r"\1", s) if s.startswith("C(") else s)
eff = coef[coef.index.str.startswith("C(") & ~coef["var"].isin(["country_g"])].copy()
save_table(eff[["var", "term", "pct", "pct_lo", "pct_hi", "p"]].round(3), "tbl_o3_adjusted_effects", "Adjusted percentage effects on annual compensation (HC3 robust 95 % CI)", "O3")
# raw vs adjusted for work arrangement
raw_remote = o3.groupby("remote_work").converted_comp_yearly.median()
R["O3"] = dict(n=int(full.nobs), nested=nest.round(4).to_dict("records"), partial_eta2=aov["partial_eta2"].round(4).to_dict(),
               final_R2=float(full.rsquared), final_adjR2=float(full.rsquared_adj), ref=REF,
               raw_remote_median=raw_remote.round(0).to_dict(), n_roles=int(o3.dev_g.nunique()),
               adj=eff[["var", "term", "pct", "pct_lo", "pct_hi", "p"]].round(3).to_dict("records"))
print(nest.round(4)); print(aov[["sum_sq", "partial_eta2"]].round(4)); eff[["var", "term", "pct", "pct_lo", "pct_hi", "p"]].round(2).sort_values("pct")

# %%
def forest(ax, d, title, color=PALETTE["blue"]):
    d = d.sort_values("pct"); y = np.arange(len(d))
    sig = (d.pct_lo > 0) | (d.pct_hi < 0)
    ax.hlines(y, d.pct_lo, d.pct_hi, color=np.where(sig, color, PALETTE["grey"]), lw=2)
    ax.scatter(d.pct, y, color=np.where(sig, color, PALETTE["grey"]), zorder=3, s=22)
    ax.axvline(0, color="black", lw=.8); ax.set_yticks(y); ax.set_yticklabels(d.term); ax.set_title(title); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Adjusted pay difference vs reference (%)")
fig, ax = plt.subplots(figsize=(7.4, 6.0))
forest(ax, eff[eff["var"] == "dev_g"], f"Job role (reference: {REF['dev']})")
savefig(fig, "fig13_role_effects", "Adjusted pay difference by job role relative to full-stack developers (country and experience held constant; 95 % HC3 CIs; grey = not significant at 5 %).", "O3")
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1, 1.2, 1]))
forest(axes[0], eff[eff["var"] == "remote_work"], f"(a) Work arrangement\n(ref: {REF['rem']})", PALETTE["green"])
d_o = eff[eff["var"] == "org_short"].copy(); d_o["o"] = d_o.term.map({k: i for i, k in enumerate(ORG_ORDER)}); d_o = d_o.sort_values("o")
y = np.arange(len(d_o)); sig = (d_o.pct_lo > 0) | (d_o.pct_hi < 0)
axes[1].hlines(y, d_o.pct_lo, d_o.pct_hi, color=np.where(sig, PALETTE["orange"], PALETTE["grey"]), lw=2); axes[1].scatter(d_o.pct, y, color=np.where(sig, PALETTE["orange"], PALETTE["grey"]), zorder=3, s=22)
axes[1].axvline(0, color="black", lw=.8); axes[1].set_ylim(-0.6, len(d_o) - 0.4); axes[1].set_yticks(y); axes[1].set_yticklabels(d_o.term); axes[1].set_title(f"(b) Organisation size\n(ref: {REF['org']})"); axes[1].set_xlabel("Adjusted pay difference vs reference (%)")
forest(axes[2], eff[eff["var"] == "ed_short"], f"(c) Education\n(ref: {REF['ed']})", PALETTE["purple"])
savefig(fig, "fig14_workmode_org_edu_effects", "Adjusted pay differences by work arrangement (a), organisation size (b) and education (c).", "O3")
plt.show()
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.bar(nest.model.str.replace(": ", "\n", regex=False), nest.R2, color=PALETTE["blue"], alpha=.35, label="Cumulative R²")
ax.bar(nest.model.str.replace(": ", "\n", regex=False), nest.delta_R2, color=PALETTE["orange"], label="Incremental ΔR²")
for i, (r_, d_) in enumerate(zip(nest.R2, nest.delta_R2)): ax.text(i, r_ + .012, f"R² = {r_:.3f}\n(Δ = +{d_:.3f})", ha="center", fontsize=8)
ax.set_ylabel("Share of variance in log(pay)"); ax.set_ylim(0, .72); ax.legend(loc="upper right"); ax.set_title("What explains pay? Nested-model variance decomposition")
plt.setp(ax.get_xticklabels(), fontsize=7.5)
savefig(fig, "fig15_variance_decomposition", "Cumulative R² and incremental ΔR² as predictor blocks are added to the OLS model for log-compensation.", "O3")
plt.show()
save_results()
