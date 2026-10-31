# %% [markdown]
# # Part A — Data inspection
# **Source:** Stack Overflow Annual Developer Survey 2024 (fielded May 2024, > 65 000 developers), distributed by
# Stack Overflow at <https://survey.stackoverflow.co/> under the Open Database Licence (ODbL).
# The *single-response, integer-coded* extract and its crosswalk were curated by Havisha Khurana for the
# **TidyTuesday** project (week 36 of 2024) and are stored in the public GitHub repository
# `rfordatascience/tidytuesday` (`data/2024/2024-09-03/`). Files in `data/raw/` are byte-for-byte copies
# (SHA-256 checksums are in `data/DATASET_DOCUMENTATION.md`).

# %%
df_raw, crosswalk, questions = load_raw()
print("Raw survey table:", df_raw.shape, "| crosswalk rows:", crosswalk.shape, "| questions:", questions.shape)
R["dataset"] = dict(rows=len(df_raw), cols=df_raw.shape[1], n_countries=int(df_raw.country.nunique()),
                    n_questions=len(questions), memory_mb=float(df_raw.memory_usage(deep=True).sum() / 1e6))
df_raw.head()

# %%
# Decode integer codes with the official crosswalk and add stateless engineered columns
df = add_features(decode(df_raw, crosswalk))
display(df[["response_id", "main_branch", "age", "country", "dev_type", "years_code_pro",
            "converted_comp_yearly", "ai_select"]].head())
print(df_raw.dtypes.value_counts())

# %% [markdown]
# ### Data dictionary
# `role` states how each column is used: **Target**, **Predictor**, **Analysis-only** (used in O4–O5 but excluded
# from models to prevent leakage) or **Excluded**.

# %%
DD = [
 ("response_id","Respondent identifier","ID","Excluded"),
 ("main_branch","Professional coding level of respondent","Nominal (5)","Predictor (AI task) / filter"),
 ("age","Age bracket","Ordinal (8)","Predictor (as age_mid)"),
 ("remote_work","Current work situation (remote / hybrid / in-person)","Nominal (3)","Predictor"),
 ("ed_level","Highest formal education completed","Ordinal (8)","Predictor"),
 ("years_code","Total years coded (incl. education); >50 coded as 51","Numeric","Predictor"),
 ("years_code_pro","Years coded professionally; >50 coded as 51","Numeric","Predictor"),
 ("dev_type","Best description of current job","Nominal (34)","Predictor"),
 ("org_size","Number of people in the organisation","Ordinal (10)","Predictor"),
 ("purchase_influence","Influence over technology purchases","Ordinal (3)","Predictor"),
 ("buildvs_buy","Customisation needed in last tool recommendation","Nominal (3)","Predictor"),
 ("country","Country of residence","Nominal (≈185)","Predictor"),
 ("currency","Day-to-day currency","Nominal","Excluded (duplicates country)"),
 ("comp_total","Total annual compensation in local currency","Numeric","Excluded (target twin)"),
 ("so_visit_freq","Stack Overflow visit frequency","Ordinal (5)","Predictor"),
 ("so_account","Has a Stack Overflow account","Nominal (3)","Predictor"),
 ("so_part_freq","Stack Overflow Q&A participation frequency","Ordinal (6)","Predictor"),
 ("so_comm","Feels part of the Stack Overflow community","Ordinal (6)","Predictor"),
 ("ai_select","Currently uses AI tools in development","Ordinal (3)","Target (O7) / Predictor (O6)"),
 ("ai_sent","Stance on AI tools in workflow","Ordinal (6)","Analysis-only (conditional question)"),
 ("ai_acc","Trust in accuracy of AI output","Ordinal (5)","Analysis-only (conditional question)"),
 ("ai_complex","AI performance on complex tasks","Ordinal (5)","Analysis-only (conditional question)"),
 ("ai_threat","Believes AI is a threat to own job","Nominal (3)","Analysis-only (conditional question)"),
 ("survey_length","Opinion on survey length","Nominal (3)","Excluded"),
 ("survey_ease","Opinion on survey difficulty","Nominal (3)","Excluded"),
 ("converted_comp_yearly","Annual compensation converted to USD","Numeric","Target (O6)"),
 ("r_used","Used the R language last year (flag)","Binary","Predictor"),
 ("r_want_to_use","Wants to use R next year (flag)","Binary","Excluded (not relevant)"),
]
data_dict = pd.DataFrame(DD, columns=["variable", "description", "type", "role"])
data_dict["missing_%"] = data_dict["variable"].map(df_raw.isna().mean() * 100).round(1)
save_table(data_dict, "tbl_data_dictionary", "Data dictionary of the 28 raw survey variables", "EDA")
data_dict

# %% [markdown]
# # Part B — Data-quality assessment
# We check (i) missing values, (ii) duplicates, (iii) logically inconsistent values and (iv) outliers —
# in that order — and document every cleaning decision.

# %%
mt = missing_table(df_raw.drop(columns="response_id"))
save_table(mt.reset_index().rename(columns={"index": "variable"}), "tbl_missing", "Missing values per variable", "EDA")
R["missing"] = dict(worst=mt.head(6)[["missing_pct"]].round(1).to_dict()["missing_pct"],
                    overall_cell_pct=float(df_raw.drop(columns="response_id").isna().mean().mean() * 100))
fig, ax = plt.subplots(figsize=(7.2, 6.2))
m = mt.sort_values("missing_pct")
ax.barh(m.index, m["missing_pct"], color=[PALETTE["red"] if v > 30 else PALETTE["blue"] for v in m["missing_pct"]])
for i, v in enumerate(m["missing_pct"]):
    ax.text(v + 0.8, i, f"{v:.1f}%", va="center", fontsize=8)
ax.set_xlabel("Missing responses (% of 65,437 respondents)"); ax.set_xlim(0, 75)
ax.set_title("Missing values by survey question")
savefig(fig, "fig01_missingness", "Percentage of missing responses per variable (red = more than 30 % missing).", "EDA")
plt.show()
mt.head(8)

# %% [markdown]
# **Interpretation.** Missingness is *structural*, not random: AI follow-up questions were only shown to people
# who use or plan to use AI; compensation is optional; and 10 000+ people typed a salary but the currency could not
# be converted (see inconsistency table). Therefore we **do not impute the target** — we model only rows where
# it is observed — and we impute *predictors* inside the CV pipeline using training-fold statistics only.

# %%
# Duplicates: exact duplicates ignoring the ID, and how complete those rows are
body = df_raw.drop(columns="response_id")
dup_mask = body.duplicated(keep=False)
dup_info = dict(n_duplicate_rows_incl_first=int(dup_mask.sum()),
                n_removable=int(body.duplicated().sum()),
                mean_answered_questions_dups=float(body[dup_mask].notna().sum(axis=1).mean()),
                mean_answered_questions_all=float(body.notna().sum(axis=1).mean()),
                id_unique=bool(df_raw.response_id.is_unique))
R["duplicates"] = dup_info
print(json.dumps(dup_info, indent=1))
print("\nExample of a flagged 'duplicate' group (all answers identical, almost all blank):")
display(df_raw[dup_mask].sort_values(list(body.columns[:6])).head(4))

# %% [markdown]
# **Decision.** The flagged "duplicates" answered only a handful of questions on average (see the printed mean
# versus the all-respondent mean) — they are *abandoned, nearly empty* surveys that happen to coincide, not double submissions. They are harmless for the
# modelling cohorts (which require a salary or an AI answer). We therefore **only remove exact duplicates inside the
# modelling cohort** (done in the cohort builder) rather than blindly dropping rows from the raw table.

# %%
incons = inconsistency_report(df)
save_table(incons, "tbl_inconsistencies", "Logical inconsistencies and suspicious values", "EDA")
R["inconsistencies"] = dict(zip(incons.check, incons.records_affected))
incons

# %%
# Outliers in compensation: three classical rules, then the effect of a log transform
comp = df["converted_comp_yearly"].dropna()
oc_raw = outlier_counts(comp); oc_raw.insert(0, "scale", "USD")
oc_log = outlier_counts(np.log(comp[comp > 0])); oc_log.insert(0, "scale", "log(USD)")
oc = pd.concat([oc_raw, oc_log], ignore_index=True)
save_table(oc, "tbl_outlier_rules", "Outliers flagged in annual compensation by three rules, raw vs log scale", "EDA")
R["outliers"] = oc.to_dict("records")
R["comp_stats_raw"] = dict(n=int(comp.size), mean=float(comp.mean()), median=float(comp.median()), sd=float(comp.std()),
                           max=float(comp.max()), skew=float(stats.skew(comp)), kurt=float(stats.kurtosis(comp)),
                           skew_log=float(stats.skew(np.log(comp[comp > 0]))))
oc

# %%
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6))
axes[0].hist(comp[comp <= comp.quantile(.99)], bins=60, color=PALETTE["blue"])
axes[0].set_title("(a) Raw USD (up to 99th pct)"); axes[0].set_xlabel("Annual compensation (USD)"); axes[0].set_ylabel("Respondents")
axes[1].hist(np.log10(comp[comp > 0]), bins=60, color=PALETTE["green"])
axes[1].set_title("(b) log₁₀ scale"); axes[1].set_xlabel("log₁₀(annual compensation in USD)")
sns.boxplot(x=np.log10(comp[comp > 0]), ax=axes[2], color=PALETTE["orange"], fliersize=2)
axes[2].set_title("(c) Box plot, log₁₀ scale"); axes[2].set_xlabel("log₁₀(annual compensation in USD)")
savefig(fig, "fig02_comp_distribution", "Distribution of annual compensation: heavily right-skewed in USD (a), far less skewed after a log transform (b), with extreme tails visible in (c).", "EDA")
plt.show()
print(f"Skewness raw = {R['comp_stats_raw']['skew']:.2f} ; after log = {R['comp_stats_raw']['skew_log']:.2f}")

# %% [markdown]
# **Cleaning policy for the salary cohort** (documented design decision):
# 1. keep **professional developers** (`main_branch = 1`) with an observed converted salary and a known country;
# 2. absolute plausibility bounds **USD 1 000 – 1 000 000** (below = probably monthly/ typo entries; above = unverifiable);
# 3. **within-country 3 × IQR fences on log(salary)** for countries with ≥ 50 salaries (a *country-aware* rule — a global
#    fence would wrongly delete every salary from low-income countries);
# 4. logically impossible experience values are **blanked, not dropped** (the row's other information is valid);
# 5. exact duplicate rows inside the cohort are removed.

# %%
cohort, attrition = build_salary_cohort(df)
dups_in_cohort = int(cohort.drop(columns="response_id").duplicated().sum())
cohort = cohort.drop_duplicates(subset=[c for c in cohort.columns if c != "response_id"]).reset_index(drop=True)
attrition.loc[len(attrition)] = ["Exact duplicate rows removed", len(cohort), dups_in_cohort]
save_table(attrition, "tbl_cohort_attrition", "Row attrition at each cleaning step for the salary cohort", "EDA")
R["cohort"] = dict(n=len(cohort), n_countries=int(cohort.country.nunique()), attrition=attrition.to_dict("records"),
                   n_india=int((cohort.country == "India").sum()), n_usa=int((cohort.country == "USA").sum()))
cohort.to_csv(DATA_PROC / "salary_cohort_clean.csv", index=False)
attrition

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.6))
a = attrition.iloc[1:].reset_index(drop=True)
ax.barh(a["step"][::-1], a["rows"][::-1], color=PALETTE["blue"])
for i, v in enumerate(a["rows"][::-1]): ax.text(v + 200, i, f"{v:,}", va="center", fontsize=8)
ax.set_xlabel("Rows remaining"); ax.set_xlim(0, 58000); ax.set_title("Salary-cohort construction (rows retained)")
savefig(fig, "fig03_cohort_attrition", "Rows retained at each step of salary-cohort construction.", "EDA")
plt.show()

# %%
# AI cohort (all respondents with a known AI-adoption answer)
ai_cohort = build_ai_cohort(df)
ai_cohort.to_csv(DATA_PROC / "ai_cohort_clean.csv", index=False)
R["ai_cohort"] = dict(n=len(ai_cohort), adoption_rate=float(ai_cohort.uses_ai.mean() * 100),
                      dist=df.ai_select.value_counts(dropna=False).rename(str).to_dict())
print(R["ai_cohort"])
print("Skewness of log(salary) in cohort:", round(float(stats.skew(cohort.log_comp)), 3))

# %% [markdown]
# # Part C — Exploratory data analysis (EDA)

# %%
desc_num = cohort[["years_code", "years_code_pro", "converted_comp_yearly", "log_comp", "age_mid", "org_mid"]].describe().T
desc_num["skew"] = cohort[desc_num.index].skew()
save_table(desc_num.reset_index().rename(columns={"index": "variable"}), "tbl_descriptives",
           "Descriptive statistics of numeric variables in the salary cohort", "EDA")
R["descriptives"] = desc_num.round(3).to_dict("index")
desc_num

# %%
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4))
order = [a for a in AGE_ORDER if a in cohort.age.unique()]
sns.countplot(y="age", data=cohort, order=order, ax=axes[0, 0], color=PALETTE["blue"]); axes[0, 0].set_title("(a) Age bracket"); axes[0, 0].set_ylabel("")
sns.histplot(cohort["years_code_pro"].dropna(), bins=30, ax=axes[0, 1], color=PALETTE["green"]); axes[0, 1].set_title("(b) Professional experience"); axes[0, 1].set_xlabel("Years coded professionally")
top_dev = cohort.dev_type.value_counts().head(8).index
sns.countplot(y="dev_type", data=cohort[cohort.dev_type.isin(top_dev)], order=top_dev, ax=axes[0, 2], color=PALETTE["orange"]); axes[0, 2].set_title("(c) Top 8 job roles"); axes[0, 2].set_ylabel("")
sns.countplot(y="remote_work", data=cohort, ax=axes[1, 0], color=PALETTE["purple"], order=cohort.remote_work.value_counts().index); axes[1, 0].set_title("(d) Work arrangement"); axes[1, 0].set_ylabel("")
sns.countplot(y="org_short", data=cohort, order=[o for o in ORG_ORDER if o in set(cohort.org_short.dropna())], ax=axes[1, 1], color=PALETTE["gold"]); axes[1, 1].set_title("(e) Organisation size"); axes[1, 1].set_ylabel("")
topc = cohort.country.value_counts().head(10)
axes[1, 2].barh(topc.index[::-1], topc.values[::-1], color=PALETTE["red"]); axes[1, 2].set_title("(f) Top 10 countries")
for ax in axes.ravel(): ax.set_xlabel("Respondents" if ax.get_xlabel() in ("", "count") else ax.get_xlabel())
savefig(fig, "fig04_univariate_profile", f"Univariate profile of the {len(cohort):,} professional developers in the salary cohort.", "EDA")
plt.show()

# %%
# Group comparison overview (EDA): median compensation (USD) by categorical groups
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8))
g = cohort.groupby("remote_work").converted_comp_yearly.median().sort_values(); axes[0].barh(g.index, g.values, color=PALETTE["blue"]); axes[0].set_title("(a) Work arrangement")
g = cohort.groupby("org_short").converted_comp_yearly.median().reindex([o for o in ORG_ORDER if o in set(cohort.org_short.dropna())]); axes[1].barh(g.index[::-1], g.values[::-1], color=PALETTE["orange"]); axes[1].set_title("(b) Organisation size")
g = cohort.groupby("ed_short").converted_comp_yearly.median().sort_values(); axes[2].barh(g.index, g.values, color=PALETTE["green"]); axes[2].set_title("(c) Education")
for ax in axes: ax.set_xlabel("Median annual compensation (USD)"); ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
savefig(fig, "fig05_group_medians", "Raw (unadjusted) median compensation by work arrangement, organisation size and education. Country mix confounds these gaps (see O3).", "EDA")
plt.show()

# %%
# Correlation structure
num_cols_eda = ["years_code", "years_code_pro", "age_mid", "org_log", "ed_rank", "so_visit_rank", "exp_gap", "pro_share", "log_comp"]
corr = cohort[num_cols_eda].corr(method="spearman")
fig, ax = plt.subplots(figsize=(7.2, 6.0))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, square=True, linewidths=.5, cbar_kws={"shrink": .8})
ax.set_title("Spearman correlation matrix (salary cohort)")
savefig(fig, "fig06_spearman_corr", "Spearman rank-correlation matrix of numeric and ordinal variables including log-compensation.", "EDA")
plt.show()
R["corr_with_logcomp"] = corr["log_comp"].drop("log_comp").round(3).to_dict()

# %%
# Association among categorical variables (Cramér's V, bias-corrected)
cat_cols_eda = ["country", "dev_type", "remote_work", "org_short", "ed_short", "age", "purchase_influence", "so_part_freq", "ai_select"]
cv = pd.DataFrame(index=cat_cols_eda, columns=cat_cols_eda, dtype=float)
sub = cohort[cat_cols_eda].dropna()
for a in cat_cols_eda:
    for b in cat_cols_eda:
        cv.loc[a, b] = 1.0 if a == b else cramers_v(sub[a], sub[b])
fig, ax = plt.subplots(figsize=(7.2, 6.0))
sns.heatmap(cv, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax, square=True, linewidths=.5, cbar_kws={"shrink": .8})
ax.set_title("Cramér's V between categorical variables")
savefig(fig, "fig07_cramers_v", "Bias-corrected Cramér's V between categorical variables (0 = independent, 1 = perfectly associated).", "EDA")
plt.show()

# %%
# Multicollinearity: Variance Inflation Factors of the numeric predictors
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_cols = ["years_code", "years_code_pro", "age_mid", "org_log", "ed_rank", "so_visit_rank"]
Xv = cohort[vif_cols].dropna(); Xv = (Xv - Xv.mean()) / Xv.std(); Xv.insert(0, "const", 1.0)
vif = pd.DataFrame({"variable": vif_cols, "VIF": [variance_inflation_factor(Xv.values, i + 1) for i in range(len(vif_cols))]})
save_table(vif, "tbl_vif", "Variance inflation factors of numeric predictors", "EDA")
R["vif"] = vif.round(2).to_dict("records"); vif

# %%
# Normality & homogeneity checks that justify the statistical tests chosen later
samp = cohort.log_comp.sample(5000, random_state=RANDOM_STATE)
k2 = stats.normaltest(samp)
top_c = cohort.country.value_counts().head(8).index
lev = stats.levene(*[cohort.loc[cohort.country == c, "log_comp"] for c in top_c])
R["assumptions"] = dict(dagostino_k2=float(k2.statistic), dagostino_p=float(k2.pvalue),
                        skew_logcomp=float(stats.skew(cohort.log_comp)), kurt_logcomp=float(stats.kurtosis(cohort.log_comp)),
                        levene_W=float(lev.statistic), levene_p=float(lev.pvalue))
print(R["assumptions"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
stats.probplot(samp, dist="norm", plot=axes[0]); axes[0].set_title("(a) Normal Q–Q plot of log(salary)")
axes[0].get_lines()[0].set_markerfacecolor(PALETTE["blue"]); axes[0].get_lines()[0].set_markersize(2.5)
sns.kdeplot(data=cohort[cohort.country.isin(top_c)], x="log_comp", hue="country", ax=axes[1], common_norm=False, palette=CAT, linewidth=1.4)
axes[1].set_title("(b) log(salary) density by country"); axes[1].set_xlabel("log(annual compensation, USD)")
savefig(fig, "fig08_normality_checks", "Q–Q plot of log-salary (a) and country-wise densities (b): log-salary is left-skewed and heavy-tailed rather than normal, with unequal location and spread across countries.", "EDA")
plt.show()

# %% [markdown]
# **Interpretation / decisions from EDA**
# * Log-salary is roughly symmetric but heavy-tailed (excess kurtosis) and its variance differs across countries →
#   we use **rank-based tests** (Kruskal–Wallis, Mann–Whitney, Spearman) and **robust (HC3) standard errors** for OLS,
#   and model `log(salary)` rather than raw USD.
# * `years_code`, `years_code_pro` and `age_mid` are strongly inter-correlated: VIFs of `years_code` and `years_code_pro`
#   are roughly 7–9 (above the conventional 5, below 10). Tree models are immune and Ridge regularisation handles it for
#   prediction; for the *inferential* OLS models (O2, O3) we deliberately use **only `years_code_pro`** so coefficients
#   stay interpretable. `exp_gap` / `pro_share` are added to expose *different* information from the same columns.
# * `country` and `dev_type` are high-cardinality → **cross-fitted target encoding** (tree models) or one-hot with rare
#   levels pooled (Ridge).

# %%
# Feature-engineering register (justification table used in the report)
fe = pd.DataFrame([
 ("age_mid", "Midpoint of age bracket", "Turns an ordinal bracket into a numeric scale usable by all models"),
 ("ed_rank", "Education level -> 1…7", "Preserves natural order of qualifications"),
 ("org_log", "log10(midpoint of organisation size)", "Company size spans 4 orders of magnitude; log makes effect approx. linear"),
 ("so_visit_rank", "Stack Overflow visit frequency -> 1…5", "Proxy for engagement / learning intensity"),
 ("exp_gap", "years_code − years_code_pro", "Years of coding before first paid job (education / hobby head-start)"),
 ("pro_share", "years_code_pro / years_code", "Share of the coding life spent professionally (career stage)"),
 ("years_pro_sq", "years_code_pro²", "Captures diminishing returns of experience in linear models"),
 ("Missing indicators", "SimpleImputer(add_indicator=True)", "Non-response itself can be informative"),
 ("Target-encoded country / dev_type", "Cross-fitted mean of target by category", "Compact numeric representation of high-cardinality variables, leakage-safe"),
], columns=["feature", "definition", "justification"])
save_table(fe, "tbl_feature_engineering", "Engineered features and their justification", "EDA")
fe
