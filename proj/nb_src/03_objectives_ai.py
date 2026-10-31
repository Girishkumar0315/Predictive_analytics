# %% [markdown]
# ## O4 — Who uses AI tools? The adoption landscape
# **Research question.** Which developer groups (age, experience, role, country, company size, Stack Overflow
# engagement) have adopted AI tools, and is India different from the rest of the world?
# **Variables:** `uses_ai` (1 = currently uses AI tools; 0 = does not), plus group factors. **Methods:** χ² test of
# independence, bias-corrected Cramér's V (effect size), Wilson 95 % CIs, two-proportion z-test.
# **Success:** adoption % with CI per group; ranking of factors by V; India-vs-world gap with p-value.

# %%
ai = ai_cohort.copy()
ai["exp_band"] = pd.Categorical(ai.exp_band, categories=bands, ordered=True)
ai = ai[ai.age != "Prefer not to say"].copy() if False else ai       # keep all rows; filter inside each factor
overall = ai.uses_ai.mean() * 100
FACTORS = {"age": "Age bracket", "exp_band": "Professional experience", "main_branch": "Coding relationship", "dev_type": "Job role",
           "country": "Country", "org_short": "Organisation size", "remote_work": "Work arrangement", "so_visit_freq": "Stack Overflow visits",
           "ed_short": "Education", "purchase_influence": "Purchasing influence"}
rows = []
for f, label in FACTORS.items():
    t = ai[[f, "uses_ai"]].dropna()
    if f == "age": t = t[t.age != "Prefer not to say"]
    if f in ("dev_type", "country"):                       # keep only groups with >= 300 respondents for stable estimates
        t = t[t[f].map(t[f].value_counts()) >= 300]
    r_ = chi2_test(t[f].astype(str), t["uses_ai"].astype(int)); r_.update(factor=f, label=label, k=t[f].nunique())
    rows.append(r_)
ftab = pd.DataFrame(rows).sort_values("cramers_v", ascending=False)
save_table(ftab[["label", "k", "n", "chi2", "dof", "p", "cramers_v"]].round(4), "tbl_o4_factor_tests", "Chi-square tests of AI adoption against each factor, ranked by Cramér's V", "O4")
def rate_table(f, min_n=300, drop=()):
    t = ai[[f, "uses_ai"]].dropna(); t = t[~t[f].isin(drop)]
    g = t.groupby(f, observed=True).uses_ai.agg(["sum", "count"]); g = g[g["count"] >= min_n]
    g["rate"] = g["sum"] / g["count"] * 100
    lo, hi = wilson_ci(g["sum"], g["count"]); g["lo"], g["hi"] = lo * 100, hi * 100
    return g.reset_index()
# India vs rest and India vs USA
ind_ai = ai[ai.country == "India"].uses_ai; rest_ai = ai[ai.country != "India"].uses_ai; usa_ai = ai[ai.country == "USA"].uses_ai
z_rest = two_prop_ztest(ind_ai.sum(), len(ind_ai), rest_ai.sum(), len(rest_ai)); z_usa = two_prop_ztest(ind_ai.sum(), len(ind_ai), usa_ai.sum(), len(usa_ai))
rt = {f: rate_table(f, drop=("Prefer not to say",)) for f in FACTORS}
R["O4"] = dict(n=len(ai), overall_rate=float(overall), factors=ftab[["factor", "label", "k", "n", "chi2", "dof", "p", "cramers_v"]].round(4).to_dict("records"),
               india=dict(n=int(len(ind_ai)), rate=float(ind_ai.mean() * 100), vs_rest=z_rest, vs_usa=z_usa, usa_rate=float(usa_ai.mean() * 100), rest_rate=float(rest_ai.mean() * 100)),
               by_age=rt["age"].round(2).to_dict("records"), by_exp=rt["exp_band"].round(2).to_dict("records"), by_branch=rt["main_branch"].round(2).to_dict("records"),
               by_so=rt["so_visit_freq"].round(2).to_dict("records"), by_org=rt["org_short"].round(2).to_dict("records"), by_remote=rt["remote_work"].round(2).to_dict("records"),
               by_role=rt["dev_type"].sort_values("rate", ascending=False).round(2).to_dict("records"), by_country=rt["country"].sort_values("rate", ascending=False).round(2).to_dict("records"),
               plan_share=float((ai_cohort.ai_select == "No, but I plan to soon").mean() * 100), never_share=float((ai_cohort.ai_select == "No, and I don't plan to").mean() * 100))
print(f"overall adoption = {overall:.1f}%  | India = {ind_ai.mean()*100:.1f}% (n={len(ind_ai)}) vs rest {rest_ai.mean()*100:.1f}% ; z-test p = {fmt_p(z_rest['p'])}")
ftab[["label", "k", "n", "chi2", "p", "cramers_v"]].round(4)

# %%
# Figure: 100 % stacked bars of the full 3-level answer by age, experience, organisation size and Stack Overflow frequency
LV = ["Yes", "No, but I plan to soon", "No, and I don't plan to"]; LVC = [PALETTE["blue"], PALETTE["gold"], PALETTE["grey"]]
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0))
def stacked(ax, f, order, title, drop=()):
    t = ai[ai[f].notna() & ~ai[f].isin(drop)]; ct = pd.crosstab(t[f], t.ai_select, normalize="index").reindex(order)[LV] * 100
    left = np.zeros(len(ct))
    for lv, c in zip(LV, LVC):
        ax.barh(range(len(ct)), ct[lv], left=left, color=c, label=lv if lv not in ax.get_legend_handles_labels()[1] else None, height=.72)
        if lv == "Yes":
            for i, v in enumerate(ct[lv]): ax.text(v / 2, i, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += ct[lv].values
    ax.set_yticks(range(len(ct))); ax.set_yticklabels([f"{o}" for o in ct.index]); ax.set_xlim(0, 100); ax.set_xlabel("% of respondents"); ax.set_title(title); ax.invert_yaxis()
stacked(axes[0, 0], "age", [a for a in AGE_ORDER if a in set(ai.age.dropna())], "(a) Age bracket", drop=("Prefer not to say",))
stacked(axes[0, 1], "exp_band", bands, "(b) Years of professional experience")
stacked(axes[1, 0], "org_short", [o for o in ORG_ORDER if o in set(ai.org_short.dropna())], "(c) Organisation size")
sov = ["Less than once per month or monthly", "A few times per month or weekly", "A few times per week", "Daily or almost daily", "Multiple times per day"]
stacked(axes[1, 1], "so_visit_freq", sov, "(d) Stack Overflow visit frequency")
axes[1, 1].set_yticklabels(["< monthly", "weekly/monthly", "few × week", "daily", "multiple × day"])
h, l = axes[0, 0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.04))
savefig(fig, "fig16_ai_adoption_factors", "AI-tool adoption in the development process by age (a), experience (b), organisation size (c) and Stack Overflow visit frequency (d); labels show % currently using AI.", "O4")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
for ax, f, title in [(axes[0], "country", "(a) Country (n ≥ 300)"), (axes[1], "dev_type", "(b) Job role (n ≥ 300)")]:
    t = rt[f].sort_values("rate"); y = np.arange(len(t))
    col = [PALETTE["orange"] if c == "India" else PALETTE["blue"] for c in t[f]]
    ax.hlines(y, t.lo, t.hi, color=col, lw=2); ax.scatter(t.rate, y, color=col, zorder=3, s=26)
    ax.axvline(overall, color=PALETTE["red"], ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(t[f]); ax.set_xlabel("% currently using AI tools (95 % Wilson CI)"); ax.set_title(title)
axes[0].text(overall + .4, 0.2, f"overall {overall:.1f}%", color=PALETTE["red"], fontsize=8)
savefig(fig, "fig17_ai_adoption_country_role", "AI adoption rate with 95 % Wilson CIs by country (a) and job role (b); dashed line = overall adoption; India in orange.", "O4")
plt.show()

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ft = ftab.sort_values("cramers_v"); ax.barh(ft.label, ft.cramers_v, color=PALETTE["purple"])
for i, (v, p_) in enumerate(zip(ft.cramers_v, ft.p)): ax.text(v + .003, i, f"V = {v:.3f}" + ("" if p_ < .001 else f"  (p = {p_:.2f})"), va="center", fontsize=8)
ax.set_xlabel("Bias-corrected Cramér's V with AI adoption"); ax.set_xlim(0, ft.cramers_v.max() * 1.3); ax.set_title("Which factors are most associated with AI adoption?")
savefig(fig, "fig18_ai_cramers_v", "Strength of association (Cramér's V) between AI adoption and each descriptive factor.", "O4")
plt.show()

# heatmap: role x age
top_roles = ai.dev_type.value_counts().head(10).index; ages = [a for a in AGE_ORDER if a in ("18-24", "25-34", "35-44", "45-54", "55-64")]
hm = ai[ai.dev_type.isin(top_roles) & ai.age.isin(ages)].pivot_table(index="dev_type", columns="age", values="uses_ai", aggfunc="mean") * 100
cnt = ai[ai.dev_type.isin(top_roles) & ai.age.isin(ages)].pivot_table(index="dev_type", columns="age", values="uses_ai", aggfunc="count")
hm = hm.where(cnt >= 30).reindex(index=hm.mean(axis=1).sort_values().index, columns=ages)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
sns.heatmap(hm, annot=True, fmt=".0f", cmap="Blues", vmin=30, vmax=85, ax=ax, linewidths=.5, cbar_kws={"label": "% using AI tools"})
ax.set_xlabel("Age bracket"); ax.set_ylabel(""); ax.set_title("AI adoption (%) by role and age (cells with n ≥ 30)")
savefig(fig, "fig19_ai_role_age_heatmap", "AI adoption by job role and age bracket (blank = fewer than 30 respondents).", "O4")
plt.show(); save_results()

# %% [markdown]
# ## O5 — AI sentiment, trust and job-threat perception
# **Research question.** Do favourability, trust in AI accuracy, belief that AI handles complex tasks, and fear of
# job loss differ by career stage, role and country? **Data note.** These four questions were asked **only** to
# people who use (or plan to use) AI tools, so inference is limited to that population — and, as proven below,
# they must not be used as predictors of adoption (leakage). **Methods:** ordinal scores (1–5), Spearman ρ with
# CI, Kruskal–Wallis with ε², χ² + Cramér's V for the categorical threat answer, Mann–Whitney for India vs rest.

# %%
# --- Proof of structural conditionality (the leakage argument) ---
cond = pd.DataFrame({q: df.groupby(df.ai_select.fillna("(unanswered)"))[q].apply(lambda s: s.notna().mean() * 100) for q in ["ai_sent", "ai_acc", "ai_complex", "ai_threat"]}).round(1)
save_table(cond.reset_index().rename(columns={"ai_select": "ai_select answer"}), "tbl_o5_conditionality", "Share of respondents answering each AI follow-up question, by their AI-use answer (percent)", "O5")
R["O5_conditionality"] = cond.to_dict("index"); cond

# %%
att = df[df.ai_select.isin(["Yes", "No, but I plan to soon"])].copy()
att["sent"] = map_prefix(att.ai_sent, SENT_SCORE); att["trust"] = map_prefix(att.ai_acc, TRUST_SCORE); att["complex_"] = map_prefix(att.ai_complex, COMPLEX_SCORE)
att["threat"] = att.ai_threat.map({"Yes": "Yes", "No": "No", "I'm not sure": "Not sure"})
att["threat_yes"] = np.where(att.threat.isna(), np.nan, (att.threat == "Yes").astype(float))
att["exp_band"] = pd.Categorical(att.exp_band, categories=bands, ordered=True)
dist = {}
for col, lab in [("ai_sent", "Sentiment"), ("ai_acc", "Trust in accuracy"), ("ai_complex", "Complex tasks"), ("ai_threat", "Threat to job")]:
    vc = att[col].value_counts(normalize=True) * 100; dist[lab] = vc.round(1).to_dict()
users = att[att.ai_select == "Yes"]
spr = {}
for v, lab in [("sent", "Favourability"), ("trust", "Trust in AI accuracy"), ("complex_", "AI on complex tasks"), ("threat_yes", "Sees AI as threat (yes=1)")]:
    spr[lab] = dict(exp=spearman(att.years_code_pro, att[v]), age=spearman(att.age_mid, att[v]))
kw = {}
for v, lab in [("sent", "Favourability"), ("trust", "Trust")]:
    gs = [g[v].dropna().values for _, g in att.groupby("exp_band", observed=True) if g[v].notna().sum() > 20]
    Hh, ph = stats.kruskal(*gs); kw[lab] = dict(H=float(Hh), p=float(ph), eps2=epsilon_squared_kruskal(Hh, sum(len(g) for g in gs)))
thr_tab = {}
for f in ["exp_band", "age", "dev_type", "country"]:
    t = att[[f, "threat_yes", "threat"]].dropna()
    if f in ("dev_type", "country"): t = t[t[f].map(t[f].value_counts()) >= 300]
    if f == "age": t = t[t.age != "Prefer not to say"]
    r_ = chi2_test(t[f].astype(str), t["threat"]); thr_tab[f] = r_
ind_t = att[att.country == "India"]; rest_t = att[att.country != "India"]
mw_india = {v: dict(india_mean=float(ind_t[v].mean()), rest_mean=float(rest_t[v].mean()), p=float(stats.mannwhitneyu(ind_t[v].dropna(), rest_t[v].dropna()).pvalue), n=int(ind_t[v].notna().sum())) for v in ["sent", "trust", "complex_"]}
z_thr = two_prop_ztest(ind_t.threat_yes.sum(), ind_t.threat_yes.notna().sum(), rest_t.threat_yes.sum(), rest_t.threat_yes.notna().sum())
um = stats.mannwhitneyu(users.sent.dropna(), att[att.ai_select != "Yes"].sent.dropna())
trust_pos = float(att.trust.dropna().ge(4).mean() * 100); trust_neg = float(att.trust.dropna().le(2).mean() * 100)
R["O5"] = dict(n_att=len(att), n_users=len(users), dist=dist, spearman=spr, kruskal=kw, threat_tests=thr_tab, india=mw_india, india_threat=z_thr,
               users_vs_planners_sent=dict(users_mean=float(users.sent.mean()), planners_mean=float(att[att.ai_select != "Yes"].sent.mean()), p=float(um.pvalue)),
               trust_positive_pct=trust_pos, trust_negative_pct=trust_neg, threat_yes_pct=float(att.threat_yes.mean() * 100),
               fav_pct=float(att.sent.dropna().ge(4).mean() * 100), complex_good_pct=float(att.complex_.dropna().ge(4).mean() * 100), complex_bad_pct=float(att.complex_.dropna().le(2).mean() * 100),
               trust_complex_rho=spearman(att.trust, att.complex_), trust_sent_rho=spearman(att.trust, att.sent))
for k, v in spr.items(): print(f"{k:28s} rho(exp) = {v['exp']['rho']:+.3f}  [{v['exp']['ci_low']:+.3f},{v['exp']['ci_high']:+.3f}] n={v['exp']['n']}")
print({k: round(v, 1) for k, v in R["O5"].items() if isinstance(v, float)})

# %%
fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.6), sharex=False)
orders = {"ai_sent": ["Very unfavorable", "Unfavorable", "Indifferent", "Unsure", "Favorable", "Very favorable"],
          "ai_acc": ["Highly distrust", "Somewhat distrust", "Neither trust nor distrust", "Somewhat trust", "Highly trust"],
          "ai_complex": ["Very poor at handling complex tasks", "Bad at handling complex tasks", "Neither good or bad at handling complex tasks", "Good, but not great at handling complex tasks", "Very well at handling complex tasks"],
          "ai_threat": ["No", "I'm not sure", "Yes"]}
short = {"Neither trust nor distrust": "Neutral", "Very poor at handling complex tasks": "Very poor", "Bad at handling complex tasks": "Bad",
         "Neither good or bad at handling complex tasks": "Neutral", "Good, but not great at handling complex tasks": "Good, not great", "Very well at handling complex tasks": "Very well"}
for ax, (col, title), c in zip(axes, [("ai_sent", "(a) Sentiment"), ("ai_acc", "(b) Trust in accuracy"), ("ai_complex", "(c) Complex tasks"), ("ai_threat", "(d) Threat to my job")], [PALETTE["blue"], PALETTE["green"], PALETTE["purple"], PALETTE["red"]]):
    vc = att[col].value_counts(normalize=True).reindex(orders[col]).fillna(0) * 100
    ax.barh([short.get(k, k) for k in vc.index], vc.values, color=c)
    for i, v in enumerate(vc.values): ax.text(v + .6, i, f"{v:.0f}%", va="center", fontsize=8)
    ax.set_xlim(0, vc.max() * 1.25); ax.set_title(title); ax.set_xlabel("% of respondents answering"); ax.invert_yaxis()
savefig(fig, "fig20_ai_attitudes_distribution", "Distribution of attitudes towards AI among respondents who use or plan to use AI tools.", "O5")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
for ax, v, lab, lv, title in [(axes[0], "ai_acc", "trust", orders["ai_acc"], "(a) Trust in AI accuracy"), (axes[1], "ai_sent", "sent", [o for o in orders["ai_sent"] if o != "Unsure"], "(b) Sentiment towards AI tools")]:
    t = att[att.exp_band.notna() & att[v].isin(lv)]; ct = pd.crosstab(t.exp_band, t[v], normalize="index").reindex(columns=lv).reindex(bands) * 100
    cols = sns.color_palette(SEQ5, len(lv)) if len(lv) == 5 else ["#a50026", "#f08a5d", "#d9d9d9", "#7fb7d6", "#1a5fa8"]
    left = np.zeros(len(ct))
    for c_, col in zip(lv, cols):
        ax.barh(range(len(ct)), ct[c_], left=left, color=col, label=short.get(c_, c_), height=.7); left += ct[c_].values
    ax.set_yticks(range(len(ct))); ax.set_yticklabels(bands); ax.set_ylabel("Years of professional experience"); ax.set_xlabel("% of respondents"); ax.set_title(title); ax.set_xlim(0, 100); ax.invert_yaxis()
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(.5, -.2))
savefig(fig, "fig21_trust_sentiment_by_experience", "Trust in AI accuracy (a) and sentiment (b) across experience bands (100 % stacked; red = negative, blue = positive).", "O5")
plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, f, title in [(axes[0], "dev_type", "(a) By job role (n ≥ 300)"), (axes[1], "age", "(b) By age bracket")]:
    t = att[[f, "threat_yes"]].dropna()
    if f == "age": t = t[t.age != "Prefer not to say"]
    g = t.groupby(f).threat_yes.agg(["sum", "count"]); g = g[g["count"] >= (300 if f == "dev_type" else 100)]; g["rate"] = g["sum"] / g["count"] * 100
    lo, hi = wilson_ci(g["sum"], g["count"]); g["lo"], g["hi"] = lo * 100, hi * 100
    g = g.sort_values("rate") if f == "dev_type" else g.reindex([a for a in AGE_ORDER if a in g.index])
    y = np.arange(len(g)); ax.hlines(y, g.lo, g.hi, color=PALETTE["red"], lw=2); ax.scatter(g.rate, y, color=PALETTE["red"], zorder=3, s=26)
    ax.axvline(att.threat_yes.mean() * 100, color=PALETTE["grey"], ls="--", lw=1); ax.set_yticks(y); ax.set_yticklabels(g.index); ax.set_xlabel("% who believe AI is a threat to their job (95 % CI)"); ax.set_title(title)
    if f == "age": ax.invert_yaxis()
savefig(fig, "fig22_threat_by_role_age", "Share of respondents who see AI as a threat to their own job, by job role (a) and age (b); dashed line = overall.", "O5")
plt.show()

cm = att[["sent", "trust", "complex_", "threat_yes", "years_code_pro", "age_mid"]].rename(columns={"sent": "Favourability", "trust": "Trust", "complex_": "AI on complex tasks", "threat_yes": "Threat (yes=1)", "years_code_pro": "Experience (yrs)", "age_mid": "Age"}).corr(method="spearman")
fig, ax = plt.subplots(figsize=(6.4, 5.2))
sns.heatmap(cm, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-.6, vmax=.6, square=True, ax=ax, linewidths=.5, cbar_kws={"shrink": .8}); ax.set_title("Spearman correlations among AI attitudes")
savefig(fig, "fig23_attitude_correlations", "Spearman correlation matrix of AI attitude scores with experience and age.", "O5")
plt.show()
R["O5"]["attitude_corr"] = cm.round(3).to_dict()
save_results()
