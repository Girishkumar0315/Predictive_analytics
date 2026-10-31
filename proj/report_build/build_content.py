"""Builds content.json for the DOCX report. Every number comes from reports/results/results.json (written by the notebook)."""
import json, re
from pathlib import Path
import pandas as pd, numpy as np

ROOT = Path(__file__).resolve().parents[1]
R = json.load(open(ROOT / "reports/results/results.json"))
REG = json.load(open(ROOT / "reports/results/registry.json"))
TB = lambda n: pd.read_csv(ROOT / f"reports/results/tables/{n}.csv")
FIGCAP = {f["file"][:-4]: f["caption"] for f in REG["figures"]}
FIGCAP["fig08_normality_checks"] = "Q–Q plot of log-salary (a) and country-wise densities (b): log-salary is left-skewed and heavy-tailed rather than normal, with unequal location and spread across countries."
TITLE = "Predicting Developer Compensation and AI-Tool Adoption: A Predictive Analysis of the Stack Overflow 2024 Developer Survey"

def n0(x): return f"{x:,.0f}"
def n1(x): return f"{x:,.1f}"
def n2(x): return f"{x:,.2f}"
def n3(x): return f"{x:,.3f}"
def pv(p): return "< 0.001" if p < 0.001 else f"= {p:.3f}"
def usd(x): return f"USD {x:,.0f}"

B = []            # blocks of the main section
FR = []           # front-matter blocks
def add(*bl): B.extend(bl)
def H1(t, **k): return {"t": "h1", "text": t, **k}
def H2(t): return {"t": "h2", "text": t}
def H3(t): return {"t": "h3", "text": t}
def P(t): return {"t": "p", "text": t}
def BL(*items): return {"t": "bullets", "items": list(items)}
def EQ(t): return {"t": "eq", "text": t}
def FIG(name, width=600): return {"t": "fig", "id": name, "file": name + ".png", "caption": FIGCAP[name], "width": width}
def TBL(id_, caption, header, rows, widths=None, align=None, note=None, size=19): return {"t": "table", "id": id_, "caption": caption, "header": header, "rows": rows, "widths": widths, "align": align, "note": note, "size": size}
def df_rows(df, fmts): return [[(f(v) if f else v) for v, f in zip(r, fmts)] for r in df.itertuples(index=False)]

# ================================================================== FRONT MATTER
COVER = [
 {"t": "center", "text": "PREDICTIVE ANALYSIS", "bold": True, "after": 0},
 {"t": "center", "text": "PROJECT REPORT", "bold": True, "after": 0},
 {"t": "center", "text": "(Project Semester [January–April 20XX — edit as per your term])", "after": 240},
 {"t": "spacer", "n": 1},
 {"t": "center", "text": TITLE.upper(), "size": 16, "bold": True, "after": 360},
 {"t": "spacer", "n": 1},
 {"t": "center", "text": "Submitted by", "after": 120},
 {"t": "center", "text": "[Name of student]", "after": 0},
 {"t": "center", "text": "Registration No. [……………]", "after": 0},
 {"t": "center", "text": "Programme and Section [……………]", "after": 0},
 {"t": "center", "text": "Course Code [……………]", "after": 240},
 {"t": "center", "text": "Under the Guidance of", "after": 120},
 {"t": "center", "text": "[Name of faculty coordinator with U.Id and designation]", "bold": True, "after": 360},
 {"t": "center", "text": "Discipline of CSE/IT", "bold": True, "after": 0},
 {"t": "center", "text": "Lovely School of [__________________]", "bold": True, "after": 0},
 {"t": "center", "text": "Lovely Professional University, Phagwara", "bold": True, "after": 0},
]
FR += [
 {"t": "frontheading", "text": "DECLARATION", "nobreak": True},
 {"t": "p", "text": "I, [Name of student], student of [Programme name] under CSE/IT Discipline at Lovely Professional University, Punjab, hereby declare that all the information furnished in this project report is based on my own intensive work and is genuine."},
 {"t": "spacer", "n": 3},
 {"t": "left", "text": "Date: [DD/MM/YYYY]\t\t\t\t\t\tSignature: [signature]"},
 {"t": "left", "text": "Registration No. [……………]\t\t\t\tName of the student: [Name]"},
 {"t": "frontheading", "text": "CERTIFICATE"},
 {"t": "p", "text": "This is to certify that [student’s name] bearing Registration no. [……………] has completed [Course Code] project titled, “" + TITLE + "” under my guidance and supervision. To the best of my knowledge, the present work is the result of his/her original development, effort and study."},
 {"t": "spacer", "n": 3},
 {"t": "left", "text": "Signature and Name of the Supervisor: [signature — to be signed by the faculty member]", "bold": True},
 {"t": "left", "text": "Designation of the Supervisor: [designation]", "bold": True},
 {"t": "left", "text": "School of [……………………………]", "bold": True},
 {"t": "left", "text": "Lovely Professional University", "after": 0},
 {"t": "left", "text": "Phagwara, Punjab.", "after": 120},
 {"t": "left", "text": "Date: [DD/MM/YYYY]"},
 {"t": "frontheading", "text": "ACKNOWLEDGEMENT"},
 {"t": "p", "text": "[Edit this page in your own words.] I express my sincere gratitude to my faculty coordinator, [Name], for guidance and constructive feedback throughout this project, and to Lovely Professional University for providing the academic environment and computing resources needed to complete it."},
 {"t": "p", "text": "I thank Stack Overflow for conducting the Annual Developer Survey and releasing the anonymised results under an open licence, and Havisha Khurana and the TidyTuesday community (R4DS Online Learning Community) for curating and publicly hosting the extract used here. I also acknowledge the authors and maintainers of the open-source Python ecosystem — NumPy, pandas, SciPy, statsmodels, scikit-learn, LightGBM, SHAP, Matplotlib and Seaborn — on which every analysis in this report is built."},
 {"t": "p", "text": "Finally, I thank my family and friends for their encouragement."},
 {"t": "frontheading", "text": "TABLE OF CONTENTS"},
 {"t": "toc"},
 {"t": "frontheading", "text": "LIST OF FIGURES"},
 {"t": "lof"},
 {"t": "frontheading", "text": "LIST OF TABLES"},
 {"t": "lot"},
]

# ================================================================== 1. INTRODUCTION
ds, co, ac = R["dataset"], R["cohort"], R["ai_cohort"]
add(H1("1. Introduction"))
add(H2("1.1 Background and motivation"))
add(P("Software development is one of the most globally distributed professions, and since 2022 it has also become the first large profession in which generative artificial-intelligence (AI) assistants are used daily by a majority of practitioners. Two questions follow naturally for any student entering the industry: *what determines how much a developer is paid?* and *who is adopting AI tools, and how do developers feel about them?* Answering these with evidence rather than anecdote requires large, real survey data and careful statistical and machine-learning analysis."))
add(P(f"The Stack Overflow Annual Developer Survey 2024 is among the largest such surveys. It contains {n0(ds['rows'])} responses from developers in {ds['n_countries']} countries and covers education, career, compensation, community participation and, for the first time in depth, AI tooling. This project uses it for a complete end-to-end *predictive analysis*: a data-quality audit, exploratory analysis, hypothesis-driven statistical analysis of eight pre-defined objectives, two supervised-learning tasks (a regression task for annual compensation and a classification task for AI-tool adoption), and an explainability and error audit of the final models."))
add(H2("1.2 Problem statement"))
add(P("Using only information that a developer could reasonably state about themselves (location, role, experience, organisation, education, work arrangement, community engagement), (a) quantify and explain differences in annual compensation, (b) describe and explain adoption of AI tools and attitudes towards them, (c) build and fairly compare machine-learning models that predict compensation and AI adoption for unseen respondents, and (d) explain what the models learned and where they fail."))
add(H2("1.3 Why this topic is relevant to a student in India"))
add(P(f"India is the third-largest respondent country in the survey ({n0(R['O4']['india']['n'])} respondents with an AI answer; {R['O1']['india_n']} in the salary cohort). Understanding how compensation scales with experience, how Indian developers’ pay compares with other markets and how fast AI tools are being adopted gives practical input to career decisions, skill planning and curriculum design. The analysis is explicitly *descriptive and predictive, not causal*: the data are observational self-reports, and the report repeatedly states what cannot be concluded."))
add(H2("1.4 Objectives"))
add(P("The objectives below were fixed *before* modelling and drive every later chapter. Each is specific, measurable and non-overlapping (O1–O3: compensation; O4–O5: AI attitudes and adoption; O6–O7: prediction; O8: explanation)."))
obj = [
 ["O1", "How much of the variation in developer pay does country explain, and where does India stand?", "Country, USD pay; Kruskal–Wallis, η², Mann–Whitney (Holm), bootstrap CI; box and CI plots", "η² / R² of country; India : USA median ratio with 95 % CI"],
 ["O2", "How strongly does professional experience raise pay and where do returns diminish?", "Years coded professionally, log-pay; Spearman ρ, quadratic log-linear OLS (HC3), delta method; hexbin/LOWESS, curves", "ρ with CI; % gain per year at 0/5/10/20 yrs; India-vs-USA slope test"],
 ["O3", "What are the net effects of role, work mode, organisation size and education after adjusting for country and experience?", "OLS (HC3), nested-model ΔR², partial η²; forest plots", "Adjusted % differences with 95 % CI; ΔR² per block"],
 ["O4", "Who uses AI tools in development?", "χ² tests, Cramér’s V, Wilson CIs, z-test; stacked bars, dot plots, heat-map", "Adoption % with CI per group; ranking of factors; India-vs-world gap"],
 ["O5", "How do sentiment, trust and job-threat perception vary with career stage, role and country?", "Spearman ρ, Kruskal–Wallis ε², χ², Mann–Whitney; distribution and trend plots", "ρ (experience ↔ trust); threat-rate gaps"],
 ["O6", "Can annual compensation be predicted (regression, ≥ 4 models)?", "Baselines, Ridge, Random Forest, HistGradientBoosting, LightGBM; CV, tuning", "RMSE(log) reduction ≥ 40 % vs baseline; beat country-median baseline; R², MAE, MdAPE"],
 ["O7", "Can current AI-tool adoption be predicted (classification, ≥ 4 models)?", "Baseline, Logistic Regression, RF, HistGB, LightGBM; CV, tuning, threshold analysis", "ROC-AUC ≥ 0.65; F1 above majority baseline; calibration"],
 ["O8", "Which inputs drive the models and where do they fail?", "Permutation importance, SHAP, partial dependence, segment error audit", "Ranked drivers; error gaps across countries/segments"],
]
add(TBL("t_obj", "Research objectives, methods and success measures", ["#", "Research question", "Variables, methods and visualisations", "Success measure"], obj, [0.5, 3.0, 3.3, 2.4], ["l", "l", "l", "l"], size=18))
add(H2("1.5 Objective-to-analysis mapping"))
mp = [["O1", "§4.1", "fig09, fig10", "tbl_o1_country_stats", "Part D, O1"], ["O2", "§4.2", "fig11, fig12", "tbl_o2_marginal", "Part D, O2"], ["O3", "§4.3", "fig13–fig15", "tbl_o3_nested_models, tbl_o3_adjusted_effects", "Part D, O3"],
      ["O4", "§4.4", "fig16–fig19", "tbl_o4_factor_tests", "Part D, O4"], ["O5", "§4.5", "fig20–fig23", "tbl_o5_conditionality", "Part D, O5"], ["O6", "§4.6", "fig24–fig26", "tbl_o6_model_comparison", "Part E, O6"],
      ["O7", "§4.7", "fig27–fig29", "tbl_o7_model_comparison", "Part E, O7"], ["O8", "§4.8", "fig30–fig36", "tbl_o8_* tables", "Part F, O8"]]
add(TBL("t_map", "Mapping of every objective to its report section, figures, result tables and notebook section", ["Objective", "Report section", "Figures", "Result tables (CSV)", "Notebook section"], mp, [0.8, 1.0, 1.3, 2.9, 1.3], ["l"] * 5, note="Figure file names refer to the PNG files in reports/figures/ (e.g. fig09 = fig09_country_boxplot.png)."))
add(H2("1.6 Tools, technologies and deliverables"))
add(P(f"All work was performed in Python {R['environment']['python']} inside a Jupyter notebook using pandas {R['environment']['pandas']}, NumPy {R['environment']['numpy']}, SciPy {R['environment']['scipy']}, statsmodels {R['environment']['statsmodels']}, scikit-learn {R['environment']['sklearn']}, LightGBM {R['environment']['lightgbm']}, SHAP, Matplotlib and Seaborn. Every random process uses the seed {R['environment']['seed']}. The project folder contains: the raw data and its documentation; the executed notebook; a reusable `src/` package (data preparation, features, modelling, statistics, plotting, prediction); all {len(REG['figures'])} figures (300 dpi PNG) and {len(REG['tables'])} result tables; the trained best models with their preprocessing pipelines; a README with set-up and GitHub instructions; viva preparation notes; and this report."))
add(H2("1.7 Organisation of the report"))
add(P("Chapter 2 documents the source of the dataset and why it was selected. Chapter 3 describes the exploratory data analysis process: inspection, quality assessment, cleaning, univariate and bivariate exploration, feature engineering and leakage control. Chapter 4 presents the eight analyses; each follows the prescribed structure *(i) Introduction, (ii) General Description, (iii) Specific Requirements, Functions and Formulas, (iv) Analysis Results, (v) Visualization and Interpretation*. Chapters 5 and 6 give the conclusion and future scope, followed by the references."))

# ================================================================== 2. SOURCE OF DATASET
add(H1("2. Source of Dataset"))
add(H2("2.1 Dataset selection process"))
add(P("The brief asks for the most suitable, real, sufficiently large, currently relevant dataset with a meaningful prediction target. The four portals named in the brief (Kaggle, data.gov.in, Mendeley Data, AI Kosh) could not be reached from the environment in which this project was built, so candidates were evaluated among openly licensed datasets that *could* be downloaded and inspected (documented, curated collections hosted on GitHub). Each candidate was actually loaded and profiled; none was judged from its title alone."))
cand = [["Stack Overflow Developer Survey 2024 (TidyTuesday extract)", "65,437 × 28", "Salary (continuous), AI adoption (binary), attitudes (ordinal)", "Very high — AI in software work; global; includes India", "**Selected**"],
        ["Netflix viewing report H1 2025", "36,121 movies / 27,803 shows × 8", "Weak (views from a few aggregate columns)", "High", "Rejected: no meaningful supervised target"],
        ["Henley Passport Index (rank by year)", "3,950 × 6", "Rank / visa-free counts (trend only)", "Medium", "Rejected: few features, time-series style"],
        ["Pokémon (2025)", "949 × 22", "Stats (game data)", "Low", "Rejected: small, not socially relevant"],
        ["Cars in Qatar", "105 × 15", "Price", "Medium", "Rejected: only 105 rows — unreliable for CV"]]
add(TBL("t_cand", "Candidate datasets evaluated (those reachable and profiled)", ["Candidate", "Rows × cols", "Prediction targets", "Topical relevance", "Decision"], cand, [2.4, 1.3, 2.2, 1.8, 1.4], ["l"] * 5, size=18))
add(P("The Stack Overflow survey was selected because it (i) is large and real, (ii) has two natural and different prediction targets (a continuous salary and a binary AI-adoption flag) plus ordinal attitude variables that allow rich statistical questions, (iii) addresses a current, widely discussed topic (AI tools in software work), (iv) is geographically diverse and includes enough Indian respondents for a country-specific analysis, and (v) has an official, documented origin and an open licence."))
add(H2("2.2 Provenance and licence"))
add(TBL("t_prov", "Provenance of the dataset", ["Item", "Value"], [
  ["Original publisher", "Stack Overflow (Annual Developer Survey, fielded May 2024)"],
  ["Official page", "https://survey.stackoverflow.co/ (2024 results: https://survey.stackoverflow.co/2024/)"],
  ["Licence", "Open Database Licence (ODbL), as stated on the survey page (re-confirm before submission)"],
  ["Extract used", "TidyTuesday 2024 week 36, curated by Havisha Khurana (single-response, integer-coded questions + crosswalk)"],
  ["Extract URL", "https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-09-03"],
  ["Repository commit at download", "b2258ae949e3c64099c4e6d00ff4b68dba1d6d0e (downloaded October 2026)"],
  ["Main file", "stackoverflow_survey_single_response.csv, SHA-256 bbfad6bf…c2446"],
  ["Companion files", "qname_levels_single_response_crosswalk.csv (code → label), stackoverflow_survey_questions.csv (question text)"],
 ], [1.6, 5.4], ["l", "l"], size=18))
add(P("**Transparency note.** Because the original download CDN was not reachable, the data were taken from the TidyTuesday GitHub copy, whose own cleaning script (stored in `data/raw/TIDYTUESDAY_README_ORIGINAL.md`) derives it from the official Stack Overflow zip. File checksums are recorded so that a reader can verify integrity. Responses are anonymised by the publisher; no personal identifiers are present."))
add(H2("2.3 Structure and data dictionary"))
add(P(f"The main table has {n0(ds['rows'])} rows (one per respondent) and {ds['cols']} columns, occupying about {n1(ds['memory_mb'])} MB in memory. Categorical answers are stored as integer codes; the crosswalk (122 rows) maps each code to its text label and was applied by the function `decode()`. {ds['n_questions']} question texts are provided. [[T:t_dict]] lists every variable, its measurement type and how it is used in this project."))
dd = TB("tbl_data_dictionary")
add(TBL("t_dict", "Data dictionary of the 28 raw variables and their role in the project", ["Variable", "Description", "Type", "Role", "Missing %"], df_rows(dd, [None, None, None, None, lambda v: f"{v:.1f}"]), [1.5, 3.0, 1.1, 1.8, 0.7], ["l", "l", "l", "l", "r"], size=16))
add(H2("2.4 Known limitations of the source"))
add(BL("**Self-selection.** Respondents are people who answer a Stack Overflow survey, not a random sample of all developers; results describe the sample, not the whole profession.",
       "**Self-reported, nominal compensation.** Pay is optional, unverified, in nominal USD (not adjusted for purchasing power) and may mix salary with bonuses.",
       "**Structural missingness.** AI follow-up questions were asked only of respondents who use or plan to use AI tools; salary is missing for about two-thirds of respondents.",
       "**Single-response extract.** Multi-select questions (programming languages, databases, AI tools used) are not in this extract, which limits feature engineering."))

# ================================================================== 3. EDA PROCESS
cs, asu = R["comp_stats_raw"], R["assumptions"]
add(H1("3. EDA Process"))
add(P("This chapter documents, in the order performed, how the raw survey was inspected, audited, cleaned and explored, and how those findings determined the modelling choices of Chapter 4. Every number is produced by the notebook (`notebooks/CA2_Predictive_Analysis.ipynb`, Parts A–C)."))
add(H2("3.1 Data inspection"))
add(P(f"After loading, the integer codes were decoded with the official crosswalk, category labels were shortened for readability (for example “United States of America” → “USA”, “Developer, full-stack” → “Full-stack dev”) and stateless helper columns were added (Section 3.8). The compensation variable `converted_comp_yearly` (USD) is available for only {n0(cs['n'])} respondents ({n1(cs['n']/ds['rows']*100)} %). Its mean is {usd(cs['mean'])} but its median only {usd(cs['median'])}; the maximum is {usd(cs['max'])}, and the skewness of {n1(cs['skew'])} shows an extremely heavy right tail."))
add(H2("3.2 Data-quality assessment"))
add(H3("3.2.1 Missing values"))
mt = TB("tbl_missing").head(10)
add(P(f"Missingness is substantial and *structured* ([[F:fig01_missingness]]). {n1(R['missing']['overall_cell_pct'])} % of all cells are empty. The worst columns are `converted_comp_yearly` and `comp_total` (optional questions), `ai_acc` and `ai_complex` (shown only to current AI users), `ai_sent`, `ai_threat`, `buildvs_buy`, `so_part_freq` and `purchase_influence`. Because the pattern is driven by survey routing rather than by chance, the target variables are never imputed; only predictors are imputed, inside the cross-validation pipeline, using training-fold medians (numeric) or an explicit “Missing” category (categorical)."))
add(FIG("fig01_missingness", 520))
add(H3("3.2.2 Duplicates"))
dp = R["duplicates"]
add(P(f"The respondent identifier is unique. Ignoring the identifier, {n0(dp['n_duplicate_rows_incl_first'])} rows belong to groups of identical answers ({n0(dp['n_removable'])} removable copies). Inspection shows they answered on average only {n1(dp['mean_answered_questions_dups'])} of 27 questions versus {n1(dp['mean_answered_questions_all'])} for the typical respondent: they are abandoned, almost-empty questionnaires that coincide by chance rather than double submissions. They are therefore *not* deleted from the raw data; exact duplicates are removed only inside the modelling cohorts (2 rows in the salary cohort)."))
add(H3("3.2.3 Inconsistent and implausible values"))
inc = TB("tbl_inconsistencies")
add(P("[[T:t_inc]] lists the logical checks. Two findings deserve emphasis. First, 504 respondents report more years of *professional* coding than *total* coding, and a further 119 report professional experience that is impossible for their age bracket; these values are *blanked and later imputed* rather than the whole respondent being dropped. Second, 10,305 respondents typed a salary in local currency but have no USD conversion, which silently removes them from any analysis based on `converted_comp_yearly`."))
add(TBL("t_inc", "Logical inconsistencies and suspicious values", ["Check", "Records affected"], df_rows(inc, [None, lambda v: n0(v)]), [5.8, 1.2], ["l", "r"]))
add(H3("3.2.4 Outliers"))
oc = TB("tbl_outlier_rules")
add(P(f"Three classical rules were compared on compensation ([[T:t_out]]). On the raw dollar scale Tukey’s fences flag {n1(oc.iloc[0].pct_flagged)} % of values and the |z| > 3 rule only {n1(oc.iloc[1].pct_flagged)} %, because the extreme values inflate the standard deviation that the z-score itself depends on (masking). After a log transform the picture changes: the z-score rule flags {n1(oc.iloc[4].pct_flagged)} %, the percentile rule {n1(oc.iloc[5].pct_flagged)} % and Tukey’s fences {n1(oc.iloc[3].pct_flagged)} %, and the skewness moves from {n1(cs['skew'])} to {n2(cs['skew_log'])} for the whole sample — the remaining outlier problem is now the *lower* tail (implausibly small salaries, down to USD 1), which is why a lower plausibility bound is part of the cleaning policy. [[F:fig02_comp_distribution]] shows the raw and log distributions."))
add(TBL("t_out", "Outliers flagged in annual compensation by three rules (raw and log scale)", ["Scale", "Rule", "Flagged", "% flagged"], df_rows(oc, [None, None, lambda v: n0(v), lambda v: f"{v:.2f}"]), [1.2, 3.6, 1.0, 1.0], ["l", "l", "r", "r"]))
add(FIG("fig02_comp_distribution"))
add(H2("3.3 Cleaning policy and cohort construction"))
add(P("Cleaning decisions were documented as explicit, reproducible rules (function `build_salary_cohort()`):"))
add({"t": "numbered", "ref": "num1", "items": [
  "Keep **professional developers** (main_branch = “I am a developer by profession”) who reported a converted salary and a country.",
  "Apply **absolute plausibility bounds** of USD 1,000–1,000,000 (smaller values are probably monthly salaries or typing errors; larger values cannot be verified).",
  "Apply **country-specific 3 × IQR fences on log-salary** for countries with at least 50 salaries. A single global fence would delete almost every salary from low-income countries and thus bias the study; country-aware fencing removes only values that are extreme *for their own market*.",
  "**Blank** logically impossible experience values (rows are kept; values are imputed later inside the pipeline).",
  "Remove exact duplicate rows inside the cohort."]})
at = TB("tbl_cohort_attrition")
add(P(f"[[T:t_att]] and [[F:fig03_cohort_attrition]] show the attrition. The final salary cohort has {n0(co['n'])} respondents from {co['n_countries']} countries, including {co['n_india']} from India and {n0(co['n_usa'])} from the USA. For the AI analyses a second cohort of {n0(ac['n'])} respondents (all with a known AI-adoption answer) is used; {n1(ac['adoption_rate'])} % of them currently use AI tools."))
add(TBL("t_att", "Row attrition at each step of salary-cohort construction", ["Step", "Rows remaining", "Rows removed"], df_rows(at, [None, lambda v: n0(v), lambda v: n0(v)]), [4.6, 1.2, 1.2], ["l", "r", "r"]))
add(FIG("fig03_cohort_attrition", 480))
add(P(f"A sensitivity analysis in Section 4.6 re-fits one model (HistGradientBoosting) under three outlier policies. It shows that the *absolute* error level depends on the policy — stricter filters leave fewer extreme salaries and therefore give lower RMSE — so error figures in this report should be read as conditional on the chosen 3 × IQR rule. The full model comparison was not repeated under the other policies."))
add(H2("3.4 Univariate exploration"))
d = R["descriptives"]
add(P(f"[[F:fig04_univariate_profile]] profiles the cohort. Respondents are mostly aged 25–44, work in full-stack and back-end roles, and are concentrated in the USA, Germany, the UK, Ukraine and India. Median professional experience is {n1(d['years_code_pro']['50%'])} years (mean {n1(d['years_code_pro']['mean'])}, maximum {n0(d['years_code_pro']['max'])}); the distribution is right-skewed with a spike at the top-code of 51 years. Remote and hybrid work dominate (about 8,700 respondents each) over in-person work (about 3,300). Compensation in the cohort has a median of {usd(d['converted_comp_yearly']['50%'])} and a mean of {usd(d['converted_comp_yearly']['mean'])}; log-salary has a skewness of {n2(asu['skew_logcomp'])}."))
add(FIG("fig04_univariate_profile"))
add(H2("3.5 Group comparisons"))
add(P("[[F:fig05_group_medians]] gives an unadjusted look at median pay by work arrangement, organisation size and education. Remote workers, employees of very large organisations and holders of postgraduate degrees earn more at face value. These raw gaps are confounded by *country mix* (for instance, remote work may be more common among developers of high-wage countries), which is why Objective O3 re-estimates them after adjustment."))
add(FIG("fig05_group_medians"))
add(H2("3.6 Correlation and association analysis"))
cl = R["corr_with_logcomp"]
add(P(f"[[F:fig06_spearman_corr]] shows Spearman correlations. Among the numeric variables, log-salary is most strongly associated with `years_code_pro` (ρ = {n2(cl['years_code_pro'])}), `years_code` (ρ = {n2(cl['years_code'])}) and `age_mid` (ρ = {n2(cl['age_mid'])}); organisation size (ρ = {n2(cl['org_log'])}) and education rank (ρ = {n2(cl['ed_rank'])}) are weaker, and Stack Overflow visit frequency is weakly *negatively* correlated (ρ = {n2(cl['so_visit_rank'])}). The three experience-type variables are strongly inter-correlated, which raises a multicollinearity concern. Variance-inflation factors ([[T:t_vif]]) of about {n1(R['vif'][0]['VIF'])} and {n1(R['vif'][1]['VIF'])} for `years_code` and `years_code_pro` are above the conventional threshold of 5 but below 10. Tree-based models are insensitive to this, Ridge regularisation absorbs it for prediction, and for the inferential regressions of O2 and O3 only `years_code_pro` is used."))
add(FIG("fig06_spearman_corr", 470))
add(TBL("t_vif", "Variance inflation factors of the numeric predictors", ["Variable", "VIF"], [[v["variable"], n2(v["VIF"])] for v in R["vif"]], [3, 1.5], ["l", "r"]))
add(P("For categorical variables, bias-corrected Cramér’s V was used ([[F:fig07_cramers_v]]). The strongest association is between organisation size and purchase influence (V ≈ 0.34; people in small organisations naturally have more say over purchases), followed by country and work arrangement (V ≈ 0.30), which confirms that raw remote-work pay gaps are confounded by country. All other pairs are weaker (V ≤ 0.19) and no pair of categorical predictors is so strongly associated as to be redundant."))
add(FIG("fig07_cramers_v", 470))
add(H2("3.7 Distributional assumptions"))
add(P(f"A D’Agostino–Pearson test rejects normality of log-salary (K² = {n0(asu['dagostino_k2'])}, p {pv(asu['dagostino_p'])}); the distribution is left-skewed (skew {n2(asu['skew_logcomp'])}) with excess kurtosis {n2(asu['kurt_logcomp'])}. Levene’s test rejects equal variances across the eight largest countries (W = {n0(asu['levene_W'])}, p {pv(asu['levene_p'])}). [[F:fig08_normality_checks]] illustrates both. With such large samples *any* departure becomes significant, so the practical consequence is what matters: we model log-salary (not raw dollars), use rank-based tests (Kruskal–Wallis, Mann–Whitney, Spearman) and heteroscedasticity-robust (HC3) standard errors."))
add(FIG("fig08_normality_checks", 500))
add(H2("3.8 Feature engineering"))
fe = TB("tbl_feature_engineering")
add(P("[[T:t_fe]] lists every engineered feature with its justification. All are *stateless* (row-wise arithmetic or recoding), so they can be computed before the train/test split without leakage. Anything that learns from data — imputation, scaling, one-hot vocabularies and target encoding — lives inside scikit-learn pipelines and is re-fitted on the training part of each cross-validation fold."))
add(TBL("t_fe", "Engineered features and their justification", ["Feature", "Definition", "Justification"], df_rows(fe, [None, None, None]), [1.7, 2.3, 3.4], ["l", "l", "l"], size=18))
add(H2("3.9 Leakage prevention and data splitting"))
lk = TB("tbl_leakage_register")
add(P("Target leakage is the most common cause of over-optimistic results. [[T:t_leak]] is the project’s *leakage register*. The most important entries are `comp_total` (the local-currency version of the salary target) and the four AI follow-up questions (`ai_sent`, `ai_acc`, `ai_complex`, `ai_threat`). The latter were asked only of respondents who already use (or plan to use) AI, so their *mere presence* reveals the adoption label; Section 4.7 measures how much they would inflate performance."))
add(TBL("t_leak", "Leakage register: columns excluded from predictors and the reason", ["Column", "Reason excluded / restricted"], df_rows(lk, [None, None]), [1.6, 5.4], ["l", "l"], size=18))
add(P(f"Both tasks use a stratified 80 / 20 hold-out split with seed {R['environment']['seed']} (salary: {n0(R['O6_setup']['n_train'])} training and {n0(R['O6_setup']['n_test'])} test rows, stratified by large country; adoption: stratified by the label). All model selection uses {R['O6_setup']['n_cv_folds']}-fold cross-validation *inside the training set*; the test set is evaluated once, after selection."))
add(H2("3.10 Summary of EDA decisions"))
add(BL("Model **log-salary**; report errors on both the log and the USD scale.",
       "Use **rank-based tests** and **HC3-robust OLS** for inference.",
       "Treat missingness as structural: never impute targets; impute predictors inside pipelines.",
       "Use **country-aware** outlier handling; blank (not drop) impossible experience values.",
       "Exclude `comp_total`, `currency` and all conditional AI follow-up variables from predictors where they leak the target.",
       "Handle high-cardinality `country` and `dev_type` with **cross-fitted target encoding** (trees) or pooled one-hot (Ridge)."))

# ================================================================== 4. ANALYSIS ON DATASET
add(H1("4. Analysis on Dataset"))
add(P("Each analysis below follows the prescribed structure: (i) Introduction, (ii) General Description, (iii) Specific Requirements, Functions and Formulas, (iv) Analysis Results, and (v) Visualization and Interpretation. All results quoted are copied programmatically from the executed notebook."))

# ---------------------------------------------------------------- O1
o1 = R["O1"]; c1 = TB("tbl_o1_country_stats")
add(H2("4.1 Objective O1 — Geographic compensation inequality and India’s position"))
add(H3("(i) Introduction"))
add(P("Where a developer lives is widely believed to be the single biggest determinant of pay. O1 turns this belief into numbers: how much of the variation in compensation is explained by country alone, and how does India compare with the other large developer markets? The answer matters for salary benchmarking, offshoring decisions and for students weighing emigration against local careers."))
add(H3("(ii) General Description"))
add(P(f"The analysis uses the salary cohort ({n0(co['n'])} professionals, {o1['n_all_countries']} countries). For pairwise comparisons only the {len(o1['top_countries'])} countries with at least 150 salary reports are retained ({n0(o1['n_top'])} respondents), so that each country median has a usable confidence interval. Pay is nominal USD as reported; it is *not* adjusted for purchasing power or taxes, so a low USD figure does not by itself mean a low standard of living."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Variables: `country`, `converted_comp_yearly` and its logarithm `log_comp`. Functions: `groupby().median()`, `scipy.stats.kruskal`, `statsmodels.formula.api.ols(\"log_comp ~ C(country)\")`, `scipy.stats.mannwhitneyu`, `statsmodels…multipletests(method=\"holm\")`, and the project helpers `eta_squared_from_groups`, `epsilon_squared_kruskal`, `cliffs_delta` and `boot_median_ci` (1,000 bootstrap resamples)."))
add(EQ("H = [12 / (N(N+1))] · Σ_{i}(R_{i}^{2} / n_{i}) − 3(N+1),     ε^{2} = H / (N − 1)     (Kruskal–Wallis and effect size)"))
add(EQ("η^{2} = SS_{between} / SS_{total} (variance in log-pay explained by country);     δ = 2U / (n_{1}n_{2}) − 1  (Cliff’s delta)"))
add(EQ("95 % bootstrap CI of a median = [2.5th, 97.5th percentile] of the medians of B = 1,000 resamples drawn with replacement"))
add(H3("(iv) Analysis Results"))
add(P(f"The Kruskal–Wallis test strongly rejects equal pay across the {len(o1['top_countries'])} large countries (H = {n0(o1['H'])}, p {pv(o1['p'])}) with a large effect size (ε² = {n2(o1['epsilon2'])}; η² = {n2(o1['eta2_top'])}). A regression of log-pay on country dummies *for all {o1['n_all_countries']} countries* gives R² = {n3(o1['R2_country_all'])} (adjusted R² = {n3(o1['adjR2_country_all'])}); in other words, **country alone explains about half of all variation in developer pay**."))
rows = [[int(r['rank']), r['country'], n0(r['n']), n0(r['median_usd']), f"{n0(r['ci_low'])}–{n0(r['ci_high'])}", n2(r['ratio_to_usa']), ("—" if pd.isna(r['cliffs_delta_india_vs']) else n2(r['cliffs_delta_india_vs']))] for _, r in c1.iterrows()]
add(P(f"[[T:t_o1]] lists the median pay of every large country. The highest median is {usd(o1['max_country']['median_usd'])} ({o1['max_country']['country']}) and the lowest {usd(o1['min_country']['median_usd'])} ({o1['min_country']['country']}), a ratio of {n1(o1['top_over_bottom'])} to 1. India’s median ({usd(o1['india_median'])}, n = {o1['india_n']}) is **{n2(o1['india_usa_ratio'])} times** the USA median (95 % bootstrap CI {n2(o1['india_usa_ratio_ci'][0])}–{n2(o1['india_usa_ratio_ci'][1])}) and ranks {o1['india_rank']} of {o1['n_ranked']}. After Holm correction, India’s pay distribution is significantly lower than that of {'all ' + str(o1['india_sig_lower_than']) + ' other countries' if o1['india_sig_lower_than']==o1['n_ranked']-1 else str(o1['india_sig_lower_than']) + ' of the other ' + str(o1['n_ranked']-1) + ' countries'} and higher than none."))
add(TBL("t_o1", "Median annual compensation (USD) by country, with 95 % bootstrap CI, ratio to the USA median and Cliff’s δ of India versus that country (negative = India lower)", ["Rank", "Country", "n", "Median (USD)", "95 % CI", "× USA", "δ (India vs)"], rows, [0.5, 1.4, 0.6, 1.1, 1.5, 0.7, 0.9], ["c", "l", "r", "r", "r", "r", "r"], size=17))
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig09_country_boxplot", 520))
add(FIG("fig10_country_median_ci", 520))
add(P(f"[[F:fig09_country_boxplot]] shows that the *whole distribution* of pay shifts with country, not just the median: the interquartile ranges of high-pay markets (USA, Israel, Switzerland) barely overlap with those of India or Brazil. [[F:fig10_country_median_ci]] confirms with confidence intervals that the ordering is statistically reliable. India’s gap to the USA is very large (Cliff’s δ close to −1: almost every Indian respondent earns less than almost every American respondent), whereas the gap to Ukraine, Russia, Turkey and Brazil is far smaller (|δ| ≈ 0.2). **Interpretation:** the location of employment is the dominant structural factor in nominal developer pay. Because these are nominal dollars, the gap overstates differences in purchasing power; and because the sample is self-selected, it should not be read as the national average for any country."))

# ---------------------------------------------------------------- O2
o2 = R["O2"]; sp = o2["spearman_all"]
add(H2("4.2 Objective O2 — The experience premium and diminishing returns"))
add(H3("(i) Introduction"))
add(P("Experience is the other classic driver of pay. O2 asks how strongly professional experience is associated with compensation, how many percent an additional year is worth at different career stages, where the returns flatten, and whether the pattern differs in India."))
add(H3("(ii) General Description"))
add(P(f"The analysis uses the {n0(o2['n'])} cohort members with a valid `years_code_pro`. Experience is modelled both non-parametrically (median pay by experience band and a LOWESS smoother) and parametrically (a quadratic log-linear regression with country fixed effects), because a straight line cannot represent diminishing returns."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `scipy.stats.spearmanr` (with a Bonett–Wright confidence interval for ρ), `statsmodels.formula.api.ols(\"log_comp ~ years_code_pro + years_pro_sq + C(country)\").fit(cov_type=\"HC3\")`, `statsmodels…lowess`, and a delta-method confidence interval for the marginal effect."))
add(EQ("ln(Pay_{i}) = β_{0} + β_{1}x_{i} + β_{2}x_{i}^{2} + γ_{country(i)} + ε_{i}"))
add(EQ("Marginal return at x:  100 · (exp(β_{1} + 2β_{2}x) − 1) % per extra year;     turning point x* = −β_{1} / (2β_{2})"))
add(EQ("Var(β_{1} + 2β_{2}x) = g′ V g,  g = (1, 2x)′,  V = HC3 covariance of (β_{1}, β_{2})   →   95 % CI = estimate ± 1.96·SE"))
add(H3("(iv) Analysis Results"))
sc = o2["spearman_by_country"]
add(P(f"Overall, pay and professional experience are moderately and positively rank-correlated (Spearman ρ = {n2(sp['rho'])}, 95 % CI {n2(sp['ci_low'])}–{n2(sp['ci_high'])}, n = {n0(sp['n'])}; bootstrap CI {n2(o2['spearman_boot_ci'][0])}–{n2(o2['spearman_boot_ci'][1])}). Within individual countries the correlation ranges from {n2(min(v['rho'] for v in sc.values()))} to {n2(max(v['rho'] for v in sc.values()))} ([[T:t_o2sp]])."))
add(TBL("t_o2sp", "Spearman correlation between professional experience and log-pay, overall and by country", ["Group", "n", "ρ", "95 % CI", "p"], [["All"] + [n0(sp['n']), n2(sp['rho']), f"{n2(sp['ci_low'])}–{n2(sp['ci_high'])}", pv(sp['p'])]] + [[k, n0(v['n']), n2(v['rho']), f"{n2(v['ci_low'])}–{n2(v['ci_high'])}", pv(v['p'])] for k, v in sc.items()], [1.5, 0.8, 0.8, 1.6, 0.9], ["l", "r", "r", "r", "r"]))
mg = o2["marginal"]
add(P(f"The quadratic model (R² = {n3(o2['r2_quad'])} including country effects) has β₁ = {n3(o2['beta1'])} and β₂ = {o2['beta2']:.5f} (both p {pv(max(o2['p_beta1'], o2['p_beta2']))}), implying a turning point at about **{n1(o2['peak_years'])} years**. [[T:t_o2m]] converts the coefficients into the estimated pay change per additional year: {n1(mg[0]['pct_per_year'])} % at entry level, {n1(mg[2]['pct_per_year'])} % at five years, {n1(mg[3]['pct_per_year'])} % at ten years and only {n1(mg[5]['pct_per_year'])} % at twenty years."))
add(TBL("t_o2m", "Estimated % change in pay per additional year of professional experience (country fixed effects; 95 % CI)", ["Years of experience", "% per extra year", "95 % CI low", "95 % CI high"], [[str(m['years']), n2(m['pct_per_year']), n2(m['ci_low']), n2(m['ci_high'])] for m in mg], [1.6, 1.4, 1.3, 1.3], ["c", "r", "r", "r"]))
sf = o2["separate_fits"]; il = o2["interaction_linear"]
add(P(f"Country-specific fits show that the *early-career* return is far steeper in India: {n1(sf['India']['pct_at_0'])} % per year at entry level versus {n1(sf['USA']['pct_at_0'])} % in the USA, {n1(sf['Germany']['pct_at_0'])} % in Germany and {n1(sf['UK']['pct_at_0'])} % in the UK; at five years the figures are {n1(sf['India']['pct_at_5'])} %, {n1(sf['USA']['pct_at_5'])} %, {n1(sf['Germany']['pct_at_5'])} % and {n1(sf['UK']['pct_at_5'])} %. A formal India-versus-USA test of the experience-slope difference (interaction term, HC3) gives a coefficient of {n3(il['coef'])} (p {pv(il['p'])}) and for the curvature {n3(il['coef_sq'])} (p {pv(il['p_sq'])})."))
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig11_experience_vs_pay"))
add(FIG("fig12_experience_marginal"))
add(P(f"[[F:fig11_experience_vs_pay]] (a) shows a wide scatter — experience explains only part of pay — but the LOWESS curve rises steeply in the first ten years and then flattens. Panel (b) shows the same pattern by experience band: the USA curve lies far above the others and flattens after about 11 years, while India’s curve starts much lower but climbs more steeply in relative terms. [[F:fig12_experience_marginal]] (a) gives the cumulative country-adjusted premium; (b) shows that the marginal return declines roughly linearly and crosses zero near 20–25 years. **Caution:** the quadratic form forces the curve to bend downwards after its turning point; the non-parametric LOWESS curve and the band medians show a *plateau*, not a decline. The correct reading is therefore “returns flatten after roughly two decades”, not “pay falls”. India’s steeper early slope is partly arithmetic — a low starting salary grows faster in percentage terms — and does not mean Indian developers ever catch up in absolute dollars."))

# ---------------------------------------------------------------- O3
o3 = R["O3"]; nest = o3["nested"]; adj = pd.DataFrame(o3["adj"])
add(H2("4.3 Objective O3 — Net effects of role, work arrangement, organisation size and education"))
add(H3("(i) Introduction"))
add(P("Raw group differences (Section 3.5) mix genuine effects with country and experience mix. O3 estimates the *adjusted* differences — comparing developers from the same country and with the same experience — and asks which blocks of characteristics add explanatory power beyond country and experience."))
add(H3("(ii) General Description"))
add(P(f"A multiple linear regression on log-pay is estimated on {n0(o3['n'])} respondents with complete information (the model includes {o3['n_roles']} role groups, country groups with at least 100 respondents plus an “other countries” group, experience and its square, organisation size, work arrangement and education). Roles and countries with fewer than 100 respondents are pooled. Reference categories are Full-stack dev, In-person, 100 to 499 employees and Bachelor’s degree. Coefficients are converted into percentage differences, 100·(e^β − 1) %."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `statsmodels.formula.api.ols` with `cov_type=\"HC3\"`; `anova_lm(typ=2)` for partial η²; `compare_f_test` for the incremental F-test of each nested block."))
add(EQ("ln(Pay) = β_{0} + country + β_{1}x + β_{2}x^{2} + role + org. size + work mode + education + ε"))
add(EQ("Adjusted difference = 100·(e^{β} − 1) %;     ΔR^{2}_{k} = R^{2}(M_{k}) − R^{2}(M_{k−1});     η^{2}_{p} = SS_{effect} / (SS_{effect} + SS_{residual})"))
add(H3("(iv) Analysis Results"))
add(P(f"[[T:t_o3n]] shows the nested models. Country alone gives R² = {n3(nest[0]['R2'])}; adding experience raises it by {n3(nest[1]['delta_R2'])}; job role adds {n3(nest[2]['delta_R2'])}, organisation size {n3(nest[3]['delta_R2'])}, work arrangement {n3(nest[4]['delta_R2'])} and education {n3(nest[5]['delta_R2'])}. Every block is statistically significant (incremental F-tests p {pv(max(n['p_F_block'] for n in nest))}), but their *practical* contribution is small: the full model reaches R² = {n3(o3['final_R2'])}. Partial η² values are {n3(o3['partial_eta2']['country_g'])} for country, {n3(o3['partial_eta2']['years_code_pro'])} for experience (linear term), and only {n3(o3['partial_eta2']['org_short'])}, {n3(o3['partial_eta2']['dev_g'])}, {n3(o3['partial_eta2']['remote_work'])} and {n3(o3['partial_eta2']['ed_short'])} for organisation size, role, work arrangement and education."))
add(TBL("t_o3n", "Nested OLS models for log-compensation: cumulative R² and incremental ΔR²", ["Model", "R²", "Adj. R²", "ΔR²", "p (F-test of block)", "n"], [[m['model'], n3(m['R2']), n3(m['adj_R2']), n3(m['delta_R2']), pv(m['p_F_block']), n0(m['n'])] for m in nest], [2.6, 0.8, 0.9, 0.8, 1.3, 0.8], ["l", "r", "r", "r", "r", "r"]))
def effrow(r): return [r['term'], f"{r['pct']:+.1f}", f"{r['pct_lo']:+.1f} to {r['pct_hi']:+.1f}", pv(r['p'])]
eo = adj[adj['var'].isin(['remote_work', 'org_short', 'ed_short'])].copy(); eo['k'] = eo['var'].map({'remote_work': 0, 'org_short': 1, 'ed_short': 2}); eo = eo.sort_values(['k', 'pct'], ascending=[True, False])
rows3 = []
for _, r in eo.iterrows(): rows3.append([{'remote_work': 'Work arrangement', 'org_short': 'Organisation size', 'ed_short': 'Education'}[r['var']]] + effrow(r))
rm = adj[(adj['var'] == 'remote_work') & (adj.term == 'Remote')].iloc[0]; hy = adj[(adj['var'] == 'remote_work') & (adj.term == 'Hybrid')].iloc[0]
add(P(f"[[T:t_o3e]] gives the adjusted effects. Compared with in-person work, **remote workers earn {n1(rm['pct'])} % more** (95 % CI {n1(rm['pct_lo'])}–{n1(rm['pct_hi'])}) and hybrid workers {n1(hy['pct'])} % more, holding country, role, experience, company size and education constant. Pay rises broadly with organisation size: relative to 100–499 employees, organisations with 10,000 or more employees pay {n1(adj[adj.term=='10,000 or more'].iloc[0]['pct'])} % more, whereas tiny organisations (2–9 employees) pay {n1(abs(adj[adj.term=='2 to 9'].iloc[0]['pct']))} % less. A Master’s degree is associated with {n1(adj[adj.term==chr(77)+'aster'+chr(39)+'s'].iloc[0]['pct'])} % higher pay than a Bachelor’s degree, whereas secondary education only is associated with {n1(abs(adj[adj.term=='Secondary'].iloc[0]['pct']))} % lower pay."))
add(TBL("t_o3e", "Adjusted percentage differences in pay (HC3 robust 95 % CI) for work arrangement, organisation size and education", ["Factor", "Category", "Difference (%)", "95 % CI (%)", "p"], rows3, [1.5, 1.6, 1.1, 1.8, 0.8], ["l", "l", "r", "r", "r"], note="References: In-person; 100 to 499 employees; Bachelor’s degree. Country, experience and role are controlled."))
rl = adj[adj['var'] == 'dev_g'].sort_values('pct', ascending=False)
add(P(f"Among roles (reference: full-stack developer), the largest adjusted premiums are for senior executives ({n1(rl[rl.term=='Senior executive'].iloc[0]['pct'])} %), engineering managers ({n1(rl[rl.term=='Engineering manager'].iloc[0]['pct'])} %), cloud-infrastructure engineers ({n1(rl[rl.term=='Cloud infra engineer'].iloc[0]['pct'])} %) and data scientists / ML specialists ({n1(rl[rl.term=='Data scientist/ML'].iloc[0]['pct'])} %); academic researchers earn {n1(abs(rl[rl.term=='Academic researcher'].iloc[0]['pct']))} % less. Front-end, desktop/enterprise, mobile and game developers do not differ significantly from full-stack developers."))
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig13_role_effects", 480))
add(FIG("fig14_workmode_org_edu_effects"))
add(FIG("fig15_variance_decomposition", 460))
add(P("[[F:fig13_role_effects]] and [[F:fig14_workmode_org_edu_effects]] show that most differences are modest compared with the country gap of O1, and that the management track and cloud/data specialisations command the clearest premiums. [[F:fig15_variance_decomposition]] makes the relative importance visible: **where you work and how long you have worked explain most of what can be explained; role, company size, work mode and education add only a few percentage points of R² each.** **Interpretation caution:** the remote-work premium is an association. Developers in lower-wage countries who work remotely for employers abroad, differences in seniority not captured by years of experience, and company type may all contribute; the data cannot separate these explanations."))

# ---------------------------------------------------------------- O4
o4 = R["O4"]; ind = o4["india"]; fa = pd.DataFrame(o4["factors"])
add(H2("4.4 Objective O4 — Who uses AI tools? The adoption landscape"))
add(H3("(i) Introduction"))
add(P("AI coding assistants spread faster than almost any earlier developer tool. O4 maps current adoption across age, experience, role, country, organisation size and Stack Overflow engagement, ranks the factors by strength of association, and tests whether India differs from the rest of the world."))
add(H3("(ii) General Description"))
add(P(f"The AI cohort contains {n0(o4['n'])} respondents who answered the question “Do you currently use AI tools in your development process?”. The answer has three levels — *Yes*, *No, but I plan to soon* ({n1(o4['plan_share'])} %) and *No, and I don’t plan to* ({n1(o4['never_share'])} %). For the tests the outcome is binary (uses AI = Yes). Groups with fewer than 300 respondents are not shown for country and role so that every interval is reasonably tight."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `pd.crosstab`, `scipy.stats.chi2_contingency`, the project helpers `cramers_v` (bias-corrected), `wilson_ci` and `two_prop_ztest`."))
add(EQ("χ^{2} = Σ (O_{ij} − E_{ij})^{2} / E_{ij},     V = √( χ̃^{2}/n / min(k−1, r−1) )    (bias-corrected Cramér’s V)"))
add(EQ("Wilson interval:  [ p̂ + z²/2n ± z√( p̂(1−p̂)/n + z²/4n² ) ] / (1 + z²/n),     z-test: z = (p̂_{1} − p̂_{2}) / √( p̂(1−p̂)(1/n_{1} + 1/n_{2}) )"))
add(H3("(iv) Analysis Results"))
add(P(f"Overall, **{n1(o4['overall_rate'])} %** of respondents use AI tools. All ten factors are statistically associated with adoption (p {pv(max(f['p'] for f in o4['factors']))}), but the effect sizes differ greatly ([[T:t_o4f]]): the strongest are age (V = {n3(fa.iloc[0].cramers_v)}), professional experience (V = {n3(fa.iloc[1].cramers_v)}), job role (V = {n3(fa.iloc[2].cramers_v)}) and country (V = {n3(fa.iloc[3].cramers_v)}); all are *small* effects (V of about 0.14–0.16), and Stack Overflow visit frequency (V = {n3(fa[fa.factor=='so_visit_freq'].iloc[0].cramers_v)}) and work arrangement (V = {n3(fa[fa.factor=='work_arrangement'].iloc[0].cramers_v) if (fa.factor=='work_arrangement').any() else n3(fa[fa.factor=='remote_work'].iloc[0].cramers_v)}) are practically irrelevant."))
add(TBL("t_o4f", "Chi-square tests of AI adoption against each factor, ranked by Cramér’s V", ["Factor", "Levels", "n", "χ²", "df", "p", "V"], [[f['label'], str(int(f['k'])), n0(f['n']), n0(f['chi2']), str(int(f['dof'])), pv(f['p']), n3(f['cramers_v'])] for f in o4['factors']], [1.9, 0.7, 0.9, 0.9, 0.6, 0.8, 0.7], ["l", "r", "r", "r", "r", "r", "r"]))
ba = {x['age']: x for x in o4['by_age']}; be = {x['exp_band']: x for x in o4['by_exp']}
add(P(f"Adoption falls steadily with age, from {n1(ba['18-24']['rate'])} % among 18–24-year-olds to {n1(ba['55-64']['rate'])} % at 55–64 and {n1(ba['65+']['rate'])} % at 65+, and with professional experience, from {n1(be['0-2']['rate'])} % (0–2 years) to {n1(be['21+']['rate'])} % (21+ years). Among roles ([[F:fig17_ai_adoption_country_role]]) AI developers ({n1(o4['by_role'][0]['rate'])} %) and data scientists / ML specialists ({n1(o4['by_role'][1]['rate'])} %) lead, whereas sysadmins and embedded developers are lowest ({n1(o4['by_role'][-1]['rate'])} % and {n1(o4['by_role'][-2]['rate'])} %). **India:** {n1(ind['rate'])} % of the {n0(ind['n'])} Indian respondents use AI tools, versus {n1(ind['rest_rate'])} % elsewhere (difference {n1(ind['vs_rest']['diff']*100)} percentage points, z = {n1(ind['vs_rest']['z'])}, p {pv(ind['vs_rest']['p'])}) and {n1(ind['usa_rate'])} % in the USA (z = {n1(ind['vs_usa']['z'])}, p {pv(ind['vs_usa']['p'])})."))
add(H3("(v) Visualization and Interpretation"))
bcn = [x["country"] for x in o4["by_country"]]; india_rank_txt = f"{bcn.index('India')+1}th of {len(bcn)} countries with at least 300 respondents"
add(FIG("fig16_ai_adoption_factors"))
add(FIG("fig17_ai_adoption_country_role"))
add(FIG("fig18_ai_cramers_v", 470))
add(FIG("fig19_ai_role_age_heatmap", 470))
add(P("[[F:fig16_ai_adoption_factors]] shows that the age and experience gradients are monotone and large, whereas organisation size has only a small inverted-U pattern (highest in organisations of 2–99 people) and Stack Overflow visit frequency has almost none. [[F:fig17_ai_adoption_country_role]] places India (orange) {india_rank_txt}, below countries such as China, Pakistan, Nigeria, Israel and Ukraine but clearly above the USA, the UK, Germany and France. [[F:fig18_ai_cramers_v]] ranks the factors, and [[F:fig19_ai_role_age_heatmap]] shows that adoption falls with age *within every one of the ten most common roles*, and that at any age adoption is highest among front-end, full-stack and mobile developers and lowest among embedded and desktop/enterprise developers. **Interpretation:** adoption is highest among the young, early-career, AI-adjacent and — notably — developers in emerging developer markets, consistent with the idea that AI tools are most attractive to people who are still building skills or who see them as a productivity lever. Because the sample is self-selected and the question asks about *current* use, these are snapshot associations, not predictions of future behaviour."))

# ---------------------------------------------------------------- O5
o5 = R["O5"]; cond = TB("tbl_o5_conditionality")
add(H2("4.5 Objective O5 — AI sentiment, trust and job-threat perception"))
add(H3("(i) Introduction"))
add(P("Adoption is only half of the story; developers’ attitudes decide whether AI tools are used wisely. O5 describes favourability, trust in the accuracy of AI output, belief that AI can handle complex tasks, and fear of job loss, and tests whether these differ with career stage, role and country."))
add(H3("(ii) General Description"))
add(P(f"Four follow-up questions were shown only to respondents who use or plan to use AI tools ([[T:t_cond]]): sentiment and job-threat to users and planners ({n0(o5['n_att'])} respondents), trust in accuracy and performance on complex tasks to *current users only* ({n0(o5['n_users'])}). All inference therefore refers to the AI-using or AI-planning population, not to all developers. Ordinal answers are scored 1–5 (e.g. “Highly distrust” = 1 … “Highly trust” = 5; “Unsure” answers are excluded); the threat question is categorical (Yes / No / Not sure)."))
add(TBL("t_cond", "Percentage of respondents answering each AI follow-up question, by their answer to “Do you use AI tools?” (proof of conditional routing)", ["AI-use answer", "ai_sent", "ai_acc", "ai_complex", "ai_threat"], [[r.iloc[0]] + [f"{v:.1f}" for v in r.iloc[1:]] for _, r in cond.iterrows()], [2.6, 1, 1, 1, 1], ["l", "r", "r", "r", "r"], note="The table is the evidence for the leakage rule of Section 3.9: the columns are never filled for respondents who neither use nor plan to use AI, and ai_acc / ai_complex are filled only for current users."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `map_prefix` (label → score), `scipy.stats.spearmanr` with Bonett–Wright CI, `scipy.stats.kruskal`, `scipy.stats.mannwhitneyu`, `chi2_test`/`cramers_v`, `wilson_ci`, `two_prop_ztest`."))
add(EQ("ρ = 1 − 6Σd_{i}^{2} / (n(n²−1)) (no ties; with ties the Pearson correlation of ranks);     ε^{2} = H/(N−1);     V as in Section 4.4"))
add(H3("(iv) Analysis Results"))
dist = o5["dist"]; sp5 = o5["spearman"]; kw = o5["kruskal"]
add(P(f"Attitudes are broadly positive but cautious. {n0(o5['fav_pct'])} % of the {n0(o5['n_att'])} respondents are favourable or very favourable towards AI tools, yet only {n0(o5['trust_positive_pct'])} % of users somewhat or highly trust the accuracy of AI output while {n0(o5['trust_negative_pct'])} % distrust it; only {n0(o5['complex_good_pct'])} % think AI handles complex tasks well, whereas {n0(o5['complex_bad_pct'])} % think it handles them badly. Just {n0(o5['threat_yes_pct'])} % believe AI is a threat to their own job. This “favourable but sceptical” profile is the headline result ([[F:fig20_ai_attitudes_distribution]]). Users are more favourable than those who only plan to use AI (mean {n2(o5['users_vs_planners_sent']['users_mean'])} vs {n2(o5['users_vs_planners_sent']['planners_mean'])}; p {pv(o5['users_vs_planners_sent']['p'])})."))
rows5 = [[k, n2(v['exp']['rho']), f"{n2(v['exp']['ci_low'])} to {n2(v['exp']['ci_high'])}", n0(v['exp']['n']), n2(v['age']['rho']), f"{n2(v['age']['ci_low'])} to {n2(v['age']['ci_high'])}"] for k, v in sp5.items()]
add(TBL("t_o5s", "Spearman correlation of attitudes with professional experience and age (all p < 0.001 given the large n)", ["Attitude", "ρ (experience)", "95 % CI", "n", "ρ (age)", "95 % CI"], rows5, [2.3, 1.0, 1.5, 0.8, 0.9, 1.5], ["l", "r", "r", "r", "r", "r"]))
add(P(f"Experience and age are associated with slightly less favourable and less trusting attitudes, but **the effects are tiny**: the largest rank correlation is ρ = {n2(min(v['exp']['rho'] for v in sp5.values()))} (experience and trust). Kruskal–Wallis tests across experience bands are significant (favourability H = {n0(kw['Favourability']['H'])}, ε² = {n3(kw['Favourability']['eps2'])}; trust H = {n0(kw['Trust']['H'])}, ε² = {n3(kw['Trust']['eps2'])}), but the ε² values show that experience band explains well under 1 % of rank variation — statistical significance without practical significance. For the threat question ([[T:t_o5t]]) the associations are small as well, strongest for country (V = {n3(o5['threat_tests']['country']['cramers_v'])}) and job role (V = {n3(o5['threat_tests']['dev_type']['cramers_v'])})."))
tt = o5["threat_tests"]
add(TBL("t_o5t", "Chi-square tests of the job-threat answer (Yes / No / Not sure) against career and context factors", ["Factor", "n", "χ²", "df", "p", "V"], [[{'exp_band': 'Experience band', 'age': 'Age bracket', 'dev_type': 'Job role (n ≥ 300)', 'country': 'Country (n ≥ 300)'}[k], n0(v['n']), n1(v['chi2']), str(v['dof']), pv(v['p']), n3(v['cramers_v'])] for k, v in tt.items()], [2.2, 0.9, 0.9, 0.6, 0.8, 0.7], ["l", "r", "r", "r", "r", "r"]))
ia = o5["india"]; it = o5["india_threat"]
add(P(f"**India versus the rest.** Indian respondents are more favourable (mean {n2(ia['sent']['india_mean'])} vs {n2(ia['sent']['rest_mean'])}, p {pv(ia['sent']['p'])}), trust AI output more (mean {n2(ia['trust']['india_mean'])} vs {n2(ia['trust']['rest_mean'])}, p {pv(ia['trust']['p'])}) and rate its performance on complex tasks higher (mean {n2(ia['complex_']['india_mean'])} vs {n2(ia['complex_']['rest_mean'])}, p {pv(ia['complex_']['p'])}); they are also somewhat more likely to see AI as a job threat ({n1(it['p1']*100)} % vs {n1(it['p2']*100)} %, p {pv(it['p'])}). Trust and favourability move together (ρ = {n2(o5['trust_sent_rho']['rho'])}), as do trust and perceived competence on complex tasks (ρ = {n2(o5['trust_complex_rho']['rho'])})."))
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig20_ai_attitudes_distribution"))
add(FIG("fig21_trust_sentiment_by_experience"))
add(FIG("fig22_threat_by_role_age"))
add(FIG("fig23_attitude_correlations", 420))
add(P("[[F:fig20_ai_attitudes_distribution]] shows the favourable-but-sceptical profile. [[F:fig21_trust_sentiment_by_experience]] shows almost identical stacked bars across experience bands — the visual counterpart of the tiny ρ values — with a slight shift towards distrust among the most experienced. [[F:fig22_threat_by_role_age]] shows that fear of job loss is highest among front-end developers, project managers, QA/test engineers and the youngest respondents and lowest among embedded developers and senior executives, but all confidence intervals sit within a narrow band around the overall rate. [[F:fig23_attitude_correlations]] shows that the attitude scores form a coherent cluster (favourability, trust and perceived competence positively correlated; threat weakly negative). **Interpretation:** developers use AI tools enthusiastically but verify their output; the data give no evidence of widespread job-loss anxiety, and career stage matters much less for attitudes than for adoption (O4). These are stated attitudes from a self-selected sample at one moment in time."))

# ---------------------------------------------------------------- O6
o6 = R["O6"]; cmpr = pd.DataFrame(o6["comparison"]).set_index("model"); bn = o6["best"]; bp = o6["best_params"]
def short(m): return m
add(H2("4.6 Objective O6 — Predicting annual compensation (regression)"))
add(H3("(i) Introduction"))
add(P("O6 asks whether a developer’s annual compensation can be predicted from the profile information that O1–O3 found to matter, and which family of models predicts it best. A useful model supports salary benchmarking and highlights which information carries predictive power."))
add(H3("(ii) General Description"))
add(P(f"The prediction target is ln(USD pay) of the {n0(co['n'])} cohort members, split into {n0(R['O6_setup']['n_train'])} training and {n0(R['O6_setup']['n_test'])} test rows. There are {R['O6_setup']['n_features_raw']} raw predictors (country, role, experience variables, age, organisation size, education, work arrangement, Stack Overflow engagement, AI use and others; Section 3.8). Six models are compared: two baselines (global median; *country median*, a domain-aware benchmark that shows how much is plain geography) and four learning algorithms of increasing flexibility — **Ridge regression**, **Random Forest**, **HistGradientBoosting** and **LightGBM**. Back-transformed predictions exp(ŷ) estimate the *conditional median* pay."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `train_test_split(stratify=…)`, `ColumnTransformer` / `Pipeline` (median imputation with missing-indicators, standardisation, one-hot encoding, ordinal encoding, cross-fitted `TargetEncoder`), `RandomizedSearchCV` (8 candidates, 3-fold) for hyper-parameter tuning on the training set only, `cross_validate` (5-fold) for the fair comparison, and the helper `reg_metrics`."))
add(EQ("Ridge:  β̂ = argmin_{β} Σ(y_{i} − x_{i}′β)^{2} + α‖β‖^{2}     (shrinks coefficients; handles correlated experience variables)"))
add(EQ("Random Forest:  ŷ(x) = (1/B) Σ_{b=1}^{B} T_{b}(x)     (average of B de-correlated bootstrap trees)"))
add(EQ("Gradient boosting:  F_{m}(x) = F_{m−1}(x) + η·h_{m}(x),  h_{m} fitted to the residuals y − F_{m−1}(x)     (HistGB and LightGBM use histogram split-finding)"))
add(EQ("RMSE = √(Σe_{i}^{2}/n),  MAE = Σ|e_{i}|/n,  R^{2} = 1 − Σe_{i}^{2}/Σ(y_{i} − ȳ)^{2},  MdAPE = median(|ŷ_{USD} − y_{USD}| / y_{USD})"))
add(P("**Pre-registered success criteria:** (a) the best model reduces the baseline test RMSE (log scale) by at least 40 %; (b) it beats the country-median baseline."))
add(H3("(iv) Analysis Results"))
mrows = []
for m, r in cmpr.iterrows():
    mrows.append([m + (" ★" if m == bn else ""), f"{r.cv_rmse_mean:.3f} ± {r.cv_rmse_std:.3f}", f"{r.cv_r2_mean:.3f}", f"{r.test_RMSE_log:.3f}", f"{r.test_R2:.3f}", n0(r.test_MAE_USD), f"{r.test_MdAPE_pct:.1f}", f"{r.test_Within25_pct:.1f}"])
add(P(f"[[T:t_o6]] gives cross-validated and hold-out results. The global-median baseline has a test R² of about zero by construction; the country-median baseline already reaches R² = {n3(cmpr.loc['Baseline: country median','test_R2'])} (RMSE {n3(cmpr.loc['Baseline: country median','test_RMSE_log'])}), confirming O1. All four learning models improve on it: Ridge R² = {n3(cmpr.loc['Ridge regression','test_R2'])}, Random Forest {n3(cmpr.loc['Random Forest','test_R2'])}, HistGradientBoosting {n3(cmpr.loc['HistGradientBoosting','test_R2'])} and **{bn} {n3(cmpr.loc[bn,'test_R2'])}**. The selected model (lowest cross-validated RMSE: {n3(cmpr.loc[bn,'cv_rmse_mean'])}) has a hold-out RMSE of {n3(cmpr.loc[bn,'test_RMSE_log'])} on the log scale (95 % bootstrap CI {n3(o6['test_rmse_ci'][0])}–{n3(o6['test_rmse_ci'][1])}), R² = {n3(cmpr.loc[bn,'test_R2'])} (95 % CI {n3(o6['test_r2_ci'][0])}–{n3(o6['test_r2_ci'][1])}), a mean absolute error of {usd(cmpr.loc[bn,'test_MAE_USD'])}, a median absolute percentage error of {n1(cmpr.loc[bn,'test_MdAPE_pct'])} %, and {n1(cmpr.loc[bn,'test_Within25_pct'])} % of predictions fall within ±25 % of the true salary."))
add(TBL("t_o6", "Regression models: 5-fold CV on the training set and hold-out test performance (★ = selected)", ["Model", "CV RMSE (log) ± SD", "CV R²", "Test RMSE (log)", "Test R²", "Test MAE (USD)", "MdAPE (%)", "Within ±25 %"], mrows, [2.1, 1.5, 0.7, 0.9, 0.7, 0.9, 0.9, 0.8], ["l", "r", "r", "r", "r", "r", "r", "r"], size=17))
lo2, hi2 = o6["paired_mse_diff_ci"]
add(P(f"**Is the winner really better?** The four learners are close. A paired bootstrap of the test squared errors between {bn} and the runner-up ({o6['runner_up']}) gives a mean-squared-error difference with 95 % CI {lo2:.4f} to {hi2:.4f}; {bn} has the lower error in {o6['paired_diff_prob_best_better']*100:.0f} % of resamples. Differences among the gradient-boosting variants are therefore small relative to the gap between learners and baselines, and the choice between them is a matter of marginal evidence, not a decisive win."))
rr, rc = o6["rmse_reduction_vs_global_pct"], o6["rmse_reduction_vs_country_pct"]
add(P(f"**Success criteria.** (a) The test RMSE is {n1(rr)} % lower than the global-median baseline — {'the target of 40 % was met' if o6['success_rmse_cut_40'] else 'the 40 % target was therefore **not met**, narrowly' if rr > 38 else 'the 40 % target was **not met**'}. (b) It is {n1(rc)} % lower than the country-median baseline — {'criterion met' if o6['success_beats_country_median'] else 'criterion not met'}. The honest reading is that the data’s predictive ceiling is roughly R² ≈ 0.6: most of what these survey variables can explain is already captured by country and experience."))
bpr = bp[bn]
add(P(f"**Tuned hyper-parameters of {bn}:** " + ", ".join(f"{k.replace('m__','')} = {v}" for k, v in bpr.items()) + f". Residuals on the log scale have mean {n3(o6['residual']['mean'])}, standard deviation {n3(o6['residual']['sd'])}, skewness {n2(o6['residual']['skew'])} and excess kurtosis {n2(o6['residual']['kurt'])}: they are centred but heavy-tailed, with a long negative tail of respondents earning far less than predicted (typically part-time, student or sub-contract earnings)."))
se = pd.DataFrame(o6["sensitivity"])
add(P(f"**Sensitivity to the outlier policy** ([[T:t_sens]]): cross-validated RMSE of the tuned HistGradientBoosting model is {n3(se.cv_rmse_log.min())}–{n3(se.cv_rmse_log.max())} across the three policies — the stricter the filter, the smaller the error, as expected, because fewer extreme points remain; the chosen 3 × IQR rule is intermediate. Absolute error levels in this report are therefore conditional on the cleaning policy; only HistGradientBoosting was re-fitted under the alternatives, so the stability of the *ranking* of models under other policies was not tested."))
add(TBL("t_sens", "Sensitivity of cross-validated performance to the outlier policy (HistGradientBoosting)", ["Policy", "n", "CV RMSE (log)", "CV R²", "CV MAE (log)"], [[r.policy, n0(r.n), n3(r.cv_rmse_log), n3(r.cv_r2), n3(r.cv_mae_log)] for r in se.itertuples()], [3.0, 0.8, 1.2, 0.9, 1.2], ["l", "r", "r", "r", "r"]))
lc = o6["learning_curve"]
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig24_regression_model_comparison"))
add(FIG("fig25_regression_diagnostics"))
add(FIG("fig26_learning_curve", 400))
add(P(f"[[F:fig24_regression_model_comparison]] shows the ordering baseline < Ridge < Random Forest < gradient boosting on every metric. [[F:fig25_regression_diagnostics]] (a) shows predictions scattered around the diagonal with a tendency to over-predict very low salaries; (b) and (c) show roughly centred residuals with no strong dependence on the fitted value. The learning curve ([[F:fig26_learning_curve]]) shows the validation RMSE still falling slowly from {n3(lc['val'][-2])} to {n3(lc['val'][-1])} as the training set grows, and a gap to the training RMSE ({n3(lc['train'][-1])}); more data, not a more complex model, is the better route to improvement. **Interpretation:** a model using only self-reported profile information predicts developer pay within about a quarter of the true value for roughly {n0(cmpr.loc[bn,'test_Within25_pct'])} % of developers, which is useful for benchmarking but not for individual salary-setting."))

# ---------------------------------------------------------------- O7
o7 = R["O7"]; cc7 = pd.DataFrame(o7["comparison"]).set_index("model"); b7 = o7["best"]; ld = R["O7_leakage_demo"]
add(H2("4.7 Objective O7 — Predicting AI-tool adoption (classification)"))
add(H3("(i) Introduction"))
add(P("O7 asks whether a developer’s current use of AI tools can be predicted from their background alone, and shows — by deliberate contrast — how badly the answer is distorted if leaky variables are allowed."))
add(H3("(ii) General Description"))
add(P(f"The target is binary (1 = currently uses AI tools; prevalence {n1(o7['positive_rate']*100)} %). The {n0(o7['n_train'])} training and {n0(o7['n_test'])} test rows are split with stratification on the label. Predictors are the same profile variables as in O6 plus the respondent’s coding relationship (professional, learner, hobbyist…) but *not* salary, and never the four conditional AI follow-up questions. Models: a majority-class baseline, **Logistic Regression**, **Random Forest**, **HistGradientBoosting** and **LightGBM**."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: the same pipelines as O6 with `StratifiedKFold`, `LogisticRegression`, `RandomForestClassifier`, `HistGradientBoostingClassifier`, `LGBMClassifier`, `roc_curve`, `precision_recall_curve`, `calibration_curve`, `cross_val_predict` (out-of-fold probabilities for threshold selection) and `clf_metrics`."))
add(EQ("Logistic regression:  P(y=1|x) = 1 / (1 + e^{−(β_{0} + x′β)})     (log-odds are linear in the features)"))
add(EQ("Precision = TP/(TP+FP),  Recall = TP/(TP+FN),  F_{1} = 2PR/(P+R),  Balanced accuracy = (TPR + TNR)/2,  Brier = Σ(p̂_{i} − y_{i})^{2}/n"))
add(EQ("ROC-AUC = P(p̂_{pos} > p̂_{neg}) = area under TPR-versus-FPR curve"))
add(P("**Pre-registered success criteria:** ROC-AUC ≥ 0.65 on the hold-out set, and F1 above the majority-class baseline."))
add(H3("(iv) Analysis Results"))
add(P(f"**Leakage demonstration.** One identical gradient-boosting model reaches a hold-out ROC-AUC of **{n3(ld['auc_honest'])}** with legitimate predictors but **{n3(ld['auc_leaky'])}** when the four conditional AI follow-up variables are added — a perfect-looking but meaningless score of +{n3(ld['auc_leaky']-ld['auc_honest'])} that arises because “was the question answered?” reveals the label ([[T:t_cond]]). All reported results exclude these variables."))
crow = []
for m, r in cc7.iterrows():
    crow.append([m + (" ★" if m == b7 else ""), f"{r.cv_roc_auc_mean:.3f} ± {r.cv_roc_auc_std:.3f}", f"{r.test_Accuracy:.3f}", f"{r.test_Precision:.3f}", f"{r.test_Recall:.3f}", f"{r.test_F1:.3f}", f"{r.test_ROC_AUC:.3f}", f"{r.test_PR_AUC:.3f}", f"{r.test_Brier:.3f}"])
add(P(f"[[T:t_o7]] compares the models. Every learning model reaches a cross-validated ROC-AUC of about 0.66–0.67, and the selected {b7} has a hold-out ROC-AUC of {n3(o7['metrics_at_05']['ROC_AUC'])} (95 % bootstrap CI {n3(o7['auc_ci'][0])}–{n3(o7['auc_ci'][1])}). Models differ by about 0.01 AUC or less, so the choice among them is not decisive; the simplest (Logistic Regression) is almost as good as the most complex."))
add(TBL("t_o7", "Classification models: 5-fold CV ROC-AUC and hold-out metrics at threshold 0.5 (★ = selected)", ["Model", "CV ROC-AUC ± SD", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC", "Brier"], crow, [2.2, 1.4, 0.65, 0.65, 0.7, 0.6, 0.8, 0.75, 0.65], ["l", "r", "r", "r", "r", "r", "r", "r", "r"], size=17))
m05, mo, mb = o7["metrics_at_05"], o7["metrics_at_opt"], o7["metrics_at_bal"]
add(P(f"**Why F1 is misleading here.** Because 61.8 % of respondents use AI, the trivial rule “predict *yes* for everybody” already scores recall 1.0 and F1 = {n3(o7['baseline_f1'])}. At the default threshold 0.5 the selected model’s F1 is {n3(m05['F1'])} — *below* that trivial baseline — so the F1 criterion is **{'met' if o7['success_f1_at_05'] else 'not met at threshold 0.5'}**; at the F1-optimal threshold {n2(o7['threshold_opt'])} (chosen on out-of-fold training predictions only) F1 rises to {n3(mo['F1'])}, {'barely above' if mo['F1']>o7['baseline_f1'] else 'not above'} the baseline, but only because the classifier then labels almost everybody a user. The ROC-AUC criterion (≥ 0.65) is **{'met' if o7['success_auc_065'] else 'not met'}**. Balanced accuracy, which weights both classes equally, is the fairer yardstick: at threshold {n2(o7['threshold_bal'])} (balanced-accuracy optimum on out-of-fold data) the selected model reaches balanced accuracy {n3(mb['BalAcc'])}, precision {n3(mb['Precision'])} and recall {n3(mb['Recall'])} ([[T:t_thr]])."))
add(TBL("t_thr", "Selected model on the hold-out set at three decision thresholds", ["Threshold rule", "Value", "Accuracy", "Bal. acc.", "Precision", "Recall", "F1"], [[n, f"{t:.2f}", f"{m['Accuracy']:.3f}", f"{m['BalAcc']:.3f}", f"{m['Precision']:.3f}", f"{m['Recall']:.3f}", f"{m['F1']:.3f}"] for n, t, m in [("Default", 0.5, m05), ("F1-optimal (OOF)", o7['threshold_opt'], mo), ("Balanced-accuracy optimal (OOF)", o7['threshold_bal'], mb)]], [2.5, 0.8, 0.9, 0.9, 0.9, 0.8, 0.7], ["l", "r", "r", "r", "r", "r", "r"]))
cf = np.array(o7["confusion"])
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig27_classification_model_comparison"))
add(FIG("fig28_roc_pr_curves", 500))
add(FIG("fig29_confusion_calibration_threshold"))
add(P(f"[[F:fig27_classification_model_comparison]] and [[F:fig28_roc_pr_curves]] show that all four learners trace almost the same ROC curve, well above chance but far from perfect. [[F:fig29_confusion_calibration_threshold]] (a) gives the confusion matrix at the balanced threshold: of {n0(cf[0].sum())} non-users {n0(cf[0,0])} are identified correctly, and of {n0(cf[1].sum())} users {n0(cf[1,1])}. Panel (b) shows the probabilities are *well calibrated* — a predicted 70 % really corresponds to roughly 70 % adopters — and panel (c) shows the precision/recall trade-off as the threshold moves. **Interpretation:** background information carries real but limited information about who uses AI tools (AUC ≈ {n2(m05['ROC_AUC'])}); individual adoption is mostly driven by factors the survey does not measure (employer policy, tasks, personal preference). The model is suitable for describing population-level propensities, not for deciding about individuals."))

# ---------------------------------------------------------------- O8
o8 = R["O8"]; er = o8["error"]
NICE = {"years_code_pro": "Professional years", "years_code": "Total years coded", "country": "Country", "dev_type": "Job role", "org_log": "Org. size (log)", "ed_rank": "Education rank", "age_mid": "Age", "remote_work": "Work arrangement", "so_visit_rank": "SO visit frequency", "so_part_freq": "SO participation", "purchase_influence": "Purchase influence", "main_branch": "Coding relationship", "so_account": "SO account", "pro_share": "Professional share of career", "exp_gap": "Years coded before 1st job", "ai_select": "Uses AI tools", "r_used": "Used R", "buildvs_buy": "Build-vs-buy", "org_short": "Org. size (category)", "ed_short": "Education", "so_comm": "SO community", "years_pro_sq": "Prof. years²"}
nm = lambda f: NICE.get(f, f)
add(H2("4.8 Objective O8 — Explainability and error analysis"))
add(H3("(i) Introduction"))
add(P("A model that cannot be explained or audited should not be trusted. O8 identifies which inputs drive each model, in which direction, and where the models make larger errors — including whether the models are less reliable for Indian developers than for others."))
add(H3("(ii) General Description"))
add(P("Three complementary explanation tools are used: *permutation importance* on the hold-out set (model-agnostic; the loss in performance when one raw column is shuffled), *SHAP values* (Shapley-value attributions; computed on the LightGBM models, which perform on par with the selected ones) and *partial dependence*. Errors of both models are then audited by segment."))
add(H3("(iii) Specific Requirements, Functions and Formulas"))
add(P("Functions: `sklearn.inspection.permutation_importance` (5 repeats), `shap.TreeExplainer(...).shap_values`, `shap.summary_plot`, a hand-written partial-dependence routine that recomputes the derived experience features consistently, and `groupby` error summaries."))
add(EQ("Permutation importance of feature j:  I_{j} = Err(X with column j shuffled) − Err(X)"))
add(EQ("Shapley value:  φ_{j} = Σ_{S⊆F∖{j}} [ |S|!(|F|−|S|−1)! / |F|! ] · ( f(S∪{j}) − f(S) ),     f(x) = φ_{0} + Σ_{j}φ_{j}"))
add(EQ("Partial dependence:  PD_{j}(v) = (1/n) Σ_{i} f(x_{i,−j}, x_{j} = v)"))
add(H3("(iv) Analysis Results"))
st, at_ = o8["salary_top"], o8["ai_top"]
add(P(f"**Salary model.** Country dominates: shuffling it raises the log-RMSE by {n3(st[0]['imp'])}, versus {n3(st[1]['imp'])} for professional years and {n3(st[2]['imp'])} for organisation size ([[T:t_imp]]). The three most important features account for {n0(o8['salary_top3_share'])} % of the total permutation importance. SHAP agrees on the ordering ({', '.join(nm(d['feature']) for d in o8['shap_salary_top'][:5])})."))
add(P(f"**Adoption model.** Importance is spread over more features: {', '.join(nm(a['feature']) for a in at_[:4])} lead, the top three accounting for {n0(o8['ai_top3_share'])} % of importance. Total years coded matters more than professional years, consistent with O4: AI adoption is associated with how long a person has been coding (a proxy for career stage and age) more than with company characteristics."))
add(TBL("t_imp", "Top permutation importances on the hold-out set (salary: increase in log-RMSE; adoption: decrease in ROC-AUC)", ["Rank", "Salary model feature", "Importance", "Adoption model feature", "Importance"], [[str(i + 1), nm(st[i]['feature']), f"{st[i]['imp']:.4f}", nm(at_[i]['feature']), f"{at_[i]['imp']:.4f}"] for i in range(8)], [0.6, 2.2, 1.1, 2.2, 1.1], ["c", "l", "r", "l", "r"]))
pe = o8["pdp_exp"]
add(P(f"**Partial dependence.** Setting every respondent’s professional experience to 0, 5, 10, 20 and 30 years, the average predicted pay is {', '.join(n0(v) for v in pe['All'])} USD for the whole test set, {', '.join(n0(v) for v in pe['USA'])} USD if all were in the USA, and {', '.join(n0(v) for v in pe['India'])} USD if all were in India. The model therefore reproduces the O2 finding of steep early gains and a plateau after about 15–20 years, and the O1 finding of a large country offset. The predicted adoption probability falls from {n1(list(o8['pdp_age'].values())[0])} % to {n1(list(o8['pdp_age'].values())[-1])} % as the age bracket midpoint moves from {list(o8['pdp_age'].keys())[0]} to {list(o8['pdp_age'].keys())[-1]}."))
bc = pd.DataFrame(er["by_country"])
add(P(f"**Error audit — salary model.** Overall the median absolute percentage error is {n1(er['overall_mdape'])} % (MAE {usd(er['overall_mae'])}). Errors differ by country ([[T:t_err]]): lowest for Germany, France and Spain, highest for Ukraine and Brazil. For India the MdAPE is {n1(er['india_mdape'])} % ({er['india_n']} test respondents) versus {n1(er['usa_mdape'])} % for the USA and {n1(er['rest_mdape'])} % for all non-Indian respondents; the mean log-residual for India is {n2(er['india_bias'])}, i.e. the model {'under' if er['india_bias']>0 else 'over'}-predicts Indian salaries on average. Errors are nearly constant across experience bands and similar across roles (MdAPE about 21–30 %), so the dominant failure mode is *country-specific pay dispersion*, not seniority."))
add(TBL("t_err", "Salary-model error by country on the hold-out set (10 largest test groups)", ["Country", "n", "MdAPE %", "MAE (USD)", "Mean log-residual", "Within ±25 %"], [[r['country'], n0(r['n']), n1(r['MdAPE_pct']), n0(r['MAE_USD']), f"{r['bias_log']:+.3f}", n1(r['within25'])] for r in er['by_country']], [1.4, 0.7, 1.0, 1.1, 1.4, 1.1], ["l", "r", "r", "r", "r", "r"]))
wr = pd.DataFrame(o8["worst"])
add(P("The largest individual errors in dollars ([[T:t_worst]]) are all *very high* earners — here all from the USA, reporting USD 0.75–1.0 million, three to six times what their profile predicts. Such pay (equity, bonuses, executive packages) cannot be inferred from the profile variables available. On the log scale the biggest errors are instead the very *low* reported salaries (the long negative residual tail noted in Section 4.6); the two kinds of error mirror each other."))
add(TBL("t_worst", "The eight largest salary-prediction errors on the hold-out set (USD)", ["Country", "Role", "Years pro", "Org. size", "Actual", "Predicted", "Error"], [[r['country'], r['dev_type'], ("—" if pd.isna(r['years_code_pro']) else n0(r['years_code_pro'])), r['org_short'], n0(r['converted_comp_yearly']), n0(r['pred_usd']), n0(r['err_usd'])] for r in o8['worst']], [1.1, 1.8, 0.8, 1.1, 0.9, 0.9, 0.9], ["l", "l", "r", "l", "r", "r", "r"], size=17))
ce = o8["clf_error"]; ci_ = ce["india"][0] if ce["india"] else None
add(P(f"**Error audit — adoption model.** Within-group ROC-AUC ranges from {n2(min(x['auc'] for x in ce['by_age']))} to {n2(max(x['auc'] for x in ce['by_age']))} across age brackets, {n2(min(x['auc'] for x in ce['by_country']))}–{n2(max(x['auc'] for x in ce['by_country']))} across the ten largest countries and {n2(min(x['auc'] for x in ce['by_role']))}–{n2(max(x['auc'] for x in ce['by_role']))} across roles; discrimination is weakest for engineering managers and for the youngest and oldest age brackets (small groups). " + (f"For India the AUC is {n2(ci_['auc'])} with accuracy {n1(ci_['accuracy'])} %, comparable with the overall model." if ci_ else "")))
dm = pd.DataFrame(R["demo"])
add(TBL("t_demo", "Illustrative predictions of the saved models (not forecasts for real individuals)", ["Profile", "Predicted salary (USD)", "P(uses AI)"], [[r['profile'], n0(r['predicted_salary_usd']), f"{r['p_uses_ai']:.2f}"] for r in R['demo']], [3.6, 1.6, 1.2], ["l", "r", "r"]))
add(H3("(v) Visualization and Interpretation"))
add(FIG("fig30_perm_importance_salary", 450)); add(FIG("fig31_perm_importance_ai", 450))
add(FIG("fig32_shap_salary", 470)); add(FIG("fig33_shap_ai", 470))
add(FIG("fig34_partial_dependence")); add(FIG("fig35_salary_error_by_segment")); add(FIG("fig36_ai_error_by_segment"))
add(P("[[F:fig30_perm_importance_salary]] and [[F:fig32_shap_salary]] show that country is overwhelmingly the most informative input; in the SHAP plot, countries with high wages (red dots at the right) push predictions up by up to roughly +0.8 on the log scale and low-wage countries down by more than 1.5. Experience variables push predictions up with a saturating effect. [[F:fig31_perm_importance_ai]] and [[F:fig33_shap_ai]] show a flatter structure for adoption. *Reading note:* in SHAP plots, colour for ordinal-encoded categorical variables (e.g. work arrangement) reflects an arbitrary code order, not a meaningful “high” or “low”. [[F:fig34_partial_dependence]] confirms the experience plateau and the age gradient. [[F:fig35_salary_error_by_segment]] shows country-specific error levels (India in orange) and flat errors across experience bands; [[F:fig36_ai_error_by_segment]] shows that classifier quality is similar across most segments. **Interpretation:** the models learned the same structure the statistical analyses of O1–O5 found; error is concentrated in markets with highly dispersed pay (Ukraine, Brazil, India), which a practitioner should treat as lower-confidence predictions."))

# ================================================================== 5. CONCLUSION
add(H1("5. Conclusion"))
add(H2("5.1 Summary of findings against the objectives"))
mgs = o2["marginal"]
add({"t": "numbered", "ref": "num2", "items": [
 f"**O1 (geography).** Country alone explains about {n0(o1['R2_country_all']*100)} % of the variation in log-pay. India’s median pay ({usd(o1['india_median'])}) is {n2(o1['india_usa_ratio'])} × the USA median and the lowest of the {o1['n_ranked']} large markets; it is significantly below {'all ' + str(o1['india_sig_lower_than']) + ' other countries' if o1['india_sig_lower_than']==o1['n_ranked']-1 else str(o1['india_sig_lower_than']) + ' of the other ' + str(o1['n_ranked']-1)}.",
 f"**O2 (experience).** Professional experience is moderately correlated with pay (ρ = {n2(o2['spearman_all']['rho'])}); the estimated gain is {n1(mgs[0]['pct_per_year'])} % per year at entry level, {n1(mgs[3]['pct_per_year'])} % at ten years and close to zero after about {n0(o2['peak_years'])} years. The early-career slope is much steeper in India ({n1(o2['separate_fits']['India']['pct_at_0'])} % vs {n1(o2['separate_fits']['USA']['pct_at_0'])} % in the USA).",
 f"**O3 (net effects).** After adjustment, remote work (+{n1(rm['pct'])} %), very large employers, management roles and cloud/data specialisations carry premiums, but role, company size, work mode and education together add only about {n0((o3['final_R2']-nest[1]['R2'])*100)} percentage points of R² beyond country and experience.",
 f"**O4 (adoption).** {n1(o4['overall_rate'])} % of respondents use AI tools; adoption falls with age and experience and is highest among AI developers and data scientists. India’s adoption ({n1(ind['rate'])} %) exceeds the rest of the world’s ({n1(ind['rest_rate'])} %).",
 f"**O5 (attitudes).** Developers are favourable ({n0(o5['fav_pct'])} %) but sceptical about accuracy (only {n0(o5['trust_positive_pct'])} % trust it); {n0(o5['threat_yes_pct'])} % see AI as a threat to their job. Career-stage effects on attitudes are statistically significant but practically negligible.",
 f"**O6 (salary prediction).** {bn} was selected (hold-out R² = {n3(cmpr.loc[bn,'test_R2'])}, MAE {usd(cmpr.loc[bn,'test_MAE_USD'])}, MdAPE {n1(cmpr.loc[bn,'test_MdAPE_pct'])} %). It beats the country-median baseline by {n1(rc)} % in RMSE but reduces the global-baseline RMSE by {n1(rr)} %, {'meeting' if o6['success_rmse_cut_40'] else 'narrowly missing'} the pre-registered 40 % target.",
 f"**O7 (adoption prediction).** {b7} reaches a hold-out ROC-AUC of {n3(o7['metrics_at_05']['ROC_AUC'])} (criterion ≥ 0.65 met) with good calibration, but F1 at threshold 0.5 is below the trivial all-yes baseline; adoption is only modestly predictable from background alone. Including leaky follow-up variables would have inflated AUC to {n3(ld['auc_leaky'])}.",
 f"**O8 (explanation).** Country is by far the most important salary predictor, followed by professional experience; for adoption, total years coded, role and country lead. Salary errors are largest for Ukraine, Brazil and India (India MdAPE {n1(er['india_mdape'])} %)."]})
add(H2("5.2 Recommendations"))
add(BL("**For students and early-career developers (including in India):** because the marginal return to experience is steepest in the first five to ten years and flattens after about two decades, early skill building — and moving into management, cloud or data specialisations where adjusted premiums are largest — is associated with the biggest pay growth. These are associations, not guarantees.",
       "**For anyone benchmarking pay:** always compare within the same country first; roughly half of the variation in nominal pay is geographic. Use the saved salary model only as a rough guide (median error about 27 %; only about 47 % of predictions fall within ±25 % of the true value).",
       "**For teams adopting AI tools:** trust in AI output is much lower than enthusiasm for it. Training should emphasise verification, particularly for complex tasks where most users rate AI performance as only moderate.",
       "**For analysts reusing this survey:** treat conditional questions (ai_sent, ai_acc, ai_complex, ai_threat) as structurally missing and never use them as predictors of the screening question; report effect sizes, not just p-values, because with tens of thousands of respondents everything is “significant”."))
add(H2("5.3 Limitations"))
add(BL("**Self-selected, self-reported data.** Results describe survey respondents, not all developers; compensation is unverified and in nominal USD without purchasing-power adjustment.",
       "**Observational design.** No causal claims can be drawn (for example about remote work, education or AI use and pay).",
       "**Single survey year and limited features.** Only single-response questions were available; languages, technologies and AI tools used could not be used as predictors.",
       "**Modest predictive ceiling.** The salary model explains about 62 % of the variance in log-pay; the adoption model’s AUC is about 0.67. More informative predictors and more data are likely to help more than more complex algorithms (learning curve, Section 4.6; the four learners perform almost identically).",
       "**Small subgroups.** India has about 1,000 salary reports, so Indian-specific estimates have wider uncertainty and may not represent all Indian developers."))

# ================================================================== 6. FUTURE SCOPE
add(H1("6. Future Scope"))
add(BL("**Multi-year analysis:** repeat the pipeline on the 2019–2025 Stack Overflow surveys to measure how AI adoption, trust and the experience premium change over time, and to test whether the models generalise across years (temporal validation).",
       "**Purchasing-power and cost-of-living adjustment:** convert salaries using PPP factors to separate genuine pay gaps from price-level differences, especially for the India-versus-USA comparison.",
       "**Richer features:** use the multi-select questions (languages, databases, cloud platforms, AI tools used, learning sources) with multi-hot or embedding representations, which should lift both models above the current ceiling.",
       "**Uncertainty-aware prediction:** quantile regression or conformal prediction to give salary *ranges* with guaranteed coverage instead of single point estimates.",
       "**Hierarchical modelling:** mixed-effects (multilevel) models with country random effects to share strength between small and large countries.",
       "**Causal inference:** propensity-score or instrumental-variable designs to study the effect of remote work or AI use on pay, which the observational analysis here cannot establish.",
       "**Ordinal and fairness analyses:** ordinal logistic regression for attitudes; formal fairness audits of the salary model across gender, age and country groups.",
       "**Deployment:** wrap the saved pipelines in a Streamlit or FastAPI application with SHAP-based per-person explanations and monitoring for data drift."))

# ================================================================== 7. REFERENCES
add(H1("7. References"))
refs = [
 "Stack Overflow, “2024 Developer Survey,” 2024. [Online]. Available: https://survey.stackoverflow.co/2024/",
 "H. Khurana and the R4DS Online Learning Community, “TidyTuesday: Stack Overflow Annual Developer Survey 2024 (week 36),” GitHub repository rfordatascience/tidytuesday, 2024. [Online]. Available: https://github.com/rfordatascience/tidytuesday/tree/main/data/2024/2024-09-03",
 "F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.",
 "G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146–3154.",
 "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.",
 "L. Breiman, “Random forests,” *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.",
 "J. H. Friedman, “Greedy function approximation: A gradient boosting machine,” *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.",
 "A. E. Hoerl and R. W. Kennard, “Ridge regression: Biased estimation for nonorthogonal problems,” *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.",
 "T. Hastie, R. Tibshirani and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. New York: Springer, 2009.",
 "S. Kaufman, S. Rosset, C. Perlich and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, art. 15, 2012.",
 "D. Micci-Barreca, “A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems,” *ACM SIGKDD Explorations*, vol. 3, no. 1, pp. 27–32, 2001.",
 "W. H. Kruskal and W. A. Wallis, “Use of ranks in one-criterion variance analysis,” *Journal of the American Statistical Association*, vol. 47, no. 260, pp. 583–621, 1952.",
 "H. Cramér, *Mathematical Methods of Statistics*. Princeton, NJ: Princeton University Press, 1946.",
 "E. B. Wilson, “Probable inference, the law of succession, and statistical inference,” *Journal of the American Statistical Association*, vol. 22, no. 158, pp. 209–212, 1927.",
 "S. Holm, “A simple sequentially rejective multiple test procedure,” *Scandinavian Journal of Statistics*, vol. 6, no. 2, pp. 65–70, 1979.",
 "N. Cliff, “Dominance statistics: Ordinal analyses to answer ordinal questions,” *Psychological Bulletin*, vol. 114, no. 3, pp. 494–509, 1993.",
 "J. G. MacKinnon and H. White, “Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties,” *Journal of Econometrics*, vol. 29, no. 3, pp. 305–325, 1985.",
 "B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York: Chapman & Hall, 1993.",
 "W. S. Cleveland, “Robust locally weighted regression and smoothing scatterplots,” *Journal of the American Statistical Association*, vol. 74, no. 368, pp. 829–836, 1979.",
 "S. Seabold and J. Perktold, “Statsmodels: Econometric and statistical modeling with Python,” in *Proc. 9th Python in Science Conference*, 2010, pp. 92–96.",
 "P. Virtanen et al., “SciPy 1.0: Fundamental algorithms for scientific computing in Python,” *Nature Methods*, vol. 17, pp. 261–272, 2020.",
 "C. R. Harris et al., “Array programming with NumPy,” *Nature*, vol. 585, pp. 357–362, 2020.",
 "W. McKinney, “Data structures for statistical computing in Python,” in *Proc. 9th Python in Science Conference*, 2010, pp. 56–61.",
 "J. D. Hunter, “Matplotlib: A 2D graphics environment,” *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90–95, 2007.",
 "M. L. Waskom, “seaborn: statistical data visualization,” *Journal of Open Source Software*, vol. 6, no. 60, art. 3021, 2021.",
]
add({"t": "numbered", "ref": "num3", "items": refs})

# ================================================================== RESOLVE CROSS-REFERENCES
fig_no, tab_no = {}, {}
for b in B:
    if b["t"] == "fig": fig_no[b["id"]] = len(fig_no) + 1
    if b["t"] == "table": tab_no[b["id"]] = len(tab_no) + 1
def resolve(s):
    s = re.sub(r"\[\[F:([a-z0-9_]+)\]\]", lambda m: f"Figure {fig_no[m.group(1)]}", s)
    return re.sub(r"\[\[T:([a-z0-9_]+)\]\]", lambda m: f"Table {tab_no[m.group(1)]}", s)
for blk in B + FR:
    for k in ("text", "caption", "note"):
        if isinstance(blk.get(k), str): blk[k] = resolve(blk[k])
    if "items" in blk: blk["items"] = [resolve(i) for i in blk["items"]]
    if "rows" in blk: blk["rows"] = [[resolve(str(c)) for c in r] for r in blk["rows"]]
missing = [f for f in FIGCAP if f not in fig_no]
print("figures used:", len(fig_no), "| tables:", len(tab_no), "| registered but unused figures:", missing)

content = {"title": TITLE, "sections": [
    {"blocks": COVER},
    {"numbering": {"fmt": "roman", "start": 1}, "blocks": FR},
    {"numbering": {"fmt": "decimal", "start": 1}, "blocks": B},
]}
json.dump(content, open(Path(__file__).parent / "content.json", "w"), ensure_ascii=False, indent=1)
print("content.json written; main blocks:", len(B))
