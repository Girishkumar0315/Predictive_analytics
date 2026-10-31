"""Data loading, label decoding, quality assessment and cohort construction.

All functions are stateless (they never learn parameters from data), so they can be
applied before the train/test split without causing leakage.  Anything that *learns*
from data (imputation, scaling, target encoding) lives in `features.py` pipelines.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_RAW, RAW_FILE, CROSSWALK_FILE, QUESTIONS_FILE

# ----------------------------------------------------------------------------------
# Label handling
# ----------------------------------------------------------------------------------
COUNTRY_SHORT = {
    "United States of America": "USA",
    "United Kingdom of Great Britain and Northern Ireland": "UK",
    "Russian Federation": "Russia",
    "Iran, Islamic Republic of...": "Iran",
    "Republic of Korea": "S. Korea",
    "Viet Nam": "Vietnam",
    "Hong Kong (S.A.R.)": "Hong Kong",
    "United Republic of Tanzania": "Tanzania",
    "Czech Republic": "Czechia",
    "The former Yugoslav Republic of Macedonia": "N. Macedonia",
    "Venezuela, Bolivarian Republic of...": "Venezuela",
    "Taiwan": "Taiwan",
}

# Ordered (low -> high) label prefixes used to build ordinal scores.
AGE_SHORT = {"Under 18 years old": "<18", "18-24 years old": "18-24", "25-34 years old": "25-34",
             "35-44 years old": "35-44", "45-54 years old": "45-54", "55-64 years old": "55-64",
             "65 years or older": "65+", "Prefer not to say": "Prefer not to say"}
AGE_MID = {"<18": 16, "18-24": 21, "25-34": 29.5, "35-44": 39.5, "45-54": 49.5, "55-64": 59.5, "65+": 67}
AGE_UPPER = {"<18": 17, "18-24": 24, "25-34": 34, "35-44": 44, "45-54": 54, "55-64": 64, "65+": 90}
AGE_ORDER = ["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

DEV_SHORT = {
    "Developer, full-stack": "Full-stack dev", "Developer, back-end": "Back-end dev",
    "Developer, front-end": "Front-end dev", "Developer, mobile": "Mobile dev",
    "Developer, desktop or enterprise applications": "Desktop/enterprise dev",
    "Developer, embedded applications or devices": "Embedded dev",
    "Developer, game or graphics": "Game/graphics dev", "Developer, QA or test": "QA/test dev",
    "Developer, AI": "AI developer", "Data scientist or machine learning specialist": "Data scientist/ML",
    "Cloud infrastructure engineer": "Cloud infra engineer", "Senior Executive (C-Suite, VP, etc.)": "Senior executive",
    "Engineer, site reliability": "Site reliability eng.", "Research & Development role": "R&D role",
    "Marketing or sales professional": "Marketing/sales", "Data or business analyst": "Data/business analyst",
    "System administrator": "Sysadmin", "Developer Experience": "Developer experience",
    "Developer Advocate": "Developer advocate", "Hardware Engineer": "Hardware engineer",
}
REMOTE_SHORT = {"Hybrid (some remote, some in-person)": "Hybrid"}

ED_RANK = {"Primary": 1, "Secondary": 2, "Some college": 3, "Associate": 4,
           "Bachelor": 5, "Master": 6, "Professional degree": 7}
ED_SHORT = {"Primary": "Primary", "Secondary": "Secondary", "Some college": "Some college",
            "Associate": "Associate", "Bachelor": "Bachelor's", "Master": "Master's",
            "Professional degree": "Doctorate/Prof.", "Something else": "Other"}

ORG_MID = {"Just me": 1, "2 to 9": 5, "10 to 19": 15, "20 to 99": 60, "100 to 499": 300,
           "500 to 999": 750, "1,000 to 4,999": 3000, "5,000 to 9,999": 7500,
           "10,000 or more": 15000}
ORG_ORDER = ["Just me", "2 to 9", "10 to 19", "20 to 99", "100 to 499", "500 to 999",
             "1,000 to 4,999", "5,000 to 9,999", "10,000 or more"]

SO_VISIT_RANK = {"Less than once per month": 1, "A few times per month": 2,
                 "A few times per week": 3, "Daily or almost": 4, "Multiple times per day": 5}

SENT_SCORE = {"Very unfavorable": 1, "Unfavorable": 2, "Indifferent": 3,
              "Favorable": 4, "Very favorable": 5}          # 'Unsure' -> NaN
TRUST_SCORE = {"Highly distrust": 1, "Somewhat distrust": 2, "Neither trust": 3,
               "Somewhat trust": 4, "Highly trust": 5}
COMPLEX_SCORE = {"Very poor": 1, "Bad at": 2, "Neither good": 3,
                 "Good, but not great": 4, "Very well": 5}

CATEGORICAL_CODED = ["main_branch", "age", "remote_work", "ed_level", "dev_type", "org_size",
                     "purchase_influence", "buildvs_buy", "so_visit_freq", "so_account",
                     "so_part_freq", "so_comm", "ai_select", "ai_sent", "ai_acc", "ai_complex",
                     "ai_threat", "survey_length", "survey_ease"]


def map_prefix(series: pd.Series, mapping: dict) -> pd.Series:
    """Map a text label to a number by matching the start of the label."""
    s = series.astype("object").where(series.notna(), "")
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    for prefix, value in mapping.items():
        out[s.str.startswith(prefix)] = value
    return out


def load_raw():
    """Return (survey, crosswalk, questions) exactly as stored on disk."""
    df = pd.read_csv(DATA_RAW / RAW_FILE)
    cw = pd.read_csv(DATA_RAW / CROSSWALK_FILE)
    qs = pd.read_csv(DATA_RAW / QUESTIONS_FILE)
    return df, cw, qs


def label_maps(cw: pd.DataFrame) -> dict:
    return {q: dict(zip(g["level"], g["label"])) for q, g in cw.groupby("qname")}


def decode(df: pd.DataFrame, cw: pd.DataFrame) -> pd.DataFrame:
    """Replace integer codes by their text labels using the official crosswalk."""
    out = df.copy()
    maps = label_maps(cw)
    for col in CATEGORICAL_CODED:
        out[col] = out[col].map(maps[col]).astype("object")
    out["country"] = out["country"].replace(COUNTRY_SHORT)
    out["age"] = out["age"].replace(AGE_SHORT)
    out["dev_type"] = out["dev_type"].replace(DEV_SHORT)
    out["remote_work"] = out["remote_work"].replace(REMOTE_SHORT)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless feature construction (row-wise arithmetic / recoding only)."""
    d = df.copy()
    d["age_mid"] = map_prefix(d["age"], AGE_MID)
    d["age_upper"] = map_prefix(d["age"], AGE_UPPER)
    d["ed_rank"] = map_prefix(d["ed_level"], ED_RANK)
    d["ed_short"] = d["ed_level"].astype("object").where(d["ed_level"].notna(), "")
    for k, v in ED_SHORT.items():
        d.loc[d["ed_short"].str.startswith(k), "ed_short"] = v
    d.loc[d["ed_level"].isna(), "ed_short"] = np.nan
    d["org_mid"] = map_prefix(d["org_size"], ORG_MID)
    d["org_short"] = d["org_size"].astype("object").where(d["org_size"].notna(), "")
    for k in ORG_ORDER:
        d.loc[d["org_short"].str.startswith(k), "org_short"] = k
    d.loc[d["org_short"].str.startswith("I don"), "org_short"] = "Don't know"
    d.loc[d["org_size"].isna(), "org_short"] = np.nan
    d["org_log"] = np.log10(d["org_mid"])
    d["so_visit_rank"] = map_prefix(d["so_visit_freq"], SO_VISIT_RANK)
    d["is_remote"] = d["remote_work"].map({"Remote": 1, "Hybrid": 0.5, "In-person": 0})
    # experience features -------------------------------------------------------------
    d["exp_gap"] = d["years_code"] - d["years_code_pro"]          # years coded before turning pro
    d["pro_share"] = d["years_code_pro"] / d["years_code"].where(d["years_code"] > 0)
    d["years_pro_sq"] = d["years_code_pro"] ** 2
    d["exp_band"] = pd.cut(d["years_code_pro"], [-0.1, 2, 5, 10, 20, 100],
                           labels=["0-2", "3-5", "6-10", "11-20", "21+"])
    d["exp_band"] = d["exp_band"].astype("object")
    # targets / flags ------------------------------------------------------------------
    d["uses_ai"] = np.where(d["ai_select"].isna(), np.nan,
                            (d["ai_select"] == "Yes").astype(float))
    d["log_comp"] = np.log(d["converted_comp_yearly"].where(d["converted_comp_yearly"] > 0))
    d["is_pro"] = (d["main_branch"] == "I am a developer by profession").astype(int)
    return d


# ----------------------------------------------------------------------------------
# Data-quality assessment
# ----------------------------------------------------------------------------------
def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    t = pd.DataFrame({"missing_n": df.isna().sum(), "missing_pct": df.isna().mean() * 100,
                      "dtype": df.dtypes.astype(str), "n_unique": df.nunique()})
    return t.sort_values("missing_pct", ascending=False)


def inconsistency_report(d: pd.DataFrame) -> pd.DataFrame:
    """Count logically impossible / suspicious records (d must contain add_features output)."""
    rows = []
    rows.append(("Professional years > total years coded",
                 int((d["years_code_pro"] > d["years_code"]).sum())))
    max_pro = d["age_upper"] - 14      # nobody is assumed to start a paid job before 14
    rows.append(("Professional years impossible for stated age bracket (age_upper - 14)",
                 int((d["years_code_pro"] > max_pro).sum())))
    rows.append(("Total years coded impossible for stated age bracket (age_upper - 4)",
                 int((d["years_code"] > d["age_upper"] - 4).sum())))
    rows.append(("Compensation below USD 1,000 per year",
                 int((d["converted_comp_yearly"] < 1000).sum())))
    rows.append(("Compensation above USD 1,000,000 per year",
                 int((d["converted_comp_yearly"] > 1_000_000).sum())))
    rows.append(("comp_total present but converted_comp_yearly missing (currency conversion failed)",
                 int((d["comp_total"].notna() & d["converted_comp_yearly"].isna()).sum())))
    rows.append(("Non-professional respondents reporting a salary",
                 int(((d["is_pro"] == 0) & d["converted_comp_yearly"].notna()).sum())))
    rows.append(("Age = 'Prefer not to say'", int((d["age"] == "Prefer not to say").sum())))
    rows.append(("Organisation size = 'I don't know' (treated as missing)",
                 int(d["org_size"].astype("object").fillna("").str.startswith("I don").sum())))
    rows.append(("years_code / years_code_pro top-coded at 51 (more than 50 years)",
                 int(((d["years_code"] == 51) | (d["years_code_pro"] == 51)).sum())))
    return pd.DataFrame(rows, columns=["check", "records_affected"])


def outlier_counts(x: pd.Series) -> pd.DataFrame:
    """Compare three classical outlier rules on a numeric series."""
    x = x.dropna()
    q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
    z = (x - x.mean()) / x.std()
    p1, p99 = x.quantile([.01, .99])
    rows = [("Tukey IQR fences (1.5 x IQR)", int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())),
            ("Z-score |z| > 3", int((z.abs() > 3).sum())),
            ("Outside 1st-99th percentile", int(((x < p1) | (x > p99)).sum()))]
    out = pd.DataFrame(rows, columns=["rule", "n_flagged"])
    out["pct_flagged"] = out["n_flagged"] / len(x) * 100
    return out


# ----------------------------------------------------------------------------------
# Cohort construction for the salary-prediction task
# ----------------------------------------------------------------------------------
def country_iqr_filter(d: pd.DataFrame, k: float = 3.0, min_n: int = 50) -> pd.Series:
    """Boolean mask: keep rows inside country-specific Tukey fences on log(comp).

    Fences are computed per country (countries with >= min_n salaries); smaller countries
    are not filtered by this rule (only by the absolute plausibility bounds).
    """
    keep = pd.Series(True, index=d.index)
    for c, g in d.groupby("country"):
        if len(g) < min_n:
            continue
        q1, q3 = g["log_comp"].quantile([.25, .75]); iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        keep.loc[g.index] = g["log_comp"].between(lo, hi)
    return keep


def build_salary_cohort(d: pd.DataFrame, lo=1_000, hi=1_000_000, k=3.0, policy="country_iqr"):
    """Return (cohort, attrition_table).  `d` must be output of add_features()."""
    steps = [("All respondents", len(d))]
    c = d[d["is_pro"] == 1]; steps.append(("Professional developers (main_branch = 1)", len(c)))
    c = c[c["converted_comp_yearly"].notna()]; steps.append(("Reported converted yearly compensation", len(c)))
    c = c[c["country"].notna()]; steps.append(("Country available", len(c)))
    c = c[c["converted_comp_yearly"].between(lo, hi)]
    steps.append((f"Absolute plausibility bounds USD {lo:,}-{hi:,}", len(c)))
    if policy == "country_iqr":
        c = c[country_iqr_filter(c, k=k)]
        steps.append((f"Within-country {k:g} x IQR fences on log(comp)", len(c)))
    c = c.copy()
    # logical inconsistencies are *repaired by blanking* (not dropped) -> kept for imputation
    bad = (c["years_code_pro"] > c["years_code"]) | (c["years_code_pro"] > c["age_upper"] - 14)
    c.loc[bad, ["years_code_pro", "exp_gap", "pro_share", "years_pro_sq", "exp_band"]] = np.nan
    steps.append(("Inconsistent professional-years values blanked (rows kept)", len(c)))
    att = pd.DataFrame(steps, columns=["step", "rows"])
    att["rows_removed"] = (-att["rows"].diff()).fillna(0).astype(int)
    return c.reset_index(drop=True), att


def build_ai_cohort(d: pd.DataFrame) -> pd.DataFrame:
    """Respondents whose AI-adoption answer is known (target must not be missing)."""
    c = d[d["uses_ai"].notna()].copy()
    bad = (c["years_code_pro"] > c["years_code"]) | (c["years_code_pro"] > c["age_upper"] - 14)
    c.loc[bad, ["years_code_pro", "exp_gap", "pro_share", "years_pro_sq", "exp_band"]] = np.nan
    return c.reset_index(drop=True)
