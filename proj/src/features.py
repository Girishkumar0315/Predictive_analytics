"""Feature specifications and leakage-safe preprocessing pipelines.

Design rule: everything that LEARNS a statistic from data (imputation medians, scaling,
one-hot vocabulary, target encoding) is inside an sklearn Pipeline, so it is re-fit on the
training part of every CV fold and never sees validation / test rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, StandardScaler,
                                   TargetEncoder)

# ------------------------------- feature specifications ---------------------------
NUM_COLS = ["years_code", "years_code_pro", "age_mid", "org_log", "ed_rank",
            "so_visit_rank", "exp_gap", "pro_share"]
LOW_CAT = ["remote_work", "purchase_influence", "buildvs_buy", "so_account",
           "so_part_freq", "so_comm", "r_used"]
HIGH_CAT = ["country", "dev_type"]
EXTRA_CAT_LINEAR = ["ed_short", "org_short"]            # one-hot only (numeric twin exists)

SALARY_SPEC = dict(num=NUM_COLS + ["years_pro_sq"], low=LOW_CAT + ["ai_select"],
                   high=HIGH_CAT, extra=EXTRA_CAT_LINEAR)
# 'main_branch' is constant in the professional cohort but informative for all respondents.
ADOPT_SPEC = dict(num=NUM_COLS + ["years_pro_sq"], low=LOW_CAT + ["main_branch"],
                  high=HIGH_CAT, extra=EXTRA_CAT_LINEAR)

# Columns that must NEVER be used as predictors (with the reason) -----------------
LEAKAGE_REGISTER = {
    "comp_total": "Local-currency twin of the target (converted_comp_yearly = comp_total x FX rate).",
    "converted_comp_yearly": "Target of the salary task.",
    "log_comp": "Transform of the target.",
    "currency": "Near-duplicate of country (information already encoded; adds sparse dummies).",
    "response_id": "Identifier - no predictive meaning.",
    "survey_length": "Opinion about the survey itself - unrelated to the prediction problem.",
    "survey_ease": "Opinion about the survey itself - unrelated to the prediction problem.",
    "ai_sent": "Asked ONLY if ai_select in {Yes, plan to} -> its mere presence reveals the target.",
    "ai_acc": "Asked ONLY if ai_select = Yes -> reveals the target of the adoption task.",
    "ai_complex": "Asked ONLY if ai_select = Yes -> reveals the target of the adoption task.",
    "ai_threat": "Asked ONLY if ai_select in {Yes, plan to} -> reveals the target.",
    "ai_select": "Target of the adoption task (allowed as a predictor in the salary task).",
}


def model_frame(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Select predictors and make categorical columns explicit strings ('Missing' for NaN).

    Filling with a constant is stateless -> safe to do before the train/test split.
    """
    cols = spec["num"] + spec["low"] + spec["high"] + spec["extra"]
    X = df[cols].copy()
    for c in spec["low"] + spec["high"] + spec["extra"]:
        X[c] = X[c].astype("object").where(X[c].notna(), "Missing").astype(str)
    for c in spec["num"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
    return X


def make_preprocessor(kind: str, spec: dict, target_type: str = "continuous") -> ColumnTransformer:
    """kind='linear' -> impute + scale numeric, one-hot everything categorical.
       kind='tree'   -> impute numeric (+ missing flags), ordinal low-card, target-encode high-card.
    """
    num_pipe = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if kind == "linear":
        num_pipe.append(("scale", StandardScaler()))
        cat = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                            sparse_output=False)
        return ColumnTransformer([
            ("num", Pipeline(num_pipe), spec["num"]),
            ("cat", cat, spec["low"] + spec["high"] + spec["extra"]),
        ], remainder="drop", verbose_feature_names_out=False)
    ordinal = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    te = TargetEncoder(target_type=target_type, smooth="auto", cv=5, random_state=42)
    return ColumnTransformer([
        ("num", Pipeline(num_pipe), [c for c in spec["num"] if c != "years_pro_sq"]),
        ("ord", ordinal, spec["low"] + ["ed_short", "org_short"]),
        ("te", te, spec["high"]),
    ], remainder="drop", verbose_feature_names_out=False)
