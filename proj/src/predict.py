"""Inference helpers: turn a plain Python dict (a developer profile) into model-ready features."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_DIR
from .data_prep import AGE_MID, ED_RANK, ORG_MID, SO_VISIT_RANK, map_prefix
from .features import SALARY_SPEC, ADOPT_SPEC, model_frame

DEFAULTS = dict(years_code=10, years_code_pro=5, age="25-34", org_short="100 to 499", ed_short="Bachelor's",
                remote_work="Hybrid", dev_type="Full-stack dev", country="USA", purchase_influence="Missing",
                buildvs_buy="Missing", so_account="Missing", so_part_freq="Missing", so_comm="Missing",
                r_used=0.0, ai_select="Yes", main_branch="I am a developer by profession",
                so_visit_freq="A few times per week")


def profile_to_frame(profile: dict) -> pd.DataFrame:
    """Build the engineered feature row for one developer (mirrors data_prep.add_features)."""
    p = {**DEFAULTS, **profile}
    d = pd.DataFrame([p])
    d["age_mid"] = map_prefix(d["age"], AGE_MID)
    d["ed_rank"] = map_prefix(d["ed_short"].replace({"Bachelor's": "Bachelor", "Master's": "Master",
                                                     "Doctorate/Prof.": "Professional degree"}), ED_RANK)
    d["org_mid"] = d["org_short"].map({"Just me": 1, "2 to 9": 5, "10 to 19": 15, "20 to 99": 60, "100 to 499": 300,
                                       "500 to 999": 750, "1,000 to 4,999": 3000, "5,000 to 9,999": 7500,
                                       "10,000 or more": 15000})
    d["org_log"] = np.log10(d["org_mid"])
    d["so_visit_rank"] = d["so_visit_freq"].map({"Less than once per month or monthly": 1, "A few times per month or weekly": 2,
                                                 "A few times per week": 3, "Daily or almost daily": 4,
                                                 "Multiple times per day": 5})
    d["exp_gap"] = d["years_code"] - d["years_code_pro"]
    d["pro_share"] = d["years_code_pro"] / d["years_code"].where(d["years_code"] > 0)
    d["years_pro_sq"] = d["years_code_pro"] ** 2
    return d


def predict_salary(profile: dict, model=None) -> float:
    """Predicted annual compensation in USD (exp of the predicted log-salary = conditional median)."""
    model = model or joblib.load(MODEL_DIR / "best_salary_model.joblib")
    X = model_frame(profile_to_frame(profile), SALARY_SPEC)
    return float(np.exp(model.predict(X)[0]))


def predict_ai_adoption(profile: dict, model=None) -> float:
    """Probability that the developer currently uses AI tools in their workflow."""
    model = model or joblib.load(MODEL_DIR / "best_ai_adoption_model.joblib")
    X = model_frame(profile_to_frame(profile), ADOPT_SPEC)
    return float(model.predict_proba(X)[0, 1])
