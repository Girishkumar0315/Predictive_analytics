# %% [markdown]
# # Predictive Analysis of Developer Compensation and AI-Tool Adoption
# ### Stack Overflow Annual Developer Survey 2024 — CA2 Predictive Analysis Project
#
# **Notebook map** — Part A: setup & data inspection · Part B: data-quality assessment & cleaning ·
# Part C: EDA · Part D: Objectives O1–O5 (statistical analysis) · Part E: Objectives O6–O7 (machine
# learning) · Part F: Objective O8 (explainability & error analysis) · Part G: conclusions.
#
# **How to run:** `jupyter nbconvert --to notebook --execute --inplace notebooks/CA2_Predictive_Analysis.ipynb`
# (about 15–25 minutes on a single CPU core; every random process uses `RANDOM_STATE = 42`).
#
# ---
# ## Project objectives (defined *before* any modelling)
# | # | Objective (measurable) | Method | Success measure |
# |---|---|---|---|
# | **O1** | Quantify how much **country** explains developer pay and where India sits | Kruskal–Wallis, η², Mann–Whitney + Holm, bootstrap CIs | η² / R² of country; India:USA median ratio with 95 % CI |
# | **O2** | Quantify the **experience → pay** relationship and its diminishing returns | Spearman ρ, log-linear quadratic OLS, bootstrap | ρ with CI; % pay gain per year at 0/5/10/20 yrs; India vs USA slope test |
# | **O3** | Estimate **net** effects of role, work mode, company size and education after controlling for country & experience | OLS (HC3 robust), nested-model ΔR², partial η² | Adjusted % differences with 95 % CIs; ΔR² per block |
# | **O4** | Map **who uses AI tools** in development (current adoption) | χ² tests, Cramér's V, Wilson CIs, two-proportion z-test | Adoption rate per group; V ranking; India vs world gap |
# | **O5** | Understand AI **sentiment, trust and job-threat** perception across career stages | Spearman/Kruskal, χ², Cramér's V | ρ (experience ↔ trust), threat-rate gaps by role |
# | **O6** | **Predict annual compensation** (regression, ≥ 4 models) | Baselines, Ridge, RF, HGB, LightGBM + CV + tuning | RMSE(log) ↓ ≥ 40 % vs global-median baseline; R², MAE (USD), MdAPE |
# | **O7** | **Predict current AI-tool adoption** (classification, ≥ 4 models) | LogReg, RF, HGB, LightGBM + CV + tuning | ROC-AUC, F1, precision/recall vs majority baseline |
# | **O8** | **Explain** the models and audit their errors | Permutation importance, SHAP, PDP, residuals by segment | Top drivers; error gaps across countries/segments |

# %%
import warnings, json, sys, time, platform
from pathlib import Path
warnings.filterwarnings("ignore")

ROOT = Path.cwd()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sklearn, statsmodels, lightgbm, scipy, joblib
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate, cross_val_predict

from src.config import *
from src.data_prep import *
from src.features import *
from src.modeling import *
from src.stats_utils import *
from src.plotting import (set_style, savefig, save_table, reset_registry, PALETTE, CAT, SEQ5)

set_style(); reset_registry()
pd.set_option("display.max_columns", 60, "display.width", 160, "display.float_format", "{:,.3f}".format)
R = {}            # master results dictionary -> reports/results/results.json (feeds the written report)

def save_results():
    def default(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.ndarray,)): return o.tolist()
        if isinstance(o, pd.DataFrame): return o.to_dict("records")
        return str(o)
    (RES_DIR / "results.json").write_text(json.dumps(R, indent=1, default=default))

R["environment"] = dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__,
                        scipy=scipy.__version__, sklearn=sklearn.__version__, statsmodels=statsmodels.__version__,
                        lightgbm=lightgbm.__version__, seed=RANDOM_STATE)
print(R["environment"])
