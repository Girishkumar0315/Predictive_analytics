"""Small, well-tested statistical helpers used by the analysis objectives."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def cramers_v(x: pd.Series, y: pd.Series, bias_correct: bool = True) -> float:
    """Cramer's V association between two categorical variables (0 = none, 1 = perfect)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum(); r, k = ct.shape
    phi2 = chi2 / n
    if bias_correct:
        phi2 = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1); k = k - (k - 1) ** 2 / (n - 1)
    return float(np.sqrt(phi2 / max(1e-12, min(k - 1, r - 1))))


def chi2_test(x: pd.Series, y: pd.Series) -> dict:
    ct = pd.crosstab(x, y)
    chi2, p, dof, _ = stats.chi2_contingency(ct, correction=False)
    return dict(chi2=float(chi2), dof=int(dof), p=float(p), n=int(ct.values.sum()),
                cramers_v=cramers_v(x, y))


def wilson_ci(successes, n, z: float = 1.96):
    """Wilson score interval for a proportion (vectorised)."""
    successes = np.asarray(successes, float); n = np.asarray(n, float)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return centre - half, centre + half


def two_prop_ztest(s1, n1, s2, n2) -> dict:
    p1, p2 = s1 / n1, s2 / n2
    p = (s1 + s2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    return dict(p1=p1, p2=p2, diff=p1 - p2, z=float(z), p=float(2 * stats.norm.sf(abs(z))))


def eta_squared_from_groups(groups: list[np.ndarray]) -> float:
    """One-way ANOVA effect size eta^2 = SS_between / SS_total."""
    allv = np.concatenate(groups); gm = allv.mean()
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    sst = ((allv - gm) ** 2).sum()
    return float(ssb / sst)


def epsilon_squared_kruskal(H: float, n: int) -> float:
    """Effect size for Kruskal-Wallis (epsilon^2 = H / (n - 1))."""
    return float(H / (n - 1))


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta = 2*AUC - 1 from the Mann-Whitney U statistic (P(a>b) - P(a<b))."""
    u = stats.mannwhitneyu(a, b, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def bootstrap_ci(func, data: pd.DataFrame, n_boot: int = 500, seed: int = 42, alpha: float = 0.05):
    """Percentile bootstrap CI of a scalar statistic func(df)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), len(data))
        vals.append(func(data.iloc[idx]))
    vals = np.asarray(vals, float)
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def spearman(x, y) -> dict:
    m = pd.concat([pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)], axis=1).dropna()
    rho, p = stats.spearmanr(m.iloc[:, 0], m.iloc[:, 1])
    n = len(m)
    # Fisher-z CI for rho
    if n > 3 and abs(rho) < 1:
        z = np.arctanh(rho); se = 1.06 / np.sqrt(n - 3)    # Bonett-Wright SE for Spearman
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = np.nan
    return dict(rho=float(rho), p=float(p), n=int(n), ci_low=float(lo), ci_high=float(hi))
