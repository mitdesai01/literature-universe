# -*- coding: utf-8 -*-
"""Policy environment (CSPP): the state's political and policy environment against migration and AI outcomes.

state-environment   cross-section: CSPP profile (2012-16 averages) against net interstate migration 2013-22 (IRS)
policy-twfe         panel: each curated CSPP variable, lagged a year, against outcomes with state and year fixed effects
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, log, wcorr, boot_ci, wls
from ..adapters.politics import CSPP_CURATED, PROFILE_YEARS
from . import needs

ENV_LABELS = {"env_" + k: v for k, v in CSPP_CURATED.items()}
OUTCOMES = {"net_migration_rate": "net interstate migration", "vr_rep_share": "Republican share of the workforce", "ai_share": "AI share of patents",
            "ai_broad_patents_per_10k_jobs": "AI patents per job", "btos_ai_use_now": "firms using AI"}


def _twfe(df, y, x):
    """y ~ x (lagged one year) with state and year fixed effects, clustered by state."""
    import statsmodels.formula.api as smf
    d = df[["state_fips", "year", y, x]].dropna().copy()
    if d.state_fips.nunique() < 15 or d.year.nunique() < 4:
        return None
    d["xz"] = (d[x] - d[x].mean()) / d[x].std()
    try:
        r = smf.ols(f"{y} ~ xz + C(state_fips) + C(year)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d.state_fips})
    except Exception:
        return None
    return {"coef": float(r.params["xz"]), "t": float(r.tvalues["xz"]), "p": float(r.pvalues["xz"]), "n": int(r.nobs)}


def _environment(s):
    envs = [c for c in s.columns if c.startswith("env_")]
    if not envs or "net_migration_rate" not in s:
        return False
    b = s.dropna(subset=["net_migration_rate", "base_returns"])
    if b.state_fips.nunique() < 40:
        return False
    g = b.groupby("state_fips")
    x = pd.DataFrame({"net_pp": g.apply(lambda d: np.average(d.net_migration_rate, weights=d.base_returns) * 100, include_groups=False),
                      "net_2020_22_pp": g.apply(lambda d: d[d.year.between(2020, 2022)].net_migration_rate.mean() * 100, include_groups=False),
                      "income_gap": g.mover_income_gap.mean(), "households": g.base_returns.mean()})
    first = s.groupby("state_fips").first()
    for c in envs + [c for c in ("aige", "state", "name", "region") if c in s]:
        x[c] = first[c]
    x = x.reset_index()
    y0, y1 = int(b.year.min()), int(b.year.max())
    rows = []
    for c in envs + (["aige"] if "aige" in x else []):
        d = x.dropna(subset=[c])
        if len(d) < 30 or d[c].nunique() < 2:
            continue
        ci = boot_ci(lambda a, b_, w: wcorr(a, b_, w), d[c].values, d.net_pp.values, d.households.values)
        rows.append({"variable": ENV_LABELS.get(c, "AI exposure of jobs (AIGE)" if c == "aige" else c), "column": c, "n": len(d),
                     "r_net": wcorr(d[c], d.net_pp, d.households), "lo": ci[0], "hi": ci[1],
                     "r_net_2020_22": wcorr(d[c], d.net_2020_22_pp, d.households), "r_income_gap": wcorr(d[c], d.income_gap, d.households)})
    corr = pd.DataFrame(rows).sort_values("r_net")
    dataset("state_env_corr", corr)
    view("state-env-corr", "bar", f"State environment ({PROFILE_YEARS[0]}-{str(PROFILE_YEARS[1])[2:]}) and net interstate migration "
         f"{y0 - 1}-{str(y1)[2:]}: correlations across states, weighted by households", "state_env_corr",
         x="variable", y=["r_net", "r_net_2020_22", "r_income_gap"], orientation="h",
         labels={"r_net": f"Net migration {y0}-{y1}", "r_net_2020_22": "Net migration 2020-22", "r_income_gap": "Movers' income gap"})
    dataset("state_environment", x)
    labels = {**ENV_LABELS, "aige": "AI exposure of jobs (AIGE)", "net_pp": f"Net interstate migration {y0}-{y1} (% of households a year)",
              "net_2020_22_pp": "Net interstate migration 2020-22 (% a year)", "income_gap": "In-movers minus out-movers ($000 AGI per return)",
              "households": "Households (returns)"}
    view("state-env-explorer", "scatter", "States: political and policy environment against net interstate migration", "state_environment",
         x=[c for c in ("env_policyeconlib_est", "env_propgoppres", "env_x_top_corporateincometaxrate", "aige", "env_inst6014_nom",
                        "env_ranney4_control", "env_perc_college", "env_hincomemed", "env_grtw") if c in x],
         y=["net_pp", "net_2020_22_pp", "income_gap"], size="households", color="region" if "region" in x else None, text="name",
         filter=["region"] if "region" in x else None, labels=labels)
    models = []
    specs = [("Policy liberalism + AIGE", ["env_policyeconlib_est", "aige"]),
             ("+ corporate tax, college share", ["env_policyeconlib_est", "aige", "env_x_top_corporateincometaxrate", "env_perc_college"]),
             ("Republican presidential share + AIGE", ["env_propgoppres", "aige"])]
    fits = {}
    for name, xs in specs:
        xs = [c for c in xs if c in x]
        d = x.dropna(subset=xs + ["net_pp"])
        m = wls(d, "net_pp", xs, "households")
        if not m:
            continue
        fits[name] = m
        for k, t in m["terms"].items():
            models.append({"model": name, "term": labels.get(k, k), "coef": t["coef"], "lo": t["coef"] - 1.96 * t["se"],
                           "hi": t["coef"] + 1.96 * t["se"], "t": t["t"], "n": m["n"]})
    dataset("state_env_models", pd.DataFrame(models))
    view("state-env-models", "coef", "Net interstate migration (points a year) per SD of each state trait, weighted by households", "state_env_models",
         x="coef", y="term", group="model", lo="lo", hi="hi", xlabel="Percentage points of households a year per SD, 95% interval")
    neg, pos = corr.head(3), corr.tail(2).iloc[::-1]
    m1 = fits.get("Policy liberalism + AIGE")
    tt = lambda m, k: f"{m['terms'][k]['coef']:+.2f} (t = {m['terms'][k]['t']:+.1f})" if m and k in m["terms"] else "n/a"  # noqa: E731
    finding("state-environment", "Households left states with liberal economic policy and AI-exposed jobs; at the state level the two are hard to separate",
            f"Across states, net interstate migration (IRS, {y0 - 1}-{str(y0)[2:]} to {y1 - 1}-{str(y1)[2:]}, weighted by households) correlates most "
            f"negatively with " + "; ".join(f"{r.variable} ({r.r_net:+.2f})" for r in neg.itertuples()) + " and most positively with "
            + "; ".join(f"{r.variable} ({r.r_net:+.2f})" for r in pos.itertuples()) +
            f". Entered together, economic policy liberalism gives {tt(m1, 'env_policyeconlib_est')} points a year per SD and AIGE "
            f"{tt(m1, 'aige')}: the two move together across states. Within states, where policy is constant, county exposure still "
            f"predicts out-migration (see the county finding).",
            theme="Policy x AI", level="state", datasets=["CSPP", "IRS SOI migration", "AIOE (AIGE)"], strength="suggestive",
            stats={r.column: round(r.r_net, 3) for r in corr.itertuples()},
            question="Which part is policy (taxes, housing regulation, labour law) and which is the kind of work a state does? "
                     "Border-county comparisons across state lines can separate them.",
            next_data="County-pair flows across state borders; state housing regulation (Wharton index); state AI legislation (NCSL)",
            caveats=[f"CSPP political variables mostly end in 2014-2017; the profile averages {PROFILE_YEARS[0]}-{PROFILE_YEARS[1]}.",
                     "51 states: correlations, not causal estimates."],
            views=["state-env-corr", "state-env-explorer", "state-env-models"], rank=5)
    return True


def run():
    s = read("panel_state_year", "PANELS")
    env = _environment(s) if s is not None else False
    if s is None or not any(c.startswith("cspp_") for c in s.columns):
        if env:
            return True
        return needs("policy", "State policy environment and AI", "Policy x AI", "state", "Correlates of State Policy (CSPP)",
                     "Do state ideology, party control or tech policy predict AI adoption, AI invention and in-migration?")
    xs = [c for c in s.columns if c.startswith("cspp_")]
    ys = [c for c in ("ai_share", "ai_broad_patents_per_10k_jobs", "btos_ai_use_now", "net_migration_rate", "vr_rep_share") if c in s and s[c].notna().sum() > 60]
    if not ys:
        return needs("policy", "State policy environment and AI", "Policy x AI", "state", "an AI outcome by state-year (PatentsView, BTOS or IRS)",
                     "Do state policies predict AI outcomes?")
    d = s.sort_values(["state_fips", "year"]).copy()
    for c in xs:
        d[c + "_lag"] = d.groupby("state_fips")[c].shift(1)
    rows = []
    for y in ys:
        for x in xs:
            r = _twfe(d, y, x + "_lag")
            if r:
                rows.append({"outcome": y, "policy_variable": x.replace("cspp_", ""), **r})
    if not rows:
        return needs("policy", "State policy environment and AI", "Policy x AI", "state", "overlapping years of CSPP and AI outcomes",
                     "Do state policies predict AI outcomes?")
    t = pd.DataFrame(rows).sort_values("p")
    m = len(t)
    t["q"] = (t.p * m / (np.arange(m) + 1)).iloc[::-1].cummin().iloc[::-1].clip(upper=1)   # Benjamini-Hochberg
    cat = read("cspp_catalog")
    if cat is not None:
        t["description"] = t.policy_variable.map(dict(zip(cat.variable, cat.description)))
    dataset("policy_twfe", t)
    view("policy-table", "table", "State policy (lagged one year) and AI outcomes, state and year fixed effects", "policy_twfe",
         columns=["outcome", "policy_variable", "description", "coef", "t", "q", "n"])
    sig = t[t.q < 0.1]
    finding("policy-twfe", "Which state policy variables move with AI outcomes within states over time",
            (f"{len(sig)} of {m} policy-outcome pairs pass a 10% false-discovery threshold. Strongest: " +
             "; ".join(f"{CSPP_CURATED.get(r.policy_variable, r.policy_variable)} -> {OUTCOMES.get(r.outcome, r.outcome)} (t = {r.t:+.1f})"
                       for r in sig.head(5).itertuples()) + "."
             if len(sig) else f"None of {m} policy-outcome pairs passes a 10% false-discovery threshold with state and year fixed effects."),
            theme="Policy x AI", level="state", datasets=["CSPP", "PatentsView/BTOS/IRS"], strength="suggestive" if len(sig) else "descriptive",
            stats={"pairs": m, "passing_fdr10": int(len(sig))},
            question="Use policy adoptions as events (staggered difference-in-differences) for the variables that pass.",
            next_data="Policy adoption dates (e.g. NCSL AI legislation, R&D tax credits)", views=["policy-table"], rank=12)
    log(f"policy: {m} two-way fixed-effects regressions")
    return True
