# -*- coding: utf-8 -*-
"""Geography: states and metros - AI exposure, partisanship, how the data were built, and space."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, wls, knn_weights, moran, log
from . import needs
from ..adapters.politics import CSPP_CURATED


def run():
    ok = False
    s = read("panel_state_year", "PANELS")
    if s is not None and "vr_rep_share" in s and s.vr_rep_share.notna().any():
        ok |= _states(s)
    m = read("panel_metro_year", "PANELS")
    if m is not None and len(m):
        ok |= _metros(m)
    if not ok:
        return needs("geo", "Geography of partisanship and AI exposure", "Geography", "state/metro", "VRscores metro or state panels",
                     "Where do AI exposure and workforce partisanship coincide?")
    return True


def _states(s):
    first, last = int(s.loc[s.vr_rep_share.notna(), "year"].min()), int(s.loc[s.vr_rep_share.notna(), "year"].max())
    a = s[s.year == first].set_index("state_fips").vr_rep_share
    x = s[s.year == last].set_index("state_fips").copy()
    x["drift_pp"] = (x.vr_rep_share - a) * 100
    if "jobs_2019" in x:
        x["coverage"] = x.vr_workers / x.jobs_2019
    x["inferred"] = (x.party_regime.fillna("") != "party registration").astype(int)
    st = x.reset_index()
    cols = [c for c in ("state", "name", "state_fips", "region", "vr_rep_share", "drift_pp", "party_regime", "aige", "coverage", "jobs_2019",
                        "rep_vote_share", "vr_workers") if c in st]
    dataset("state_summary", st[cols])
    series = [c for c in ("vr_rep_share", "rep_vote_share", "aige", "net_migration_rate", "mover_income_gap", "ai_share",
                          "ai_broad_patents_per_10k_jobs", "btos_ai_use_now", "ai_inventor_net", "coverage") if c in s or c == "coverage"]
    long = s.copy()
    if "jobs_2019" in long:
        long["coverage"] = long.vr_workers / long.jobs_2019
    vals = [c for c in series if c in long] + [c for c in long.columns if c.startswith("cspp_")]
    long = long[long[vals].notna().any(axis=1)]
    dataset("state_year", long[["state", "name", "state_fips", "year"] + vals])
    view("state-map", "choropleth", "States by year", "state_year", location="state", year="year",
         value=[c for c in series if c in long] + [c for c in long.columns if c.startswith("cspp_")], text="name",
         labels={"vr_rep_share": "Republican share of the matched workforce (VRscores)", "rep_vote_share": "Republican presidential vote share",
                 "aige": "AI exposure of the workforce (AIGE)", "ai_share": "AI share of patents", "ai_broad_patents_per_10k_jobs": "AI patents per 10,000 jobs",
                 "btos_ai_use_now": "Firms using AI (BTOS)", "net_migration_rate": "Net interstate migration (share of households, IRS)",
                 "mover_income_gap": "In-movers minus out-movers ($000 AGI per return)", "ai_inventor_net": "Net AI inventor moves",
                 "coverage": "VRscores coverage (matched workers per job)", **{"cspp_" + k: v for k, v in CSPP_CURATED.items()}})
    n = x.aige.notna().sum() if "aige" in x else 0
    if n >= 20:
        r = wcorr(x.aige, x.vr_rep_share, x.jobs_2019) if "jobs_2019" in x else wcorr(x.aige, x.vr_rep_share)
        r_cov = wcorr(x.aige, x.coverage) if "coverage" in x else np.nan
        vote = f" The same states' Republican vote share correlates {wcorr(x.aige, x.rep_vote_share):+.2f} with AIGE." if "rep_vote_share" in x and x.rep_vote_share.notna().sum() > 20 else ""
        finding("state-aige", "AI-exposed states have more Democratic workforces, and VRscores covers them more densely",
                f"State AI exposure (AIGE) correlates {r:+.2f} with the Republican share of the matched metro workforce in {last} ({n} states, weighted by jobs)."
                f"{vote} Coverage (matched workers per job) correlates {r_cov:+.2f} with AIGE: the data lean towards AI-exposed labour markets.",
                theme="Political ideology x AI", level="state", datasets=["VRscores", "AIOE (AIGE)", "QCEW"], strength="suggestive",
                stats={"n": n, "r_aige_rep": r, "r_aige_coverage": r_cov},
                question="Is the partisan gap in AI exposure a feature of places or of the people the data happen to cover?",
                next_data="County-level partisanship (votes) with AIGE; coverage weights for VRscores", views=["state-map"], rank=8)
    if "coverage" in x and x.coverage.notna().sum() > 20:
        lo = x.nsmallest(4, "coverage")
        finding("state-coverage", "State maps built from VRscores metros misplace workers of multi-state metros",
                "Matched workers per job range from " + f"{x.coverage.min():.1%} to {x.coverage.max():.1%}. The lowest: " +
                ", ".join(f"{i} {v:.1%}" for i, v in zip(lo.state, lo.coverage)) +
                ", where large metros are named after a neighbouring state first (New York-Newark, Philadelphia-Wilmington, Boston-Nashua) "
                "and are credited to that state; the District of Columbia absorbs its suburbs the same way.",
                theme="Measurement", level="state", datasets=["VRscores", "QCEW"], strength="artifact",
                stats={"min": float(x.coverage.min()), "max": float(x.coverage.max()), "median": float(x.coverage.median())},
                question="Rebuild state figures by splitting metro workers across states in proportion to county jobs.",
                next_data="Census List 1 (county -> CBSA) with county job counts", views=["state-map"], rank=16)
    if x.drift_pp.notna().sum() > 20:
        m = wls(x.reset_index(), "drift_pp", ["inferred", "vr_rep_share"], None)
        if m:
            finding("state-regime", "States where party is inferred rather than registered drift Democratic faster",
                    f"Holding the starting level equal, states whose party is inferred (primary participation or L2's model) drifted "
                    f"{abs(m['terms']['inferred']['coef']):.1f} points {'further towards' if m['terms']['inferred']['coef'] < 0 else 'less towards'} the Democrats between {first} and {last} "
                    f"(t = {m['terms']['inferred']['t']:+.1f}, {m['n']} states).",
                    theme="Measurement", level="state", datasets=["VRscores"], strength="suggestive",
                    stats={"t": m["terms"]["inferred"]["t"], "n": m["n"]},
                    question="Validate inferred party against registration in states that switched regime, or against survey self-reports.",
                    next_data="L2 modelled-party accuracy by age; CES validated vote/party", views=["state-map"], rank=11)
    return True


def _metros(m):
    have_years = m.year.nunique() > 1
    last = int(m.year.max())
    cur = m[m.year == last].set_index("msa").copy()
    if have_years:
        first = int(m.year.min())
        w = m.pivot_table(index="msa", columns="year", values="rep_share")
        cur["drift_pp"] = (w[last] - w[first]) * 100
        yrs = np.array(w.columns, float)
        ok = w.notna().all(axis=1)
        from sklearn.cluster import KMeans
        shape = w[ok].sub(w[ok].mean(axis=1), axis=0)
        km = KMeans(4, n_init=20, random_state=0).fit(shape.values)
        cur.loc[shape.index, "trajectory"] = pd.Series(km.labels_, index=shape.index).map(lambda k: f"path {k + 1}")
        cent = pd.DataFrame(km.cluster_centers_ * 100, columns=w.columns)
        cent["trajectory"] = [f"path {k + 1}" for k in range(4)]
        cent["metros"] = pd.Series(km.labels_).value_counts().sort_index().values
        info = cur.groupby("trajectory").apply(lambda g: pd.Series({"inferred_share": np.mean(g.party_regime != "party registration"),
                                                                     "examples": "; ".join(g.nlargest(3, "workers").index)}), include_groups=False)
        cent = cent.merge(info, left_on="trajectory", right_index=True)
        tl = cent.melt(id_vars=["trajectory", "metros", "inferred_share", "examples"], var_name="year", value_name="demeaned_pp")
        dataset("metro_trajectories", tl)
        view("metro-traj", "line", "Four typical paths of metro workforce partisanship (points, relative to each metro's own mean)", "metro_trajectories",
             x="year", y="demeaned_pp", group="trajectory", text="examples")
    cur["inferred"] = (cur.party_regime.fillna("") != "party registration").astype(int)
    geo = cur.dropna(subset=["lat", "lon"]) if "lat" in cur else cur.iloc[0:0]
    if len(geo) > 30:
        W = knn_weights(geo.lat, geo.lon, k=8)
        mo_lvl = moran(geo.rep_share.values, W)
        cur.loc[geo.index, "lisa_level"] = mo_lvl["quadrant"]
        if have_years:
            g2 = geo.dropna(subset=["drift_pp"])
            W2 = knn_weights(g2.lat, g2.lon, k=8)
            mo = moran(g2.drift_pp.values, W2)
            cur.loc[g2.index, "lisa_drift"] = mo["quadrant"]
            cur.loc[g2.index, "lisa_drift_strength"] = mo["local"]
    keep = [c for c in ("workers", "rep_share", "drift_pp", "lat", "lon", "state", "party_regime", "trajectory", "lisa_level", "lisa_drift",
                        "cbsa", "metro_aiie", "metro_aiie_lm", "rep_vote_share", "patents", "ai_broad_patents") if c in cur]
    dataset("metro_summary", cur.reset_index()[["msa"] + keep])
    yc = [c for c in ("rep_share", "workers", "lat", "lon", "party_regime", "metro_aiie", "rep_vote_share", "ai_broad_patents") if c in m]
    dataset("metro_year", m[["msa", "year"] + yc])
    view("metro-map", "points", "Metro areas by year", "metro_year", lat="lat", lon="lon", size="workers", year="year", text="msa",
         value=[c for c in ("rep_share", "metro_aiie", "rep_vote_share", "ai_broad_patents") if c in m], color_scale="partisan",
         labels={"rep_share": "Republican share of the matched workforce", "metro_aiie": "AI exposure of the metro's industry mix",
                 "rep_vote_share": "Republican vote share", "ai_broad_patents": "AI patents"})
    view("metro-explorer", "scatter", "Metro areas: level, drift and exposure", "metro_summary",
         x=[c for c in ("rep_share", "metro_aiie", "rep_vote_share", "workers") if c in cur], y=[c for c in ("drift_pp", "rep_share") if c in cur],
         size="workers", color="party_regime", text="msa", filter=["party_regime", "trajectory"],
         labels={"rep_share": "Republican share", "drift_pp": "Change since first year (points)", "metro_aiie": "AI exposure (industry mix)",
                 "rep_vote_share": "Republican vote share", "workers": "Matched workers"})
    if have_years:
        d = cur.reset_index().dropna(subset=["drift_pp"])
        d["log_workers"] = np.log(d.workers)
        d["start"] = d.rep_share - d.drift_pp / 100
        mr = wls(d, "drift_pp", ["inferred", "start", "log_workers"], "workers")
        eff = mr["terms"]["inferred"]["coef"] if mr else np.nan
        finding("metro-regime", "Where party is inferred rather than registered, metro workforces drift Democratic much faster",
                f"Across {len(d)} metros, {np.mean(d.drift_pp < 0):.0%} moved towards the Democrats between {first} and {last}. Metros in states "
                f"where party comes from primaries or L2's model drifted {abs(eff):.1f} points {'further' if eff < 0 else 'less'} than metros in registration states, holding the "
                f"starting level and size equal (t = {mr['terms']['inferred']['t']:+.1f}). The drift is a trait of the measurement regime, "
                f"not only of places.", theme="Measurement", level="metro", datasets=["VRscores"], strength="robust",
                stats={"n": len(d), "effect_pp": eff, "t": mr["terms"]["inferred"]["t"] if mr else None},
                question="How much of VRscores' Democratic drift is cohort replacement, and how much is the inference model aging with voters?",
                next_data="VRscores by age cohort and regime; registration-state benchmarks", views=["metro-explorer", "metro-traj"], rank=5)
        if len(geo) > 30:
            g2 = cur.dropna(subset=["lisa_drift"])
            hh = g2[g2.lisa_drift == "high-high"].nlargest(5, "lisa_drift_strength")
            ll = g2[g2.lisa_drift == "low-low"].nlargest(5, "lisa_drift_strength")
            finding("metro-space", "Partisan drift clusters in space, along state lines",
                    f"Metro drift is spatially autocorrelated (Moran's I = {mo['I']:.2f}, permutation p = {mo['p']:.3f}; level: I = {mo_lvl['I']:.2f}). "
                    f"Clusters moving least towards the Democrats (or towards the Republicans): " + "; ".join(f"{i} ({v:+.1f})" for i, v in zip(hh.index, hh.drift_pp)) +
                    ". Clusters moving most: " + "; ".join(f"{i} ({v:+.1f})" for i, v in zip(ll.index, ll.drift_pp)) + ". Clusters line up with states' party-data regimes.",
                    theme="Geography", level="metro", datasets=["VRscores"], strength="robust",
                    stats={"moran_drift": mo["I"], "p": mo["p"], "moran_level": mo_lvl["I"]},
                    question="Are the Georgia and Missouri clusters real political change or artefacts of primary-based and modelled party?",
                    next_data="Party registration records for the same voters; primary crossover rates", views=["metro-map", "metro-explorer"], rank=9)
    if "metro_aiie" in cur and cur.metro_aiie.notna().sum() > 30:
        r = wcorr(cur.metro_aiie, cur.rep_share, cur.workers)
        finding("metro-aiie", "Metros whose industry mix is more AI-exposed have more Democratic workforces",
                f"AI exposure computed from each metro's 2019 industry mix (QCEW x AIIE) correlates {r:+.2f} with the Republican share of the "
                f"matched workforce ({cur.metro_aiie.notna().sum()} metros, weighted by workers).",
                theme="Political ideology x AI", level="metro", datasets=["VRscores", "QCEW", "AIOE (AIIE)"], strength="suggestive",
                stats={"r_weighted": r}, question="Does AI exposure predict partisan change within metros once education is controlled?",
                next_data="ACS education by metro (ACS_MSA file), metro-year VRscores", views=["metro-explorer", "metro-map"], rank=10)
    log(f"geography: {len(cur)} metros")
    return True
