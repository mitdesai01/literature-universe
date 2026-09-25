# -*- coding: utf-8 -*-
"""Landscape: every state-level measure at once - components, archetypes and anomalies."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, log
from . import needs
from ..adapters.politics import CSPP_CURATED

LABELS = {"vr_rep_share": "Workforce Republican share", "drift_pp": "Workforce drift", "rep_vote_share": "Republican vote share",
          "aige": "AI exposure", "coverage": "VRscores coverage", "ai_share": "AI share of patents", "ai_broad_patents_per_10k_jobs": "AI patents per job",
          "btos_ai_use_now": "Firms using AI", "net_migration_rate": "Net migration", "ai_inventor_net": "Net AI inventor moves",
          "gross_migration_rate": "Gross migration", "mover_income_gap": "Movers' income gap",
          **{"cspp_" + k: v for k, v in CSPP_CURATED.items()}}


def run():
    s = read("panel_state_year", "PANELS")
    if s is None:
        return needs("landscape", "State archetypes", "Landscape", "state", "the state panel", "Which states look alike across politics, AI and innovation?")
    feats = {}
    for c in [c for c in LABELS if c in s] + [c for c in s.columns if c.startswith("cspp_")]:
        k = s.dropna(subset=[c])
        if k.state_fips.nunique() >= 40:
            feats[c] = k[k.year >= k.year.max() - 2].groupby("state_fips")[c].mean()
    if "vr_rep_share" in s:
        a = s.dropna(subset=["vr_rep_share"])
        f, l_ = a.year.min(), a.year.max()
        feats["drift_pp"] = (a[a.year == l_].set_index("state_fips").vr_rep_share - a[a.year == f].set_index("state_fips").vr_rep_share) * 100
        if "jobs_2019" in s:
            j = s.groupby("state_fips").jobs_2019.first()
            feats["coverage"] = a[a.year == l_].set_index("state_fips").vr_workers / j
    X = pd.DataFrame(feats)
    X = X.loc[:, X.notna().mean() > 0.8]
    if X.shape[1] < 3 or len(X) < 20:
        return needs("landscape", "State archetypes", "Landscape", "state", "three or more state measures", "Which states look alike?")
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.ensemble import IsolationForest
    Z = (X - X.mean()) / X.std()
    Z = Z.fillna(0)
    pca = PCA(n_components=min(3, Z.shape[1])).fit(Z)
    P = pca.transform(Z)
    k = max(range(3, 7), key=lambda k_: silhouette_score(Z, KMeans(k_, n_init=10, random_state=0).fit_predict(Z)))
    lab = KMeans(k, n_init=30, random_state=0).fit_predict(Z)
    iso = IsolationForest(n_estimators=500, random_state=0).fit(Z)
    out = pd.DataFrame({"state_fips": X.index, "pc1": P[:, 0], "pc2": P[:, 1], "archetype": [f"type {i + 1}" for i in lab],
                        "anomaly": -iso.score_samples(Z)})
    gs = read("geo_state")
    if gs is not None:
        out = out.merge(gs[["state_fips", "state", "name", "region"]], on="state_fips", how="left")
    out = out.merge(X.reset_index().rename(columns={"index": "state_fips"}), on="state_fips", how="left")
    dataset("landscape", out)
    load = pd.DataFrame({"measure": [LABELS.get(c, c.replace("cspp_", "CSPP: ")) for c in X.columns], "pc1": pca.components_[0], "pc2": pca.components_[1]})
    dataset("landscape_loadings", load)
    view("landscape", "scatter", "States on the first two components of all state measures", "landscape", x=["pc1"] + list(X.columns),
         y=["pc2"] + list(X.columns), color="archetype", text="name", size=None, labels={**LABELS, "pc1": "Component 1", "pc2": "Component 2"})
    view("landscape-loadings", "bar", "What the two components are made of", "landscape_loadings", x="measure", y=["pc1", "pc2"])
    arch = out.groupby("archetype").apply(lambda g: ", ".join(g.state.astype(str)), include_groups=False)
    odd = out.nlargest(5, "anomaly")
    finding("landscape", "State archetypes across politics, AI and innovation",
            f"{X.shape[1]} state measures ({', '.join(LABELS.get(c, c) for c in X.columns[:8])}{'...' if X.shape[1] > 8 else ''}) reduce to components "
            f"explaining {pca.explained_variance_ratio_[0]:.0%} and {pca.explained_variance_ratio_[1]:.0%} of the variation. {k} archetypes: " +
            "; ".join(f"{a}: {m}" for a, m in arch.items()) + ". Most unusual profiles (isolation forest): " + ", ".join(odd.state.astype(str)) + ".",
            theme="Landscape", level="state", datasets=["all state-level sources"], strength="descriptive",
            stats={"measures": list(X.columns), "explained": [float(v) for v in pca.explained_variance_ratio_[:2]]},
            question="Do archetypes predict how states respond to AI (adoption, invention, policy)?",
            next_data="More state measures: AI legislation, broadband, university R&D", views=["landscape", "landscape-loadings"], rank=19)
    log(f"landscape: {X.shape[1]} measures, {k} archetypes")
    return True
