# -*- coding: utf-8 -*-
"""Occupations: how AI exposure lines up with workforce partisanship, and what explains it."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, boot_ci, wls, zscore, log, LAB
from . import needs, coef_rows

APP_LABELS = {"exp_abstract_strategy_games": "Strategy games", "exp_real_time_video_games": "Video games",
              "exp_image_recognition": "Image recognition", "exp_visual_question_answering": "Visual Q&A",
              "exp_generating_images": "Image generation", "exp_reading_comprehension": "Reading comprehension",
              "exp_language_modeling": "Language modeling", "exp_translation": "Translation",
              "exp_speech_recognition": "Speech recognition", "exp_instrumental_track_recognition": "Music recognition"}
CONTROLS = {"log_salary": "Median wage (log)", "req_education": "Required education", "female": "Female share", "white": "White share"}


def run():
    p = read("panel_occupation", "PANELS")
    if p is None or "aioe" not in p or p.aioe.notna().sum() < 30:
        return needs("occ-ai-partisanship", "AI exposure and partisanship across occupations", "Political ideology x AI", "occupation",
                     "VRscores occupations and AIOE", "Are AI-exposed occupations politically distinctive?")
    d = p[(p.workers >= LAB["SETTINGS"]["min_workers"]) & p.aioe.notna()].copy()
    apps = [c for c in APP_LABELS if c in d]
    if len(apps) >= 3:
        from sklearn.decomposition import PCA
        ok = d[apps].notna().all(axis=1)
        X = d.loc[ok, apps].apply(zscore)
        pca = PCA(n_components=3).fit(X)
        sc = pca.transform(X)
        sign = np.sign(pca.components_[1][apps.index("exp_image_recognition")]) if "exp_image_recognition" in apps else 1
        d.loc[ok, "pc_general"], d.loc[ok, "pc_perception_vs_language"] = sc[:, 0] * np.sign(pca.components_[0].sum()), sc[:, 1] * sign
        load = pd.DataFrame({"application": [APP_LABELS[a] for a in apps], "general": pca.components_[0] * np.sign(pca.components_[0].sum()),
                             "perception_vs_language": pca.components_[1] * sign})
        dataset("occ_pca_loadings", load)
        view("occ-pca", "bar", "How the ten AI applications load on the two exposure dimensions", "occ_pca_loadings",
             x="application", y=["perception_vs_language", "general"], note=f"explained variance: {pca.explained_variance_ratio_[0]:.0%} and {pca.explained_variance_ratio_[1]:.0%}")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        k = max(range(3, 7), key=lambda k_: silhouette_score(X, KMeans(k_, n_init=10, random_state=0).fit_predict(X)))
        d.loc[ok, "exposure_cluster"] = KMeans(k, n_init=20, random_state=0).fit_predict(X).astype(str)
    w = d.workers
    r_raw, r_w = wcorr(d.aioe, d.rep_share), wcorr(d.aioe, d.rep_share, w)
    ci_w = boot_ci(lambda x, y, ww: wcorr(x, y, ww), d.aioe.values, d.rep_share.values, w.values)
    per_app = pd.DataFrame([{"measure": APP_LABELS.get(c, c), "r_workers": wcorr(d[c], d.rep_share, w), "r_occupations": wcorr(d[c], d.rep_share)}
                            for c in ["aioe", "aioe_lm", "aioe_ig"] + apps if c in d])
    per_app["measure"] = per_app.measure.replace({"aioe": "AIOE (all applications)", "aioe_lm": "Language-model AIOE", "aioe_ig": "Image-generation AIOE"})
    dataset("occ_exposure_corr", per_app)
    view("occ-corr", "bar", "Correlation of each AI exposure measure with the Republican share of the workforce", "occ_exposure_corr",
         x="measure", y=["r_occupations", "r_workers"], orientation="h", note="r_workers weights occupations by matched workers")
    ctrl = [c for c in CONTROLS if c in d and d[c].notna().sum() > 60]
    m0 = wls(d, "rep_share", ["aioe"], "workers")
    m1 = wls(d, "rep_share", ["aioe"] + [c for c in ("log_salary", "req_education") if c in ctrl], "workers")
    m2 = wls(d, "rep_share", ["aioe"] + ctrl, "workers", fe="soc2")
    rows = coef_rows(m0, "AIOE only") + coef_rows(m1, "+ wage, education") + coef_rows(m2, "+ gender, race, occupation group")
    names = {"aioe": "AI exposure (AIOE)", **CONTROLS, "pc_perception_vs_language": "Perception vs language exposure",
             "pc_general": "General AI exposure (component 1)"}
    for r in rows:
        r["coef"], r["lo"], r["hi"] = r["coef"] * 100, r["lo"] * 100, r["hi"] * 100
        r["term"] = names.get(r["term"], r["term"])
    if "pc_perception_vs_language" in d:
        m3 = wls(d, "rep_share", ["pc_perception_vs_language"], "workers")
        m4 = wls(d, "rep_share", ["pc_perception_vs_language", "pc_general"] + ctrl, "workers", fe="soc2")
        for lab, m in (("perception vs language, alone", m3), ("perception vs language + controls + group", m4)):
            for r in coef_rows(m, lab, names):
                r["coef"], r["lo"], r["hi"] = r["coef"] * 100, r["lo"] * 100, r["hi"] * 100
                rows.append(r)
    dataset("occ_models", pd.DataFrame(rows))
    view("occ-models", "coef", "Republican share (points) per standard deviation of each measure, weighted by workers", "occ_models",
         x="coef", y="term", group="model", lo="lo", hi="hi")
    if m2:
        d.loc[m2["resid"].index, "resid"] = m2["resid"]
    keep = ["occ_key", "title", "group", "soc_link", "link_method", "workers", "rep_share", "change_pp", "aioe", "aioe_lm", "aioe_ig",
            "pc_general", "pc_perception_vs_language", "exposure_cluster", "cognitive_share", "log_salary", "req_education",
            "female", "white", "creative_weight", "resid", "dyn_change", "daioe_z", "daioe_genai_z", "daioe_rise_z", "frs21_aioe",
            "open24_human_E1_E2", "webb19_ai_score", "webb19_robot_score", "fo17_p_computerisation"] + apps
    dataset("occ_explorer", d[[c for c in keep if c in d]])
    view("occ-explorer", "scatter", "Occupations: AI exposure against the Republican share of the workforce", "occ_explorer",
         x=[c for c in ["aioe", "aioe_lm", "aioe_ig", "pc_perception_vs_language", "pc_general", "req_education", "log_salary",
                        "cognitive_share", "female", "dyn_change", "daioe_z", "daioe_genai_z", "daioe_rise_z", "open24_human_E1_E2",
                        "webb19_ai_score", "webb19_robot_score", "fo17_p_computerisation"] + apps if c in d],
         y=["rep_share", "resid", "change_pp"], size="workers", color="group", text="title", filter=["group", "exposure_cluster"],
         labels={**{k: v for k, v in APP_LABELS.items()}, "aioe": "AI exposure (AIOE)", "aioe_lm": "Language-model exposure",
                 "aioe_ig": "Image-generation exposure", "rep_share": "Republican share", "resid": "Unexplained partisanship",
                 "pc_perception_vs_language": "Perception minus language exposure", "pc_general": "General AI exposure",
                 "req_education": "Required education", "log_salary": "Median wage (log)", "change_pp": "Change since first year (points)",
                 "cognitive_share": "Cognitive share of abilities", "female": "Female share", "dyn_change": "Change in dynamic exposure"})
    coef = lambda m, t: (m["terms"][t]["coef"] * 100 if m else np.nan)  # noqa: E731
    tval = lambda m, t: (m["terms"][t]["t"] if m else np.nan)  # noqa: E731
    finding("occ-ai-education", "AI exposure looks Democratic across occupations, but the link runs through education",
            f"Across {len(d)} occupations, more AI-exposed ones have more Democratic workforces (r = {r_raw:+.2f}); weighted by workers "
            f"the link nearly vanishes (r = {r_w:+.2f}, 95% CI {ci_w[0]:+.2f} to {ci_w[1]:+.2f}). With wage and education held equal, "
            f"exposure's coefficient is {coef(m1, 'aioe'):+.1f} points per SD (t = {tval(m1, 'aioe'):+.1f}); adding gender, race and "
            f"occupation group gives {coef(m2, 'aioe'):+.1f} (t = {tval(m2, 'aioe'):+.1f}).",
            theme="Political ideology x AI", level="occupation", datasets=["VRscores", "AIOE"],
            strength="robust" if abs(r_raw) > 0.2 else "descriptive",
            stats={"n": len(d), "r_unweighted": r_raw, "r_weighted": r_w, "ci_weighted": ci_w,
                   "aioe_with_wage_edu_pp": coef(m1, "aioe"), "aioe_full_pp": coef(m2, "aioe"), "n_full": m2["n"] if m2 else None},
            question="Will generative AI's labor-market effects land on educated, Democratic-leaning workers, and does exposure change political attitudes or sorting?",
            next_data="Dynamic AIOE (exposure after 2022), VRscores occupation panel by year, worker panels (CPS, CES) with party",
            caveats=["Demographic controls exist for fewer occupations, so the full model has a smaller sample.",
                     "VRscores partisanship is party identity from voter files, not ideology."],
            views=["occ-explorer", "occ-models", "occ-corr"], rank=1)
    if m1 and m2 and "req_education" in m1["terms"] and "log_salary" in m1["terms"]:
        e1, s1 = m1["terms"]["req_education"], m1["terms"]["log_salary"]
        e2, s2, f2 = (m2["terms"].get(k, {"coef": np.nan, "t": np.nan}) for k in ("req_education", "log_salary", "female"))
        pay_holds = abs(s2["t"]) >= 2
        finding("occ-education-vs-pay", "Education pulls occupations Democratic and pay pulls them Republican" +
                ("" if pay_holds else "; the pay effect runs through who does the job"),
                f"With both in the model, one SD more required education goes with {e1['coef'] * 100:+.1f} points Republican share (t = {e1['t']:+.1f}) "
                f"and one SD higher pay with {s1['coef'] * 100:+.1f} (t = {s1['t']:+.1f}), {m1['n']} occupations. Adding gender and race shares and "
                f"occupation group leaves education at {e2['coef'] * 100:+.1f} (t = {e2['t']:+.1f}) and pay at {s2['coef'] * 100:+.1f} (t = {s2['t']:+.1f}); "
                f"the female share is the strongest single predictor ({f2['coef'] * 100:+.1f} points per SD, t = {f2['t']:+.1f}).",
                theme="Political ideology x AI", level="occupation", datasets=["VRscores", "AIOE inputs (O*NET, OES wages, CPS demographics)"],
                strength="robust" if abs(e2["t"]) >= 2 else "suggestive", stats={"n_wage_edu": m1["n"], "n_full": m2["n"], "r2_full": m2["r2"]},
                    question="Is AI's partisan footprint just the education realignment, or does exposure add something within education levels?",
                    next_data="Worker-level data with education, occupation and party", views=["occ-models"], rank=3)
    if "pc_perception_vs_language" in d:
        r_pc = wcorr(d.pc_perception_vs_language, d.rep_share, w)
        m4t = m4["terms"]["pc_perception_vs_language"] if m4 else {"coef": np.nan, "t": np.nan}
        finding("occ-two-ais", "Two kinds of AI exposure: language-AI jobs lean Democratic, perception-AI jobs Republican, until you control for education",
                f"Beyond overall exposure, occupations differ in which AI touches them: the second principal component contrasts image "
                f"recognition and games with language modelling, translation and reading. It correlates {r_pc:+.2f} with Republican share "
                f"(weighted), but with wage, education, gender, race and occupation group it is {m4t['coef'] * 100:+.1f} points per SD "
                f"(t = {m4t['t']:+.1f}).", theme="Political ideology x AI", level="occupation", datasets=["AIOE inputs (O*NET abilities)", "VRscores"],
                strength="fragile" if abs(m4t["t"]) < 2 else "suggestive", stats={"r_weighted": r_pc, "coef_controls_pp": m4t["coef"] * 100, "t": m4t["t"]},
                question="Does the generative-AI wave (language) shift the politics of exposure towards Democratic-leaning professions compared with automation and vision AI?",
                next_data="Exposure measures by AI generation (Dynamic AIOE), task-level exposure", views=["occ-pca", "occ-explorer"], rank=4)
    if "resid" in d:
        big = d[d.workers >= 5000].dropna(subset=["resid"])
        hi, lo = big.nlargest(5, "resid"), big.nsmallest(5, "resid")
        finding("occ-anomalies", "Occupations far more partisan than their exposure, pay, education and demographics predict",
                "More Republican than predicted: " + "; ".join(f"{t} ({r * 100:+.0f})" for t, r in zip(hi.title, hi.resid)) +
                ". More Democratic: " + "; ".join(f"{t} ({r * 100:+.0f})" for t, r in zip(lo.title, lo.resid)) + ".",
                theme="Anomalies", level="occupation", datasets=["VRscores", "AIOE"], strength="descriptive",
                question="What do the outliers share (mission, religion, public-facing work, organisational setting)?",
                next_data="Occupation x employer-type data; values and mission measures", views=["occ-explorer"], rank=20)
    if "dyn_change" in d and d.dyn_change.notna().sum() > 30:
        r_dyn = wcorr(d.dyn_change, d.rep_share, w)
        finding("occ-dynamic", "Where AI exposure grew fastest, and whose workers those are",
                f"Using {p.dyn_measure.dropna().iloc[0] if 'dyn_measure' in p else 'dynamic exposure'}, the growth in exposure correlates "
                f"{r_dyn:+.2f} with the Republican share of the workforce (weighted by workers, {d.dyn_change.notna().sum()} occupations).",
                theme="Political ideology x AI", level="occupation", datasets=["Dynamic AIOE", "VRscores"], strength="suggestive",
                stats={"r_weighted": r_dyn}, question="Does rising exposure predict partisan drift within occupations?",
                next_data="VRscores occupation panel by year", views=["occ-explorer"], rank=5)
    log(f"occupations: {len(d)} occupations analysed")
    return True
