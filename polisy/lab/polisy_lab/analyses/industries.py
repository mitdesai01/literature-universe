# -*- coding: utf-8 -*-
"""Industries: AI exposure, occupational composition, drift, and (with BTOS) actual AI use."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, boot_ci, wls, log
from . import needs, coef_rows


def run():
    p = read("panel_industry", "PANELS")
    if p is None or "aiie" not in p or p.aiie.notna().sum() < 20:
        return needs("ind-ai-partisanship", "AI exposure and partisanship across industries", "Political ideology x AI", "industry",
                     "VRscores industries and AIIE", "Are AI-exposed industries politically distinctive?")
    d = p.dropna(subset=["aiie", "rep_share"]).copy()
    d["sector2"] = d.naics4.str[:2].replace({"32": "31", "33": "31", "45": "44", "49": "48"})
    w = d.workers
    out = {}
    for c in [x for x in ("aiie", "aiie_lm", "aiie_ig") if x in d]:
        out[c] = (wcorr(d[c], d.rep_share, w), boot_ci(lambda x, y, ww: wcorr(x, y, ww), d[c].values, d.rep_share.values, w.values))
    ctrl = [c for c in ("ind_education", "ind_log_wage") if c in d and d[c].notna().sum() > 30]
    m0 = wls(d, "rep_share", ["aiie"], "workers")
    m1 = wls(d, "rep_share", ["aiie"] + ctrl, "workers")
    m2 = wls(d, "rep_share", ["aiie"] + ctrl, "workers", fe="sector2")
    m3 = wls(d, "rep_share", ["aiie_lm"] + ctrl, "workers", fe="sector2") if "aiie_lm" in d else None
    names = {"aiie": "AI exposure (AIIE)", "aiie_lm": "Language-model AIIE", "ind_education": "Education of the workforce",
             "ind_log_wage": "Median wage (log)"}
    rows = []
    for lab, m in (("AIIE only", m0), ("+ education, wage", m1), ("+ sector", m2), ("language-model AIIE + all", m3)):
        for r in coef_rows(m, lab, names):
            r["coef"], r["lo"], r["hi"] = r["coef"] * 100, r["lo"] * 100, r["hi"] * 100
            rows.append(r)
    dataset("ind_models", pd.DataFrame(rows))
    view("ind-models", "coef", "Republican share (points) per SD, industries weighted by workers", "ind_models", x="coef", y="term", group="model", lo="lo", hi="hi")
    keep = ["naics4", "title", "sector", "workers", "rep_share", "change_pp", "aiie", "aiie_lm", "aiie_ig", "ind_education", "ind_log_wage",
            "rep_pred", "culture_gap", "occ_coverage", "btos_ai_use_sector"]
    dataset("ind_explorer", d[[c for c in keep if c in d]])
    labels = {"aiie": "AI exposure (AIIE)", "aiie_lm": "Language-model AIIE", "aiie_ig": "Image-generation AIIE", "rep_share": "Republican share",
              "change_pp": "Change 2012-2024 (points)", "ind_education": "Education of the workforce", "ind_log_wage": "Median wage (log)",
              "rep_pred": "Predicted from occupation mix", "culture_gap": "Actual minus predicted", "btos_ai_use_sector": "AI use (BTOS, sector)"}
    view("ind-explorer", "scatter", "Industries: AI exposure against the Republican share of the workforce", "ind_explorer",
         x=[c for c in ("aiie", "aiie_lm", "aiie_ig", "ind_education", "ind_log_wage", "rep_pred", "btos_ai_use_sector") if c in d],
         y=[c for c in ("rep_share", "change_pp", "culture_gap") if c in d], size="workers", color="sector", text="title",
         filter=["sector"], labels=labels)
    t = lambda m, k: m["terms"][k]["t"] if m else np.nan  # noqa: E731
    cf = lambda m, k: m["terms"][k]["coef"] * 100 if m else np.nan  # noqa: E731
    r_a, ci_a = out["aiie"]
    lm_txt = f" Language-model exposure is stronger (r = {out['aiie_lm'][0]:+.2f})." if "aiie_lm" in out else ""
    within = m2 and abs(t(m2, "aiie")) >= 2
    lm_within = m3 and abs(t(m3, "aiie_lm")) >= 2
    title = ("AI-exposed industries have more Democratic workforces, even within sectors" if within else
             "AI-exposed industries have more Democratic workforces, mostly because their workers are highly educated" +
             ("; language-model exposure keeps a modest link within sectors" if lm_within else ""))
    finding("ind-ai-partisanship", title,
            f"Across {len(d)} four-digit industries, AI exposure correlates {r_a:+.2f} with the Republican share of the matched workforce "
            f"(weighted; 95% CI {ci_a[0]:+.2f} to {ci_a[1]:+.2f}).{lm_txt} Exposure tracks education closely, and with education and wages "
            f"held equal the coefficient falls to {cf(m1, 'aiie'):+.1f} points per SD (t = {t(m1, 'aiie'):+.1f}); within sectors it is "
            f"{cf(m2, 'aiie'):+.1f} (t = {t(m2, 'aiie'):+.1f})" + (f" and {cf(m3, 'aiie_lm'):+.1f} for language-model exposure (t = {t(m3, 'aiie_lm'):+.1f})." if m3 else "."),
            theme="Political ideology x AI", level="industry", datasets=["VRscores", "AIOE (AIIE)", "OES staffing"],
            strength="suggestive" if m2 and abs(t(m2, "aiie")) >= 2 else "fragile",
            stats={"n": len(d), "r_weighted": r_a, "ci": ci_a, "coef_edu_wage_pp": cf(m1, "aiie"), "coef_sector_pp": cf(m2, "aiie"),
                   "corr_aiie_education": float(d[["aiie", "ind_education"]].corr().iloc[0, 1]) if "ind_education" in d else None},
            question="Is there an industry-level (firm culture, geography) component of AI exposure's partisan footprint beyond the education of the workforce?",
            next_data="Employer-level VRscores linked to NAICS and location; BTOS adoption by industry", views=["ind-explorer", "ind-models"], rank=2)
    if "rep_pred" in d and d.culture_gap.notna().sum() > 30:
        c = d.dropna(subset=["culture_gap"])
        r_c = wcorr(c.rep_pred, c.rep_share, c.workers)
        sd_gap = np.sqrt(np.average((c.culture_gap - np.average(c.culture_gap, weights=c.workers)) ** 2, weights=c.workers))
        sd_act = np.sqrt(np.average((c.rep_share - np.average(c.rep_share, weights=c.workers)) ** 2, weights=c.workers))
        big = c[c.workers >= 20000]
        g = pd.concat([big.nsmallest(8, "culture_gap"), big.nlargest(8, "culture_gap")])
        dataset("ind_culture_gap", g[["title", "sector", "workers", "rep_share", "rep_pred", "culture_gap"]].assign(culture_gap_pp=lambda z: z.culture_gap * 100))
        view("ind-gap", "bar", "Industries more Democratic (left) or Republican (right) than their occupations predict (points)", "ind_culture_gap",
             x="culture_gap_pp", y="title", orientation="h", color="sector")
        gm = wls(c, "culture_gap", ["aiie"] + ctrl, "workers")
        finding("ind-composition", "Occupational mix explains most of an industry's partisanship; the rest points to place and mission",
                f"Predicting each industry's Republican share from national occupation shares and its staffing pattern gives r = {r_c:+.2f} "
                f"with the actual share ({len(c)} industries with 70%+ of jobs linked). The remaining gap has an SD of {sd_gap * 100:.1f} points "
                f"(actual: {sd_act * 100:.1f}). Most Democratic relative to their occupations: " + ", ".join(big.nsmallest(3, "culture_gap").title) +
                "; most Republican: " + ", ".join(big.nlargest(3, "culture_gap").title) + "." +
                (f" The gap is unrelated to AI exposure once education is held equal (t = {gm['terms']['aiie']['t']:+.1f})." if gm else ""),
                theme="Organizations", level="industry", datasets=["VRscores", "OES staffing", "AIOE"], strength="robust",
                stats={"r_pred_actual": r_c, "sd_gap_pp": sd_gap * 100, "sd_actual_pp": sd_act * 100},
                question="How much of the residual is geography (where the industry sits) versus organisational culture or mission?",
                next_data="Industry x metro employment (QCEW) with metro partisanship; employer-level data", views=["ind-gap", "ind-explorer"], rank=6)
    if "change_pp" in d and d.change_pp.notna().sum() > 30:
        c = d.dropna(subset=["change_pp"])
        md = wls(c, "change_pp", ["aiie", "rep_share"], "workers")
        finding("ind-drift", "Nearly every industry drifted Democratic, and AI exposure does not predict by how much",
                f"{(c.change_pp < 0).mean():.0%} of {len(c)} industries moved towards the Democrats between the first and last year "
                f"(weighted mean {np.average(c.change_pp, weights=c.workers):+.1f} points). Exposure does not predict the size of the move "
                f"(t = {md['terms']['aiie']['t']:+.1f}); industries that started more Republican moved more (t = {md['terms']['rep_share']['t']:+.1f}), "
                f"consistent with party being fixed at 2024 while younger cohorts enter.",
                theme="Measurement", level="industry", datasets=["VRscores", "AIOE"], strength="robust",
                stats={"n": len(c), "share_moving_dem": float((c.change_pp < 0).mean())},
                question="Separate cohort replacement from real change: does drift vanish once workforce age is held constant?",
                next_data="VRscores by worker age cohort; a second voter-file vintage", views=["ind-explorer"], rank=12)
    if "btos_ai_use_sector" in d and d.btos_ai_use_sector.notna().sum() > 10:
        s = d.dropna(subset=["btos_ai_use_sector"]).groupby("sector2").apply(lambda g: pd.Series({
            "aiie": np.average(g.aiie, weights=g.workers), "ai_use": g.btos_ai_use_sector.iloc[0],
            "rep_share": np.average(g.rep_share, weights=g.workers), "workers": g.workers.sum(),
            "sector": g.sector.iloc[0] if "sector" in g else g.name}), include_groups=False).reset_index()
        b = np.polyfit(s.aiie, s.ai_use, 1)
        s["adoption_gap"] = s.ai_use - np.polyval(b, s.aiie)
        dataset("sector_adoption", s)
        view("sector-adoption", "scatter", "Sectors: AI exposure against the share of firms using AI (BTOS)", "sector_adoption",
             x=["aiie"], y=["ai_use", "adoption_gap"], size="workers", color="rep_share", text="sector",
             labels={"aiie": "AI exposure (AIIE)", "ai_use": "Firms using AI (%)", "adoption_gap": "Use beyond what exposure predicts", "rep_share": "Republican share"})
        r_e = float(s[["aiie", "ai_use"]].corr().iloc[0, 1])
        r_p = float(s[["adoption_gap", "rep_share"]].corr().iloc[0, 1])
        finding("sector-exposure-adoption", "AI exposure versus actual AI use by sector",
                f"Across {len(s)} sectors, exposure (AIIE) correlates {r_e:+.2f} with the share of firms reporting AI use (BTOS, latest year). "
                f"The use left unexplained by exposure correlates {r_p:+.2f} with the sector's Republican share. Leaders beyond exposure: "
                + ", ".join(s.nlargest(3, "adoption_gap").sector.astype(str)) + "; laggards: " + ", ".join(s.nsmallest(3, "adoption_gap").sector.astype(str)) + ".",
                theme="AI adoption", level="industry", datasets=["BTOS", "AIOE (AIIE)", "VRscores"], strength="suggestive" if len(s) >= 15 else "descriptive",
                stats={"n_sectors": len(s), "r_exposure_use": r_e, "r_gap_partisanship": r_p},
                question="Do workforce politics or leadership shape how fast exposed industries actually adopt AI?",
                next_data="BTOS by subsector and firm size; firm-level adoption (e.g. job postings mentioning AI)", views=["sector-adoption"], rank=7)
    log(f"industries: {len(d)} industries analysed")
    return True
