# -*- coding: utf-8 -*-
"""Exposure measures and their dynamics: which occupations each AI or automation measure points at politically
(the DAIOE SOC 2018 panel carries AIOE, GPT exposure, Webb's patent scores and Frey-Osborne side by side), and how
dynamic exposure (DAIOE, 2010-2024) moved relative to occupations' partisanship."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import read, finding, dataset, view, wcorr, boot_ci, wls, log
from . import needs

# measure -> (label, family). Families group measures built the same way.
MEASURES = {
    "frs21_aioe": ("AIOE (Felten, Raj & Seamans)", "AI benchmarks"),
    "exp_cumul": ("DAIOE, all applications", "AI benchmarks"),
    "exp_cumul_genai": ("DAIOE, generative AI", "AI benchmarks"),
    "open24_human_E1": ("GPT exposure, direct (Eloundou et al.)", "Language models"),
    "open24_human_E1_E2": ("GPT exposure with tools (Eloundou et al.)", "Language models"),
    "open24_gpt_automation": ("GPT automation potential (Eloundou et al.)", "Language models"),
    "webb19_ai_score": ("AI patents (Webb)", "Patents"),
    "webb19_software_score": ("Software patents (Webb)", "Patents"),
    "webb19_robot_score": ("Robot patents (Webb)", "Patents"),
    "fo17_p_computerisation": ("Computerisation risk (Frey & Osborne)", "Routine automation"),
}
CONTROLS = ["req_education", "log_salary"]


def run():
    p = read("panel_occupation", "PANELS")
    have = [m for m in MEASURES if p is not None and m in p and p[m].notna().sum() > 100]
    if not have:
        return needs("occ-waves", "Which occupations each AI and automation measure points at", "Political ideology x AI", "occupation",
                     "the DAIOE SOC 2018 panel (daioe_panel_soc2018)", "Do different AI exposure measures agree about who is exposed?")
    _waves(p, have)
    _dynamics(p)
    log(f"exposure: {len(have)} measures compared")
    return True


def _waves(p, have):
    d0 = p.dropna(subset=["rep_share", "workers"])
    ctrl = [c for c in CONTROLS if c in d0]
    rows, models = [], []
    for m in have:
        lab, fam = MEASURES[m]
        d = d0.dropna(subset=[m])
        r_w, r_u = wcorr(d[m], d.rep_share, d.workers), wcorr(d[m], d.rep_share)
        ci = boot_ci(lambda x, y, w: wcorr(x, y, w), d[m].values, d.rep_share.values, d.workers.values)
        m0 = wls(d, "rep_share", [m], "workers")
        dc = d.dropna(subset=ctrl)
        m1 = wls(dc, "rep_share", [m] + ctrl, "workers")
        m2 = wls(dc.dropna(subset=["soc2"]), "rep_share", [m] + ctrl, "workers", fe="soc2")
        row = {"measure": lab, "variable": m, "family": fam, "n": len(d), "r_workers": r_w, "lo": ci[0], "hi": ci[1], "r_occupations": r_u}
        for tag, mod in (("raw", m0), ("ctrl", m1), ("fe", m2)):
            row[f"coef_{tag}"] = mod["terms"][m]["coef"] * 100 if mod else np.nan
            row[f"t_{tag}"] = mod["terms"][m]["t"] if mod else np.nan
        rows.append(row)
        for tag, name, mod in (("raw", "Measure alone", m0), ("ctrl", "+ education, wage", m1), ("fe", "+ occupation group", m2)):
            if mod:
                t = mod["terms"][m]
                models.append({"model": name, "term": lab, "coef": t["coef"] * 100, "lo": (t["coef"] - 1.96 * t["se"]) * 100,
                               "hi": (t["coef"] + 1.96 * t["se"]) * 100, "t": t["t"], "n": mod["n"]})
    w = pd.DataFrame(rows).sort_values("r_workers")
    dataset("occ_waves", w)
    dataset("occ_waves_models", pd.DataFrame(models))
    view("occ-waves", "bar", "Correlation of each exposure measure with the Republican share of an occupation's workers", "occ_waves",
         x="measure", y=["r_occupations", "r_workers"], orientation="h",
         labels={"r_occupations": "Occupations weighted equally", "r_workers": "Weighted by workers"})
    view("occ-waves-models", "coef", "Republican share (points) per SD of each measure, workers-weighted", "occ_waves_models",
         x="coef", y="term", group="model", lo="lo", hi="hi", xlabel="Points of Republican share per SD, 95% interval")
    sig = lambda col, sign: [r.measure for r in w.itertuples() if abs(getattr(r, "t_" + col)) >= 2 and np.sign(getattr(r, "coef_" + col)) == sign]  # noqa: E731
    rep_side, dem_side = sig("ctrl", 1), sig("ctrl", -1)
    fam = w.groupby("family").r_workers.mean()
    title = ("AI exposure measures disagree about who is exposed: " +
             ("benchmark and language-model measures lean Democratic, patent and automation measures Republican"
              if fam.get("Language models", 0) < 0 < fam.get("Routine automation", 0) else "the partisan lean depends on the measure"))
    finding("occ-waves", title,
            f"Across {int(w.n.max())} occupations (weighted by workers), the correlation with the Republican share of workers runs from "
            f"{w.r_workers.min():+.2f} ({w.iloc[0].measure}) to {w.r_workers.max():+.2f} ({w.iloc[-1].measure}). Weighting occupations equally, "
            f"the benchmark and GPT measures sit near {w[w.family.isin(['AI benchmarks', 'Language models'])].r_occupations.mean():+.2f}, "
            f"patent and automation measures near {w[w.family.isin(['Patents', 'Routine automation'])].r_occupations.mean():+.2f}. "
            f"Holding education and wage equal, measures with a clear Republican lean (|t| of 2 or more): {', '.join(rep_side) or 'none'}; "
            f"clear Democratic lean: {', '.join(dem_side) or 'none'}.",
            theme="Political ideology x AI", level="occupation", datasets=["VRscores", "DAIOE (SOC 2018 panel)", "AIOE"],
            strength="robust" if rep_side and dem_side else "suggestive",
            stats={"n": int(w.n.max()), "r_min": w.r_workers.min(), "r_max": w.r_workers.max(),
                   **{r.variable: round(r.r_workers, 3) for r in w.itertuples()}},
            question="Which measure predicts what actually happens to workers (employment, wages, hours) since 2022, and do exposed "
                     "occupations shift politically, as robot exposure did after 2016?",
            next_data="VRscores occupation panel by year; OEWS employment and wages 2019-2025; CPS occupation panels",
            caveats=["VRscores occupations are linked to SOC 2018 by title, then code; 12% of workers are not linked.",
                     "Webb's and Frey-Osborne's scores come from older SOC versions mapped in the DAIOE panel."],
            views=["occ-waves", "occ-waves-models"], rank=2)


def _dynamics(p):
    dx = read("occ_daioe")
    if dx is None or "z_allapps" not in dx:
        return
    d0 = p.dropna(subset=["soc_link", "rep_share", "workers"])
    occ = d0.groupby("soc_link").apply(lambda g: pd.Series({"rep_share": np.average(g.rep_share, weights=g.workers),
                                                             "workers": g.workers.sum()}), include_groups=False)
    x = dx.merge(occ, left_on="soc", right_index=True)
    rows = []
    for y, g in x.groupby("year"):
        row = {"year": int(y), "r_all": wcorr(g.z_allapps, g.rep_share, g.workers)}
        if "z_genai" in g and g.z_genai.notna().sum() > 50:
            row["r_genai"] = wcorr(g.z_genai, g.rep_share, g.workers)
        for c, lab in (("z_lngmod", "r_language"), ("z_imgrec", "r_images")):
            if c in g and g[c].notna().sum() > 50:
                row[lab] = wcorr(g[c], g.rep_share, g.workers)
        rows.append(row)
    t = pd.DataFrame(rows)
    dataset("daioe_partisan_trend", t)
    view("daioe-trend", "line", "Correlation of DAIOE exposure (standing within each year) with the Republican share, 2010-2024", "daioe_partisan_trend",
         x="year", y=[c for c in ("r_all", "r_genai", "r_language", "r_images") if c in t],
         labels={"r_all": "All applications", "r_genai": "Generative AI", "r_language": "Language modelling", "r_images": "Image recognition"})
    # which occupations moved: change in standing since 2012
    keep = ["title", "group", "workers", "rep_share", "daioe", "daioe_z", "daioe_rise_z", "daioe_genai_z", "daioe_genai_rise_z", "aioe", "req_education"]
    e = p.dropna(subset=["daioe_rise_z", "rep_share"])
    dataset("occ_daioe_explorer", e[[c for c in keep if c in e]])
    view("daioe-explorer", "scatter", "Occupations: change in relative AI exposure against the Republican share of workers", "occ_daioe_explorer",
         x=[c for c in ("daioe_rise_z", "daioe_genai_rise_z", "daioe_z", "daioe_genai_z") if c in e], y=["rep_share"], size="workers",
         color="group", text="title", filter=["group"],
         labels={"daioe_rise_z": "Change in standing since 2012 (SD)", "daioe_genai_rise_z": "Change in generative-AI standing since 2022 (SD)",
                 "daioe_z": "DAIOE standing, latest year (SD)", "daioe_genai_z": "Generative-AI standing, latest year (SD)"})
    r_rise = wcorr(e.daioe_rise_z, e.rep_share, e.workers)
    ci = boot_ci(lambda a, b, w: wcorr(a, b, w), e.daioe_rise_z.values, e.rep_share.values, e.workers.values)
    same = float(x.pivot_table(index="soc", columns="year", values="daioe_allapps").corr().iloc[0, -1])
    up = e.nlargest(5, "daioe_rise_z")
    a, z = t.iloc[0], t.iloc[-1]
    finding("occ-daioe", "Dynamic exposure mostly rises for everyone; where it rose most, workforces are more Democratic",
            f"DAIOE grows about {np.exp(p.daioe_growth.mean()):.0f}-fold between 2012 and {int(z.year)} for every occupation alike (levels in {int(a.year)} "
            f"and {int(z.year)} correlate {same:.2f}), so the change is mostly a shared trend. Relative standing moved a little: occupations "
            f"that gained standing since 2012 have more Democratic workforces (r = {r_rise:+.2f}, 95% CI {ci[0]:+.2f} to {ci[1]:+.2f}, weighted by "
            f"workers), led by {', '.join(up.title.astype(str).str.split(',').str[0])}. The correlation of standing with the Republican share "
            f"drifted from {a.r_all:+.2f} ({int(a.year)}) to {z.r_all:+.2f} ({int(z.year)}).",
            theme="Political ideology x AI", level="occupation", datasets=["VRscores", "DAIOE (SOC 2010)"], strength="descriptive",
            stats={"r_rise": r_rise, "ci": ci, "levels_corr": same, "r_first": a.r_all, "r_last": z.r_all},
            question="Does a rise in an occupation's exposure precede changes in who works in it (entry by younger, more Democratic cohorts)?",
            next_data="VRscores occupation panel by year and age; OEWS by year",
            caveats=["Partisanship is measured once (2024), so this is a cross-section of changes, not a panel."],
            views=["daioe-trend", "daioe-explorer"], rank=14)
