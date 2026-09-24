# -*- coding: utf-8 -*-
"""10 Question 2: what kind of occupations lean Republican, and does it show up in places?

What it does: (a) correlates occupational Republican share with wages, job zone and AI
exposure; (b) tracks the three inventive occupation groups over time; (c) relates metro
workplace partisanship to the local share of employment in inventive occupations.
Why: the 16-point spread inside the inventive occupations is the report's most
promotable finding, and the San Jose versus Huntsville contrast suggests mission rather
than skill is doing the work.
Expect: output/tables/10_*.csv and figures/10_*.png.
Diagnostics: the metro correlation should be near -0.39 with the 2024 OEWS file; if it
is near zero, the CBSA crosswalk is failing and module 04 should be rerun.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polisy_core import CONFIG, paths, log, save, weighted_corr, INVENTIVE_SOC2


def main():
    P = paths()
    occ_path, metro_path = P["PANELS"] / "occupation_year.parquet", P["PANELS"] / "metro_year.parquet"
    if not occ_path.exists():
        log("occupation panel missing; run 06 first")
        return None
    o = pd.read_parquet(occ_path)
    last = o[o.year == o.year.max()].copy()
    rows = []
    for var, label, logged in [("a_median", "median annual wage", True), ("job_zone", "O*NET job zone", False),
                               ("aioe", "AI exposure", False), ("tot_emp", "US employment", True)]:
        if var not in last:
            continue
        d = last[["rep_share", var, "tp"]].dropna()
        if len(d) < 30:
            continue
        x = np.log(d[var]) if logged else d[var]
        rows.append({"correlate": label, "n": len(d), "pearson_r": np.corrcoef(x, d.rep_share)[0, 1],
                     "worker_weighted_r": weighted_corr(x, d.rep_share, d.tp)})
    if rows:
        save(pd.DataFrame(rows), "10_occupation_correlates")

    o["soc2"] = o.occ_code.str[:2]
    o["group"] = o.soc2.map(INVENTIVE_SOC2).fillna("other occupations")
    grp = o.groupby(["group", "year"], as_index=False)[["dem", "rep"]].sum()
    grp["rep_share"] = grp.rep / (grp.rep + grp.dem)
    save(grp, "10_inventive_groups")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for g, d in grp.groupby("group"):
        ax.plot(d.year, 100 * d.rep_share, marker="o", lw=3 if g != "other occupations" else 1.5, label=g)
    ax.set_ylabel("Republican share (%)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("The inventive occupations are three different political worlds")
    fig.savefig(Path(CONFIG["OUT"]) / "figures" / "10_inventive_groups.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

    oews_metro = P["CANONICAL"] / "oews_msa.parquet"
    if metro_path.exists() and oews_metro.exists():
        om = pd.read_parquet(oews_metro)
        om = om[om.year == om.year.max()].copy()
        om["soc2"] = om.occ_code.str[:2]
        tot = om.groupby("cbsa", as_index=False).tot_emp.sum().rename(columns={"tot_emp": "total_emp"})
        inv = (om[om.soc2.isin(INVENTIVE_SOC2)].groupby("cbsa", as_index=False).tot_emp.sum()
               .rename(columns={"tot_emp": "inventive_emp"}))
        loc = tot.merge(inv, on="cbsa", how="left")
        loc["inventive_share"] = loc.inventive_emp / loc.total_emp
        m = pd.read_parquet(metro_path)
        m = m[m.year == m.year.max()].merge(loc, on="cbsa", how="inner").dropna(subset=["inventive_share", "rep_share"])
        if len(m) > 20:
            r = m.inventive_share.corr(m.rep_share)
            save(m[["msa", "cbsa", "rep_share", "inventive_share", "total_emp", "tp"]]
                 .sort_values("inventive_share", ascending=False), "10_metro_inventive",
                 f"correlation with workplace Republican share = {r:.3f}")
            fig, ax = plt.subplots(figsize=(7.5, 5))
            ax.scatter(100 * m.inventive_share, 100 * m.rep_share, s=np.sqrt(m.tp) / 6, alpha=.5, color="#2f5d9a")
            for _, row in pd.concat([m.nlargest(3, "inventive_share"), m.nlargest(3, "rep_share")]).iterrows():
                ax.annotate(str(row.msa)[:28], (100 * row.inventive_share, 100 * row.rep_share), fontsize=7)
            ax.set_xlabel("local employment in inventive occupations (%)")
            ax.set_ylabel("Republican share of metro workers (%)")
            ax.set_title(f"Technical metros and workplace politics (r = {r:.2f})")
            fig.savefig(Path(CONFIG["OUT"]) / "figures" / "10_metro_inventive.png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            log(f"metro inventive-share correlation: {r:.3f}")
    return True


if __name__ == "__main__":
    main()
