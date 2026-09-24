# -*- coding: utf-8 -*-
"""08 Descriptives: the tables every later question refers back to.

What it does: coverage by occupation, industry and metro; the partisan distribution by
year; and the reliability curve that sets the minimum employer size.
Why: these are the numbers that go in the limitations section of any paper using
VRscores, and they should exist before any question is asked.
Expect: output/tables/08_*.csv and figures/08_*.png.
Diagnostics: over-representation ratios above 20 are real but concentrate in small
occupations; the reliability threshold should land near 30 to 40 two-party workers.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polisy_core import CONFIG, paths, con, log, save, q, vr_view


def main():
    P = paths()
    c = con()
    for kind in ("employer", "metro", "industry", "occupation"):
        pq = P["CANONICAL"] / f"vr_{kind}.parquet"
        if pq.exists():
            vr_view(c, f"vr_{kind}", pq, kind)

    occ = P["PANELS"] / "occupation_year.parquet"
    o = pd.read_parquet(occ) if occ.exists() else None
    if o is not None and "tot_emp" not in o:
        log("coverage by occupation skipped: the occupation panel has no OEWS employment (module 03 found no OEWS files)")
    elif o is not None:
        last = o[o.year == o.year.max()].dropna(subset=["tot_emp"]).copy()
        last = last[(last.tot_emp >= 5000) & (last.tp >= 500)]
        last["vr_share"] = last.tp / last.tp.sum()
        last["bls_share"] = last.tot_emp / last.tot_emp.sum()
        last["over_representation"] = last.vr_share / last.bls_share
        save(pd.concat([last.nlargest(15, "over_representation"), last.nsmallest(15, "over_representation")])
             [["occ_code", "rep_share", "tp", "tot_emp", "over_representation"]],
             "08_coverage_by_occupation", "above 1 = over-covered by VRscores")

    if (P["CANONICAL"] / "vr_employer.parquet").exists():
        ly = q(c, "SELECT max(year) AS y FROM vr_employer").y.iloc[0]
        d = q(c, f"SELECT rep AS r, tp AS n FROM vr_employer WHERE year = {ly} AND tp >= 5")
        p = d.r / d.n
        pbar = d.r.sum() / d.n.sum()
        tau2 = max(float(p.var() - (pbar * (1 - pbar) / d.n).mean()), 1e-4)
        need = {rr: pbar * (1 - pbar) / tau2 * rr / (1 - rr) for rr in (0.7, 0.8, 0.9)}
        rel = pd.DataFrame([{"reliability": k, "two_party_workers_needed": round(v),
                             "share_of_employers": float((d.n >= v).mean()),
                             "share_of_workers": float(d.loc[d.n >= v, "n"].sum() / d.n.sum())}
                            for k, v in need.items()])
        save(rel, "08_reliability_thresholds", f"between-employer SD = {np.sqrt(tau2):.3f}")
        grid = np.logspace(0.7, 4, 200)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grid, tau2 / (tau2 + pbar * (1 - pbar) / grid), color="#1f2733", lw=2.5)
        for k, v in need.items():
            ax.scatter([v], [k], color="#2e7d74", zorder=3)
            ax.annotate(f"{k:.0%} at {v:.0f}", (v, k), textcoords="offset points", xytext=(8, -4), fontsize=8)
        ax.set_xscale("log"); ax.set_xlabel("two-party workers"); ax.set_ylabel("reliability")
        ax.set_title(f"How large an employer must be for its score to be signal ({ly})")
        fig.savefig(Path(CONFIG["OUT"]) / "figures" / "08_reliability.png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        log("figure 08_reliability.png")
    return True


if __name__ == "__main__":
    main()
