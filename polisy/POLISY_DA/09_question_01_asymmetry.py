# -*- coding: utf-8 -*-
"""09 Question 1: is the partisan asymmetry real?

What it does: recomputes party-specific exposure under four variants (incl. leaners,
registered only, balanced employers, 25+ two-party workers) for every year, then splits
the metro trend by how each state records party.
Why: the report found Republican exposure rising (1.102 to 1.123) while the Democratic
figure fell (1.069 to 1.062). Before that is a finding it has to survive measurement
choices, employer entry and exit, and the inferred-party states.
Expect: output/tables/09_exposure_variants.csv, 09_exposure_change.csv, figure
09_asymmetry.png.
Diagnostics: the asymmetry counts as robust only if the Republican change is positive
and the Democratic change negative in all four variants.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from polisy_core import CONFIG, paths, con, log, save, q, vr_view, exposure_by_party


def main():
    P = paths()
    c = con()
    vr_view(c, "vr_employer", P["CANONICAL"] / "vr_employer.parquet", "employer")
    years = [int(y) for y in q(c, "SELECT DISTINCT year FROM vr_employer ORDER BY 1").year]
    n_years = len(years)
    balanced = set(q(c, f"""SELECT unit FROM vr_employer GROUP BY unit
                            HAVING count(DISTINCT year) = {n_years}""").unit)
    rows = []
    for y in years:
        d = q(c, f"SELECT unit, dem, rep, dem_raw, rep_raw, tp FROM vr_employer WHERE year = {y}")
        variants = {
            "incl. leaners": (d.dem, d.rep),
            "registered only": (d.dem_raw, d.rep_raw),
            "balanced employers": (d.loc[d.unit.isin(balanced), "dem"], d.loc[d.unit.isin(balanced), "rep"]),
            "25+ two-party workers": (d.loc[d.tp >= 25, "dem"], d.loc[d.tp >= 25, "rep"]),
        }
        for label, (dd, rr) in variants.items():
            de, re_, comb = exposure_by_party(dd, rr)
            rows.append({"variant": label, "year": y, "dem_exposure": de,
                         "rep_exposure": re_, "combined": comb})
    T = save(pd.DataFrame(rows), "09_exposure_variants")
    first, last = years[0], years[-1]
    chg = (T[T.year == last].set_index("variant")[["dem_exposure", "rep_exposure"]] -
           T[T.year == first].set_index("variant")[["dem_exposure", "rep_exposure"]]).reset_index()
    chg.columns = ["variant", "dem_change", "rep_change"]
    chg["asymmetry_holds"] = (chg.rep_change > 0) & (chg.dem_change < 0)
    save(chg, "09_exposure_change", f"{first} to {last}; holds in {chg.asymmetry_holds.sum()} of {len(chg)} variants")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v, g in T.groupby("variant"):
        ax.plot(g.year, g.rep_exposure, marker="o", label=f"Republican, {v}")
        ax.plot(g.year, g.dem_exposure, marker="s", ls="--", alpha=.7, label=f"Democratic, {v}")
    ax.set_ylabel("own-party exposure (1 = random mixing)")
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax.set_title("Does the partisan asymmetry survive every variant?")
    fig.savefig(Path(CONFIG["OUT"]) / "figures" / "09_asymmetry.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    log(f"asymmetry holds in {chg.asymmetry_holds.sum()} of {len(chg)} variants")
    return T


if __name__ == "__main__":
    main()
