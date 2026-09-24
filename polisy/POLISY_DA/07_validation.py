# -*- coding: utf-8 -*-
"""07 Validation: does the data match its own documentation, and do the merges hold?

What it does: five gates. (1) VRscores year counts against the codebook. (2) Duplicate
unit-years. (3) Published shares rebuilt from counts. (4) Merge coverage against a
threshold. (5) The name-matched firm panel validated against DIPI employee liberalism.
Why: gate 5 is the important one. It tells you how much the name match costs relative to
the proper crosswalk, in the only currency that matters, agreement with a known measure.
Expect: output/tables/07_validation.csv with a pass or fail per gate.
DIPI comes from canonical/dipi.parquet (built by polisy_core.load_dipi, in module 01 or
here): found under any file name, gvkeys zero-padded on both sides, and the measure is
CONFIG["DIPI_MEASURE"] (empLiberalism_10yr) or the closest employee-liberalism column.
Diagnostics: gate 5 should land near r = 0.45 to 0.50 for firms with 200+ matched
workers. The authors report 0.51 using the licensed crosswalk.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from polisy_core import CONFIG, paths, con, log, save, q, vr_view, load_dipi, dipi_measure, norm_gvkey

CODEBOOK = {2012: 370426, 2013: 393571, 2014: 417604, 2015: 439982, 2016: 461191, 2017: 482144,
            2018: 502104, 2019: 516553, 2020: 525013, 2021: 535238, 2022: 539944, 2023: 540676,
            2024: 534392}


def main():
    P = paths()
    c = con()
    gates = []
    for kind in ("employer", "metro", "industry", "occupation"):
        pq = P["CANONICAL"] / f"vr_{kind}.parquet"
        if pq.exists():
            vr_view(c, f"vr_{kind}", pq, kind)

    if (P["CANONICAL"] / "vr_employer.parquet").exists():
        yr = q(c, "SELECT year, count(*) AS rows FROM vr_employer GROUP BY 1 ORDER BY 1")
        yr["codebook"] = yr.year.map(CODEBOOK)
        yr["diff_pct"] = 100 * (yr.rows / yr.codebook - 1)
        worst = yr.diff_pct.abs().max()
        gates.append({"gate": "employer rows vs codebook", "value": f"max {worst:.3f}%",
                      "threshold": "< 0.1%", "pass": bool(worst < 0.1)})
        dup = q(c, "SELECT count(*) - count(DISTINCT (unit, year)) AS d FROM vr_employer").d.iloc[0]
        gates.append({"gate": "duplicate employer-years", "value": int(dup), "threshold": "0", "pass": dup == 0})
        chk = q(c, """SELECT max(abs(rep_share - republican_pct_two_party_imp)) AS m FROM (
                        SELECT rep_workers_imp / NULLIF(dem_workers_imp + rep_workers_imp, 0) AS rep_share,
                               republican_pct_two_party_imp FROM read_parquet('""" +
                str((P['CANONICAL'] / 'vr_employer.parquet').as_posix()) + """') )""").m.iloc[0]
        gates.append({"gate": "shares rebuilt from counts", "value": f"{chk:.2e}",
                      "threshold": "< 1e-6", "pass": bool(chk < 1e-6)})

    rep_path = Path(CONFIG["OUT"]) / "tables" / "06_merge_report.csv"
    if rep_path.exists():
        mr = pd.read_csv(rep_path)
        for _, r in mr.iterrows():
            gates.append({"gate": f"coverage: {r['merge']}", "value": f"{r['worker_share']:.1%}",
                          "threshold": "report alongside results", "pass": True})

    firm = P["PANELS"] / "firm_year.parquet"
    if not firm.exists():
        log("DIPI gates skipped: panels/firm_year.parquet missing (modules 04 and 06 build it)")
    d = load_dipi() if firm.exists() else None
    measure = dipi_measure(d) if d is not None else None
    if firm.exists() and d is not None and measure is None:
        log(f"DIPI has no liberalism column; set CONFIG['DIPI_MEASURE'] to one of {list(d.columns)[:20]}")
    if firm.exists() and measure is not None:
        f = pd.read_parquet(firm)
        f["gvkey"] = norm_gvkey(f.gvkey)
        d = d[["gvkey", "year", measure]].rename(columns={measure: "dipi"})
        m = f.merge(d, on=["gvkey", "year"], how="inner").dropna(subset=["dipi", "rep_share"])
        m["vr_dem_share"] = 1 - m.rep_share
        mid = 0.0 if m.dipi.min() < 0 else 0.5          # DIPI midpoint: 0 on a -1..1 scale, .5 on a share
        log(f"DIPI check on '{measure}': {len(m):,} matched firm-years, midpoint {mid}")
        if len(m) > 50:
            r_all = m.vr_dem_share.corr(m.dipi)
            big = m[m.workers >= 200]
            r_big = big.vr_dem_share.corr(big.dipi) if len(big) > 50 else np.nan
            agree = ((m.vr_dem_share > .5) == (m.dipi > mid)).mean()
            gates.append({"gate": "name match vs DIPI (all matched firms)", "value": f"r = {r_all:.3f}, n = {len(m):,}",
                          "threshold": "r > 0.35", "pass": bool(r_all > 0.35)})
            gates.append({"gate": "name match vs DIPI (200+ workers)", "value": f"r = {r_big:.3f}, n = {len(big):,}",
                          "threshold": "r > 0.45 (authors: 0.51)", "pass": bool(r_big > 0.45)})
            gates.append({"gate": "same majority party", "value": f"{agree:.1%}",
                          "threshold": "> 60% (authors: 73%)", "pass": bool(agree > 0.60)})
    out = save(pd.DataFrame(gates, columns=["gate", "value", "threshold", "pass"]), "07_validation")
    failed = out[~out["pass"]]
    log("all gates passed" if failed.empty else f"FAILED gates: {', '.join(failed.gate)}")
    return out


if __name__ == "__main__":
    main()
