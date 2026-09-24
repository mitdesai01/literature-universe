# -*- coding: utf-8 -*-
"""02 VRscores cleaning: raw ZIPs to canonical Parquet.

What it does: converts the four VRscores panels to one Parquet each, adding the year
from the file name where the panel does not carry it (industry and occupation).
Why: the employer panel is 6.26m rows; Parquet plus DuckDB makes every later query
seconds rather than minutes, and pandas never sees the whole thing.
Expect: data/canonical/vr_employer.parquet (about 450 MB), vr_metro, vr_industry,
vr_occupation, and output/tables/02_vr_year_counts.csv.
Finding the files: all four panels download as dataverse_files.zip, so they are told
apart by the files inside (employer_panel_year_*, msa_panel_year_*, naics_panel_year_*,
occupation_panel_year_*), not by name. Renamed zips, "(1)" copies and unzipped folders
all work; module 01 shows which file each panel came from.
Diagnostics: year counts against the codebook; duplicate unit-year rows must be zero;
occupation codes that are not SOC-shaped ("Retired", "unknown", "On Leave") are reported
and dropped downstream, not here.
"""
import pandas as pd
from polisy_core import FILES, paths, con, log, save, locate, missing_hint, panel_to_parquet, vr_view, q

PANELS = {"employer": "VR_EMPLOYER", "metro": "VR_MSA",
          "industry": "VR_INDUSTRY", "occupation": "VR_OCCUPATION"}


def main():
    P = paths()
    c = con()
    out = {}
    for kind, key in PANELS.items():
        src = locate(key)["path"]
        if src is None:
            log(f"{kind}: skipped. {missing_hint(key)}")
            continue
        pq = panel_to_parquet(c, src, P["CANONICAL"] / f"vr_{kind}.parquet", FILES[key]["members"])
        out[kind] = pq
        vr_view(c, f"vr_{kind}", pq, kind)
        dup = q(c, f"SELECT count(*) - count(DISTINCT (unit, year)) AS dups FROM vr_{kind}").iloc[0, 0]
        yrs = q(c, f"SELECT year, count(*) AS rows, sum(tp) AS two_party_workers FROM vr_{kind} GROUP BY 1 ORDER BY 1")
        yrs.insert(0, "panel", kind)
        save(yrs, f"02_year_counts_{kind}", f"{dup} duplicate unit-years")
        if dup:
            log(f"WARNING {kind}: {dup} duplicate unit-year rows")
    return out


if __name__ == "__main__":
    main()
