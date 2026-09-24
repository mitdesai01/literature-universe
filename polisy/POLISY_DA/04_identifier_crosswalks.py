# -*- coding: utf-8 -*-
"""04 Identifier crosswalks: metro, occupation, industry, and the firm name match.

What it does: builds the four key tables every later merge uses.
  keys/cw_msa_cbsa.csv     VRscores metro name  -> CBSA code (+ state, party regime)
  keys/cw_occupation.csv   VRscores O*NET code  -> SOC6 (invalid codes flagged)
  keys/cw_industry.csv     VRscores NAICS6      -> NAICS4 (OEWS resolution)
  keys/cw_vrid_gvkey.csv   VRscores employer    -> gvkey, by name with evidence
Why: merges fail silently when keys are built inline. Building them once, with a match
score per row, makes every later join auditable.
Expect: match rates near 97% for metros, 100% for occupations after dropping invalid
codes, 71% for industries, and roughly 35% of Compustat firms for the name match.
Diagnostics: review every metro row with match_score < 3 by hand once; check the firm
match against DIPI (module 07).
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd
from polisy_core import (CONFIG, paths, con, log, save, q, sqlp, parse_msa, party_regime,
                         norm_name, vr_view)

try:
    from rapidfuzz import fuzz, process
    HAVE_RF = True
except ImportError:
    HAVE_RF = False


def cbsa_reference(P):
    """Census delineation file if present, else the OEWS metro file (codes + titles)."""
    delin = P["RAW"] / "list1_2023.xlsx"
    if delin.exists():
        raw = pd.read_excel(delin, skiprows=2, dtype=str)
        raw.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in raw.columns]
        code = next(c for c in raw.columns if c.startswith("cbsa_code"))
        title = next(c for c in raw.columns if c.startswith("cbsa_title"))
        typ = next((c for c in raw.columns if "metropolitan" in c and "division" not in c), None)
        st = next((c for c in raw.columns if c.startswith("fips_state")), None)
        cty = next((c for c in raw.columns if c.startswith("fips_county")), None)
        d = raw.rename(columns={code: "cbsa", title: "cbsa_title", typ: "cbsa_type",
                                st: "state_fips", cty: "county_fips3"})
        d = d[d.cbsa.str.match(r"^\d{5}$", na=False)].copy()
        if "state_fips" in d and "county_fips3" in d:
            d["county_fips"] = d.state_fips.str.zfill(2) + d.county_fips3.str.zfill(3)
            d[["cbsa", "county_fips"]].dropna().to_csv(P["KEYS"] / "cw_cbsa_county.csv", index=False)
        log("CBSA reference: Census delineation file")
        return d[["cbsa", "cbsa_title"]].drop_duplicates()
    oews = P["CANONICAL"] / "oews_msa.parquet"
    if oews.exists():
        d = pd.read_parquet(oews, columns=["cbsa", "area_title"]).drop_duplicates()
        log("CBSA reference: OEWS metro file (no county links; election merges need the Census file)")
        return d.rename(columns={"area_title": "cbsa_title"})
    log("no CBSA reference available")
    return None


def build_metro_crosswalk(P, vr_msa_names, ref):
    parsed = ref.cbsa_title.map(parse_msa)
    ref = ref.assign(cities=[p[0] for p in parsed], states=[p[1] for p in parsed])
    by_state = {}
    for r in ref.itertuples(index=False):
        for s in r.states:
            by_state.setdefault(s, []).append(r)
    rows = []
    for name in pd.Series(vr_msa_names).dropna().unique():
        cities, states = parse_msa(name)
        best, score = None, 0
        for cand in by_state.get(states[0] if states else "", []):
            s = 3 if cities and cand.cities and cities[0] == cand.cities[0] else (2 if set(cities) & set(cand.cities) else 0)
            if s > score:
                best, score = cand, s
        rows.append({"msa": name, "cbsa": best.cbsa if best is not None else None,
                     "cbsa_title": best.cbsa_title if best is not None else None,
                     "match_score": score, "state": states[0] if states else None,
                     "party_regime": party_regime(states[0] if states else None)})
    cw = pd.DataFrame(rows)
    cw.to_csv(P["KEYS"] / "cw_msa_cbsa.csv", index=False)
    log(f"metro crosswalk: {cw.cbsa.notna().mean():.1%} matched, {(cw.match_score == 3).mean():.1%} on the principal city")
    return cw


def build_firm_crosswalk(P, c, year, min_workers=25, min_score=90):
    """Name match VRscores employers to Compustat gvkeys, with size and timing evidence."""
    comp = pd.read_csv(CONFIG["COMPUSTAT"], usecols=["gvkey", "fyear", "conm", "state", "emp"], low_memory=False)
    comp = comp[comp.fyear == year].dropna(subset=["conm"]).drop_duplicates("gvkey")
    comp["gvkey"] = comp.gvkey.astype(str).str.zfill(6)
    comp["name_norm"] = comp.conm.map(norm_name)
    emp = q(c, f"""SELECT unit AS vrid, company_name, workers, tp FROM vr_employer
                   WHERE year = {year} AND workers >= {min_workers}""")
    emp["name_norm"] = emp.company_name.map(norm_name)
    emp = emp[emp.name_norm != ""]
    exact = comp.merge(emp, on="name_norm", how="inner").assign(method="exact", score=100.0)
    rows = [exact]
    if HAVE_RF:
        pool = emp.name_norm.to_numpy()
        blocks = {}
        for i, n in enumerate(pool):
            blocks.setdefault(n.split()[0], []).append(i)
        fuzzy = []
        unmatched = comp[~comp.gvkey.isin(exact.gvkey)]
        for f in unmatched.itertuples(index=False):
            idx = blocks.get(f.name_norm.split()[0] if f.name_norm else "", [])
            if not idx:
                continue
            hits = process.extract(f.name_norm, pool[idx], scorer=fuzz.token_sort_ratio,
                                   limit=3, score_cutoff=min_score)
            for _, sc, j in hits:
                k = idx[j]
                fuzzy.append({"gvkey": f.gvkey, "conm": f.conm, "state": f.state, "emp": f.emp,
                              "name_norm": f.name_norm, "vrid": emp.vrid.iloc[k],
                              "company_name": emp.company_name.iloc[k], "workers": emp.workers.iloc[k],
                              "tp": emp.tp.iloc[k], "method": "fuzzy", "score": float(sc)})
        if fuzzy:
            rows.append(pd.DataFrame(fuzzy))
    cand = pd.concat(rows, ignore_index=True)
    emp_total = q(c, f"SELECT sum(workers) FROM vr_employer WHERE year = {year}").iloc[0, 0]
    cand["size_ratio"] = cand.workers / (pd.to_numeric(cand.emp, errors="coerce") * 1000)
    cand["size_ok"] = cand.size_ratio.between(0.002, 1.5)
    cand["tier"] = np.where((cand.method == "exact") & cand.size_ok.fillna(False), "high",
                   np.where(cand.method == "exact", "medium",
                   np.where(cand.size_ok.fillna(False) & (cand.score >= 95), "medium", "low")))
    cand["accept"] = pd.NA          # fill by hand for anything below high
    cand.to_csv(P["KEYS"] / "cw_vrid_gvkey.csv", index=False)
    high = cand[cand.tier == "high"]
    log(f"firm crosswalk {year}: {cand.gvkey.nunique():,} Compustat firms with a candidate, "
        f"{high.gvkey.nunique():,} at high confidence, covering {high.workers.sum() / emp_total:.1%} of VRscores workers")
    return cand


def main(year=2015):
    P = paths()
    c = con()
    for kind in ("employer", "metro", "industry", "occupation"):
        pq = P["CANONICAL"] / f"vr_{kind}.parquet"
        if pq.exists():
            vr_view(c, f"vr_{kind}", pq, kind)
    ref = cbsa_reference(P)
    if ref is not None and (P["CANONICAL"] / "vr_metro.parquet").exists():
        names = q(c, "SELECT DISTINCT unit AS msa FROM vr_metro").msa
        build_metro_crosswalk(P, names, ref)
    if (P["CANONICAL"] / "vr_occupation.parquet").exists():
        occ = q(c, "SELECT DISTINCT unit AS onet_code FROM vr_occupation")
        occ["valid"] = occ.onet_code.str.match(r"^\d{2}-\d{4}")
        occ["occ_code"] = occ.onet_code.str[:7].where(occ.valid)
        occ.to_csv(P["KEYS"] / "cw_occupation.csv", index=False)
        log(f"occupation crosswalk: {occ.valid.mean():.1%} valid SOC codes, dropping {(~occ.valid).sum()} non-occupations")
    if (P["CANONICAL"] / "vr_industry.parquet").exists():
        ind = q(c, "SELECT DISTINCT unit AS naics6 FROM vr_industry")
        ind["naics4"] = ind.naics6.str.zfill(6).str[:4]
        ind.to_csv(P["KEYS"] / "cw_industry.csv", index=False)
        log(f"industry crosswalk: {ind.naics4.nunique()} four-digit industries")
    if Path(CONFIG["COMPUSTAT"]).exists() and (P["CANONICAL"] / "vr_employer.parquet").exists():
        build_firm_crosswalk(P, c, year)
    return True


if __name__ == "__main__":
    main()
