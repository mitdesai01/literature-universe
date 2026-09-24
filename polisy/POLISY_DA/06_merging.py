# -*- coding: utf-8 -*-
"""06 Merging: build the analysis panels.

What it does: joins external data onto each VRscores grain and writes four panels.
  panels/metro_year.parquet       VRscores metro + CBSA + ACS + vote + regime
  panels/occupation_year.parquet  VRscores occupation + OEWS + O*NET job zone + AIOE
  panels/industry_year.parquet    VRscores industry + OEWS 4-digit staffing
  panels/firm_year.parquet        VRscores employers matched to gvkey (high tier only)
Why: analysis modules should never touch raw files or repeat joins.
Expect: four parquet files plus output/tables/06_merge_report.csv giving, for each merge,
matched rows and the share of VRscores workers covered.
Inputs come through polisy_core.locate (county returns, O*NET, AIOE), so their download
names and folders do not matter; module 01 shows which files were used.
Diagnostics: every row of the merge report must carry a coverage share; anything below
50% gets stated next to the result it produces.
"""
import json
import zipfile
import numpy as np
import pandas as pd
from polisy_core import (paths, con, log, save, q, vr_view, locate, missing_hint, read_table, pick,
                         digits)

ACS_MAP = {"B01003_001E": "population", "B19013_001E": "median_hh_income", "B23025_004E": "employed",
           "B15003_022E": "bachelors", "B15003_001E": "pop25_denom", "B01002_001E": "median_age"}


def load_acs(P):
    frames = []
    for f in sorted(P["RAW"].glob("acs1_*.json")):
        raw = json.loads(f.read_text())
        df = pd.DataFrame(raw[1:], columns=raw[0]).rename(columns={**ACS_MAP, raw[0][-1]: "cbsa"})
        for c in ACS_MAP.values():
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["year"] = int(f.stem.split("_")[1])
        df["share_bachelors"] = df.bachelors / df.pop25_denom
        frames.append(df[["cbsa", "year", "population", "median_hh_income", "employed",
                          "median_age", "share_bachelors"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def load_votes(P):
    loc = locate("COUNTYPRES")
    cw = P["KEYS"] / "cw_cbsa_county.csv"
    if loc["path"] is None:
        log("votes: skipped. " + missing_hint("COUNTYPRES"))
        return None
    if not cw.exists():
        log("votes: skipped, keys/cw_cbsa_county.csv missing; module 04 builds it from the CBSA reference. "
            + missing_hint("CBSA_REFERENCE"))
        return None
    d = read_table(loc["path"], loc["member"], header_hint="candidatevotes",
                   columns={"year", "countyfips", "party", "candidatevotes", "mode"})
    wanted = {"year": "year", "countyfips": "county_fips", "party": "party",
              "candidatevotes": "candidatevotes", "mode": "mode"}
    d = d.rename(columns={pick(d, k): v for k, v in wanted.items() if pick(d, k) is not None})
    d["votes"] = pd.to_numeric(d.candidatevotes, errors="coerce")
    d["year"] = pd.to_numeric(d.year, errors="coerce")
    d["party"] = d.party.str.strip().str.upper()
    d = d[d.party.isin(["DEMOCRAT", "REPUBLICAN"])]
    d["county_fips"] = digits(d.county_fips).str.extract(r"(\d+)")[0].str.zfill(5)
    d = d.dropna(subset=["year", "county_fips"])
    if "mode" in d:                  # 2020 and 2024 report some counties by voting mode;
        mode = d["mode"].fillna("TOTAL").str.strip().str.upper()   # keep TOTAL where it exists
        total = mode.isin(["TOTAL", "TOTAL VOTES"])
        has_total = total.groupby([d.year, d.county_fips]).transform("any")
        d = d[total | ~has_total]
    g = (d.groupby(["year", "county_fips", "party"], as_index=False).votes.sum()
           .pivot_table(index=["year", "county_fips"], columns="party", values="votes").reset_index())
    link = pd.read_csv(cw, dtype=str)
    m = g.merge(link, on="county_fips", how="inner")
    out = m.groupby(["year", "cbsa"], as_index=False)[["DEMOCRAT", "REPUBLICAN"]].sum()
    out["rep_vote_share"] = out.REPUBLICAN / (out.DEMOCRAT + out.REPUBLICAN)
    return out


def main():
    P = paths()
    c = con()
    report = []
    for kind in ("employer", "metro", "industry", "occupation"):
        pq = P["CANONICAL"] / f"vr_{kind}.parquet"
        if pq.exists():
            vr_view(c, f"vr_{kind}", pq, kind)

    # ---- metro panel
    if (P["CANONICAL"] / "vr_metro.parquet").exists():
        metro = q(c, "SELECT unit AS msa, year, tp, dem, rep, rep_share, rep_share_raw FROM vr_metro")
        metro["year"] = metro.year.astype("int64")
        cwp = P["KEYS"] / "cw_msa_cbsa.csv"
        if cwp.exists():
            cw = pd.read_csv(cwp, dtype={"msa": str, "cbsa": str})
            m = metro.merge(cw[["msa", "cbsa", "state", "party_regime", "match_score"]], on="msa", how="left")
        else:
            log("metro panel: keys/cw_msa_cbsa.csv missing (module 04 needs a CBSA reference); "
                "written without CBSA codes, so no ACS or vote merge")
            m = metro.assign(cbsa=pd.NA, state=pd.NA, party_regime=pd.NA, match_score=0)
        acs = load_acs(P)
        if acs is not None:
            m = m.merge(acs, on=["cbsa", "year"], how="left")
            report.append({"merge": "metro x ACS", "rows": len(m),
                           "matched": m.population.notna().sum(),
                           "worker_share": m.loc[m.population.notna(), "tp"].sum() / m.tp.sum()})
        votes = load_votes(P) if m.cbsa.notna().any() else None
        if votes is not None:
            votes["year"] = votes.year.astype("int64")
            votes["vote_year"] = votes.year
            m["cbsa"], votes["cbsa"] = m.cbsa.astype(object), votes.cbsa.astype(object)   # merge_asof wants one key dtype
            m = pd.merge_asof(m.sort_values("year"), votes.sort_values("year"),
                              left_on="year", right_on="year", by="cbsa", direction="backward")
            report.append({"merge": "metro x votes", "rows": len(m),
                           "matched": m.rep_vote_share.notna().sum(),
                           "worker_share": m.loc[m.rep_vote_share.notna(), "tp"].sum() / m.tp.sum()})
        m.to_parquet(P["PANELS"] / "metro_year.parquet", index=False)

    # ---- occupation panel
    if (P["CANONICAL"] / "vr_occupation.parquet").exists():
        occ = q(c, "SELECT unit AS onet_code, year, tp, dem, rep, rep_share FROM vr_occupation")
        occ = occ[occ.onet_code.str.match(r"^\d{2}-\d{4}")].copy()
        occ["occ_code"] = occ.onet_code.str[:7]
        oe = P["CANONICAL"] / "oews_national.parquet"
        if oe.exists():
            nat = pd.read_parquet(oe)[["occ_code", "tot_emp", "a_median", "year"]]
            nat = nat[nat.year == nat.year.max()].drop(columns="year")
            occ = occ.merge(nat, on="occ_code", how="left")
            report.append({"merge": "occupation x OEWS", "rows": len(occ),
                           "matched": occ.tot_emp.notna().sum(),
                           "worker_share": occ.loc[occ.tot_emp.notna(), "tp"].sum() / occ.tp.sum()})
        jz = locate("ONET")["path"]
        if jz is not None:
            j = None
            if jz.suffix.lower() == ".zip":
                with zipfile.ZipFile(jz) as zf:
                    name = next((n for n in zf.namelist() if n.lower().endswith("job zones.txt")), None)
                    if name:
                        j = pd.read_csv(zf.open(name), sep="\t", dtype=str)
            elif (jz / "Job Zones.txt").exists():
                j = pd.read_csv(jz / "Job Zones.txt", sep="\t", dtype=str)
            if j is not None:
                j.columns = [x.strip().lower().replace("*", "").replace("-", "_").replace(" ", "_") for x in j.columns]
                code = next(x for x in j.columns if "soc_code" in x or x == "onetsoc_code")
                j["occ_code"] = j[code].str[:7]
                j["job_zone"] = pd.to_numeric(j.job_zone, errors="coerce")
                occ = occ.merge(j.groupby("occ_code", as_index=False).job_zone.mean(), on="occ_code", how="left")
        else:
            log("occupation panel: no O*NET job zones. " + missing_hint("ONET"))
        ai = locate("AIOE")
        if ai["path"] is not None:
            a = read_table(ai["path"], ai["member"], header_hint="aioe")
            a.columns = [str(x).strip().lower() for x in a.columns]
            code = next((x for x in a.columns if "soc" in x or "code" in x), None)
            val = next((x for x in a.columns if "aioe" in x), None)
            if code and val:
                a = a[[code, val]].rename(columns={code: "occ_code", val: "aioe"})
                a["aioe"] = pd.to_numeric(a.aioe, errors="coerce")
                a["occ_code"] = a.occ_code.str.strip().str[:7]
                occ = occ.merge(a.groupby("occ_code", as_index=False).aioe.mean(), on="occ_code", how="left")
        occ.to_parquet(P["PANELS"] / "occupation_year.parquet", index=False)

    # ---- industry panel
    if (P["CANONICAL"] / "vr_industry.parquet").exists() and (P["CANONICAL"] / "oews_industry.parquet").exists():
        ind = q(c, "SELECT unit AS naics6, year, tp, dem, rep, rep_share FROM vr_industry")
        ind["naics4"] = ind.naics6.str.zfill(6).str[:4]
        oi = pd.read_parquet(P["CANONICAL"] / "oews_industry.parquet")
        oi = oi[oi.year == oi.year.max()]
        oi["soc2"] = oi.occ_code.str[:2]
        tot = oi.groupby("naics4", as_index=False).tot_emp.sum().rename(columns={"tot_emp": "industry_emp"})
        inv = (oi[oi.soc2.isin({"15", "17", "19"})].groupby("naics4", as_index=False).tot_emp.sum()
               .rename(columns={"tot_emp": "inventive_emp"}))
        staff = tot.merge(inv, on="naics4", how="left")
        staff["inventive_share"] = staff.inventive_emp.fillna(0) / staff.industry_emp
        ind = ind.merge(staff, on="naics4", how="left")
        report.append({"merge": "industry x OEWS", "rows": len(ind),
                       "matched": ind.industry_emp.notna().sum(),
                       "worker_share": ind.loc[ind.industry_emp.notna(), "tp"].sum() / ind.tp.sum()})
        ind.to_parquet(P["PANELS"] / "industry_year.parquet", index=False)

    # ---- firm panel (high-tier name matches only)
    cwf = P["KEYS"] / "cw_vrid_gvkey.csv"
    if cwf.exists() and (P["CANONICAL"] / "vr_employer.parquet").exists():
        link = pd.read_csv(cwf, dtype={"gvkey": str, "vrid": str})
        use = link[(link.accept == 1) | ((link.accept.isna()) & (link.tier == "high"))][["gvkey", "vrid"]]
        c.register("link_df", use)
        firm = q(c, """SELECT l.gvkey, e.year, count(*) AS n_vrids, sum(e.workers) AS workers,
                              sum(e.dem) AS dem, sum(e.rep) AS rep
                       FROM vr_employer e JOIN link_df l ON e.unit = l.vrid GROUP BY 1, 2""")
        firm["rep_share"] = firm.rep / (firm.dem + firm.rep)
        total = q(c, "SELECT sum(workers) AS w FROM vr_employer").w.iloc[0]
        report.append({"merge": "employer x gvkey (names)", "rows": len(firm),
                       "matched": firm.gvkey.nunique(), "worker_share": firm.workers.sum() / total})
        firm.to_parquet(P["PANELS"] / "firm_year.parquet", index=False)
    if report:
        save(pd.DataFrame(report), "06_merge_report", "coverage share belongs next to every result")
    return True


if __name__ == "__main__":
    main()
