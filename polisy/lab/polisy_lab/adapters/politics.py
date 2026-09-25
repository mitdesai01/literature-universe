# -*- coding: utf-8 -*-
"""Politics sources -> canonical tables.

vrscores   vr_occupation_year, vr_industry_year, vr_metro_year, vr_state_year,
           vr_employer_summary, vr_sorting; from the POLISY_DA Parquet panels when they
           exist (codes, every year), else from a VRscores HTML report (titles, what its
           figures carry). The `source` column says which.
elections  votes_county_year from the MIT county returns POLISY_DA already finds
cspp       cspp_state_year (every numeric variable) and cspp_catalog (names, labels, themes)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import log, write, find_col, squash, pc, to_state_fips, STATE_ABBR, LAB
from ..sources import discover

VR_STATE = re.compile(r"\s([A-Z]{2})(?:-[A-Z]{2})*(?:\s+(?:MSA|Metro Area|Micro Area|CSA))?\s*$")


def msa_state(name):
    m = VR_STATE.search(str(name))
    return m.group(1) if m else None


def msa_multistate(name):
    m = re.search(r"\s([A-Z]{2}(?:-[A-Z]{2})+)(?:\s+(?:MSA|Metro Area|Micro Area))?\s*$", str(name))
    return bool(m)


# --------------------------------------------------------------------------- VRscores
def adapt_vrscores():
    canon = Path(pc.CONFIG["CANONICAL"])
    if (canon / "vr_employer.parquet").exists() or (canon / "vr_occupation.parquet").exists():
        return _vr_from_panels(canon)
    rep = discover("vrscores", "report")
    for p, _ in rep:
        if _is_vr_report(p):
            return _vr_from_report(p)
    log("vrscores: neither POLISY_DA panels (run modules 01-02) nor a VRscores HTML report were found")
    return False


def _is_vr_report(p):
    try:
        with open(p, "rb") as fh:
            head = fh.read(4_000_000)
        return b"fig-4_6_sunburst" in head or b"fig-4_4_msa_bubbles" in head or (b"VRscores" in head and b"Plotly.newPlot" in head)
    except OSError:
        return False


def _vr_from_panels(canon):
    """Aggregate the POLISY_DA canonical panels (they carry NAICS / O*NET codes and all years)."""
    c = pc.con()
    out = {}
    for kind in ("employer", "metro", "industry", "occupation"):
        p = canon / f"vr_{kind}.parquet"
        if p.exists():
            pc.vr_view(c, f"vr_{kind}", p, kind)
            out[kind] = True
    if out.get("occupation"):
        d = pc.q(c, """SELECT unit, year, sum(dem) dem, sum(rep) rep, sum(tp) tp FROM vr_occupation GROUP BY 1, 2""")
        d["soc"] = d.unit.str[:7].where(d.unit.str.match(r"^\d{2}-\d{4}"))
        d = d.dropna(subset=["soc"]).groupby(["soc", "year"], as_index=False)[["dem", "rep", "tp"]].sum()
        d["rep_share"], d["workers"], d["occ_key"], d["title"], d["group"] = d.rep / d.tp, d.tp, d.soc, None, None
        write(d.assign(source="panels"), "vr_occupation_year")
    if out.get("industry"):
        d = pc.q(c, """SELECT unit, year, sum(dem) dem, sum(rep) rep, sum(tp) tp FROM vr_industry GROUP BY 1, 2""")
        d["naics6"] = pc.digits(d.unit).str.zfill(6)
        d = d.groupby(["naics6", "year"], as_index=False)[["dem", "rep", "tp"]].sum()
        d["rep_share"], d["workers"], d["ind_key"], d["title"] = d.rep / d.tp, d.tp, d.naics6, None
        write(d.assign(source="panels"), "vr_industry_year")
    if out.get("metro"):
        d = pc.q(c, """SELECT unit AS msa, year, dem, rep, tp FROM vr_metro""")
        d["rep_share"], d["workers"] = d.rep / d.tp, d.tp
        cw = Path(pc.CONFIG["KEYS"]) / "cw_msa_cbsa.csv"
        if cw.exists():
            k = pd.read_csv(cw, dtype={"msa": str, "cbsa": str})[["msa", "cbsa"]]
            d = d.merge(k, on="msa", how="left")
        _vr_state_from_metro(d)
        write(d.assign(source="panels"), "vr_metro_year")
    if out.get("employer"):
        _vr_employer_summary(c)
    _vr_sorting(c, out)
    return True


def _vr_state_from_metro(m):
    m = m.assign(state=m.msa.map(msa_state), multi_state=m.msa.map(msa_multistate))
    s = m.dropna(subset=["state"]).groupby(["state", "year"], as_index=False).agg(dem=("dem", "sum"), rep=("rep", "sum"),
                                                                                   multi_state_share=("multi_state", "mean"))
    s["rep_share"], s["workers"] = s.rep / (s.dem + s.rep), s.dem + s.rep
    s["party_regime"] = s.state.map(pc.party_regime)
    s["state_fips"] = s.state.map(STATE_ABBR)
    write(s.assign(source="metro panel"), "vr_state_year", note="metro workers assigned to the first state in the metro name")


def _vr_employer_summary(c, min_tp=200):
    years = pc.q(c, "SELECT min(year) a, max(year) b, count(DISTINCT year) n FROM vr_employer").iloc[0]
    d = pc.q(c, f"""
        WITH e AS (SELECT unit, year, any_value(company_name) name, sum(dem) dem, sum(rep) rep, sum(tp) tp
                   FROM vr_employer GROUP BY 1, 2)
        SELECT unit AS vrid, any_value(name) AS employer, count(*) AS years, avg(tp) AS workers,
               sum(rep) / NULLIF(sum(tp), 0) AS avg_rep_share,
               regr_slope(rep / NULLIF(tp, 0), year) AS slope
        FROM e GROUP BY 1 HAVING count(*) = {int(years.n)} AND avg(tp) >= {min_tp}""")
    d["change_pp"] = d.slope * (years.b - years.a) * 100
    d["class"] = np.select([d.change_pp <= -5, d.change_pp >= 5], ["Drifted Democratic (5+ points)", "Drifted Republican (5+ points)"],
                           "Stable (within 5 points)")
    write(d.drop(columns="slope").assign(source="panels"), "vr_employer_summary", note=f"employers in all {int(years.n)} years, {min_tp}+ two-party workers")


def _dissimilarity(dem, rep):
    D, R = dem.sum(), rep.sum()
    return 0.5 * np.abs(dem / D - rep / R).sum()


def _vr_sorting(c, have):
    rows = []
    for kind in ("employer", "metro", "industry", "occupation"):
        if not have.get(kind):
            continue
        d = pc.q(c, f"SELECT unit, year, dem, rep FROM vr_{kind} WHERE dem + rep > 0")
        for y, g in d.groupby("year"):
            de, re_, comb = pc.exposure_by_party(g.dem, g.rep)
            rows += [{"dimension": kind, "measure": "over_exposure", "year": int(y), "value": comb},
                     {"dimension": kind, "measure": "over_exposure_dem", "year": int(y), "value": de},
                     {"dimension": kind, "measure": "over_exposure_rep", "year": int(y), "value": re_},
                     {"dimension": kind, "measure": "dissimilarity", "year": int(y), "value": _dissimilarity(g.dem, g.rep)}]
    if rows:
        write(pd.DataFrame(rows).assign(source="panels"), "vr_sorting")


def _vr_from_report(path):
    """Tables recovered from the figures of a VRscores HTML report (2024 cross-sections, titles, trends)."""
    s = Path(path).read_text(encoding="utf-8", errors="ignore")
    dec = json.JSONDecoder()
    figs = {}

    def skip(i):
        while s[i] in " \n\r\t,":
            i += 1
        return i
    for m in re.finditer(r'Plotly\.newPlot\(\s*"(fig-[^"]+)"', s):
        fid, i = m.group(1), skip(m.end())
        try:
            data, i = dec.raw_decode(s, i)
            layout, i = dec.raw_decode(s, skip(i))
        except ValueError:
            continue
        frames = None
        fm = re.search(r'Plotly\.addFrames\(\s*[\'"]%s[\'"]\s*,' % re.escape(fid), s[i:i + 200000])
        if fm:
            frames, _ = dec.raw_decode(s, skip(i + fm.end()))
        figs[fid] = {"data": data, "frames": frames}
    log(f"vrscores: {len(figs)} figures read from {Path(path).name}")
    if "fig-4_6_sunburst" in figs:
        t = figs["fig-4_6_sunburst"]["data"][0]
        o = pd.DataFrame({"id": t["ids"], "title": t["labels"], "group": t["parents"],
                          "workers": [x[0] for x in t["customdata"]], "rep_share": [x[1] for x in t["customdata"]]})
        o = o[o.id.str.count("/") == 1].assign(year=2024, occ_key=lambda d: d.title, soc=None, source="report")
        write(o.drop(columns="id"), "vr_occupation_year", note="2024 only; titles, no SOC codes")
    if "fig-4_6_groups_time" in figs:
        rows = [{"group": tr["name"], "year": x, "rep_share": y} for tr in figs["fig-4_6_groups_time"]["data"]
                if tr.get("name") and tr.get("x") for x, y in zip(tr["x"], tr["y"])]
        write(pd.DataFrame(rows).assign(source="report"), "vr_occgroup_year")
    if "fig-4_5_treemap" in figs:
        t = figs["fig-4_5_treemap"]["data"][0]
        i = pd.DataFrame({"id": t["ids"], "title": t["labels"], "parent": t["parents"],
                          "workers": [x[0] for x in t["customdata"]], "rep_share": [x[1] for x in t["customdata"]]})
        i = i[i.id.str.count("/") == 2].assign(sector=lambda d: d.parent.str.split("/").str[-1], year=2024)
        ch = {}
        if "fig-4_5_industries_3d" in figs:
            ch = {h: z for tr in figs["fig-4_5_industries_3d"]["data"] for h, z in zip(tr.get("hovertext", []), tr["z"])}
        i["change_pp"] = i.title.map(ch)
        i["ind_key"], i["naics6"], i["source"] = i.title, None, "report"
        write(i.drop(columns=["id", "parent"]), "vr_industry_year", note="2024 only; titles, no NAICS codes; change since 2012 for 500+ worker industries")
    if "fig-4_5_sector_heatmap" in figs:
        t = figs["fig-4_5_sector_heatmap"]["data"][0]
        h = pd.DataFrame(t["z"], index=t["y"], columns=t["x"])
        h.index.name = "sector"
        write(h.reset_index().melt(id_vars="sector", var_name="year", value_name="rep_share").assign(source="report"), "vr_sector_year")
    if "fig-4_4_msa_bubbles" in figs and figs["fig-4_4_msa_bubbles"]["frames"]:
        rows = []
        for fr in figs["fig-4_4_msa_bubbles"]["frames"]:
            t = fr["data"][0]
            rows += [{"msa": n, "year": int(fr["name"]), "workers": cd[0], "rep_share": cd[1], "lat": cd[2], "lon": cd[3]}
                     for n, cd in zip(t["hovertext"], t["customdata"])]
        m = pd.DataFrame(rows)
        cw = Path(pc.CONFIG["KEYS"]) / "cw_msa_cbsa.csv"
        if cw.exists():
            m = m.merge(pd.read_csv(cw, dtype={"msa": str, "cbsa": str})[["msa", "cbsa"]], on="msa", how="left")
        write(m.assign(source="report"), "vr_metro_year")
    if "fig-4_4_state_map" in figs and figs["fig-4_4_state_map"]["frames"]:
        rows = []
        for fr in figs["fig-4_4_state_map"]["frames"]:
            t = fr["data"][0]
            rows += [{"state": st, "year": int(fr["name"]), "workers": cd[0], "rep_share": cd[1], "party_regime": cd[2]}
                     for st, cd in zip(t["locations"], t["customdata"])]
        st = pd.DataFrame(rows).assign(state_fips=lambda d: d.state.map(STATE_ABBR), source="report")
        write(st, "vr_state_year", note="metro workers assigned to the first state in the metro name")
    if "fig-5_4_trajectories_3d" in figs:
        rows = [{"employer": h, "class": tr["name"], "avg_rep_share": x, "change_pp": y, "workers": 10 ** z}
                for tr in figs["fig-5_4_trajectories_3d"]["data"] for h, x, y, z in zip(tr["hovertext"], tr["x"], tr["y"], tr["z"])]
        write(pd.DataFrame(rows).assign(source="report"), "vr_employer_summary", note="the report's sample of large stable employers")
    if "fig-5_1_sorting" in figs:
        rows = [{"dimension": {"Employers": "employer", "Metro areas": "metro", "Industries": "industry", "Occupations": "occupation"}.get(tr.get("name"), tr.get("name")),
                 "measure": "over_exposure" if tr.get("xaxis", "x") == "x" else "dissimilarity_excess",
                 "year": x, "value": (1 + y / 100) if tr.get("xaxis", "x") == "x" else y}
                for tr in figs["fig-5_1_sorting"]["data"] if tr.get("x") for x, y in zip(tr["x"], tr["y"])]
        write(pd.DataFrame(rows).assign(source="report"), "vr_sorting")
    return True


# --------------------------------------------------------------------------- elections
def adapt_elections():
    loc = pc.locate("COUNTYPRES")
    if loc["path"] is None:
        log("elections: county presidential returns not found (see POLISY_DA FILES['COUNTYPRES'])")
        return False
    log(f"elections: using {Path(loc['path']).name}{' :: ' + loc['member'] if loc.get('member') else ''} (found by POLISY_DA as COUNTYPRES)")
    d = pc.read_table(loc["path"], loc["member"], header_hint="candidatevotes",
                      columns={"year", "countyfips", "party", "candidatevotes", "mode"})
    ren = {pc.pick(d, k): v for k, v in {"year": "year", "countyfips": "county_fips", "party": "party",
                                         "candidatevotes": "votes", "mode": "mode"}.items() if pc.pick(d, k)}
    d = d.rename(columns=ren)
    d["votes"] = pd.to_numeric(d.votes, errors="coerce")
    d["year"] = pd.to_numeric(d.year, errors="coerce")
    d["party"] = d.party.str.strip().str.upper()
    d["county_fips"] = pc.digits(d.county_fips).str.extract(r"(\d+)")[0].str.zfill(5)
    d = d.dropna(subset=["year", "county_fips"])
    if "mode" in d:
        mode = d["mode"].fillna("TOTAL").str.upper().str.strip()
        tot = mode.isin(["TOTAL", "TOTAL VOTES"])
        d = d[tot | ~tot.groupby([d.year, d.county_fips]).transform("any")]
    d = d[d.party.isin(["DEMOCRAT", "REPUBLICAN"])]
    v = d.pivot_table(index=["county_fips", "year"], columns="party", values="votes", aggfunc="sum").reset_index()
    v = v.rename(columns={"DEMOCRAT": "dem_votes", "REPUBLICAN": "rep_votes"})
    v["rep_vote_share"] = v.rep_votes / (v.dem_votes + v.rep_votes)
    v["year"] = v.year.astype(int)
    v["state_fips"] = v.county_fips.str[:2]
    write(v, "votes_county_year")
    return True


# --------------------------------------------------------------------------- CSPP
THEMES = {
    "ideology": r"ideolog|liberal|conservat|citi|inst6|nominate|\bcf\b|ranney",
    "party control": r"democrat|republican|gov.?party|trifecta|control|majority|legislat|senate|house|divided",
    "policy": r"policy|regulat|tax|minimum.?wage|right.?to.?work|union|license|medicaid|expansion|spending",
    "economy": r"gdp|income|unemploy|poverty|employment|wage|manufact|business",
    "innovation": r"patent|r.?&.?d|research|innovat|science|technolog|broadband|internet|startup|venture|university",
    "education": r"educat|college|degree|school|tuition",
    "population": r"population|immigra|foreign.?born|urban|age|race|hispanic|black|white",
}


# Variables checked against known states (e.g. propgoppres for Texas = 55.5 in 2012, McCain 2008; 57.2 from 2014, Romney
# 2012) and the CSPP codebook descriptions. These are the ones the lab uses by name; everything else stays in the catalog.
CSPP_CURATED = {
    "propgoppres": "Republican share of the last presidential vote (%)",
    "propgopleg": "Republican share of state legislators (%)",
    "ranney4_control": "Democratic control of state government (Ranney index, 0-1)",
    "inst6014_nom": "State government ideology (0 conservative to 100 liberal)",
    "policyeconlib_est": "Economic policy liberalism (Caughey and Warshaw)",
    "policysociallib_est": "Social policy liberalism (Caughey and Warshaw)",
    "masseconlib_est": "Public economic liberalism (Caughey and Warshaw)",
    "masssociallib_est": "Public social liberalism (Caughey and Warshaw)",
    "grtw": "Right-to-work law (1 = yes)",
    "perc_college": "Adults with a college degree (%)",
    "x_top_corporateincometaxrate": "Top corporate income tax rate (%)",
    "hincomemed": "Median household income ($)",
    "incomepcap": "Income per capita ($)",
    "unemployment": "Unemployment rate (%)",
}
PROFILE_YEARS = (2012, 2016)   # the state environment before the outcomes (migration 2013-22, VRscores 2012-24)


def _cspp_profile(out):
    """One row per state: each curated variable averaged over PROFILE_YEARS, else its last value up to the window's end."""
    have = [c for c in CSPP_CURATED if c in out]
    if not have:
        return None
    w = out[(out.year >= PROFILE_YEARS[0]) & (out.year <= PROFILE_YEARS[1])].groupby("state_fips")[have].mean()
    last = out[out.year <= PROFILE_YEARS[1] + 1].sort_values("year").groupby("state_fips")[have].last()
    prof = w.combine_first(last).reset_index()
    return prof


def adapt_cspp():
    files = discover("cspp", "data")
    if not files:
        log("cspp: no Correlates of State Policy file found")
        return False
    best = None
    for p, mem in files:
        try:
            d = (pd.read_stata(p) if str(p).lower().endswith(".dta") else pc.read_table(p, mem))
        except Exception as e:
            log(f"cspp: {Path(p).name} unreadable ({e})")
            continue
        if best is None or d.shape[1] > best[1].shape[1]:
            best = (p, d)
    if best is None or best[1].shape[1] < 20:
        log("cspp: no wide state-year table found")
        return False
    p, d = best
    yc = find_col(d, [r"year"], "year", True, "cspp")
    sc = find_col(d, [r"statefips", r"fips", r"st", r"stateabbrev", r"state", r"statename"], "state", True, "cspp")
    d["year"] = pd.to_numeric(d[yc], errors="coerce")
    d["state_fips"] = to_state_fips(d[sc])
    d = d.dropna(subset=["year", "state_fips"])
    num = {}
    for c in d.columns:
        if c in (yc, sc, "year", "state_fips"):
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        if v.notna().sum() >= 0.2 * max(d[c].notna().sum(), 1) and v.notna().any():
            num[str(c)] = v
    out = pd.concat([d[["state_fips", "year"]].astype({"year": int}), pd.DataFrame(num, index=d.index)], axis=1)
    out = out.groupby(["state_fips", "year"], as_index=False).first()
    write(out, "cspp_state_year", note=f"{len(num)} numeric variables, {int(out.year.min())}-{int(out.year.max())}")
    labels = {}
    if str(p).lower().endswith(".dta"):
        with pd.read_stata(p, iterator=True) as r:
            labels = r.variable_labels()
    for cp, cm in discover("cspp", "codebook"):
        try:
            cb = pc.read_table(cp, cm, header_hint="variable")
            vc = find_col(cb, [r"variable", r"varname", r"name"], "variable", False, "cspp codebook")
            dc = find_col(cb, [r"description", r"variablelabel", r"label", r"definition", r"shortdescription"], "description", False, "cspp codebook")
            if vc and dc:
                labels.update(dict(zip(cb[vc].astype(str), cb[dc].astype(str))))
        except Exception as e:
            log(f"cspp: codebook {Path(cp).name} unreadable ({e})")
    recent = out[out.year >= 2010]
    cat = pd.DataFrame({"variable": list(num)})
    cat["description"] = cat.variable.map(lambda v: labels.get(v) or CSPP_CURATED.get(v, ""))
    cat["curated"] = cat.variable.isin(list(CSPP_CURATED))
    txt = (cat.variable + " " + cat.description).str.lower()
    for t, rx in THEMES.items():
        cat[t] = txt.str.contains(rx, regex=True)
    cat["coverage_2010plus"] = [float(recent[v].notna().mean()) for v in cat.variable]
    cat["last_year"] = [int(out.loc[out[v].notna(), "year"].max()) if out[v].notna().any() else None for v in cat.variable]
    write(cat, "cspp_catalog", note=f"{(cat.description != '').mean():.0%} of variables have a description")
    prof = _cspp_profile(out)
    if prof is not None:
        write(prof, "cspp_state_profile", note=f"curated variables averaged over {PROFILE_YEARS[0]}-{PROFILE_YEARS[1]}")
    return True
