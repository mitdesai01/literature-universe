# -*- coding: utf-8 -*-
"""link: canonical tables -> analysis panels, with a diagnostic line for every join.

panel_occupation   VRscores occupations x AI exposure (codes, else O*NET titles, else fuzzy titles), DAIOE levels and
                   growth, and the other exposure measures of the DAIOE SOC 2018 panel
panel_industry     VRscores industries (NAICS-4) x AIIE x staffing (education, wages, occupation mix) x BTOS sector AI use
panel_state_year   state x year: workforce partisanship, votes, AIGE, jobs, BTOS AI use, patents, IRS migration, CSPP
                   (curated variables by year, cspp_*, and the 2012-16 state profile, env_*)
panel_county_year  county x year: IRS migration, AIGE, votes, patents, jobs, CBSA
panel_metro_year   VRscores metro x year: CBSA, AI exposure from the metro's industry mix, patents, votes
"""
from __future__ import annotations

import re
import zipfile

import numpy as np
import pandas as pd

from .core import log, read, write, diagnostic, pc, squash, find_col, to_state_fips

NAICS_SECTORS = {"11": "agriculture forestry fishing and hunting", "21": "mining quarrying and oil and gas extraction",
                 "22": "utilities", "23": "construction", "31": "manufacturing", "32": "manufacturing", "33": "manufacturing",
                 "42": "wholesale trade", "44": "retail trade", "45": "retail trade", "48": "transportation and warehousing",
                 "49": "transportation and warehousing", "51": "information", "52": "finance and insurance",
                 "53": "real estate and rental and leasing", "54": "professional scientific and technical services",
                 "55": "management of companies and enterprises", "56": "administrative and support and waste management and remediation services",
                 "61": "educational services", "62": "health care and social assistance", "71": "arts entertainment and recreation",
                 "72": "accommodation and food services", "81": "other services except public administration", "92": "public administration"}


def asof(left, right, by, right_year="vote_year"):
    """Latest right-hand value at or before each left-hand year (e.g. the last election), per `by`.
    Key columns are cast to object on both sides: merge_asof refuses mixed string dtypes."""
    left, right = left.copy(), right.copy()
    left[by], right[by] = left[by].astype(object), right[by].astype(object)
    left["year"], right[right_year] = left.year.astype("int64"), right[right_year].astype("int64")
    return pd.merge_asof(left.sort_values("year"), right.sort_values(right_year), left_on="year", right_on=right_year,
                         by=by, direction="backward")


def norm_title(s):
    s = str(s).lower().replace("&", " and ")
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\b(all other|except.*$)", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _title_link(left_titles, right, key, title_col, cutoff=92):
    """left titles -> right keys: exact normalised title first, then fuzzy (token-set ratio)."""
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        process = None
    lut = {}
    for k, t in zip(right[key], right[title_col]):
        lut.setdefault(norm_title(t), k)
    out = {}
    for t in pd.Series(left_titles).dropna().unique():
        n = norm_title(t)
        if n in lut:
            out[t] = (lut[n], "exact title")
        elif process is not None:
            hit = process.extractOne(n, list(lut), scorer=fuzz.token_set_ratio, score_cutoff=cutoff)
            if hit:
                out[t] = (lut[hit[0]], "fuzzy title")
    return out


def _onet_titles():
    loc = pc.locate("ONET")
    if loc["path"] is None:
        return None
    try:
        with zipfile.ZipFile(loc["path"]) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith("occupation data.txt"))
            d = pd.read_csv(zf.open(name), sep="\t", dtype=str)
    except Exception:
        return None
    c, t = find_col(d, [r"onetsoccode"], "", False), find_col(d, [r"title"], "", False)
    return d[[c, t]].rename(columns={c: "onet", t: "title"}) if c and t else None


# --------------------------------------------------------------------------- occupations
def panel_occupation():
    vr, ex = read("vr_occupation_year"), read("occ_exposure")
    if vr is None or ex is None:
        log("panel_occupation: needs vr_occupation_year and occ_exposure")
        return None
    last, first = vr.year.max(), vr.year.min()
    cur = vr[vr.year == last].copy()
    if first < last:
        base = vr[vr.year == first][["occ_key", "rep_share"]].rename(columns={"rep_share": "rep_share_first"})
        cur = cur.merge(base, on="occ_key", how="left")
        cur["change_pp"] = (cur.rep_share - cur.rep_share_first) * 100
    cur["soc_link"], cur["link_method"] = None, None
    if cur.get("soc") is not None and cur.soc.notna().any():
        hit = cur.soc.isin(ex.soc)
        cur.loc[hit, ["soc_link", "link_method"]] = np.c_[cur.loc[hit, "soc"], np.repeat("SOC code", hit.sum())]
        if (~hit).any():
            ot = _onet_titles()
            if ot is not None:
                ot["soc"] = ot.onet.str[:7]
                tmap = dict(zip(ot.soc, ot.title))
                cur.loc[~hit, "title"] = cur.loc[~hit, "soc"].map(tmap)
    todo = cur.soc_link.isna() & cur.title.notna()
    if todo.any():
        tl = _title_link(cur.loc[todo, "title"], ex, "soc", "title")
        cur.loc[todo, "soc_link"] = cur.loc[todo, "title"].map(lambda t: tl.get(t, (None, None))[0])
        cur.loc[todo, "link_method"] = cur.loc[todo, "title"].map(lambda t: tl.get(t, (None, None))[1])
    p = cur.merge(ex.drop(columns=["title"], errors="ignore").rename(columns={"soc": "soc_link"}), on="soc_link", how="left")
    dx = read("occ_daioe")
    if dx is not None and len(dx):
        p = p.merge(_daioe_features(dx), left_on="soc_link", right_index=True, how="left")
        diagnostic("VRscores occupations -> DAIOE", "panel_occupation", "occ_daioe (SOC 2010)", "SOC 2010 code",
                   p.loc[p.daioe.notna(), "workers"].sum(), p.workers.sum(), "matched workers")
    else:
        dyn = read("occ_exposure_dynamic")
        if dyn is not None and len(dyn):
            main = dyn.groupby("measure").soc.nunique().idxmax()
            d = dyn[dyn.measure == main].dropna(subset=["value"]).sort_values("period")
            ends = d.groupby("soc").value.agg(dyn_first="first", dyn_last="last")
            ends["dyn_change"] = ends.dyn_last - ends.dyn_first
            p = p.merge(ends, left_on="soc_link", right_index=True, how="left")
            p["dyn_measure"] = main
    m18 = read("occ_measures_soc2018")
    if m18 is not None and len(m18):
        p = _link_soc2018(p, m18)
    p["soc2"] = p.soc_link.astype(str).str[:2].where(p.soc_link.notna())
    if "salary_median" in p:
        p["log_salary"] = np.log(p.salary_median)
    ok = p.aioe.notna()
    diagnostic("VRscores occupations -> AIOE", "vr_occupation_year", "occ_exposure", "SOC code or occupation title",
               p.loc[ok, "workers"].sum(), p.workers.sum(), "matched workers",
               note="; ".join(f"{'not linked' if pd.isna(k) else k}: {v:.1%}" for k, v in (p.groupby("link_method", dropna=False).workers.sum() / p.workers.sum()).items()))
    write(p, "panel_occupation", where="PANELS")
    return p


MEASURES_2018 = ["frs21_aioe", "open24_human_E1", "open24_human_E1_E2", "open24_gpt_automation", "webb19_ai_score",
                 "webb19_software_score", "webb19_robot_score", "fo17_p_computerisation", "exp_cumul", "exp_cumul_genai",
                 "exp_cumul_lngmod", "exp_cumul_imgrec", "cognitive_abilities", "physical_abilities", "social_skills"]


def _daioe_features(dx):
    """Per SOC 2010 occupation: DAIOE level and standing in the last year, growth since 2012, and the change in
    standing since 2022 (the generative-AI years)."""
    y1 = int(dx.year.max())
    y0 = 2012 if (dx.year == 2012).any() else int(dx.year.min())
    w = dx.pivot_table(index="soc", columns="year", values=[c for c in ("daioe_allapps", "daioe_genai", "z_allapps", "z_genai") if c in dx])
    f = pd.DataFrame(index=w.index)
    f["daioe"] = w[("daioe_allapps", y1)]
    f["daioe_z"] = w[("z_allapps", y1)]
    f["daioe_growth"] = np.log(w[("daioe_allapps", y1)] / w[("daioe_allapps", y0)])
    f["daioe_rise_z"] = w[("z_allapps", y1)] - w[("z_allapps", y0)]
    if ("daioe_genai", y1) in w:
        f["daioe_genai"] = w[("daioe_genai", y1)]
        f["daioe_genai_z"] = w[("z_genai", y1)]
        if ("z_genai", 2022) in w and y1 > 2022:
            f["daioe_genai_rise_z"] = w[("z_genai", y1)] - w[("z_genai", 2022)]
    return f


def _link_soc2018(p, m18):
    """VRscores occupations -> SOC 2018 (exact title, then the same SOC code, then fuzzy title), for the other
    exposure measures in the DAIOE SOC 2018 panel."""
    tl = _title_link(p.title.dropna().unique(), m18, "soc2018", "title")
    exact = {t: k for t, (k, how) in tl.items() if how == "exact title"}
    fuzzy = {t: k for t, (k, how) in tl.items() if how == "fuzzy title"}
    codes = set(m18.soc2018)
    p["soc2018"] = p.title.map(exact)
    p["soc2018_link"] = np.where(p.soc2018.notna(), "exact title", None)
    same = p.soc2018.isna() & p.soc_link.isin(codes)
    p.loc[same, ["soc2018", "soc2018_link"]] = np.c_[p.loc[same, "soc_link"], np.repeat("same SOC code", same.sum())]
    fz = p.soc2018.isna() & p.title.isin(list(fuzzy))
    p.loc[fz, "soc2018"] = p.loc[fz, "title"].map(fuzzy)
    p.loc[fz, "soc2018_link"] = "fuzzy title"
    keep = ["soc2018"] + [c for c in MEASURES_2018 if c in m18]
    p = p.merge(m18[keep].drop_duplicates("soc2018"), on="soc2018", how="left")
    ok = p[[c for c in MEASURES_2018 if c in p]].notna().any(axis=1)
    diagnostic("VRscores occupations -> SOC 2018 exposure measures", "panel_occupation", "occ_measures_soc2018 (DAIOE)",
               "SOC 2018 title or code", p.loc[ok, "workers"].sum(), p.workers.sum(), "matched workers",
               note="; ".join(f"{'not linked' if pd.isna(k) else k}: {v:.1%}" for k, v in
                              (p.groupby("soc2018_link", dropna=False).workers.sum() / p.workers.sum()).items()))
    return p


# --------------------------------------------------------------------------- industries
def _naics6_from_titles(titles):
    try:
        import naics
    except ImportError:
        log("panel_industry: pip install naics to link industry titles to NAICS codes")
        return {}
    codes = pd.DataFrame([(k, v) for k, v in naics.NAICS_CODES.items() if len(k) == 6], columns=["naics6", "title"])
    codes["key"] = codes.title.map(norm_title)
    uniq = codes[~codes.key.duplicated(keep=False)]
    lut = dict(zip(uniq.key, uniq.naics6))
    return {t: lut.get(norm_title(t)) for t in pd.Series(titles).dropna().unique()}


def panel_industry():
    vr, ex = read("vr_industry_year"), read("ind_exposure")
    if vr is None or ex is None:
        log("panel_industry: needs vr_industry_year and ind_exposure")
        return None
    if "naics6" not in vr or vr.naics6.isna().all():
        m = _naics6_from_titles(vr.title)
        vr["naics6"] = vr.title.map(m)
        diagnostic("VRscores industry titles -> NAICS-6", "vr_industry_year", "NAICS 2017 titles", "industry title",
                   vr.loc[vr.naics6.notna(), "workers"].sum(), vr.workers.sum(), "matched workers")
    vr["naics4"] = vr.naics6.str[:4]
    last, first = vr.year.max(), vr.year.min()
    agg = lambda g: pd.Series({"workers": g.workers.sum(), "rep_share": np.average(g.rep_share, weights=g.workers),  # noqa: E731
                               "n_detail": len(g), "sector": g.sector.iloc[0] if "sector" in g else None})
    cur = vr[(vr.year == last) & vr.naics4.notna()].groupby("naics4").apply(agg, include_groups=False).reset_index()
    if "change_pp" in vr and vr.change_pp.notna().any():
        ch = vr[(vr.year == last) & vr.change_pp.notna() & vr.naics4.notna()].groupby("naics4").apply(
            lambda g: np.average(g.change_pp, weights=g.workers), include_groups=False).rename("change_pp")
        cur = cur.merge(ch, on="naics4", how="left")
    elif first < last:
        b = vr[(vr.year == first) & vr.naics4.notna()].groupby("naics4").apply(
            lambda g: np.average(g.rep_share, weights=g.workers), include_groups=False).rename("rep_first")
        cur = cur.merge(b, on="naics4", how="left")
        cur["change_pp"] = (cur.rep_share - cur.rep_first) * 100
    p = cur.merge(ex, on="naics4", how="left")
    p["title"] = p.title.astype(str).str.replace(r"\s*\([^)]*only\)", "", regex=True).where(p.title.notna())
    diagnostic("VRscores industries (NAICS-4) -> AIIE", "vr_industry_year", "ind_exposure", "naics4",
               p.loc[p.aiie.notna(), "workers"].sum(), p.workers.sum(), "matched workers")
    occ, staff = read("panel_occupation", "PANELS"), read("oes_staffing")
    if occ is not None and staff is not None:
        r = occ.dropna(subset=["soc_link"]).groupby("soc_link").rep_share.mean()
        s = staff.assign(rep_occ=staff.soc.map(r))
        cov = s.groupby("naics4").apply(lambda g: g.loc[g.rep_occ.notna(), "emp"].sum() / g.emp.sum(), include_groups=False).rename("occ_coverage")
        pred = s.dropna(subset=["rep_occ"]).groupby("naics4").apply(lambda g: np.average(g.rep_occ, weights=g.emp), include_groups=False).rename("rep_pred")
        p = p.merge(pred, on="naics4", how="left").merge(cov, on="naics4", how="left")
        p["culture_gap"] = (p.rep_share - p.rep_pred).where(p.occ_coverage >= 0.7)
    bt = read("btos_ai")
    if bt is not None and (bt.level == "sector").any():
        s = bt[(bt.level == "sector") & (bt.measure == "ai_use_now")]
        s = s[s.year == s.year.max()].groupby("geo").rate.mean()
        code = {}
        for g in s.index:
            m = re.match(r"^\s*(\d{2})(?:\s*-\s*(\d{2}))?", str(g))
            if m:
                lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
                code.update({str(k): s[g] for k in range(lo, hi + 1)})
            else:
                for k, name in NAICS_SECTORS.items():
                    if squash(name) in squash(g) or squash(g) in squash(name):
                        code.setdefault(k, s[g])
        p["btos_ai_use_sector"] = p.naics4.str[:2].map(code)
        diagnostic("industries -> BTOS sector AI use", "panel_industry", "btos_ai (sector)", "NAICS 2-digit",
                   p.loc[p.btos_ai_use_sector.notna(), "workers"].sum(), p.workers.sum(), "matched workers")
    write(p, "panel_industry", where="PANELS")
    return p


# --------------------------------------------------------------------------- states
def _flows_by(flows, key):
    f = flows.copy()
    mig = f[f.origin != f.dest]
    stay = f[f.origin == f.dest].groupby(["dest", "year"]).returns.sum().rename("stayers")
    inn = mig.groupby(["dest", "year"]).agg(in_returns=("returns", "sum"), in_agi=("agi", "sum"))
    out = mig.groupby(["origin", "year"]).agg(out_returns=("returns", "sum"), out_agi=("agi", "sum"))
    out.index.names = ["dest", "year"]
    t = pd.concat([stay, inn, out], axis=1).reset_index().rename(columns={"dest": key})
    t["net_returns"] = t.in_returns.fillna(0) - t.out_returns.fillna(0)
    t["net_agi"] = t.in_agi.fillna(0) - t.out_agi.fillna(0)
    base = t.stayers + t.out_returns.fillna(0)
    t["net_migration_rate"] = t.net_returns / base
    t["gross_migration_rate"] = (t.in_returns.fillna(0) + t.out_returns.fillna(0)) / base
    return t


def _cspp_pick(cat, per_theme=3):
    picks = {}
    for theme in ("ideology", "party control", "policy", "innovation", "economy", "education"):
        if theme not in cat:
            continue
        c = cat[cat[theme] & (cat.coverage_2010plus > 0.5)].sort_values(["coverage_2010plus", "last_year"], ascending=False)
        picks[theme] = list(c.variable.head(per_theme))
    return picks


def panel_state_year(years=range(2000, 2026)):
    gs = read("geo_state")
    if gs is None:
        log("panel_state_year: needs geo_state (run the geography adapter)")
        return None
    p = gs.merge(pd.DataFrame({"year": list(years)}), how="cross")
    vr = read("vr_state_year")
    if vr is not None:
        p = p.merge(vr[["state_fips", "year", "rep_share", "workers", "party_regime"]].rename(
            columns={"rep_share": "vr_rep_share", "workers": "vr_workers"}), on=["state_fips", "year"], how="left")
        p["party_regime"] = p.groupby("state_fips").party_regime.transform(lambda s: s.ffill().bfill())
    v = read("votes_county_year")
    if v is not None:
        sv = v.groupby(["state_fips", "year"], as_index=False)[["dem_votes", "rep_votes"]].sum()
        sv["rep_vote_share"] = sv.rep_votes / (sv.dem_votes + sv.rep_votes)
        p = asof(p, sv[["state_fips", "year", "rep_vote_share"]].rename(columns={"year": "vote_year"}), "state_fips")
    se = read("state_exposure")
    if se is not None:
        p = p.merge(se[["state_fips", "aige"]], on="state_fips", how="left")
    qt = read("qcew_area_total")
    if qt is not None:
        j = qt[qt.area_type == "state"].assign(state_fips=lambda d: d.area.str[:2]).groupby("state_fips").jobs_2019.sum()
        p["jobs_2019"] = p.state_fips.map(j)
    bt = read("btos_ai")
    if bt is not None and (bt.level == "state").any():
        b = bt[(bt.level == "state")].copy()
        b["state_fips"] = to_state_fips(b.geo).values
        w = b.groupby(["state_fips", "year", "measure"]).rate.mean().unstack("measure").add_prefix("btos_").reset_index()
        p = p.merge(w, on=["state_fips", "year"], how="left")
    ps = read("patents_state_year")
    if ps is not None:
        p = p.merge(ps, on=["state_fips", "year"], how="left")
        if "jobs_2019" in p:
            for c in ("patents", "ai_patents", "ai_broad_patents"):
                p[c + "_per_10k_jobs"] = p[c] / p.jobs_2019 * 1e4
        p["ai_share"] = p.ai_broad_patents / p.patents
    ist = read("irs_state_year")
    if ist is not None:
        keep = ["net_migration_rate", "gross_migration_rate", "net_returns", "net_agi", "net_agi_rate", "in_agi_per_return",
                "out_agi_per_return", "mover_income_gap", "base_returns"]
        p = p.merge(ist[["state_fips", "year"] + [c for c in keep if c in ist]], on=["state_fips", "year"], how="left")
    else:
        fl = read("irs_flows_state")
        if fl is not None:
            p = p.merge(_flows_by(fl, "state_fips"), on=["state_fips", "year"], how="left")
    mv = read("inventor_moves")
    if mv is not None:
        a = mv[mv.ai]
        io = pd.concat([a.groupby(["dest", "year"]).moves.sum().rename("ai_inventor_in"),
                        a.groupby(["origin", "year"]).moves.sum().rename("ai_inventor_out").rename_axis(["dest", "year"])], axis=1).reset_index()
        io["ai_inventor_net"] = io.ai_inventor_in.fillna(0) - io.ai_inventor_out.fillna(0)
        p = p.merge(io.rename(columns={"dest": "state_fips"}), on=["state_fips", "year"], how="left")
    cs, cat = read("cspp_state_year"), read("cspp_catalog")
    if cs is not None and cat is not None:
        cur = list(cat.variable[cat.curated]) if "curated" in cat else []
        picks = {"curated": cur} if cur else _cspp_pick(cat)
        keep = sorted({v for vs in picks.values() for v in vs if v in cs})
        if keep:
            p = p.merge(cs[["state_fips", "year"] + keep].rename(columns={v: "cspp_" + v for v in keep}), on=["state_fips", "year"], how="left")
            log("panel_state_year: CSPP variables used: " + "; ".join(f"{t}: {', '.join(v)}" for t, v in picks.items() if v))
    prof = read("cspp_state_profile")
    if prof is not None:
        p = p.merge(prof.rename(columns={c: "env_" + c for c in prof.columns if c != "state_fips"}), on="state_fips", how="left")
    write(p.sort_values(["state_fips", "year"]), "panel_state_year", where="PANELS")
    return p


# --------------------------------------------------------------------------- counties
def panel_county_year(years=range(2000, 2026)):
    gc = read("geo_county")
    parts = [x for x in (read("votes_county_year"), read("patents_county_year"), read("irs_county_year"), read("county_exposure")) if x is not None]
    if gc is None and not parts:
        log("panel_county_year: needs geo_county, votes, patents, migration or AIGE")
        return None
    base = gc[["county_fips", "state_fips", "county_name", "cbsa", "cbsa_title"]] if gc is not None else \
        pd.DataFrame({"county_fips": pd.concat([x.county_fips for x in parts]).unique()}).assign(
            state_fips=lambda d: d.county_fips.str[:2], county_name=None, cbsa=None, cbsa_title=None)
    p = base.merge(pd.DataFrame({"year": list(years)}), how="cross")
    v = read("votes_county_year")
    if v is not None:
        p = asof(p, v[["county_fips", "year", "rep_vote_share"]].rename(columns={"year": "vote_year"}), "county_fips")
    pt = read("patents_county_year")
    if pt is not None:
        p = p.merge(pt, on=["county_fips", "year"], how="left")
    ce = read("county_exposure")
    if ce is not None:
        p = p.merge(ce[["county_fips", "aige"]], on="county_fips", how="left")
    qt = read("qcew_area_total")
    if qt is not None:
        p["jobs_2019"] = p.county_fips.map(qt[qt.area_type == "county"].set_index("area").jobs_2019)
    icy = read("irs_county_year")
    if icy is not None:
        keep = ["net_migration_rate", "gross_migration_rate", "net_returns", "net_agi", "net_agi_rate", "in_agi_per_return",
                "out_agi_per_return", "stay_agi_per_return", "mover_income_gap", "base_returns"]
        p = p.merge(icy[["county_fips", "year"] + [c for c in keep if c in icy]], on=["county_fips", "year"], how="left")
    else:
        fl = read("irs_flows_county")
        if fl is not None:
            p = p.merge(_flows_by(fl, "county_fips"), on=["county_fips", "year"], how="left")
    names = read("irs_county_names")
    if names is not None and "county_name" in p:
        p["county_name"] = p.county_name.fillna(p.county_fips.map(dict(zip(names.county_fips, names.county_name))))
    write(p.sort_values(["county_fips", "year"]), "panel_county_year", where="PANELS")
    return p


# --------------------------------------------------------------------------- metros
def _vr_msa_to_cbsa(msas, cbsa_titles):
    """VRscores metro names (older OMB titles) -> CBSA codes by principal city and state."""
    ref = cbsa_titles.dropna().drop_duplicates("cbsa")
    parsed = {c: pc.parse_msa(t) for c, t in zip(ref.cbsa, ref.cbsa_title)}
    out = {}
    for m in msas:
        cities, states = pc.parse_msa(m)
        best, score = None, 0
        for c, (cc, ss) in parsed.items():
            if not states or states[0] not in ss:
                continue
            s = 3 if cities and cc and cities[0] == cc[0] else (2 if set(cities) & set(cc) else 0)
            if s > score:
                best, score = c, s
        out[m] = best
    return out


def panel_metro_year():
    vr = read("vr_metro_year")
    if vr is None:
        log("panel_metro_year: needs vr_metro_year")
        return None
    gc = read("geo_county")
    if ("cbsa" not in vr or vr.cbsa.isna().all()) and gc is not None and gc.cbsa.notna().any():
        m = _vr_msa_to_cbsa(vr.msa.unique(), gc[["cbsa", "cbsa_title"]])
        vr["cbsa"] = vr.msa.map(m)
    if "cbsa" in vr:
        diagnostic("VRscores metros -> CBSA", "vr_metro_year", "Census CBSA titles", "principal city + state",
                   vr[vr.year == vr.year.max()].cbsa.notna().sum(), vr[vr.year == vr.year.max()].msa.nunique(), "metros")
    p = vr.copy()
    qi, ie = read("qcew_area_industry"), read("ind_exposure")
    if qi is not None and ie is not None and "cbsa" in p:
        q = qi[qi.area_type == "msa"].merge(ie[["naics4", "aiie"] + [c for c in ("aiie_lm",) if c in ie]], on="naics4", how="inner")
        ex = q.groupby("area").apply(lambda g: pd.Series({
            "metro_aiie": np.average(g.aiie, weights=g.emp) if g.emp.sum() > 0 else np.nan,
            "metro_aiie_lm": np.average(g.aiie_lm.fillna(g.aiie), weights=g.emp) if "aiie_lm" in g and g.emp.sum() > 0 else np.nan,
            "metro_jobs_private_2019": g.emp.sum()}), include_groups=False)
        p = p.merge(ex, left_on="cbsa", right_index=True, how="left")
    pt = read("patents_county_year")
    if pt is not None and gc is not None and "cbsa" in p:
        mp = pt.merge(gc[["county_fips", "cbsa"]], on="county_fips").groupby(["cbsa", "year"], as_index=False)[["patents", "ai_patents", "ai_broad_patents"]].sum()
        p = p.merge(mp, on=["cbsa", "year"], how="left")
    v = read("votes_county_year")
    if v is not None and gc is not None and "cbsa" in p:
        mv = v.merge(gc[["county_fips", "cbsa"]], on="county_fips").groupby(["cbsa", "year"], as_index=False)[["dem_votes", "rep_votes"]].sum()
        mv["rep_vote_share"] = mv.rep_votes / (mv.dem_votes + mv.rep_votes)
        p = asof(p, mv[["cbsa", "year", "rep_vote_share"]].rename(columns={"year": "vote_year"}), "cbsa")
    p["state"] = p.msa.map(lambda n: (pc.parse_msa(n)[1] or [None])[0])
    p["party_regime"] = p.state.map(pc.party_regime)
    write(p, "panel_metro_year", where="PANELS")
    return p


def build_panels():
    out = {}
    for f in (panel_occupation, panel_industry, panel_state_year, panel_county_year, panel_metro_year):
        try:
            out[f.__name__] = f()
        except Exception as e:  # a failed panel must not stop the others; the log says why
            log(f"{f.__name__} failed: {type(e).__name__}: {e}")
            out[f.__name__] = None
    return out
