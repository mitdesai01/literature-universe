# -*- coding: utf-8 -*-
"""AI exposure and adoption sources -> canonical tables.

aioe          occ_exposure, ind_exposure, county_exposure, state_exposure, oes_staffing,
              qcew_area_industry, qcew_area_total
dynamic_aioe  occ_exposure_dynamic (soc x period x measure), structure discovered at run time
btos          btos_long (every question) and btos_ai (AI use now / expected, by geography)
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import log, write, find_col, squash, pc, diagnostic, to_state_fips, wcorr
from ..sources import discover

COGNITIVE = {"categoryflexibility", "deductivereasoning", "flexibilityofclosure", "fluencyofideas", "inductivereasoning",
             "informationordering", "mathematicalreasoning", "memorization", "numberfacility", "oralcomprehension",
             "oralexpression", "originality", "perceptualspeed", "problemsensitivity", "selectiveattention",
             "spatialorientation", "speedofclosure", "timesharing", "visualization", "writtencomprehension", "writtenexpression"}
SENSORY = {"auditoryattention", "depthperception", "farvision", "glaresensitivity", "hearingsensitivity", "nearvision",
           "nightvision", "peripheralvision", "soundlocalization", "speechclarity", "speechrecognition", "visualcolordiscrimination"}


def expand_naics(code, title):
    """BLS and AIIE publish some industries as combinations, e.g. 3250A1 'Chemical Manufacturing
    (3251, 3252, 3253, and 3259 only)'. Return the member 4-digit codes, else the code's first four digits."""
    m = re.search(r"\(([^)]*?)\s+only\)", str(title))
    members = re.findall(r"\b(\d{4})\b", m.group(1)) if m else []
    return members or [str(code)[:4]]


def _one(source, role):
    hit = discover(source, role)
    return hit[0] if hit else (None, None)


def _read_any(path, member=None):
    if path is None:
        return None
    ext = Path(member or path.name).suffix.lower()
    if ext == ".dta" and member:
        with zipfile.ZipFile(path) as zf:
            return pd.read_stata(io.BytesIO(zf.read(member)))
    if ext == ".dta":
        return pd.read_stata(path)
    return pc.read_table(path, member)


def _slug(s):
    return re.sub(r"[^a-z]+", "_", str(s).lower()).strip("_")


# --------------------------------------------------------------------------- AIOE family
def application_exposure(abilities, matrix, valid_soc=None):
    """Rebuild AIOE (Felten, Raj & Seamans 2021) for all ten AI applications together and for
    each one alone, plus the cognitive and sensory ability shares.

    Occupation k: sum_j (IM_jk/5 * LV_jk/7) * relatedness_j / sum_j (IM_jk/5 * LV_jk/7), averaged
    over O*NET-SOC codes sharing a 6-digit SOC, then standardised. With all ten applications this
    reproduces the published AIOE (r > 0.9999 on the 2021 release).
    """
    ab = abilities.copy()
    c_occ = find_col(ab, [r"onetsoccode", r"onetsoc.*", r"soccode"], "O*NET-SOC code", True, "abilities")
    c_el = find_col(ab, [r"elementname"], "ability name", True, "abilities")
    c_sc = find_col(ab, [r"scaleid"], "scale (IM/LV)", True, "abilities")
    c_v = find_col(ab, [r"datavalue", r"value"], "value", True, "abilities")
    ab["el"] = ab[c_el].map(squash)
    ab["imp"] = np.where(ab[c_sc].astype(str).str.upper() == "IM", pd.to_numeric(ab[c_v], errors="coerce") / 5, np.nan)
    ab["lvl"] = np.where(ab[c_sc].astype(str).str.upper() == "LV", pd.to_numeric(ab[c_v], errors="coerce") / 7, np.nan)
    g = ab.groupby([c_occ, "el"], as_index=False).agg(imp=("imp", "max"), lvl=("lvl", "max"))
    g["w"] = g.imp * g.lvl
    m = matrix.copy()
    c_app = find_col(m, [r"applications?", r"application(name)?"], "application", True, "mturk matrix")
    m = m.set_index(m[c_app].astype(str)).drop(columns=[c for c in m.columns if squash(c) in ("applications", "application", "applicationid")])
    m.columns = [squash(c) for c in m.columns]
    W = g.pivot_table(index=c_occ, columns="el", values="w", aggfunc="first")
    common = [c for c in m.columns if c in W.columns]
    missing = sorted(set(m.columns) - set(common))
    if missing:
        log(f"  abilities missing from O*NET file: {missing}")
    W = W[common]
    base = W.sum(axis=1)
    per_app = pd.DataFrame({a: (W * m.loc[a, common]).sum(axis=1) for a in m.index})
    parts = pd.concat([per_app, per_app.sum(axis=1).rename("ALL"), base.rename("_base"),
                       W[[c for c in common if c in COGNITIVE]].sum(axis=1).rename("_cog"),
                       W[[c for c in common if c in SENSORY]].sum(axis=1).rename("_sen")], axis=1)
    soc = parts.index.astype(str).str[:7]
    agg = parts.groupby(soc).mean()
    if valid_soc is not None:
        agg = agg[agg.index.isin(set(valid_soc))]
    out = agg.drop(columns=["_base", "_cog", "_sen"]).div(agg["_base"], axis=0)
    out = (out - out.mean()) / out.std(ddof=1)
    out.columns = ["exp_" + _slug(c) if c != "ALL" else "aioe_rebuilt" for c in out.columns]
    out["cognitive_share"] = agg["_cog"] / agg["_base"]
    out["sensory_share"] = agg["_sen"] / agg["_base"]
    out.index.name = "soc"
    return out.reset_index()


def adapt_aioe():
    ap, _ = _one("aioe", "appendix")
    if ap is None:
        loc = pc.locate("AIOE")
        ap = loc["path"]
    if ap is None:
        log("aioe: no AIOE_DataAppendix.xlsx found; run fetch or add it to a search folder")
        return False
    xl = pd.ExcelFile(ap)
    sheet = {squash(s): s for s in xl.sheet_names}
    A = pc.read_table(ap, header_hint="soccode")
    occ = A.rename(columns={find_col(A, [r"soccode", r"soc"], "SOC", True, "Appendix A"): "soc",
                            find_col(A, [r"occupationtitle", r"title"], "title", True, "Appendix A"): "title",
                            find_col(A, [r"aioe"], "AIOE", True, "Appendix A"): "aioe"})[["soc", "title", "aioe"]]
    occ["aioe"] = pd.to_numeric(occ.aioe, errors="coerce")
    occ = occ[occ.soc.astype(str).str.fullmatch(r"\d{2}-\d{4}")]
    for role, col in (("lm", "aioe_lm"), ("ig", "aioe_ig")):
        p, _ = _one("aioe", role)
        if p is not None:
            x = pd.ExcelFile(p)
            d = x.parse(x.sheet_names[0])
            d = d.rename(columns={find_col(d, [r"soccode"], "SOC", True, role): "soc",
                                  find_col(d, [r".*aioe"], "exposure", True, role): col})
            occ = occ.merge(d[["soc", col]].assign(**{col: lambda z: pd.to_numeric(z[col], errors="coerce")}), on="soc", how="left")
    abil, _ = _one("aioe", "abilities")
    mt, _ = _one("aioe", "mturk")
    if abil is not None and mt is not None:
        apps = application_exposure(_read_any(abil), _read_any(mt), valid_soc=occ.soc)
        occ = occ.merge(apps, on="soc", how="left")
        r = occ[["aioe", "aioe_rebuilt"]].corr().iloc[0, 1]
        log(f"aioe: rebuilt AIOE vs published r = {r:.5f} ({occ.aioe_rebuilt.notna().sum()} occupations); "
            f"{sum(c.startswith('exp_') for c in occ.columns)} application-specific exposures")
        diagnostic("AIOE rebuild", "published AIOE", "rebuilt from O*NET abilities", "soc",
                   occ.aioe_rebuilt.notna().sum(), len(occ), "occupations", note=f"r = {r:.5f}")
    for role, cols in (("salary", {"median": [r"mediansalary\d*", r"median.*"], "mean": [r"meansalary\d*"]}),
                       ("education", {"req_education": [r"occreqeducation", r".*education.*"]}),
                       ("creative", {"creative_weight": [r"avgcreativeweight", r".*creative.*"]}),
                       ("representation", {"female": [r"meanfemale"], "white": [r"meanwhite"], "black": [r"meanblack"],
                                           "asian": [r"meanasian"], "hispanic": [r"meanhispanic"]})):
        p, mem = _one("aioe", role)
        if p is None:
            continue
        d = _read_any(p, mem)
        k = find_col(d, [r"occcode", r"soccode", r"soc"], "SOC", True, role)
        keep = {}
        for new, pats in cols.items():
            c = find_col(d, pats, new, False, role)
            if c:
                keep[c] = ("salary_" + new) if role == "salary" else new
        d = d[[k] + list(keep)].rename(columns={k: "soc", **keep})
        for c in keep.values():
            d[c] = pd.to_numeric(d[c], errors="coerce")
        occ = occ.merge(d.drop_duplicates("soc"), on="soc", how="left")
    occ["soc2"] = occ.soc.str[:2]
    write(occ, "occ_exposure", note=f"{occ.shape[1]} columns")

    B = pc.read_table(ap, header_hint="aiie")
    ind = B.rename(columns={find_col(B, [r"naics"], "NAICS", True, "Appendix B"): "naics4",
                            find_col(B, [r"industrytitle", r"title"], "title", True, "Appendix B"): "title",
                            find_col(B, [r"aiie"], "AIIE", True, "Appendix B"): "aiie"})[["naics4", "title", "aiie"]]
    ind["naics4"] = pc.digits(ind.naics4).str.zfill(4)
    ind["naics4"] = [expand_naics(c, t) for c, t in zip(ind.naics4, ind.title)]
    ind = ind.explode("naics4")
    ind["aiie"] = pd.to_numeric(ind.aiie, errors="coerce")
    ind = ind.groupby("naics4", as_index=False).agg(title=("title", "first"), aiie=("aiie", "mean"))
    for role, col in (("lm", "aiie_lm"), ("ig", "aiie_ig")):
        p, _ = _one("aioe", role)
        if p is not None:
            x = pd.ExcelFile(p)
            d = x.parse(x.sheet_names[1] if len(x.sheet_names) > 1 else x.sheet_names[0])
            if find_col(d, [r"naics"]) is None:
                continue
            tcol = find_col(d, [r"naicsdescription", r"title", r"description"], "", False)
            d = d.rename(columns={find_col(d, [r"naics"], "NAICS", True, role): "naics4", find_col(d, [r".*aiie"], "AIIE", True, role): col})
            d["naics4"] = [expand_naics(pc.digits(pd.Series([c])).iloc[0][:4] if str(c)[:4].isdigit() else c, t)
                           for c, t in zip(d.naics4, d[tcol] if tcol else [""] * len(d))]
            d = d.explode("naics4")
            d[col] = pd.to_numeric(d[col], errors="coerce")
            ind = ind.merge(d.groupby("naics4", as_index=False)[col].mean(), on="naics4", how="left")
    staff = adapt_oes_staffing(occ)
    if staff is not None:
        ind = ind.merge(staff, on="naics4", how="left")
    write(ind, "ind_exposure")

    C = pc.read_table(ap, header_hint="aige")
    geo = C.rename(columns={find_col(C, [r"fipscode", r"fips"], "FIPS", True, "Appendix C"): "fips",
                            find_col(C, [r"geographicarea", r"area", r"name"], "name", True, "Appendix C"): "name",
                            find_col(C, [r"aige"], "AIGE", True, "Appendix C"): "aige"})[["fips", "name", "aige"]]
    geo["fips"] = pc.digits(geo.fips).str.zfill(5)
    geo["aige"] = pd.to_numeric(geo.aige, errors="coerce")
    write(geo[~geo.fips.str.endswith("000")].rename(columns={"fips": "county_fips"}), "county_exposure")
    st = geo[geo.fips.str.endswith("000")].assign(state_fips=lambda d: d.fips.str[:2])[["state_fips", "name", "aige"]]
    write(st, "state_exposure")
    adapt_qcew()
    return True


def adapt_oes_staffing(occ):
    p, mem = _one("aioe", "oes_staffing")
    if p is None:
        return None
    d = _read_any(p, mem)
    c_n = find_col(d, [r"naics"], "NAICS", True, "OES staffing")
    c_o = find_col(d, [r"occcode"], "SOC", True, "OES staffing")
    c_g = find_col(d, [r"ogroup"], "occupation level", False, "OES staffing")
    c_e = find_col(d, [r"totemp"], "employment", True, "OES staffing")
    c_w = find_col(d, [r"amedian"], "median wage", False, "OES staffing")
    if c_g:
        d = d[d[c_g].astype(str).str.lower() == "detailed"]
    c_t = find_col(d, [r"naicstitle"], "industry title", False, "OES staffing")
    s = pd.DataFrame({"naics4": [expand_naics(c, t) for c, t in zip(d[c_n].astype(str), d[c_t] if c_t else [""] * len(d))],
                      "soc": d[c_o].astype(str),
                      "emp": pd.to_numeric(d[c_e].astype(str).str.replace(",", ""), errors="coerce"),
                      "a_median": pd.to_numeric(d[c_w].astype(str).str.replace(",", ""), errors="coerce") if c_w else np.nan})
    s = s.dropna(subset=["emp"]).explode("naics4")
    s = s.groupby(["naics4", "soc"], as_index=False).agg(emp=("emp", "sum"), a_median=("a_median", "mean"))
    write(s, "oes_staffing")
    o = s.merge(occ[["soc"] + [c for c in ("req_education", "aioe", "aioe_lm") if c in occ]], on="soc", how="left")
    def wavg(g, c):
        k = g[c].notna() & g.emp.gt(0)
        return np.average(g.loc[k, c], weights=g.loc[k, "emp"]) if k.any() else np.nan
    return (o.groupby("naics4").apply(lambda g: pd.Series({
        "ind_emp_oes": g.emp.sum(),
        "ind_education": wavg(g, "req_education") if "req_education" in g else np.nan,
        "ind_log_wage": np.log(wavg(g, "a_median")) if g.a_median.notna().any() else np.nan,
        "ind_aioe_from_staffing": wavg(g, "aioe")}), include_groups=False).reset_index())


def adapt_qcew():
    """QCEW 2019 employment (AIOE input) by area and 4-digit industry: counties, states, metros."""
    p, mem = _one("aioe", "qcew")
    if p is None:
        log("aioe: QCEW county x industry file not found (Input/county_naics_2019); area employment skipped")
        return
    cols = ["area_fips", "own_code", "industry_code", "agglvl_code", "annual_avg_emplvl"]
    try:
        q = pd.read_stata(p, columns=cols) if str(p).endswith(".dta") else pc.read_table(p, mem)[cols]
    except Exception as e:
        log(f"aioe: QCEW file unreadable ({e})")
        return
    q["agg"] = q.agglvl_code.astype(str)
    q["area"] = q.area_fips.astype(str)
    q["emp"] = pd.to_numeric(q.annual_avg_emplvl, errors="coerce").fillna(0)
    kind = {"7": "county", "5": "state", "4": "msa", "1": "national"}
    q["area_type"] = q["agg"].str[0].map(kind)
    q.loc[q.area_type == "msa", "area"] = q.area.str.replace("C", "", regex=False).str.zfill(4) + "0"   # C1018 -> CBSA 10180
    tot = q[q["agg"].isin(["73", "53", "43", "13"])].groupby(["area_type", "area"], as_index=False).emp.sum()
    write(tot.rename(columns={"emp": "jobs_2019"}), "qcew_area_total", note="jobs summed over supersectors and ownerships")
    ind = q[q["agg"].isin(["76", "56", "46", "16"])].assign(naics4=lambda d: d.industry_code.astype(str).str[:4])
    ind = ind.groupby(["area_type", "area", "naics4"], as_index=False).emp.sum()
    write(ind, "qcew_area_industry", note="2019 annual average employment")


# --------------------------------------------------------------------------- Dynamic AIOE (DAIOE)
DAIOE_APPS = ["stratgames", "videogames", "imgrec", "imgcompr", "imggen", "readcompr", "lngmod", "translat", "speechrec"]


def _daioe_best(files):
    """The 2024 refresh over the frozen 2010-2023 vintage; text files over Stata over Excel (faster to read)."""
    speed = {".tsv": 0, ".csv": 0, ".txt": 0, ".dta": 1, ".parquet": 1, ".xlsx": 2}
    key = lambda f: ("refresh" not in str(f[1] or f[0]).lower(), speed.get(Path(str(f[1] or f[0])).suffix.lower(), 3))  # noqa: E731
    return sorted(files, key=key)[0] if files else None


def adapt_dynamic_aioe():
    """DAIOE (v1.0.0 layout) when present: occupation x year exposure on SOC 2010, and the SOC 2018 panel that also
    carries other published exposure measures. Any other occupation x time table falls back to layout discovery."""
    done = False
    f = _daioe_best(discover("dynamic_aioe", "soc2010"))
    if f:
        p, m = f
        d = pc.read_table(p, m)
        code = find_col(d, [r"occcodesoc2010", r"soc2010", r"soccode"], "SOC 2010 code", True, "daioe")
        title = find_col(d, [r"occtitlesoc2010", r"title"], "title", False, "daioe")
        vals = [c for c in d.columns if squash(c).startswith("daioe")]
        x = pd.DataFrame({"soc": d[code].astype(str).str.strip().str[:7], "year": pd.to_numeric(d["year"], errors="coerce")})
        if title:
            x["title"] = d[title]
        for c in vals:
            x[c] = pd.to_numeric(d[c], errors="coerce")
        x = x.dropna(subset=["year"])
        x = x[x.soc.str.fullmatch(r"\d{2}-\d{4}")].copy()
        x["year"] = x.year.astype(int)
        # The index is cumulative (levels rise every year), so standing within a year is kept alongside the level.
        for c in ("daioe_allapps", "daioe_genai", "daioe_lngmod", "daioe_imgrec", "daioe_imggen", "daioe_readcompr"):
            if c in x:
                g = x.groupby("year")[c]
                x["z_" + c[6:]] = (x[c] - g.transform("mean")) / g.transform("std")
        vintage = "2024 refresh" if "refresh" in str(m or p).lower() else "frozen 2010-2023"
        write(x, "occ_daioe", note=f"DAIOE {vintage} ({Path(str(m or p)).name}): {x.soc.nunique()} SOC 2010 occupations, "
                                   f"{x.year.min()}-{x.year.max()}")
        long = x.melt(id_vars=["soc", "year"], value_vars=vals, var_name="measure", value_name="value").dropna(subset=["value"])
        long["period"] = long.year.astype(str)
        write(long, "occ_exposure_dynamic", note=f"DAIOE {vintage}, one row per occupation, year and measure")
        done = True
    f18 = discover("dynamic_aioe", "soc2018")
    if f18:
        p, m = f18[0]
        d = pc.read_table(p, m)
        code = find_col(d, [r"soc2018code"], "SOC 2018 code", True, "daioe soc2018")
        ttl = find_col(d, [r"soc2018title"], "SOC 2018 title", False, "daioe soc2018")
        y = pd.DataFrame({"soc2018": d[code].astype(str).str.strip().str[:7], "title": d[ttl] if ttl else None})
        for c in d.columns:
            if c not in (code, ttl):
                y[c] = pd.to_numeric(d[c], errors="coerce")
        y = y.dropna(subset=["year"])
        y["year"] = y.year.astype(int)
        write(y, "occ_soc2018_panel", note=f"DAIOE SOC 2018 panel: {y.soc2018.nunique()} occupations, {y.year.min()}-{y.year.max()}")
        last = y[y.year == y.year.max()].drop(columns=["year"])
        write(last, "occ_measures_soc2018", note=f"one row per SOC 2018 occupation; cumulative DAIOE as of {y.year.max()}")
        done = True
    if done:
        return True
    files = discover("dynamic_aioe", "data")
    if not files:
        log("dynamic_aioe: no files found")
        return False
    frames = []
    for p, mem in files:
        try:
            d = pc.read_table(p, mem)
        except Exception as e:
            log(f"dynamic_aioe: {Path(p).name} {mem or ''} unreadable ({e})")
            continue
        long = _dynamic_long(d, f"{Path(p).name}{'::' + mem if mem else ''}")
        if long is not None:
            frames.append(long)
    if not frames:
        log("dynamic_aioe: files found but no occupation x time structure recognised; see results/profiles")
        return False
    out = pd.concat(frames, ignore_index=True)
    write(out, "occ_exposure_dynamic", note=f"{out.measure.nunique()} measures, periods {out.period.min()}-{out.period.max()}")
    return True


def _dynamic_long(d, label):
    soc_col = None
    for c in d.columns:
        v = d[c].astype(str).str.strip().str[:7]
        if v.str.fullmatch(r"\d{2}-\d{4}").mean() > 0.8:
            soc_col = c
            break
    if soc_col is None:
        log(f"  dynamic_aioe {label}: no column of SOC codes")
        return None
    t_col = find_col(d, [r"year", r"period", r"date", r"quarter", r"month", r"time", r"wave"], "time", False, label)
    wide_years = [c for c in d.columns if re.fullmatch(r"(19|20)\d{2}([q_\-]\d{1,2})?", squash(c))]
    ids = {soc_col} | ({t_col} if t_col else set())
    num = [c for c in d.columns if c not in ids and pd.to_numeric(d[c], errors="coerce").notna().mean() > 0.8 and c not in wide_years]
    if t_col:
        long = d[[soc_col, t_col] + num].melt(id_vars=[soc_col, t_col], var_name="measure", value_name="value")
        long = long.rename(columns={soc_col: "soc", t_col: "period"})
    elif wide_years:
        long = d[[soc_col] + wide_years].melt(id_vars=[soc_col], var_name="period", value_name="value").rename(columns={soc_col: "soc"})
        long["measure"] = "exposure"
    else:
        log(f"  dynamic_aioe {label}: no time column; kept as a cross-section")
        long = d[[soc_col] + num].melt(id_vars=[soc_col], var_name="measure", value_name="value").rename(columns={soc_col: "soc"})
        long["period"] = "static"
    long["soc"] = long.soc.astype(str).str.strip().str[:7]
    long["value"] = pd.to_numeric(long.value, errors="coerce")
    long["period"] = long.period.astype(str)
    long["year"] = pd.to_numeric(long.period.str.extract(r"((?:19|20)\d{2})")[0], errors="coerce")
    long["file"] = label
    log(f"  dynamic_aioe {label}: soc='{soc_col}', time='{t_col or ('wide years' if wide_years else 'none')}', measures={sorted(long.measure.unique())[:6]}")
    return long.dropna(subset=["value"])


# --------------------------------------------------------------------------- BTOS
GEO_COLS = [("state", [r"state", r"statename", r"st"]), ("sector", [r"sector", r"naics(sector)?", r"naics2"]),
            ("subsector", [r"subsector", r"naics3"]), ("msa", [r"msa", r"cbsa", r"metro.*"]),
            ("size", [r"empsize", r"employmentsize.*", r"size.*"])]


def adapt_btos():
    files = discover("btos", "data")
    if not files:
        log("btos: no files found")
        return False
    frames = []
    for p, mem in files:
        for sheet_name, d in _btos_tables(p, mem):
            long = _btos_long(d, f"{Path(p).name}{'::' + sheet_name if sheet_name else ''}", Path(p).stem)
            if long is not None:
                frames.append(long)
    if not frames:
        log("btos: files found but no question x answer x period table recognised; see results/profiles")
        return False
    bl = pd.concat(frames, ignore_index=True)
    bl = bl[~bl.is_se] if "is_se" in bl else bl
    write(bl.drop(columns=["is_se"], errors="ignore"), "btos_long", note=f"{bl.question.nunique()} questions")
    ai = bl[bl.question.str.contains(r"artificial intelligence|\bAI\b", case=False, regex=True, na=False)].copy()
    if ai.empty:
        log("btos: no question mentions Artificial Intelligence")
        return True
    q = ai.question.str.lower()
    ai["measure"] = np.select([q.str.contains("next six months|next 6 months|will|expect|plan"),
                               q.str.contains("last two weeks|last 2 weeks|did this business use|used|use")],
                              ["ai_use_next6m", "ai_use_now"], "ai_other")
    yes = ai[ai.answer.map(squash).isin(["yes"])]
    out = yes.groupby(["level", "geo", "measure", "period"], as_index=False).agg(rate=("value", "mean"), year=("year", "first"),
                                                                                 date=("date", "first"), question=("question", "first"))
    write(out, "btos_ai", note=f"measures {sorted(out.measure.unique())}, levels {sorted(out.level.unique())}")
    return True


def _btos_tables(p, mem):
    ext = Path(mem or p.name).suffix.lower()
    if ext in (".xlsx", ".xls"):
        src = io.BytesIO(zipfile.ZipFile(p).read(mem)) if mem else p
        with pd.ExcelFile(src) as book:
            for sh in book.sheet_names:
                top = book.parse(sh, header=None, nrows=30, dtype=str).values.tolist()
                row = pc._header_row(top, "question", exact=False)
                if row is not None:
                    yield sh, book.parse(sh, header=row, dtype=str)
    elif p.suffix.lower() == ".zip" and not mem:
        for n in pc.members(p).values():
            if Path(n).suffix.lower() in (".csv", ".xlsx"):
                yield from _btos_tables(p, n)
    else:
        yield None, pc.read_table(p, mem, header_hint="question")


def _btos_long(d, label, stem):
    d = d.dropna(how="all")
    q = find_col(d, [r"question", r"questiontext"], "question", False, label)
    a = find_col(d, [r"answer", r"answertext", r"response"], "answer", False, label)
    if not q or not a:
        return None
    qid = find_col(d, [r"questionid", r"qid"], "", False)
    per_cols = [c for c in d.columns if re.fullmatch(r"(19|20)\d{2}\s?[-_/]?\s?\d{1,2}", str(c).strip()) or re.fullmatch(r"\d{6}", str(c).strip())
                or re.fullmatch(r"\d{4}-\d{2}-\d{2}.*", str(c).strip())]
    geo_level, geo_col = "national", None
    for lvl, pats in GEO_COLS:
        c = find_col([x for x in d.columns if x not in (q, a, qid)], pats, "", False)
        if c is not None and c not in per_cols:
            geo_level, geo_col = lvl, c
            break
    if geo_col is None:
        low = stem.lower()
        geo_level = next((lvl for lvl, _ in GEO_COLS if lvl in low), "national")
    ids = [c for c in (geo_col, qid, q, a) if c]
    if per_cols:
        long = d[ids + per_cols].melt(id_vars=ids, var_name="period", value_name="raw")
    else:
        pc_ = find_col(d, [r"period", r"collectionperiod", r"smpdt", r"date", r"week"], "period", False, label)
        vc = find_col(d, [r"estimate", r"percent", r"value", r"share"], "estimate", False, label)
        if not pc_ or not vc:
            log(f"  btos {label}: question/answer found but no period columns")
            return None
        long = d[ids + [pc_, vc]].rename(columns={pc_: "period", vc: "raw"})
    long = long.rename(columns={q: "question", a: "answer", **({qid: "question_id"} if qid else {}), **({geo_col: "geo"} if geo_col else {})})
    if "geo" not in long:
        long["geo"] = "US"
    if "question_id" not in long:
        long["question_id"] = pd.factorize(long.question)[0]
    long["value"] = pd.to_numeric(long.raw.astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False).str.strip(), errors="coerce")
    long["period"] = long.period.astype(str).str.strip()
    yr = long.period.str.extract(r"^((?:19|20)\d{2})")[0]
    pn = long.period.str.extract(r"^(?:19|20)\d{2}\s?[-_/]?\s?(\d{1,2})$")[0]
    long["year"] = pd.to_numeric(yr, errors="coerce")
    long["date"] = pd.to_datetime(long.year.astype("Int64").astype(str) + "-01-01", errors="coerce") + pd.to_timedelta(
        (pd.to_numeric(pn, errors="coerce").fillna(1) - 1) * 14, unit="D")
    iso = pd.to_datetime(long.period, errors="coerce", format="%Y-%m-%d")
    long["date"] = iso.fillna(long["date"])
    long["level"] = geo_level
    long["is_se"] = bool(re.search(r"standard.?error|\bse\b", label, re.I))
    long["file"] = label
    if geo_level == "state":
        long["state_fips"] = to_state_fips(long.geo)
    log(f"  btos {label}: level={geo_level}, {long.question.nunique()} questions, {len(per_cols) or 'long-format'} periods")
    return long.dropna(subset=["value"])
