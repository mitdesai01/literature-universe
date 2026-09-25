# -*- coding: utf-8 -*-
"""core: configuration, IO, column roles, statistics and the finding record for the POLISY lab.

Design rules (they extend the POLISY_DA rules in polisy_core):
  * every input is found with polisy_core's finder, by name and by content, never by a
    hard-coded path; FILES entries for the lab's sources are registered in sources.py
  * no adapter assumes a column name: roles are matched by pattern (find_col) and every
    match is logged, so a changed download shows up as a log line, not a silent error
  * adapters write canonical Parquet tables with one declared grain each (CANONICAL)
  * analyses read canonical tables and panels only, and return findings plus the tidy
    data behind every chart; the site is rendered from those outputs alone
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent.parent / "POLISY_DA", _HERE.parent, Path("/content/polisy_code")):
    if (_cand / "polisy_core.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
import polisy_core as pc  # noqa: E402

__version__ = "0.1.0 (2026-09-25)"

# --------------------------------------------------------------------------- config
LAB_ROOT = Path(os.environ.get("POLISY_LAB_ROOT", pc.ROOT / "lab"))
LAB = {
    "RAW": LAB_ROOT / "raw",              # downloads made by the lab (inputs you supply are searched for)
    "CANONICAL": LAB_ROOT / "canonical",  # one parquet per source table, one grain each (GRAINS)
    "PANELS": LAB_ROOT / "panels",        # linked panels: occupation, industry, state_year, county_year, metro_year
    "RESULTS": LAB_ROOT / "results",      # findings.json, views/*.json, tables/*.csv, profiles, diagnostics
    "SITE": LAB_ROOT / "site",            # the built POLISY site (GitHub Pages ready)
    "TMP": LAB_ROOT / "_tmp",
    "SETTINGS": {
        "ai_cpc": "broad",               # AI patent definition: "narrow" (G06N) or "broad" (see patents adapter)
        "min_workers": 500,              # smallest VRscores unit used in occupation/industry analyses
        "bootstrap": 1000,               # bootstrap draws for confidence intervals
        "seed": 20260925,
    },
}
GRAINS = {
    "occ_exposure": ("soc",),
    "ind_exposure": ("naics4",),
    "county_exposure": ("county_fips",),
    "occ_exposure_dynamic": ("soc", "period", "measure"),
    "occ_daioe": ("soc", "year"),
    "occ_soc2018_panel": ("soc2018", "year"),
    "occ_measures_soc2018": ("soc2018",),
    "irs_county_year": ("county_fips", "year"),
    "irs_state_year": ("state_fips", "year"),
    "irs_county_names": ("county_fips",),
    "cspp_state_profile": ("state_fips",),
    "cspp_catalog": ("variable",),
    "qcew_area_industry": ("area", "naics4"),
    "oes_staffing": ("naics4", "soc"),
    "geo_state": ("state_fips",),
    "geo_county": ("county_fips",),
    "vr_occupation_year": ("occ_key", "year"),
    "vr_industry_year": ("ind_key", "year"),
    "vr_metro_year": ("msa", "year"),
    "vr_state_year": ("state", "year"),
    "vr_employer_summary": ("employer",),
    "vr_sorting": ("dimension", "measure", "year"),
    "votes_county_year": ("county_fips", "year"),
    "btos_long": ("level", "geo", "question_id", "answer", "period"),
    "btos_ai": ("level", "geo", "measure", "period"),
    "cspp_state_year": ("state_fips", "year"),
    "irs_flows_state": ("origin", "dest", "year"),
    "irs_flows_county": ("origin", "dest", "year"),
    "patents": ("patent_id",),
    "patent_places": ("patent_id", "inventor_id"),
    "patents_county_year": ("county_fips", "year"),
    "patents_state_year": ("state_fips", "year"),
    "ai_cpc_edges": ("a", "b", "period"),
    "inventor_moves": ("origin", "dest", "year", "ai"),
}


def log(msg):
    pc.log(msg)


def dirs():
    for k in ("RAW", "CANONICAL", "PANELS", "RESULTS", "SITE", "TMP"):
        Path(LAB[k]).mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "views", "profiles"):
        (Path(LAB["RESULTS"]) / sub).mkdir(exist_ok=True)
    return {k: Path(LAB[k]) for k in ("RAW", "CANONICAL", "PANELS", "RESULTS", "SITE", "TMP")}


def canon(name):
    return Path(LAB["CANONICAL"]) / f"{name}.parquet"


def panel_path(name):
    return Path(LAB["PANELS"]) / f"{name}.parquet"


def write(df, name, where="CANONICAL", note=""):
    """Write one canonical table or panel; checks its declared grain when there is one."""
    p = Path(LAB[where]) / f"{name}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    grain = GRAINS.get(name)
    if grain and set(grain) <= set(df.columns):
        dup = int(df.duplicated(list(grain)).sum())
        if dup:
            log(f"WARNING {name}: {dup:,} rows repeat the grain {grain}")
    df.to_parquet(p, index=False)
    log(f"{where.lower()} {name}: {len(df):,} rows{' - ' + note if note else ''}")
    return p


def read(name, where="CANONICAL", columns=None):
    p = Path(LAB[where]) / f"{name}.parquet"
    return pd.read_parquet(p, columns=columns) if p.exists() else None


def have(*names, where="CANONICAL"):
    return all((Path(LAB[where]) / f"{n}.parquet").exists() for n in names)


# --------------------------------------------------------------------------- column roles
def squash(s):
    return pc.squash(s)


def find_col(cols, patterns, label="", required=False, source=""):
    """First column whose squashed name matches one of `patterns` (regexes, tried in order).

    Why: none of the open datasets promise stable column names across releases; every
    adapter goes through here, and the match is logged so a renamed column is visible.
    """
    cols = list(cols.columns) if hasattr(cols, "columns") else list(cols)
    sq = {c: squash(c) for c in cols}
    for pat in patterns:                       # first pass: the whole squashed name
        rx = re.compile(pat)
        for c in cols:
            if rx.fullmatch(sq[c]):
                if label:
                    log(f"  {source + ': ' if source else ''}{label} -> column '{c}'")
                return c
    for pat in patterns:                       # second pass: pattern anywhere in the name
        rx = re.compile(pat)
        for c in cols:
            if rx.search(sq[c]):
                if label:
                    log(f"  {source + ': ' if source else ''}{label} -> column '{c}' (partial match)")
                return c
    if required:
        raise KeyError(f"{source}: no column for {label or patterns}; columns are {cols[:40]}")
    return None


STATE_ABBR = {"AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09", "DE": "10", "DC": "11",
              "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18", "IA": "19", "KS": "20", "KY": "21",
              "LA": "22", "ME": "23", "MD": "24", "MA": "25", "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30",
              "NE": "31", "NV": "32", "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
              "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47", "TX": "48", "UT": "49",
              "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55", "WY": "56", "PR": "72"}
FIPS_ABBR = {v: k for k, v in STATE_ABBR.items()}
STATE_NAMES = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
               "connecticut": "CT", "delaware": "DE", "districtofcolumbia": "DC", "florida": "FL", "georgia": "GA",
               "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
               "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
               "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
               "nebraska": "NE", "nevada": "NV", "newhampshire": "NH", "newjersey": "NJ", "newmexico": "NM",
               "newyork": "NY", "northcarolina": "NC", "northdakota": "ND", "ohio": "OH", "oklahoma": "OK",
               "oregon": "OR", "pennsylvania": "PA", "rhodeisland": "RI", "southcarolina": "SC", "southdakota": "SD",
               "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
               "westvirginia": "WV", "wisconsin": "WI", "wyoming": "WY", "puertorico": "PR"}
REGIONS = {"Northeast": "CT ME MA NH RI VT NJ NY PA", "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
           "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX", "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA"}
REGION_OF = {s: r for r, ss in REGIONS.items() for s in ss.split()}


def to_state_fips(values):
    """State names, postal codes or numeric codes -> 2-digit FIPS strings (NaN when unknown)."""
    out = []
    for v in pd.Series(values).astype(str):
        s = v.strip()
        if re.fullmatch(r"\d+(\.0+)?", s):
            out.append(str(int(float(s))).zfill(2))
        elif s.upper() in STATE_ABBR:
            out.append(STATE_ABBR[s.upper()])
        else:
            out.append(STATE_ABBR.get(STATE_NAMES.get(squash(s)), np.nan))
    return pd.Series(out, index=getattr(values, "index", None))


def zfill_code(s, n):
    return pc.digits(s).str.zfill(n).where(pc.digits(s).str.fullmatch(r"\d+").fillna(False).astype(bool))


# --------------------------------------------------------------------------- statistics
def wcorr(x, y, w=None):
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    k = ~(np.isnan(x) | np.isnan(y) | np.isnan(w))
    x, y, w = x[k], y[k], w[k]
    if len(x) < 3 or w.sum() == 0:
        return np.nan
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return float(np.average((x - mx) * (y - my), weights=w) / math.sqrt(vx * vy)) if vx > 0 and vy > 0 else np.nan


def boot_ci(stat, *arrays, n=None, seed=None, level=0.95):
    """Percentile bootstrap CI of stat(*resampled arrays); rows are resampled together."""
    n = n or LAB["SETTINGS"]["bootstrap"]
    rng = np.random.default_rng(seed or LAB["SETTINGS"]["seed"])
    arrays = [np.asarray(a) for a in arrays]
    idx = np.arange(len(arrays[0]))
    vals = [stat(*(a[s] for a in arrays)) for s in (rng.choice(idx, len(idx)) for _ in range(n))]
    a = (1 - level) / 2
    return tuple(float(v) for v in np.nanpercentile(vals, [100 * a, 100 * (1 - a)]))


def zscore(s):
    s = pd.Series(s, dtype=float)
    return (s - s.mean()) / s.std(ddof=1)


def wls(df, y, xs, weights=None, fe=None, robust=True):
    """Weighted least squares with optional fixed effects; returns a small dict per term.

    Coefficients are reported per standard deviation of each continuous x (z-scored first), so
    effects are comparable across measures; 0/1 indicators stay as they are, so their coefficient
    is the difference between the two groups. y stays in its own units.
    """
    import statsmodels.api as sm
    d = df.dropna(subset=[y] + list(xs) + ([weights] if weights else []) + ([fe] if fe else [])).copy()
    if len(d) < len(xs) + 5:
        return None
    X = pd.DataFrame({x: d[x].astype(float) if set(pd.unique(d[x].dropna())) <= {0, 1} else zscore(d[x]) for x in xs}, index=d.index)
    if fe:
        X = pd.concat([X, pd.get_dummies(d[fe].astype(str), prefix="fe", drop_first=True, dtype=float)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    model = sm.WLS(d[y].astype(float), X, weights=d[weights].astype(float) if weights else np.ones(len(d)))
    r = model.fit(cov_type="HC1") if robust else model.fit()
    return {"n": int(r.nobs), "r2": float(r.rsquared),
            "terms": {x: {"coef": float(r.params[x]), "se": float(r.bse[x]), "t": float(r.tvalues[x]),
                          "p": float(r.pvalues[x])} for x in xs},
            "resid": r.resid}


def haversine_km(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    a = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 12742 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def knn_weights(lat, lon, k=8):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    D = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    W = np.zeros_like(D)
    for i in range(len(lat)):
        W[i, np.argsort(D[i])[1:k + 1]] = 1
    return W / W.sum(1, keepdims=True)


def moran(x, W, perms=999, seed=None):
    """Global Moran's I with a permutation p-value, and local (LISA) statistics."""
    rng = np.random.default_rng(seed or LAB["SETTINGS"]["seed"])
    z = zscore(x).to_numpy()
    I = float(z @ W @ z / (z @ z))
    sims = np.array([(zp @ W @ zp) / (zp @ zp) for zp in (rng.permutation(z) for _ in range(perms))])
    lag = W @ z
    quad = np.select([(z > 0) & (lag > 0), (z < 0) & (lag < 0), (z > 0) & (lag < 0), (z < 0) & (lag > 0)],
                     ["high-high", "low-low", "high-low", "low-high"], "none")
    return {"I": I, "p": float((np.sum(sims >= I) + 1) / (perms + 1)), "local": z * lag, "quadrant": quad}


def gini(x, w=None):
    x = np.asarray(x, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw, cxw = np.cumsum(w), np.cumsum(x * w)
    return float(1 - 2 * np.sum(w * (cxw - x * w / 2)) / (cw[-1] * cxw[-1])) if cxw[-1] > 0 else np.nan


# --------------------------------------------------------------------------- findings and views
RESULTS = {"findings": [], "views": {}, "datasets": {}, "diagnostics": [], "catalog": {}, "profiles": {}, "run": {}}
STRENGTH = ("robust", "suggestive", "fragile", "artifact", "descriptive", "needs data")


def finding(fid, title, claim, *, theme, level, datasets, strength, stats=None, question="", next_data="",
            caveats=(), views=(), rank=50):
    """One entry for the findings board.

    strength: robust (survives the controls and variants tried), suggestive (holds but
    thin or not yet stress-tested), fragile (sign or size depends on choices), artifact
    (most likely produced by how the data were built), descriptive (a fact, no claim),
    needs data (the analysis could not run with the inputs present).
    """
    assert strength in STRENGTH, strength
    f = {"id": fid, "title": title, "claim": claim, "theme": theme, "level": level, "datasets": list(datasets),
         "strength": strength, "stats": _clean(stats or {}), "question": question, "next_data": next_data,
         "caveats": list(caveats), "views": list(views), "rank": rank}
    RESULTS["findings"] = [x for x in RESULTS["findings"] if x["id"] != fid] + [f]
    return f


def view(vid, kind, title, dataset, **spec):
    """Register a chart the site can draw. `dataset` names a table registered with dataset()."""
    RESULTS["views"][vid] = {"id": vid, "kind": kind, "title": title, "dataset": dataset, **_clean(spec)}
    return vid


def dataset(did, df, columns=None, digits=4, note=""):
    """Register the tidy table behind one or more views; floats are rounded to keep the site small."""
    d = df[columns] if columns else df
    d = d.copy()
    num = d.select_dtypes("number").columns
    d[num] = d[num].replace([np.inf, -np.inf], np.nan)
    recs = []
    for row in d.itertuples(index=False):
        rec = {}
        for c, v in zip(d.columns, row):
            if isinstance(v, (float, np.floating)):
                rec[c] = None if math.isnan(v) else float(f"{v:.{digits}g}")
            elif isinstance(v, (np.integer,)):
                rec[c] = int(v)
            elif isinstance(v, (pd.Timestamp,)):
                rec[c] = v.isoformat()[:10]
            elif v is None or (isinstance(v, float) and math.isnan(v)):
                rec[c] = None
            else:
                rec[c] = v if isinstance(v, (int, str, bool)) else str(v)
        recs.append(rec)
    RESULTS["datasets"][did] = {"id": did, "columns": list(d.columns), "rows": recs, "note": note}
    return did


def diagnostic(step, left, right, key, matched, total, unit="rows", note=""):
    """One line of the linkage report: how much of `left` found a partner in `right`."""
    share = matched / total if total else np.nan
    RESULTS["diagnostics"] = [x for x in RESULTS["diagnostics"] if x["step"] != step] + [
        {"step": step, "left": left, "right": right, "key": key, "matched": int(matched), "total": int(total),
         "share": None if total == 0 else round(float(share), 4), "unit": unit, "note": note}]
    log(f"link {step}: {matched:,} of {total:,} {unit} ({share:.1%})" if total else f"link {step}: nothing to link")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if (o is None or math.isnan(float(o))) else float(f"{float(o):.5g}")
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _merge_previous(old):
    """Keep what earlier runs produced for stages this run skipped (e.g. run only "analyze", keep the catalog)."""
    ran = set(RESULTS["run"].get("stages") or ())
    for key, stage in (("catalog", "inventory"), ("profiles", "profile")):
        if stage not in ran and not RESULTS[key] and old.get(key):
            RESULTS[key] = old[key]
    if not {"adapt", "link"} <= ran and old.get("diagnostics"):     # partial re-link: newer steps win
        steps = {d["step"] for d in RESULTS["diagnostics"]}
        RESULTS["diagnostics"] = [d for d in old["diagnostics"] if d["step"] not in steps] + RESULTS["diagnostics"]
    if "adapt" not in ran and old.get("run", {}).get("adapters"):
        RESULTS["run"]["adapters"] = old["run"]["adapters"]
    only = RESULTS["run"].get("analyses_only")
    if "analyze" not in ran or only:                                  # no or some analyses: merge by id
        new_ids = {f["id"] for f in RESULTS["findings"]}
        RESULTS["findings"] = [f for f in old.get("findings", []) if f["id"] not in new_ids] + RESULTS["findings"]
        for key in ("views", "datasets"):
            RESULTS[key] = {**old.get(key, {}), **RESULTS[key]}


def save_results():
    R = Path(LAB["RESULTS"])
    R.mkdir(parents=True, exist_ok=True)
    f = R / "results.json"
    if f.exists():
        try:
            _merge_previous(json.loads(f.read_text()))
        except (ValueError, KeyError, TypeError) as e:
            log(f"results: could not merge the previous results.json ({e}); writing this run only")
    RESULTS["run"].update({"version": __version__, "finished": time.strftime("%Y-%m-%d %H:%M")})
    RESULTS["findings"].sort(key=lambda x: (x.get("rank") or 99, x["id"]))
    with open(f, "w") as fh:
        json.dump(RESULTS, fh, separators=(",", ":"))
    pd.DataFrame(RESULTS["findings"]).drop(columns=["views"], errors="ignore").to_csv(R / "findings.csv", index=False)
    pd.DataFrame(RESULTS["diagnostics"]).to_csv(R / "linkage_diagnostics.csv", index=False)
    ran = RESULTS["run"].get("stages") or []
    if "analyze" in ran or RESULTS["findings"]:
        log(f"results: {len(RESULTS['findings'])} findings, {len(RESULTS['views'])} views, "
            f"{len(RESULTS['datasets'])} datasets -> {f}")
    else:
        log(f"results: saved after {', '.join(ran) or 'this step'} -> {f}")
    return f
