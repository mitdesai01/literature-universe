# -*- coding: utf-8 -*-
"""polisy_core: shared configuration, IO and keys for the POLISY data pipeline.

Design rules enforced here:
  * nothing large is ever read into pandas; DuckDB reads the raw files and returns
    aggregates, Polars handles medium frames, pandas only holds final tables
  * every raw file is converted once to Parquet in DATA/canonical and read from there
  * every canonical table has exactly one grain, declared in GRAINS below
  * every merge is a left join onto a VRscores grain and keeps a match indicator

Run modules in numeric order; each writes into DATA and prints its diagnostics.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import polars as pl
    HAVE_POLARS = True
except ImportError:
    HAVE_POLARS = False

# --------------------------------------------------------------------------- config
ROOT = Path(os.environ.get("POLISY_ROOT", "/content/polisy"))
CONFIG = {
    "RAW": ROOT / "data" / "raw",              # downloads land here, never edited
    "CANONICAL": ROOT / "data" / "canonical",  # one parquet per source, one grain each
    "KEYS": ROOT / "data" / "keys",            # crosswalks
    "PANELS": ROOT / "data" / "panels",        # analysis-ready panels
    "OUT": ROOT / "output",                    # tables and figures
    # VRscores inputs, as downloaded
    "VR_EMPLOYER": "/content/dataverse_files.zip",
    "VR_MSA": "/content/dataverse_files__msa.zip",
    "VR_INDUSTRY": "/content/dataverse_files_industries.zip",
    "VR_OCCUPATION": "/content/dataverse_files_Occupation.zip",
    # thesis files
    "COMPUSTAT": "/content/Compustat_Final.csv",
    "DIPI": "/content/Organizational_Leadership_File.csv",
    "COUNTYPRES": "/content/countypres_2000-2024.csv",
    # external
    "OEWS_YEARS": [2024],
    "CENSUS_API_KEY": "",
    "USER_AGENT": "academic research (University of Groningen)",
}

GRAINS = {
    "vr_employer": ("vrid", "year"),
    "vr_metro": ("msa", "year"),
    "vr_industry": ("naics6", "year"),
    "vr_occupation": ("occ_code", "year"),
    "oews_national": ("occ_code", "year"),
    "oews_metro": ("cbsa", "occ_code", "year"),
    "oews_industry": ("naics4", "occ_code", "year"),
    "acs_metro": ("cbsa", "year"),
    "vote_county": ("county_fips", "year"),
    "compustat": ("gvkey", "fyear"),
    "dipi": ("gvkey", "year"),
}

PRIMARY_STATES = {"GA", "IL", "IN", "MI", "MS", "OH", "SC", "TN", "TX", "VA", "WA"}
MODELED_STATES = {"AL", "HI", "MN", "MO", "MT", "ND", "WI", "VT"}
INVENTIVE_SOC2 = {"15": "Computer & mathematical", "17": "Architecture & engineering",
                  "19": "Life, physical & social science"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def paths():
    for k in ("RAW", "CANONICAL", "KEYS", "PANELS", "OUT"):
        Path(CONFIG[k]).mkdir(parents=True, exist_ok=True)
    (Path(CONFIG["OUT"]) / "tables").mkdir(exist_ok=True)
    (Path(CONFIG["OUT"]) / "figures").mkdir(exist_ok=True)
    return {k: Path(CONFIG[k]) for k in ("RAW", "CANONICAL", "KEYS", "PANELS", "OUT")}


def con():
    """One DuckDB connection with a spill directory, so 6M-row scans never OOM."""
    c = duckdb.connect()
    tmp = Path(CONFIG["CANONICAL"]).parent / "_duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    c.execute(f"SET temp_directory = '{tmp.as_posix()}'")
    c.execute("SET preserve_insertion_order = false")
    return c


def sqlp(p):
    return str(Path(p).as_posix()).replace("'", "''")


def q(c, sql):
    return c.execute(sql).df()


def q1(c, sql):
    r = c.execute(sql).fetchone()
    return None if r is None else r[0]


def save(df, name, note=""):
    """Every table the pipeline produces lands in output/tables as CSV."""
    p = Path(CONFIG["OUT"]) / "tables" / f"{name}.csv"
    df.to_csv(p, index=False)
    log(f"table {name}.csv ({len(df):,} rows){' - ' + note if note else ''}")
    return df


# ------------------------------------------------------------------- reading raw VRscores
def sniff_delim(path):
    """VRscores ships .tab files that are comma separated; never trust the extension."""
    with open(path, "rb") as fh:
        head = fh.readline(200000).decode("utf-8", "ignore")
    counts = {",": head.count(","), "\t": head.count("\t"), ";": head.count(";")}
    return max(counts, key=counts.get)


YEAR_RE = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")


def zip_to_parquet(c, zip_path, out_path, add_year_from_name=True):
    """Unpack one member at a time, convert with DuckDB, delete the unpacked copy.

    Why: the employer ZIP holds 13 files of up to 175 MB uncompressed. This keeps peak
    disk use at one file and peak memory near zero.
    """
    out_path = Path(out_path)
    if out_path.exists():
        return out_path
    tmp = out_path.parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist()
                   if n.lower().endswith((".csv", ".tab")) and "__MACOSX" not in n]
        for name in sorted(members):
            dest = tmp / Path(name).name
            with zf.open(name) as fin, open(dest, "wb") as fout:
                fout.write(fin.read())
            delim = sniff_delim(dest)
            reader = (f"read_csv('{sqlp(dest)}', header=true, auto_detect=true, "
                      f"sample_size=-1, delim='{delim}')")
            cols = [x.lower() for x in q(c, f"DESCRIBE SELECT * FROM {reader}")["column_name"]]
            m = YEAR_RE.search(Path(name).name)
            add = f", {int(m.group(1))} AS year" if (add_year_from_name and "year" not in cols and m) else ""
            part = tmp / (Path(name).stem + ".parquet")
            c.execute(f"COPY (SELECT *{add} FROM {reader}) TO '{sqlp(part)}' (FORMAT parquet, COMPRESSION zstd)")
            dest.unlink()
            parts.append(part)
    lst = "[" + ", ".join(f"'{sqlp(p)}'" for p in parts) + "]"
    c.execute(f"COPY (SELECT * FROM read_parquet({lst}, union_by_name=true)) "
              f"TO '{sqlp(out_path)}' (FORMAT parquet, COMPRESSION zstd)")
    for p in parts:
        p.unlink()
    log(f"canonical {out_path.name}: {q1(c, f'SELECT count(*) FROM read_parquet(\'{sqlp(out_path)}\')'):,} rows")
    return out_path


def vr_view(c, name, parquet, kind):
    """Register a VRscores panel under canonical column names."""
    key = {"employer": "vrid", "metro": "msa", "industry": "naics_code", "occupation": "onet_code"}[kind]
    extra = "company_name, employee_count AS workers, avg_match_quality AS match_q," if kind == "employer" else ""
    c.execute(f"""CREATE OR REPLACE VIEW {name} AS SELECT
        CAST({key} AS VARCHAR) AS unit, CAST(year AS INTEGER) AS year, {extra}
        CAST(dem_workers_raw AS DOUBLE) AS dem_raw, CAST(rep_workers_raw AS DOUBLE) AS rep_raw,
        CAST(dem_workers_imp AS DOUBLE) AS dem, CAST(rep_workers_imp AS DOUBLE) AS rep,
        CAST(dem_workers_imp + rep_workers_imp AS DOUBLE) AS tp,
        rep_workers_imp / NULLIF(dem_workers_imp + rep_workers_imp, 0) AS rep_share,
        rep_workers_raw / NULLIF(dem_workers_raw + rep_workers_raw, 0) AS rep_share_raw
        FROM read_parquet('{sqlp(parquet)}')""")
    return name


# ------------------------------------------------------------------- canonical keys
def norm_city(s):
    s = str(s).lower().replace("st.", "saint").replace("ste.", "sainte")
    return re.sub(r"[^a-z ]", "", s).strip()


def parse_msa(name):
    """'Boston-Cambridge-Quincy MA-NH MSA' -> (cities, states)."""
    s = str(name).strip()
    m = re.search(r"\s([A-Z]{2}(?:-[A-Z]{2})*)(?:\s+(?:MSA|Metro Area|Micro Area))?\s*$", s)
    states = m.group(1).split("-") if m else []
    base = s[:m.start()] if m else s
    return [norm_city(c) for c in re.split(r"[-/]", base) if c.strip()], states


def party_regime(state):
    if not state:
        return "unknown"
    if state in PRIMARY_STATES:
        return "primary-based"
    if state in MODELED_STATES:
        return "modelled by L2"
    return "party registration"


LEGAL_TAIL = {"inc", "incorporated", "corp", "corporation", "co", "company", "companies", "llc",
              "ltd", "limited", "plc", "lp", "llp", "holdings", "holding", "group", "the", "and"}
ABBR = {"international": "intl", "technologies": "tech", "technology": "tech", "systems": "sys",
        "services": "svcs", "communications": "comm", "manufacturing": "mfg",
        "pharmaceuticals": "pharma", "laboratories": "labs", "industries": "inds",
        "financial": "finl", "national": "natl", "american": "amer", "america": "amer",
        "corporation": "corp", "company": "co", "incorporated": "inc"}


def norm_name(s):
    """Canonical company name, used on both sides of every name match."""
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\(.*?\)", " ", s.lower())
    s = re.sub(r"/[a-z]{2,4}/?", " ", s)
    s = re.sub(r"-\s*(cl|class)\s*[a-z]\b", " ", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [ABBR.get(t, t) for t in s.split()]
    out, run = [], []
    for t in toks:                                  # "u s steel" -> "us steel"
        if len(t) == 1 and t.isalpha():
            run.append(t)
            continue
        if run:
            out.append("".join(run)); run = []
        out.append(t)
    if run:
        out.append("".join(run))
    while out and out[0] == "the":
        out = out[1:]
    while out and out[-1] in LEGAL_TAIL:
        out.pop()
    return " ".join(out)


# ------------------------------------------------------------------- shared analysis
def exposure_by_party(dem, rep):
    """Leave-one-out own-party coworker exposure divided by the national party share.

    Returns (democratic ratio, republican ratio, combined). 1.00 = random mixing. The
    leave-one-out form makes the benchmark exactly 1 for units of any size, which is why
    small employers do not bias it.
    """
    d, r = np.round(np.asarray(dem, float)), np.round(np.asarray(rep, float))
    t = d + r
    keep = t >= 2
    d, r, t = d[keep], r[keep], t[keep]
    D, R, T = d.sum(), r.sum(), t.sum()
    if D == 0 or R == 0:
        return np.nan, np.nan, np.nan
    ed = (d * (d - 1) / (t - 1)).sum() / D
    er = (r * (r - 1) / (t - 1)).sum() / R
    return ed / (D / T), er / (R / T), (D * ed / (D / T) + R * er / (R / T)) / T


def weighted_corr(x, y, w):
    x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    return cov / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))


def coverage_note(matched_workers, total_workers):
    """Every merged result carries this line. Non-negotiable."""
    return f"covers {matched_workers / total_workers:.1%} of VRscores matched workers"
