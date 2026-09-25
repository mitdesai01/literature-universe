# -*- coding: utf-8 -*-
"""sources: the lab's dataset registry, file discovery, downloads and schema profiles.

Adding a dataset = one SOURCES entry (what it is, where it comes from, how to recognise its
files) + one adapter (adapters/) that turns the files into canonical tables. Everything else
(discovery, download, profiling, the data catalog on the site) works from this entry.
"""
from __future__ import annotations

import io
import json
import re
import shutil
import os
import subprocess
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .core import LAB, log, pc, squash, RESULTS, dirs

UA = {"User-Agent": "Mozilla/5.0 (academic research; POLISY lab)"}

# Each source: title, theme, publisher, url (landing page), access, terms, grain, keys, and
# `files`: {role: {"names": regex on the cleaned file name, "kinds": extensions,
#                  "tokens": header cells that must all be present ("~x" = contained),
#                  "many": several files (e.g. one per year)}}.
SOURCES = {
    "aioe": {
        "title": "AI Occupational, Industry and Geographic Exposure (AIOE, AIIE, AIGE)",
        "theme": "AI exposure", "publisher": "Felten, Raj & Seamans (2021, Strategic Management Journal)",
        "url": "https://github.com/AIOE-Data/AIOE", "access": "open; module fetch clones the GitHub repository",
        "grain": "occupation (6-digit SOC), industry (4-digit NAICS), county (FIPS)", "keys": ["soc", "naics4", "county_fips"],
        "files": {
            "appendix": {"names": r"aioe_dataappendix", "kinds": (".xlsx",), "tokens": ("~aioe",)},
            "lm": {"names": r"language_modeling_aioe", "kinds": (".xlsx",), "tokens": ("~languagemodelingaioe",)},
            "ig": {"names": r"image_generation_aioe", "kinds": (".xlsx",), "tokens": ("~imagegenerationaioe",)},
            "abilities": {"names": r"abilities", "kinds": (".dta", ".csv"), "tokens": ("onetsoccode", "elementname", "scaleid")},
            "mturk": {"names": r"mturk", "kinds": (".dta", ".csv"), "tokens": ("applications", "oralcomprehension")},
            "salary": {"names": r"salary", "kinds": (".dta", ".csv"), "tokens": ("occcode", "~salary")},
            "education": {"names": r"education", "kinds": (".dta", ".csv"), "tokens": ("occcode", "~education")},
            "creative": {"names": r"creative", "kinds": (".dta", ".csv"), "tokens": ("occcode", "~creative")},
            "representation": {"names": r"representation", "kinds": (".dta", ".csv"), "tokens": ("occcode", "~female")},
            "oes_staffing": {"names": r"oes_4dig", "kinds": (".dta", ".zip", ".csv"), "tokens": ("naics", "occcode", "totemp")},
            "qcew": {"names": r"county_naics", "kinds": (".dta", ".csv"), "tokens": ("areafips", "agglvlcode", "~emplvl")},
        }},
    "dynamic_aioe": {
        "title": "Dynamic AI Occupational Exposure (DAIOE)", "theme": "AI exposure", "publisher": "DAIOE v1.0.0, Zenodo record 21873968",
        "url": "https://zenodo.org/records/21873968", "access": "open; module fetch lists the record's files through the Zenodo API",
        "grain": "occupation (SOC 2010, SOC 2018) x year, 2010-2024", "keys": ["soc", "soc2018", "year"],
        "files": {
            # DAIOE v1.0.0 (Zenodo 21873968): occupation x year panels on several classifications. The US SOC 2010
            # panel links to AIOE and VRscores; the SOC 2018 panel adds other published exposure measures.
            "soc2010": {"names": r"daioe_soc2010", "kinds": pc.TABLES, "tokens": ("occcodesoc2010", "year"), "many": True},
            "soc2018": {"names": r"daioe_panel_soc2018|daioe.*soc2018", "kinds": pc.TABLES, "tokens": ("soc2018code", "year"), "many": True},
            # anything else that looks like an occupation x time exposure table
            "data": {"names": r"dynamic_?ai|dyn_?aioe|aioe_?dyn|time_?varying", "kinds": pc.TABLES + (".zip",),
                     "tokens": ("~soc",), "many": True}}},
    "btos": {
        "title": "Business Trends and Outlook Survey (AI use)", "theme": "AI adoption", "publisher": "US Census Bureau",
        "url": "https://www.census.gov/hfp/btos/data_downloads", "access": "open; module fetch reads the download page for file links",
        "grain": "geography (national, sector, state, MSA, size) x question x answer x biweekly period", "keys": ["naics2", "state_fips", "period"],
        "files": {"data": {"names": r"btos|business_?trends|response_?estimates|^(national|sector|subsector|state|msa|"
                                    r"employment_?size.*|empsize|state_?by_?sector|state_?sector|top_?25.*)$",
                           "kinds": (".xlsx", ".csv", ".zip"), "tokens": ("~question", "~answer"), "many": True}}},
    "cspp": {
        "title": "Correlates of State Policy", "theme": "policy & politics", "publisher": "IPPSR, Michigan State University (Jordan & Grossmann)",
        "url": "https://ippsr.msu.edu/public-policy/correlates-state-policy", "access": "open; module fetch reads the project page for file links",
        "grain": "state x year", "keys": ["state_fips", "year"],
        "files": {"data": {"names": r"correlates|cspp|state_?policy", "kinds": (".csv", ".dta", ".xlsx", ".zip"), "tokens": ("year",)},
                  "codebook": {"names": r"codebook|variable", "kinds": (".xlsx", ".csv"), "tokens": ("~variable",)}}},
    "irs_migration": {
        "title": "IRS SOI migration data (state and county flows)", "theme": "migration", "publisher": "IRS Statistics of Income",
        "url": "https://www.irs.gov/statistics/soi-tax-stats-migration-data", "access": "open; module fetch tries the yearly CSV links",
        "grain": "origin x destination x year pair (households = returns, people = exemptions, AGI)", "keys": ["state_fips", "county_fips", "year"],
        "files": {"state": {"names": r"^state_?(in|out)_?flow_?\d{4}", "kinds": (".csv",), "tokens": ("~y1statefips", "~y2statefips"), "many": True},
                  "county": {"names": r"^county_?(in|out)_?flow_?\d{4}", "kinds": (".csv",), "tokens": ("~y1countyfips", "~y2countyfips"), "many": True}}},
    "patentsview": {
        "title": "PatentsView granted patents (patents, CPC, inventors, locations)", "theme": "innovation", "publisher": "USPTO PatentsView",
        "url": "https://patentsview.org/download/data-download-tables", "access": "open, large; use your local copies (fetch can download them)",
        "grain": "patent; patent x inventor; location", "keys": ["patent_id", "state_fips", "county_fips", "year"],
        "files": {"patent": {"names": r"^g_patent(_tsv)?$", "kinds": (".zip", ".tsv", ".csv", ".parquet"), "tokens": ("patentid", "~patentdate")},
                  "cpc": {"names": r"^g_cpc_current(_tsv)?$", "kinds": (".zip", ".tsv", ".csv", ".parquet"), "tokens": ("patentid", "~cpc")},
                  "inventor": {"names": r"^g_inventor_disambiguated(_tsv)?$", "kinds": (".zip", ".tsv", ".csv", ".parquet"), "tokens": ("patentid", "~locationid")},
                  "location": {"names": r"^g_location_disambiguated(_tsv)?$", "kinds": (".zip", ".tsv", ".csv", ".parquet"), "tokens": ("locationid", "~latitude")}}},
    "vrscores": {
        "title": "VRscores workforce partisanship (employer, metro, industry, occupation panels)", "theme": "politics at work",
        "publisher": "Kagan, Frake & Hurst (Organization Science 2026)", "url": "https://dataverse.harvard.edu",
        "access": "run POLISY_DA modules 01-04 first (their canonical Parquet files are read), or supply the VRscores HTML report",
        "grain": "employer/metro/industry/occupation x year", "keys": ["soc", "naics6", "msa", "cbsa", "year"],
        "files": {"report": {"names": r"vrscores|report", "kinds": (".html",), "tokens": ()}}},
    "elections": {
        "title": "County presidential returns 2000-2024", "theme": "politics", "publisher": "MIT Election Data and Science Lab",
        "url": "https://doi.org/10.7910/DVN/VOQCHQ", "access": "found by POLISY_DA (COUNTYPRES)", "grain": "county x election year",
        "keys": ["county_fips", "year"], "files": {}},
    "geography": {
        "title": "Census CBSA delineation (county -> metro)", "theme": "geography", "publisher": "US Census Bureau / OMB",
        "url": "https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html",
        "access": "found by POLISY_DA (CBSA_REFERENCE)", "grain": "county", "keys": ["county_fips", "cbsa"], "files": {}},
}


# --------------------------------------------------------------------------- discovery
EXTRA_EXTS = (".html", ".htm", ".7z")


def _lab_items():
    """polisy_core's index of the search folders plus the lab's own download folder, 4 levels deep,
    plus the file types polisy_core does not index (.html reports, .7z archives)."""
    items = list(pc._index())
    seen = {str(i["path"]) for i in items}
    for root, depth in pc._roots():
        for kind, p in pc._walk(root, depth):
            if kind == "file" and p.suffix.lower() in EXTRA_EXTS and str(p) not in seen:
                seen.add(str(p))
                try:
                    items.append({"path": p, "kind": p.suffix.lower(), "clean": pc.clean_name(p.stem)[0],
                                  "size": p.stat().st_size, "mtime": p.stat().st_mtime})
                except OSError:
                    pass
    raw = Path(LAB["RAW"])
    if raw.exists():
        for kind, p in pc._walk(raw, 4, cap=2000):
            if str(p) in seen:
                continue
            clean, ext = pc.clean_name(p.name)
            if kind == "file" and p.suffix.lower() in EXTRA_EXTS:
                clean, ext = pc.clean_name(p.stem)[0], p.suffix.lower()
            k = "dir" if kind == "dir" else ext
            if k:
                try:
                    items.append({"path": p, "kind": k, "clean": clean, "size": p.stat().st_size, "mtime": p.stat().st_mtime})
                except OSError:
                    pass
    return items


_ZIPS = {}


def _zip_tables(path):
    """Table files inside a zip: (name as stored, clean name, extension, size). Cached per zip version."""
    try:
        st = Path(path).stat()
        key = (str(path), st.st_size, st.st_mtime)
    except OSError:
        return []
    if key not in _ZIPS:
        out = []
        try:
            with zipfile.ZipFile(path) as zf:
                for i in zf.infolist():
                    n = i.filename
                    if n.endswith("/") or "__MACOSX" in n or Path(n).name.startswith("."):
                        continue
                    clean, ext = pc.clean_name(Path(n).name)
                    if ext in pc.TABLES:
                        out.append((n, clean, ext, i.file_size))
        except (OSError, zipfile.BadZipFile):
            pass
        _ZIPS[key] = out
    return _ZIPS[key]


def _fits(path, member, tokens):
    if not tokens:
        return True
    t = pc._tokens(path, member, rows=6)
    return bool(t) and all(pc._has(t, w) for w in tokens)


def discover(source, role):
    """Files for one role of one source: CONFIG override first, then the search folders.

    Returns a list of (path, member) pairs, best first. Every table inside a zip is a
    candidate of its own, matched by its own name (or by the zip's name), so one download
    holding many files (IRS: one per year and direction) is found without unzipping.
    Several files are returned only for roles marked many=True.
    """
    spec = SOURCES[source]["files"][role]
    key = f"LAB_{source}_{role}".upper()
    forced = pc.CONFIG.get(key)
    if forced:
        paths = [Path(p) for p in (forced if isinstance(forced, (list, tuple)) else [forced])]
        found = [(p, None) for p in paths if p.exists()]
        if found:
            return found
        log(f"{key}: CONFIG path(s) not found, searching instead")
    rx = re.compile(spec["names"], re.I)
    tokens = spec.get("tokens", ())
    hits, seen = [], set()

    def add(path, member, clean, size, mtime):
        ident = (clean, size, member and Path(member).parent.name)
        if ident in seen or not _fits(path, member, tokens):
            return
        seen.add(ident)
        hits.append((path, member, mtime))
    for it in _lab_items():
        if it["kind"] == ".zip":
            zip_named = bool(rx.search(it["clean"]))
            for name, clean, ext, size in _zip_tables(it["path"]):
                if ext in spec["kinds"] and (zip_named or rx.search(clean)):
                    add(it["path"], name, clean, size, it["mtime"])
        elif it["kind"] in spec["kinds"] and rx.search(it["clean"]):
            add(it["path"], None, it["clean"], it["size"], it["mtime"])
    hits.sort(key=lambda h: h[2], reverse=True)
    out = [(p, m) for p, m, _ in hits]
    return out if spec.get("many") else out[:1]


def unpack_archives():
    """Extract .7z archives the adapters need (the AIOE repository ships QCEW as county_naics_2019.7z)."""
    raw = dirs()["RAW"] / "unpacked"
    for it in _lab_items():
        if it["kind"] != ".7z" or not re.search(r"county_naics|oes|abilities", it["clean"]):
            continue
        dest = raw / it["clean"]
        if dest.exists() and any(dest.iterdir()):
            continue
        try:
            import py7zr
            with py7zr.SevenZipFile(it["path"]) as z:
                z.extractall(dest)
            log(f"unpacked {it['path'].name} -> {dest}")
        except Exception as e:
            log(f"cannot unpack {it['path'].name} ({e}); pip install py7zr")


def inventory():
    """Which files were found for every source and role (the data catalog on the site)."""
    unpack_archives()
    rows = []
    for s, meta in SOURCES.items():
        for role in meta["files"]:
            found = discover(s, role)
            rows.append({"source": s, "role": role, "files": len(found),
                         "paths": "; ".join(str(p) + (f" :: {m}" if m else "") for p, m in found[:6]) + (" ..." if len(found) > 6 else "")})
    inv = pd.DataFrame(rows)
    RESULTS["catalog"] = {s: {k: v for k, v in meta.items() if k != "files"} | {
        "roles": inv[inv.source == s][["role", "files", "paths"]].to_dict("records")} for s, meta in SOURCES.items()}
    return inv


# --------------------------------------------------------------------------- downloads
def _get(url, dest=None, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if dest:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(data)
    return data


def _links(page_url, pattern=r'href="([^"]+\.(?:xlsx|csv|zip|dta|xls))"'):
    html = _get(page_url).decode("utf-8", "ignore")
    out = []
    for h in re.findall(pattern, html, flags=re.I):
        out.append(h if h.startswith("http") else urllib.parse.urljoin(page_url, h))
    return sorted(set(out))


def fetch_aioe(raw):
    dest = raw / "aioe"
    if discover("aioe", "abilities") and discover("aioe", "appendix"):
        log("aioe: already present")
        return
    if shutil.which("git"):
        try:
            subprocess.run(["git", "clone", "--depth", "1", "https://github.com/AIOE-Data/AIOE", str(dest)],
                           check=True, capture_output=True, timeout=900, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"})
            log(f"aioe: cloned into {dest}")
        except Exception as e:
            log(f"aioe: git clone failed ({e}); downloading files one by one")
    if not (dest / "AIOE_DataAppendix.xlsx").exists():
        base = "https://raw.githubusercontent.com/AIOE-Data/AIOE/main/"
        for f in ["AIOE_DataAppendix.xlsx", "Language Modeling AIOE and AIIE.xlsx", "Image Generation AIOE and AIIE.xlsx",
                  "Input/abilities_2020.dta", "Input/mturk_mapping_matrix.dta", "Input/application_list.dta",
                  "Input/occ_title_2020.dta", "Input/oes_4dig_naics.zip", "Input/county_naics_2019.7z",
                  "Generative AI/occ_salary_data_2021.dta", "Generative AI/occ_required_education.dta",
                  "Generative AI/occ_creative_weight.dta", "Generative AI/occ_representation.dta"]:
            try:
                _get(base + urllib.parse.quote(f), dest / f)
            except Exception as e:
                log(f"aioe: {f} failed ({e})")
    for z in dest.rglob("*.7z"):
        if not z.with_suffix(".dta").exists():
            try:
                import py7zr
                with py7zr.SevenZipFile(z) as a:
                    a.extractall(z.parent)
                log(f"aioe: extracted {z.name}")
            except Exception as e:
                log(f"aioe: cannot extract {z.name} ({e}); pip install py7zr")


def fetch_zenodo(raw, record="21873968"):
    if any(discover("dynamic_aioe", r) for r in ("soc2010", "soc2018", "data")):
        log("dynamic_aioe: already present")
        return
    try:
        meta = json.loads(_get(f"https://zenodo.org/api/records/{record}"))
    except Exception as e:
        log(f"dynamic_aioe: Zenodo API unreachable ({e}). Download the files from https://zenodo.org/records/{record} "
            f"into {raw / 'dynamic_aioe'}")
        return
    for f in meta.get("files", []):
        name, url, size = f.get("key"), f.get("links", {}).get("self"), f.get("size", 0)
        if not name or not url:
            continue
        if size and size > 3e9:
            log(f"dynamic_aioe: skipping {name} ({size / 1e9:.1f} GB); download it by hand if needed")
            continue
        try:
            _get(url, raw / "dynamic_aioe" / name, timeout=900)
            log(f"dynamic_aioe: {name} ({size / 1e6:.1f} MB)")
        except Exception as e:
            log(f"dynamic_aioe: {name} failed ({e})")


def fetch_page_files(raw, source, page, keep=r"."):
    if discover(source, "data"):
        log(f"{source}: already present")
        return
    try:
        links = [u for u in _links(page) if re.search(keep, u, re.I)]
    except Exception as e:
        log(f"{source}: cannot read {page} ({e}). Download the data files by hand into {raw / source}")
        return
    log(f"{source}: {len(links)} file links on {page}")
    for u in links:
        try:
            _get(u, raw / source / Path(urllib.parse.urlparse(u).path).name, timeout=600)
        except Exception as e:
            log(f"{source}: {u} failed ({e})")


def fetch_irs(raw, first=2011, last=2023):
    have = {pc.clean_name(Path(m or p).name)[0] for p, m in discover("irs_migration", "state") + discover("irs_migration", "county")}
    got = 0
    for y in range(first, last):
        yy = f"{y % 100:02d}{(y + 1) % 100:02d}"
        for lvl in ("state", "county"):
            for d in ("inflow", "outflow"):
                name = f"{lvl}{d}{yy}.csv"
                if pc.clean_name(name)[0] in have:
                    continue
                try:
                    _get(f"https://www.irs.gov/pub/irs-soi/{name}", raw / "irs_migration" / name, timeout=300)
                    got += 1
                except Exception:
                    pass
    log(f"irs_migration: {got} new files downloaded ({len(have)} already present)")


def fetch_patentsview(raw):
    base = "https://s3.amazonaws.com/data.patentsview.org/download/"
    for role, f in [("location", "g_location_disambiguated.tsv.zip"), ("patent", "g_patent.tsv.zip"),
                    ("cpc", "g_cpc_current.tsv.zip"), ("inventor", "g_inventor_disambiguated.tsv.zip")]:
        if discover("patentsview", role):
            continue
        try:
            _get(base + f, raw / "patentsview" / f, timeout=3600)
            log(f"patentsview: {f} downloaded")
        except Exception as e:
            log(f"patentsview: {f} failed ({e}); copy your local file into a search folder")


def fetch_all(patentsview=False):
    """Download what can be downloaded; never re-download a file that is already found."""
    raw = dirs()["RAW"]
    fetch_aioe(raw)
    fetch_zenodo(raw)
    fetch_page_files(raw, "btos", "https://www.census.gov/hfp/btos/data_downloads", keep=r"\.(xlsx|csv|zip)$")
    fetch_page_files(raw, "cspp", "https://ippsr.msu.edu/public-policy/correlates-state-policy", keep=r"correlates|cspp|codebook")
    fetch_irs(raw)
    if patentsview:
        fetch_patentsview(raw)


# --------------------------------------------------------------------------- profiles
def profile_file(path, member=None, nrows=20000):
    """Columns, types, missingness, distinct counts and examples of one file (a sample of rows)."""
    path = Path(path)
    ext = Path(member or path.name).suffix.lower()
    try:
        if ext in (".tsv",) or (ext == ".zip" and member and member.lower().endswith(".tsv")):
            df = _read_tsv_sample(path, member, nrows)
        elif ext == ".dta" and not member:
            with pd.read_stata(path, iterator=True) as r:
                df = r.read(nrows)
        elif ext == ".html":
            return {"file": str(path), "kind": "html report", "size_mb": round(path.stat().st_size / 1e6, 1)}
        else:
            df = pc.read_table(path, member)
            df = df.head(nrows)
    except Exception as e:
        return {"file": str(path), "error": f"{type(e).__name__}: {e}"}
    cols = []
    for c in df.columns[:400]:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        cols.append({"name": str(c), "non_null": round(float(s.notna().mean()), 3), "distinct": int(s.nunique(dropna=True)),
                     "numeric": bool(num.notna().sum() >= 0.9 * s.notna().sum() and s.notna().any()),
                     "examples": [str(v)[:40] for v in s.dropna().unique()[:3]]})
    return {"file": str(path) + (f" :: {member}" if member else ""), "rows_sampled": len(df), "columns": len(df.columns),
            "fields": cols, "size_mb": round(path.stat().st_size / 1e6, 1)}


def _read_tsv_sample(path, member, nrows):
    if member:
        with zipfile.ZipFile(path) as zf, zf.open(member) as fh:
            head = b"".join(fh.readline() for _ in range(nrows + 1))
        return pd.read_csv(io.BytesIO(head), sep="\t", dtype=str, on_bad_lines="skip", quoting=3)
    return pd.read_csv(path, sep="\t", dtype=str, nrows=nrows, on_bad_lines="skip", quoting=3)


def profile_all():
    out = {}
    for s, meta in SOURCES.items():
        for role in meta["files"]:
            done = set()
            for p, m in discover(s, role):
                stem = Path(m or p).stem.lower()
                if stem in done:                 # the same table in another format (e.g. .tsv, .dta and .xlsx)
                    continue
                if len(done) >= 4:
                    break
                done.add(stem)
                pr = profile_file(p, m)
                out[f"{s}/{role}/{Path(m or p).name}"] = pr
                if "fields" in pr:
                    log(f"profile {s}/{role}: {Path(p).name}: {pr['columns']} columns, e.g. " +
                        ", ".join(f['name'] for f in pr["fields"][:8]))
    RESULTS["profiles"] = out
    (Path(LAB["RESULTS"]) / "profiles" / "profiles.json").write_text(json.dumps(out, indent=1, default=str))
    return out
