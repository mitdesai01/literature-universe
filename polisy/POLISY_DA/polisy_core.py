# -*- coding: utf-8 -*-
"""polisy_core: shared configuration, IO and keys for the POLISY data pipeline.

Design rules enforced here:
  * nothing large is ever read into pandas; DuckDB reads the raw files and returns
    aggregates, Polars handles medium frames, pandas only holds final tables
  * every raw file is converted once to Parquet in DATA/canonical and read from there
  * every canonical table has exactly one grain, declared in GRAINS below
  * every merge is a left join onto a VRscores grain and keeps a match indicator
  * no module builds the path to a download itself: every input is located by
    locate()/find(), which know the file's download name, looser variants of that name,
    and what has to be inside it (FILES below)

Run module 05 (downloads) first, then 01 to 10 in numeric order: run("05"), run("01"), ...
Each writes into DATA and prints its diagnostics. show_files() prints which file is used
for every input.
"""
from __future__ import annotations

import io
import json
import os
import re
import shutil
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

__version__ = "2026-09-24 file finder"

# --------------------------------------------------------------------------- config
ROOT = Path(os.environ.get("POLISY_ROOT", "/content/polisy"))
CONFIG = {
    "RAW": ROOT / "data" / "raw",              # downloads land here, never edited
    "CANONICAL": ROOT / "data" / "canonical",  # one parquet per source, one grain each
    "KEYS": ROOT / "data" / "keys",            # crosswalks
    "PANELS": ROOT / "data" / "panels",        # analysis-ready panels
    "OUT": ROOT / "output",                    # tables and figures

    # folders searched for inputs after RAW, each one subfolder level deep. Add the folder
    # you keep downloads in, or set POLISY_SEARCH="/a:/b" before importing.
    "SEARCH_DIRS": ["/content", "/content/drive/MyDrive", Path.home() / "Downloads"],

    # Inputs. A path here is used when it exists and holds the right content; otherwise the
    # file is searched for by name and content (FILES says what each looks like). None =
    # always search. Set a path to force one file.
    "VR_EMPLOYER": "/content/dataverse_files.zip",
    "VR_MSA": "/content/dataverse_files__msa.zip",
    "VR_INDUSTRY": "/content/dataverse_files_industries.zip",
    "VR_OCCUPATION": "/content/dataverse_files_Occupation.zip",
    "COMPUSTAT": "/content/Compustat_Final.csv",
    "DIPI": "/content/Organizational_Leadership_File.csv",
    "COUNTYPRES": "/content/countypres_2000-2024.csv",
    "CBSA_REFERENCE": "/content/list1_2023.xlsx",
    "OEWS_NATIONAL": None,                     # a path, or {year: path}
    "OEWS_MSA": None,
    "OEWS_INDUSTRY": None,
    "ONET": None,
    "AIOE": None,

    # external
    "OEWS_YEARS": [2024],
    "DIPI_MEASURE": "empLiberalism_10yr",      # the DIPI column module 07 validates against
    "CENSUS_API_KEY": "",
    "USER_AGENT": "academic research (University of Groningen)",
}
if os.environ.get("POLISY_SEARCH"):
    CONFIG["SEARCH_DIRS"] = os.environ["POLISY_SEARCH"].split(os.pathsep) + CONFIG["SEARCH_DIRS"]

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


# ------------------------------------------------------------------- finding input files
TABLES = (".csv", ".tab", ".tsv", ".txt", ".xlsx", ".xlsm", ".xls", ".dta", ".parquet")
CSV_LIKE = (".csv", ".tab", ".tsv", ".txt")
EXCEL = (".xlsx", ".xlsm", ".xls")

# Every input the pipeline reads, declared once.
#   what         what the file is
#   download_as  the name it has when it comes off the download page
#   names        regex for looser names, tried on clean_name(): lower case, "(1)" copy
#                suffixes dropped, spaces and dashes turned into "_"
#   kinds        extensions accepted; "dir" is an unzipped folder
#   members      regex a file inside the zip or folder must match, or
#   columns      header cells the table must have ("~x" = some cell containing x); the
#                first rows are searched, so title rows above the header are fine
#   prefer       among several fitting tables, prefer the one with this column
# {yy} and {yyyy} stand for the OEWS year.
FILES = {
    "VR_EMPLOYER": {
        "what": "VRscores employer panel", "download_as": "dataverse_files.zip",
        "names": r"dataverse_files|vrscores|employer", "kinds": (".zip", "dir"),
        "members": r"^employer_panel_year_\d{4}\.(tab|csv|parquet)$",
        "source": "Harvard Dataverse, VRscores data set, employer panel files, Download all",
        "used_by": "02"},
    "VR_MSA": {
        "what": "VRscores metro panel", "download_as": "dataverse_files.zip",
        "names": r"dataverse_files|vrscores|msa|metro", "kinds": (".zip", "dir"),
        "members": r"^msa_panel_year_\d{4}\.(tab|csv|parquet)$",
        "source": "Harvard Dataverse, VRscores data set, MSA panel files, Download all",
        "used_by": "02"},
    "VR_INDUSTRY": {
        "what": "VRscores industry panel", "download_as": "dataverse_files.zip",
        "names": r"dataverse_files|vrscores|industr|naics", "kinds": (".zip", "dir"),
        "members": r"^naics_panel_year_\d{4}\.(tab|csv|parquet)$",
        "source": "Harvard Dataverse, VRscores data set, industry panel files, Download all",
        "used_by": "02"},
    "VR_OCCUPATION": {
        "what": "VRscores occupation panel", "download_as": "dataverse_files.zip",
        "names": r"dataverse_files|vrscores|occupation", "kinds": (".zip", "dir"),
        "members": r"^occupation_panel_year_\d{4}\.(tab|csv|parquet)$",
        "source": "Harvard Dataverse, VRscores data set, occupation panel files, Download all",
        "used_by": "02"},
    "COMPUSTAT": {
        "what": "Compustat annual fundamentals", "download_as": "Compustat_Final.csv",
        "names": r"compustat|funda|comp_final", "kinds": TABLES + (".zip",),
        "columns": ("gvkey", "fyear", "conm"), "exclude": ("~liberal",),
        "source": "WRDS, Compustat Fundamentals Annual (WRDS names the file with a random code)",
        "used_by": "04"},
    "DIPI": {
        "what": "DIPI organizational leadership file", "download_as": "Organizational_Leadership_File.csv",
        "names": r"organi[sz]ational_?leadership|dipi|leadership_?file|political_?ideolog",
        "kinds": TABLES + (".zip",), "columns": ("~gvkey", "~liberal"), "prefer": "~empliberal",
        "source": "DIPI open data, Mannor & Busenbark (2025), tiny.cc/politicalideology",
        "used_by": "01, 07"},
    "COUNTYPRES": {
        "what": "MIT county presidential returns", "download_as": "countypres_2000-2024.csv",
        "names": r"countypres|county_?pres", "kinds": TABLES + (".zip",),
        "columns": ("countyfips", "candidatevotes", "party"),
        "source": "Harvard Dataverse, MIT Election Lab, doi:10.7910/DVN/VOQCHQ",
        "used_by": "06"},
    "CBSA_REFERENCE": {
        "what": "Census CBSA delineation file, List 1", "download_as": "list1_2023.xlsx",
        "names": r"^list_?1($|_)|delineation|cbsa_?2_?fips|cbsa_.*count", "kinds": (".xlsx", ".xls", ".csv"),
        "columns": ("cbsacode", "~fipscounty"),
        "source": "census.gov, Metropolitan and Micropolitan Delineation Files, List 1 (July 2023); module 05 downloads it",
        "used_by": "01, 04 (06 through keys/cw_cbsa_county.csv)"},
    "OEWS_NATIONAL": {
        "what": "BLS OEWS national estimates", "download_as": "oesm{yy}nat.zip",
        "names": r"^oesm{yy}nat|^national_m{yyyy}_dl", "kinds": (".zip", ".xlsx", "dir"),
        "members": r"^national_m{yyyy}_dl\.xlsx$",
        "source": "bls.gov/oes/tables.htm, May {yyyy}, National (zip)", "used_by": "03"},
    "OEWS_MSA": {
        "what": "BLS OEWS metro area estimates", "download_as": "oesm{yy}ma.zip",
        "names": r"^oesm{yy}ma|^msa_m{yyyy}_dl", "kinds": (".zip", ".xlsx", "dir"),
        "members": r"^msa_m{yyyy}_dl\.xlsx$",
        "source": "bls.gov/oes/tables.htm, May {yyyy}, Metropolitan and nonmetropolitan area (zip)",
        "used_by": "03"},
    "OEWS_INDUSTRY": {
        "what": "BLS OEWS 4-digit industry estimates", "download_as": "oesm{yy}in4.zip",
        "names": r"^oesm{yy}in4|^nat4d_m{yyyy}_dl", "kinds": (".zip", ".xlsx", "dir"),
        "members": r"^nat4d_m{yyyy}_dl\.xlsx$",
        "source": "bls.gov/oes/tables.htm, May {yyyy}, National industry-specific (zip)",
        "used_by": "03"},
    "ONET": {
        "what": "O*NET database, text files", "download_as": "db_29_0_text.zip",
        "names": r"^db_\d+_\d+_text|^onet", "kinds": (".zip", "dir"),
        "members": r"^job zones\.txt$",
        "source": "onetcenter.org/database.html, Text zip; module 05 downloads it", "used_by": "06"},
    "AIOE": {
        "what": "AI occupational exposure (AIOE) scores", "download_as": "AIOE_DataAppendix.xlsx",
        "names": r"aioe", "kinds": (".xlsx", ".csv"), "columns": ("~aioe",),
        "source": "github.com/AIOE-Data/AIOE; module 05 downloads it", "used_by": "06"},
}

_COPY = re.compile(r"(?:[\s_-]*\(\d+\)|[\s_-]+copy(?:[\s_-]*\d+)?)$", re.I)
_SKIP_DIRS = {"sample_data", "drive", "__macosx", "canonical", "panels", "keys", "output",
              "_tmp", "_duckdb_tmp", "site-packages", "node_modules"}
_CACHE = {"index": None, "time": 0.0, "peek": {}}


def squash(s):
    """'FIPS County Code' -> 'fipscountycode': how column names are compared."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def clean_name(name):
    """'Copy of Organizational Leadership File (1).CSV' -> ('organizational_leadership_file', '.csv')."""
    name = str(name).strip()
    ext = Path(name).suffix.lower()
    if ext not in TABLES + (".zip",):
        ext = ""
    stem = _COPY.sub("", name[:len(name) - len(ext)] if ext else name)
    stem = re.sub(r"^copy of\s+", "", stem, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_"), ext


def spec(key, year=None):
    """FILES[key] with the OEWS year filled in."""
    s = dict(FILES[key])
    if year is None and "{yy" in s["download_as"]:
        year = CONFIG["OEWS_YEARS"][-1]
    if year is not None:
        yy, yyyy = f"{int(year) % 100:02d}", str(int(year))
        s = {k: v.replace("{yyyy}", yyyy).replace("{yy}", yy) if isinstance(v, str) else v
             for k, v in s.items()}
    return s


def members(path):
    """{lower-case base name: name as stored} for the files in a zip or folder; a plain file
    is its own only member."""
    p = Path(path)
    try:
        if p.is_dir():
            return {e.name.lower(): e.name for e in os.scandir(p)
                    if e.is_file() and not e.name.startswith(".")}
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                return {Path(n).name.lower(): n for n in zf.namelist()
                        if not n.endswith("/") and "__MACOSX" not in n and not Path(n).name.startswith(".")}
    except (OSError, zipfile.BadZipFile):
        return {}
    return {p.name.lower(): p.name}


def _roots():
    """(folder, depth) pairs to search: RAW two levels deep, then SEARCH_DIRS one level."""
    out, seen = [], set()
    for i, d in enumerate([CONFIG["RAW"]] + list(CONFIG["SEARCH_DIRS"])):
        p = Path(d).expanduser()
        try:
            if not p.is_dir() or p.resolve() in seen:
                continue
            seen.add(p.resolve())
        except OSError:
            continue
        out.append((p, 2 if i == 0 else 1))
    return out


def _walk(root, depth, cap=400):
    """Files and folders under root, `depth` folder levels down, skipping pipeline output."""
    found, todo, visited = [("dir", root)], [(root, 0)], 0
    while todo and visited < cap:
        folder, level = todo.pop()
        visited += 1
        try:
            entries = list(os.scandir(folder))
        except OSError:
            continue
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    if e.name.lower() in _SKIP_DIRS:
                        continue
                    found.append(("dir", Path(e.path)))
                    if level < depth:
                        todo.append((Path(e.path), level + 1))
                elif e.is_file():
                    found.append(("file", Path(e.path)))
            except OSError:
                continue
    return found


def _index(refresh=False, ttl=30):
    """Everything under the search folders; rescanned when older than `ttl` seconds."""
    if not refresh and _CACHE["index"] is not None and time.time() - _CACHE["time"] < ttl:
        return _CACHE["index"]
    items, seen = [], set()
    for root, depth in _roots():
        for kind, p in _walk(root, depth):
            try:
                rp = p.resolve()
                if rp in seen:
                    continue
                seen.add(rp)
                st = p.stat()
            except OSError:
                continue
            clean, ext = clean_name(p.name)
            k = "dir" if kind == "dir" else ext
            if k:
                items.append({"path": p, "kind": k, "clean": clean, "size": st.st_size, "mtime": st.st_mtime})
    _CACHE.update(index=items, time=time.time())
    return items


def _delim(text):
    counts = {d: text.count(d) for d in (",", "\t", ";", "|")}
    return max(counts, key=counts.get)


def _text_tokens(raw, rows):
    lines = [x for x in raw.decode("utf-8-sig", "replace").splitlines() if x.strip()][:rows]
    if not lines:
        return set()
    d = _delim("\n".join(lines))
    return {squash(x) for line in lines for x in line.split(d)} - {""}


def _read_tokens(path, member, rows):
    ext = Path(member or path.name).suffix.lower()
    try:
        if member is not None:
            with zipfile.ZipFile(path) as zf:
                if ext in CSV_LIKE:
                    with zf.open(member) as fh:
                        return _text_tokens(fh.read(1 << 18), rows)
                if zf.getinfo(member).file_size > 50e6:
                    return None
                src = io.BytesIO(zf.read(member))
        elif ext in CSV_LIKE:
            with open(path, "rb") as fh:
                return _text_tokens(fh.read(1 << 18), rows)
        else:
            src = path
        if ext in EXCEL:
            books = pd.read_excel(src, sheet_name=None, header=None, nrows=rows, dtype=str)
            return {squash(v) for df in list(books.values())[:12] for v in df.to_numpy().ravel()
                    if isinstance(v, str)} - {""}
        if ext == ".parquet":
            import pyarrow.parquet as papq
            return {squash(c) for c in papq.read_schema(src).names}
        if ext == ".dta":
            with pd.read_stata(src, iterator=True) as r:
                return {squash(c) for c in r.variable_labels()}
    except Exception:
        return None
    return None


def _tokens(path, member=None, rows=15):
    """Squashed cells of a table's first rows (every sheet of a workbook), or None when it
    cannot be read. Cached per file version."""
    path = Path(path)
    try:
        st = path.stat()
    except OSError:
        return None
    ck = (str(path), member, st.st_size, st.st_mtime)
    if ck not in _CACHE["peek"]:
        _CACHE["peek"][ck] = _read_tokens(path, member, rows)
    return _CACHE["peek"][ck]


def _has(tokens, want):
    return any(want[1:] in t for t in tokens) if want.startswith("~") else want in tokens


def _check(s, path, kind):
    """(verdict, member): verdict is True/False once the contents were looked at and None
    when they could not be read; member is the fitting table inside a zip, if any."""
    if "members" in s:
        rx = re.compile(s["members"], re.I)
        return any(rx.search(n) for n in members(path)), None
    if "columns" not in s:
        return None, None

    def fits(t):
        return all(_has(t, w) for w in s["columns"]) and not any(_has(t, w) for w in s.get("exclude", ()))

    if kind == ".zip":
        want, rx = clean_name(s["download_as"])[0], re.compile(s["names"], re.I)
        best = None
        for low, name in members(path).items():
            t = _tokens(path, name) if Path(low).suffix in TABLES else None
            if t and fits(t):
                clean = clean_name(low)[0]
                score = (_has(t, s["prefer"]) if s.get("prefer") else False, clean == want, bool(rx.search(clean)))
                if best is None or score > best[0]:
                    best = (score, name)
        return (True, best[1]) if best else (False, None)
    t = _tokens(path)
    return (None, None) if t is None else (fits(t), None)


def _expects(s):
    if "members" in s:
        return f"files named like {s['members']}"
    return "columns " + ", ".join(w.replace("~", "*") for w in s["columns"])


def _vintage(clean):
    return max((int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", clean)), default=0)


def _worth_peeking(it):
    """Looked at by contents alone, when nothing with a fitting name exists."""
    return it["kind"] in ("dir", ".zip", ".parquet", ".dta") + CSV_LIKE or (it["kind"] in EXCEL and it["size"] < 20e6)


def _ident(it, s):
    """Candidates with one ident are copies of the same download, not a real choice."""
    if it["kind"] in ("dir", ".zip") and "members" in s:
        rx = re.compile(s["members"], re.I)
        return tuple(sorted(n for n in members(it["path"]) if rx.search(n)))
    return it["clean"], it["size"]


def _candidates(s, items, by_name):
    """Fitting files, best first, and the ones whose name fit but whose contents did not."""
    want = clean_name(s["download_as"])[0]
    rx = re.compile(s["names"], re.I)
    hits, rejected = [], []
    for it in items:
        if it["kind"] not in s["kinds"]:
            continue
        level = 3 if it["clean"] == want else 2 if rx.search(it["clean"]) else 0
        if bool(level) != by_name or (not level and not _worth_peeking(it)):
            continue
        ok, member = _check(s, it["path"], it["kind"])
        if ok is False and level:
            rejected.append(it["path"])
        if ok is False or (ok is None and not level):
            continue
        prefer = int(_has(_tokens(it["path"], member) or set(), s["prefer"])) if s.get("prefer") else 0
        hits.append({**it, "member": member, "rank": (level or 1, ok is True, prefer, _vintage(it["clean"])),
                     "how": {3: "download name", 2: "similar name", 0: "file contents"}[level],
                     "ident": _ident(it, s)})
    # ties: a zip or file before an unzipped folder (which may be half extracted), then newest
    return sorted(hits, key=lambda h: (h["rank"], h["kind"] != "dir", h["mtime"]), reverse=True), rejected


def locate(key, year=None, refresh=False):
    """Where input `key` is, and how it was found.

    Order: CONFIG[key] when that path exists and its contents fit; otherwise the search
    folders, first by download name, then by a similar name ("(1)" copies, spaces,
    capitals, another extension), then by contents alone (zip members or header columns).
    Returns a dict: path (None when missing), member (the table inside a zip), found_by,
    status (ok; AMBIGUOUS when two different files fit equally well; MISSING),
    alternatives and note.
    """
    s = spec(key, year)
    if year is None and "{yy" in FILES[key]["download_as"]:
        year = CONFIG["OEWS_YEARS"][-1]
    res = {"key": key, "year": year, "what": s["what"], "download_as": s["download_as"],
           "used_by": s["used_by"], "path": None, "member": None, "found_by": "",
           "status": "MISSING", "alternatives": "", "note": ""}
    want = CONFIG.get(key)
    if isinstance(want, dict):
        want = want.get(year)
    if want:
        p = Path(str(want)).expanduser()
        if p.exists():
            ok, member = _check(s, p, "dir" if p.is_dir() else p.suffix.lower())
            if ok is not False:
                res.update(path=p, member=member, found_by="CONFIG path", status="ok")
                return res
            res["note"] = f"CONFIG['{key}'] = {p} is not the {s['what']} (no {_expects(s)}), so it was ignored. "
        else:
            res["note"] = f"CONFIG['{key}'] = {p} does not exist. "
    for attempt in (0, 1):
        items = _index(refresh=refresh or attempt == 1)
        hits, rejected = _candidates(s, items, by_name=True)
        if not hits:
            hits = _candidates(s, items, by_name=False)[0]
        if hits and hits[0]["path"].exists():
            break
    else:
        if rejected:
            res["note"] += (f"Found {', '.join(p.name for p in rejected[:3])} by name, but without "
                            f"{_expects(s)}; check it is the right download.")
        return res
    best = hits[0]
    res.update(path=best["path"], member=best["member"], found_by=best["how"], status="ok")
    others = [h for h in hits[1:] if h["ident"] != best["ident"]]
    if others and others[0]["rank"] == best["rank"]:
        res["status"] = "AMBIGUOUS"
    res["alternatives"] = "; ".join(str(h["path"]) for h in others[:3])
    return res


def find(key, year=None):
    """Path of input `key` (a file, a zip or an unzipped folder), or None. See locate()."""
    return locate(key, year)["path"]


def missing_hint(key, year=None):
    """What to download and where to put it, for an input that was not found."""
    s = spec(key, year)
    where = ", ".join(str(p) for p, _ in _roots()) or "a folder listed in CONFIG['SEARCH_DIRS']"
    return (f"{key} not found. Needed: {s['what']}, downloaded as '{s['download_as']}' from {s['source']}. "
            f"Put it in one of {where} (any file name works) or set CONFIG['{key}'] to its path.")


def as_year(v):
    """A year from a table cell: None for missing, else an int."""
    return None if v is None or pd.isna(v) else int(v)


def file_table():
    """One row per input (per OEWS year): found or not, which file, and how it was found."""
    _index(refresh=True)
    rows = []
    for key, s in FILES.items():
        for y in (CONFIG["OEWS_YEARS"] if "{yy" in s["download_as"] else [None]):
            r = locate(key, y)
            rows.append({"input": key if y is None else f"{key} {y}", "key": key, "year": y,
                         "status": r["status"], "found_by": r["found_by"],
                         "path": "" if r["path"] is None else str(r["path"]), "member": r["member"] or "",
                         "download_name": r["download_as"], "what": r["what"], "used_by": r["used_by"],
                         "alternatives": r["alternatives"], "note": r["note"].strip()})
    return pd.DataFrame(rows)


def show_files():
    """Print which file every input resolves to, and what to do about the ones that failed."""
    t = file_table()
    view = t.assign(file=[p + (f" :: {m}" if m else "") for p, m in zip(t.path, t.member)])
    with pd.option_context("display.max_colwidth", 90, "display.width", 250):
        print(view[["input", "status", "found_by", "file"]].to_string(index=False))
    for r in t.itertuples(index=False):
        if r.status == "MISSING":
            log(missing_hint(r.key, as_year(r.year)) + (f" ({r.note})" if r.note else ""))
            continue
        if r.status == "AMBIGUOUS":
            log(f"{r.input}: more than one file fits; using {r.path}. Others: {r.alternatives}. "
                f"Set CONFIG['{r.key}'] to choose.")
        if r.note:
            log(f"{r.input}: {r.note} Using {r.path}.")
    return t


# ------------------------------------------------------------------- reading input tables
def _head(src, n=1 << 18):
    if isinstance(src, io.BytesIO):
        return src.getvalue()[:n]
    with open(src, "rb") as fh:
        return fh.read(n)


def _header_row(rows, hint, exact=True):
    """Index of the header row among a table's first rows (lists of cells): the row with a
    cell equal to `hint` (or containing it, exact=False); without a hint, the first row at
    least half as full as the fullest one, which skips title rows."""
    if not rows:
        return None
    if hint:
        def hit(v):
            return isinstance(v, str) and (squash(v) == hint if exact else hint in squash(v))
        return next((i for i, row in enumerate(rows) if any(hit(v) for v in row)), None)
    filled = np.array([sum(isinstance(v, str) and v.strip() != "" for v in row) for row in rows])
    return int(np.argmax(filled >= max(2, filled.max() / 2)))


def _passes(hint):
    """Header searches in order: exact cell, cell containing the hint, then no hint."""
    return ([(hint, True), (hint, False)] if hint else []) + [(None, True)]


def read_table(path, member=None, header_hint=None, columns=None):
    """Read one input table as text, whatever its format: csv, tab, tsv, txt, xlsx, xls, dta
    or parquet, or any of these inside a zip (member says which).

    header_hint: a squashed column name marking the header row, for files with title rows
    above it (the Census delineation file has two); in a workbook, the sheet holding it.
    columns: squashed names to keep; everything else is dropped while reading.
    """
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            if member is None:
                member = next((n for n in zf.namelist() if Path(n).suffix.lower() in TABLES
                               and "__MACOSX" not in n), None)
            if member is None:
                raise ValueError(f"{path.name} holds no table")
            data = io.BytesIO(zf.read(member))
        return _read(data, Path(member).suffix.lower(), header_hint, columns, member)
    return _read(path, path.suffix.lower(), header_hint, columns, path.name)


def _read(src, ext, header_hint, columns, label):
    keep = (lambda c: squash(c) in columns) if columns else None
    if ext in CSV_LIKE:
        lines = _head(src).decode("utf-8-sig", "replace").splitlines()[:40]
        sep = _delim("\n".join(x for x in lines if x.strip()))
        cells = [x.split(sep) for x in lines]
        skip = next(r for r in (_header_row(cells, h, e) for h, e in _passes(header_hint)) if r is not None) if cells else 0
        for enc in ("utf-8-sig", "latin-1"):
            try:
                if isinstance(src, io.BytesIO):
                    src.seek(0)
                return pd.read_csv(src, sep=sep, dtype=str, skiprows=skip, usecols=keep,
                                   encoding=enc, low_memory=False)
            except UnicodeDecodeError:
                continue
    if ext in EXCEL:
        with pd.ExcelFile(src) as book:
            tops = {sh: book.parse(sh, header=None, nrows=40, dtype=str).values.tolist() for sh in book.sheet_names}
            for hint, exact in _passes(header_hint):
                for sheet, top in tops.items():
                    row = _header_row(top, hint, exact)
                    if row is not None:
                        return book.parse(sheet, header=row, dtype=str, usecols=keep)
        raise ValueError(f"{label}: the workbook has no data")
    if ext == ".dta":
        df = pd.read_stata(src, convert_categoricals=False)
    elif ext == ".parquet":
        df = pd.read_parquet(src)
    else:
        raise ValueError(f"{label}: cannot read {ext or 'extension-less'} files")
    if keep:
        df = df[[c for c in df.columns if keep(c)]]
    return df.astype(str).where(df.notna())


def pick(df, *names, contains=()):
    """The first column of df whose squashed name is one of `names`, else the first whose
    squashed name contains every string in `contains`; None if there is none."""
    sq = {squash(c): c for c in df.columns}
    hit = next((sq[n] for n in names if n in sq), None)
    if hit is None and contains:
        hit = next((c for s, c in sq.items() if all(x in s for x in contains)), None)
    return hit


def digits(s):
    """Codes read as text or as numbers ('01', 1.0, '10180.0') as plain digit strings."""
    return pd.Series(s).astype(str).str.strip().str.replace(r"\.0+$", "", regex=True)


def norm_gvkey(s):
    """gvkey as Compustat writes it, six digits with leading zeros: 1004, '1004.0' and
    '001004' all become '001004'. Anything that is not a number becomes missing."""
    s = digits(s)
    return s.where(s.str.fullmatch(r"\d{1,6}").fillna(False).astype(bool)).str.zfill(6)


def _source_id(loc):
    p = Path(loc["path"])
    if p.is_dir():
        rx = re.compile(loc.get("member") or ".", re.I)
        size = sum(f.stat().st_size for f in p.iterdir() if f.is_file() and rx.search(f.name.lower()))
    else:
        size = p.stat().st_size
    return {"member": loc.get("member"), "size": size}


def _stamp(out, loc):
    Path(f"{out}.source.json").write_text(json.dumps(_source_id(loc)))


def _up_to_date(out, loc):
    """True when `out` was built from this very source; an output built before sources
    were recorded is trusted once, and recorded."""
    out, side = Path(out), Path(f"{out}.source.json")
    if not out.exists():
        return False
    if not side.exists():
        _stamp(out, loc)
        return True
    return json.loads(side.read_text()) == _source_id(loc)


def dipi_measure(df):
    """The DIPI column module 07 validates against: CONFIG['DIPI_MEASURE'] when present,
    else the closest employee-liberalism column (10-year window first), else None."""
    cols = {squash(c): c for c in df.columns}
    want = squash(CONFIG.get("DIPI_MEASURE") or "")
    if want in cols:
        return cols[want]
    for test in (("emp", "liberal", "10"), ("emp", "liberal"), ("liberal",)):
        hit = next((c for s, c in cols.items() if all(t in s for t in test)), None)
        if hit is not None:
            return hit
    return None


_ID_COLUMNS = {"cusip", "cik", "tic", "ticker", "sic", "sich", "naics", "naicsh", "permno", "permco",
               "conm", "coname", "company", "companyname", "name", "firm", "firmname"}


def load_dipi(refresh=False):
    """DIPI at its declared grain (gvkey, year), built once into canonical/dipi.parquet.

    Why: the download's name, format (csv, tab, xlsx, zip) and column spelling (GVKEY or
    gvkey, year or fyear) all vary; this settles them in one place. gvkeys are zero-padded
    to six digits as in Compustat, or no firm would ever match. Rebuilt when the source
    file changes; returns None when no DIPI file is found.
    """
    loc = locate("DIPI")
    if loc["path"] is None:
        log(missing_hint("DIPI"))
        return None
    out = Path(CONFIG["CANONICAL"]) / "dipi.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not refresh and _up_to_date(out, loc):
        return pd.read_parquet(out)
    raw = read_table(loc["path"], loc["member"], header_hint="gvkey")
    gv = pick(raw, "gvkey", contains=("gvkey",))
    yr = pick(raw, "year", "fyear", "fiscalyear", "datayear", "yr")
    if gv is None or yr is None:
        raise ValueError(f"DIPI file {loc['path'].name} needs a gvkey and a year column; "
                         f"its columns are {list(raw.columns)[:25]}")
    d = raw.rename(columns={gv: "gvkey", yr: "year"})
    d["gvkey"] = norm_gvkey(d.gvkey)
    d["year"] = pd.to_numeric(d.year, errors="coerce")
    d = d.dropna(subset=["gvkey", "year"]).astype({"year": int})
    for col in d.columns.drop(["gvkey", "year"]):
        if squash(col) in _ID_COLUMNS or not d[col].notna().any():
            continue
        num = pd.to_numeric(d[col], errors="coerce")
        if num.notna().sum() >= 0.95 * d[col].notna().sum():
            d[col] = num
    d = d.drop_duplicates()
    dup = int(d.duplicated(["gvkey", "year"]).sum())
    if dup:
        log(f"WARNING DIPI: {dup:,} repeated gvkey-years; averaged to keep one row per firm-year")
        rest = [c for c in d.columns if c not in ("gvkey", "year")]
        d = d.groupby(["gvkey", "year"], as_index=False).agg(
            {c: "mean" if pd.api.types.is_numeric_dtype(d[c]) else "first" for c in rest})
    d.to_parquet(out, index=False)
    _stamp(out, loc)
    log(f"canonical dipi.parquet: {len(d):,} firm-years, {d.gvkey.nunique():,} firms, "
        f"{d.year.min()}-{d.year.max()}, measure {dipi_measure(d)} (from {loc['path'].name})")
    return d


def cbsa_delineation():
    """Census List 1 (or NBER's cbsa2fipsxw copy of it) as (one row per CBSA, one row per
    county), or None when no file is found.

    Raises ValueError, naming the columns it did find, when the file has no 'CBSA Code'
    header. The county table is None for a file without county codes (List 2 is principal
    cities, not counties; the election merge needs List 1).
    """
    loc = locate("CBSA_REFERENCE")
    if loc["path"] is None:
        return None
    raw = read_table(loc["path"], loc["member"], header_hint="cbsacode")
    code, title = pick(raw, "cbsacode"), pick(raw, "cbsatitle", "cbsaname")
    if code is None or title is None:
        raise ValueError(f"{loc['path'].name} has no 'CBSA Code' and 'CBSA Title' header; "
                         f"columns: {list(raw.columns)[:12]}")
    typ = pick(raw, contains=("metropolitan", "micropolitan"))
    st = pick(raw, "fipsstatecode", "statefips", "fipsstate")
    cty = pick(raw, "fipscountycode", "countyfips", "fipscounty")
    d = pd.DataFrame({"cbsa": digits(raw[code]), "cbsa_title": raw[title].str.strip(),
                      "cbsa_type": raw[typ].str.strip() if typ else None})
    ok = d.cbsa.str.fullmatch(r"\d{5}").fillna(False).astype(bool)
    cbsa = d[ok].drop_duplicates("cbsa").reset_index(drop=True)
    county = None
    if st and cty:
        c = pd.DataFrame({"cbsa": d.cbsa, "county_fips": digits(raw[st]).str.zfill(2) + digits(raw[cty]).str.zfill(3)})
        for out_col, names in (("county_name", ("countycountyequivalent", "countyname")),
                               ("state_name", ("statename",)), ("central_outlying", ("centraloutlyingcounty",))):
            col = pick(raw, *names)
            c[out_col] = raw[col].str.strip() if col else None
        keep = ok & c.county_fips.str.fullmatch(r"\d{5}").fillna(False).astype(bool)
        county = c[keep].drop_duplicates(["cbsa", "county_fips"]).reset_index(drop=True)
    return cbsa, county


# ------------------------------------------------------------------- reading raw VRscores
def sniff_delim(path):
    """VRscores ships .tab files that are comma separated; never trust the extension."""
    with open(path, "rb") as fh:
        head = fh.readline(200000).decode("utf-8", "ignore")
    counts = {",": head.count(","), "\t": head.count("\t"), ";": head.count(";")}
    return max(counts, key=counts.get)


YEAR_RE = re.compile(r"(?<!\d)(20[0-4]\d)(?!\d)")


def panel_to_parquet(c, src, out_path, pattern=None, add_year_from_name=True):
    """One VRscores panel (a zip, its unzipped folder, or loose files) to one Parquet.

    Unpacks one member at a time, converts it with DuckDB, deletes the unpacked copy.
    Why: the employer ZIP holds 13 files of up to 175 MB uncompressed. This keeps peak
    disk use at one file and peak memory near zero. `pattern` (FILES[key]["members"]) picks
    the panel's files, so other files next to them are left alone. The Parquet is rebuilt
    when the source changes, so a corrected download is picked up without deleting anything.
    """
    src, out_path = Path(src), Path(out_path)
    pattern = pattern or r"\.(csv|tab|parquet)$"
    if _up_to_date(out_path, {"path": src, "member": pattern}):
        return out_path
    rx = re.compile(pattern, re.I)
    tmp = out_path.parent / "_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    zf = zipfile.ZipFile(src) if src.suffix.lower() == ".zip" else None
    try:
        if zf is not None:
            names = sorted(n for n in zf.namelist() if rx.search(Path(n).name) and "__MACOSX" not in n)
        else:
            names = sorted(str(f) for f in src.iterdir() if f.is_file() and rx.search(f.name))
        if not names:
            raise ValueError(f"{src} holds no files named like {pattern}")
        for name in names:
            if zf is not None:
                dest = tmp / Path(name).name
                with zf.open(name) as fin, open(dest, "wb") as fout:
                    shutil.copyfileobj(fin, fout, 1 << 24)
            else:
                dest = Path(name)
            if dest.suffix.lower() == ".parquet":
                reader = f"read_parquet('{sqlp(dest)}')"
            else:
                reader = (f"read_csv('{sqlp(dest)}', header=true, auto_detect=true, "
                          f"sample_size=-1, delim='{sniff_delim(dest)}')")
            cols = [x.lower() for x in q(c, f"DESCRIBE SELECT * FROM {reader}")["column_name"]]
            m = YEAR_RE.search(Path(name).name)
            add = f", {int(m.group(1))} AS year" if (add_year_from_name and "year" not in cols and m) else ""
            part = tmp / (Path(name).stem + ".part.parquet")
            c.execute(f"COPY (SELECT *{add} FROM {reader}) TO '{sqlp(part)}' (FORMAT parquet, COMPRESSION zstd)")
            if zf is not None:
                dest.unlink()
            parts.append(part)
    finally:
        if zf is not None:
            zf.close()
    lst = "[" + ", ".join(f"'{sqlp(p)}'" for p in parts) + "]"
    c.execute(f"COPY (SELECT * FROM read_parquet({lst}, union_by_name=true)) "
              f"TO '{sqlp(out_path)}' (FORMAT parquet, COMPRESSION zstd)")
    for p in parts:
        p.unlink()
    _stamp(out_path, {"path": src, "member": pattern})
    n = q1(c, f"SELECT count(*) FROM read_parquet('{sqlp(out_path)}')")
    log(f"canonical {out_path.name}: {n:,} rows from {src.name}")
    return out_path


zip_to_parquet = panel_to_parquet   # the name earlier notebook cells import


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


# ------------------------------------------------------------------- running modules
def run(step, **kwargs):
    """Run module `step` ("01" to "10") from the files next to this one and return what its
    main() returns, e.g. run("04", year=2016). Same as running the file itself."""
    import runpy
    here = Path(__file__).resolve().parent
    hits = sorted(here.glob(f"{int(step):02d}_*.py"))
    if not hits:
        raise FileNotFoundError(f"no module {int(step):02d}_*.py in {here}")
    log(f"--- {hits[0].name}")
    return runpy.run_path(str(hits[0]), run_name="polisy_module")["main"](**kwargs)
