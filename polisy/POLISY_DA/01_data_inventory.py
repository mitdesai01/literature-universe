# -*- coding: utf-8 -*-
"""01 Data inventory.

What it does: finds every input, whatever it came to be called between the download page
and Colab, lists what is inside each file, and reads DIPI and the CBSA reference once, so
a file that is there but unusable fails here rather than in module 04 or 07.
Why: nothing downstream should run on a half-complete download, and the VRscores ZIPs
mix .tab and .csv members that look wrong until you check them. File names drift: "(1)"
copies, renamed or unzipped zips, WRDS's random names. polisy_core.FILES says what each
input looks like; this module shows which file was matched to it, and how.
Expect: output/tables/01_files.csv (one row per input), 01_inventory.csv (every file
inside every zip), 01_read_checks.csv, canonical/dipi.parquet and a printed summary.
Diagnostics: every input should say ok. MISSING prints the download name and where to
get it; AMBIGUOUS means two different files fit, so set CONFIG[key] to pick one; "similar
name" and "file contents" matches are fine but worth one look at the file column.
Compare employer row counts per year against the VRscores codebook (370,426 in 2012
rising to 534,392 in 2024, 6,258,838 in total) once module 02 has run.
"""
import re
import zipfile
from pathlib import Path
import pandas as pd
import polisy_core
from polisy_core import (paths, log, save, show_files, spec, as_year, locate, load_dipi, dipi_measure,
                         cbsa_delineation)

CODEBOOK_ROWS = {2012: 370426, 2013: 393571, 2014: 417604, 2015: 439982, 2016: 461191,
                 2017: 482144, 2018: 502104, 2019: 516553, 2020: 525013, 2021: 535238,
                 2022: 539944, 2023: 540676, 2024: 534392}


def contents(src, pattern=None):
    """Inventory rows for the files inside a zip or folder (those matching `pattern`), or
    for the file itself."""
    p = Path(src)
    rx = re.compile(pattern, re.I) if pattern else None
    if p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p) as zf:
            return [{"path": i.filename, "format": Path(i.filename).suffix.lstrip("."),
                     "mb_compressed": round(i.compress_size / 1e6, 2), "mb_raw": round(i.file_size / 1e6, 2)}
                    for i in zf.infolist() if not i.is_dir()]
    if p.is_dir():
        return [{"path": f.name, "format": f.suffix.lstrip("."), "mb_raw": round(f.stat().st_size / 1e6, 2)}
                for f in sorted(p.iterdir()) if f.is_file() and (rx is None or rx.search(f.name))]
    return [{"path": p.name, "format": p.suffix.lstrip("."), "mb_raw": round(p.stat().st_size / 1e6, 2)}]


def read_checks():
    """Read DIPI and the CBSA reference end to end; a readable file is not a usable one."""
    out = []
    try:
        d = load_dipi() if locate("DIPI")["path"] is not None else None
        if d is not None:
            m = dipi_measure(d)
            out.append({"input": "DIPI", "readable": m is not None,
                        "detail": f"{len(d):,} firm-years, {d.gvkey.nunique():,} firms, {d.year.min()}-{d.year.max()}; "
                                  + (f"module 07 uses column '{m}'" if m else
                                     "no liberalism column: set CONFIG['DIPI_MEASURE'] to the column to use")})
    except Exception as e:
        out.append({"input": "DIPI", "readable": False, "detail": f"{type(e).__name__}: {e}"})
    try:
        got = cbsa_delineation()
        if got is not None:
            cbsa, county = got
            out.append({"input": "CBSA_REFERENCE", "readable": county is not None,
                        "detail": f"{len(cbsa):,} CBSAs, " + (f"{len(county):,} county links" if county is not None
                                                             else "no county codes: this is not List 1")})
    except Exception as e:
        out.append({"input": "CBSA_REFERENCE", "readable": False, "detail": f"{type(e).__name__}: {e}"})
    return pd.DataFrame(out, columns=["input", "readable", "detail"])


def main():
    paths()
    log(f"polisy_core {polisy_core.__version__} from {polisy_core.__file__}")
    files = show_files()
    save(files, "01_files", "which file each input resolved to, and how")
    rows = []
    for r in files.itertuples(index=False):
        if r.status == "MISSING":
            rows.append({"asset": r.input, "path": r.download_name, "status": "MISSING"})
            continue
        for item in contents(r.path, spec(r.key, as_year(r.year)).get("members")):
            rows.append({"asset": r.input, "status": r.status.lower(), "found_by": r.found_by, **item})
    inv = pd.DataFrame(rows)
    save(inv, "01_inventory")
    checks = save(read_checks(), "01_read_checks", "DIPI and CBSA reference read end to end")
    for c in checks.itertuples(index=False):
        log(f"{c.input}: {'ok' if c.readable else 'NOT USABLE'} - {c.detail}")
    missing = files[files.status == "MISSING"]
    if len(missing):
        log(f"MISSING inputs: {', '.join(missing.input)}")
    log(f"{(files.status != 'MISSING').sum()} of {len(files)} inputs found, "
        f"{inv.get('mb_raw', pd.Series(dtype=float)).sum():,.0f} MB raw")
    return files


if __name__ == "__main__":
    main()
