# -*- coding: utf-8 -*-
"""03 BLS OEWS cleaning.

What it does: reads the OEWS national, metro and industry zips, keeps detailed
occupations only, and writes three canonical Parquet tables.
Why: OEWS stacks totals, major, minor, broad and detailed rows in one column. Summing
without filtering O_GROUP == 'detailed' double counts employment by roughly a factor of
three. Metro rows are AREA_TYPE 4, not 2.
Expect: canonical/oews_national.parquet (about 830 occupations), oews_metro.parquet
(about 141k rows), oews_industry.parquet (4-digit NAICS, about 83k rows).
Finding the files: BLS names them oesm24nat.zip, oesm24ma.zip and oesm24in4.zip. They
can sit in data/raw or any search folder, as "(1)" copies, or unzipped (a folder such as
oesm24nat/, or the bare national_M2024_dl.xlsx); module 01 shows what was found.
Diagnostics: total detailed employment should be close to the published US total
(about 151m in May 2024); metro count should be near 390.
"""
import io
import re
import zipfile
from pathlib import Path
import pandas as pd
from polisy_core import CONFIG, paths, log, save, locate, spec, members, missing_hint

LEVELS = {"national": ("OEWS_NATIONAL", "national_m"), "msa": ("OEWS_MSA", "msa_m"),
          "industry": ("OEWS_INDUSTRY", "nat4d_m")}


def read_level(year: int, level: str):
    key, prefer = LEVELS[level]
    src = locate(key, year)["path"]
    if src is None:
        log(f"OEWS {level} {year}: {missing_hint(key, year)}")
        return None
    names = members(src)
    sheets = [n for low, n in names.items() if low.endswith((".xlsx", ".xls")) and "description" not in low]
    exact = [n for low, n in names.items() if re.search(spec(key, year)["members"], low)]
    member = next(iter(exact), None) or next((n for n in sheets if prefer in Path(n).name.lower()),
                                             sheets[0] if sheets else None)
    if member is None:
        log(f"OEWS {level} {year}: no spreadsheet inside {src}")
        return None
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as zf:
            df = pd.read_excel(io.BytesIO(zf.read(member)), dtype=str)
    else:
        df = pd.read_excel(src / member if src.is_dir() else src, dtype=str)
    log(f"OEWS {level} {year}: reading {Path(member).name} from {src.name}")
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("tot_emp", "a_median", "a_mean"):
        if col in df:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
    if "o_group" in df:
        df = df[df.o_group.str.lower() == "detailed"]
    if level == "msa" and "area_type" in df:
        df = df[df.area_type.astype(str).str.strip() == "4"]
    if level == "industry" and "i_group" in df:
        df = df[df.i_group.str.contains("digit", case=False, na=False)]
    keep = [c for c in ["area", "area_title", "naics", "naics_title", "occ_code", "occ_title",
                        "tot_emp", "a_median"] if c in df]
    out = df[keep].copy()
    out["year"] = year
    if level == "msa":
        out = out.rename(columns={"area": "cbsa"})
    if level == "industry":
        out["naics4"] = out["naics"].astype(str).str.replace(r"[^0-9]", "", regex=True).str[:4]
    log(f"OEWS {level} {year}: {len(out):,} detailed rows")
    return out


def main():
    P = paths()
    for level in LEVELS:
        frames = [read_level(y, level) for y in CONFIG["OEWS_YEARS"]]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(P["CANONICAL"] / f"oews_{level}.parquet", index=False)
        if level == "national":
            save(pd.DataFrame([{"year": y, "occupations": (df.year == y).sum(),
                                "total_employment": df.loc[df.year == y, "tot_emp"].sum()}
                               for y in sorted(df.year.unique())]), "03_oews_national_check",
                 "compare total employment with the published US total")
        if level == "msa":
            log(f"metros in OEWS: {df.cbsa.nunique()}")
    return True


if __name__ == "__main__":
    main()
