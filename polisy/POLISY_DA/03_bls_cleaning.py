# -*- coding: utf-8 -*-
"""03 BLS OEWS cleaning.

What it does: reads the OEWS national, metro and industry zips, keeps detailed
occupations only, and writes three canonical Parquet tables.
Why: OEWS stacks totals, major, minor, broad and detailed rows in one column. Summing
without filtering O_GROUP == 'detailed' double counts employment by roughly a factor of
three. Metro rows are AREA_TYPE 4, not 2.
Expect: canonical/oews_national.parquet (about 830 occupations), oews_metro.parquet
(about 141k rows), oews_industry.parquet (4-digit NAICS, about 83k rows).
Diagnostics: total detailed employment should be close to the published US total
(about 151m in May 2024); metro count should be near 390.
"""
import zipfile
from pathlib import Path
import pandas as pd
from polisy_core import CONFIG, paths, log, save

LEVELS = {"national": ("nat", "national_m"), "msa": ("ma", "msa_m"), "industry": ("in4", "nat4d_m")}


def read_level(raw: Path, year: int, level: str):
    suffix, prefer = LEVELS[level]
    zpath = raw / f"oesm{str(year)[2:]}{suffix}.zip"
    if not zpath.exists():
        log(f"OEWS {level} {year}: {zpath.name} not found in {raw}")
        return None
    with zipfile.ZipFile(zpath) as zf:
        sheets = [n for n in zf.namelist() if n.lower().endswith((".xlsx", ".xls"))
                  and "file_description" not in n.lower()]
        member = next((n for n in sheets if prefer in Path(n).name.lower()), sheets[0])
        df = pd.read_excel(zf.open(member), dtype=str)
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
        frames = [read_level(P["RAW"], y, level) for y in CONFIG["OEWS_YEARS"]]
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
