# -*- coding: utf-8 -*-
"""05 Downloads: fetch the open datasets, skip politely when blocked.

What it does: downloads O*NET, AIOE, the Census delineation file, ACS metro tables and
(optionally) PatentsView into data/raw.
Why: every later module reads from data/raw, so a blocked download should fail here with
an instruction, not halfway through an analysis.
Expect: files in data/raw and a printed line per source. Anything that fails prints the
manual URL to use instead.
Diagnostics: run 01 again afterwards; every source you intend to use should say ok.
"""
import json
import urllib.request
from pathlib import Path
import pandas as pd
from polisy_core import CONFIG, paths, log

SOURCES = {
    "onet": ("https://www.onetcenter.org/dl_files/database/db_29_0_text.zip", "onet_db.zip",
             "https://www.onetcenter.org/database.html"),
    "aioe": ("https://raw.githubusercontent.com/AIOE-Data/AIOE/master/AIOE_DataAppendix.csv", "aioe.csv",
             "https://github.com/AIOE-Data/AIOE"),
    "cbsa_delineation": ("https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/"
                         "2023/delineation-files/list1_2023.xlsx", "list1_2023.xlsx",
                         "https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html"),
}
ACS_VARS = ["B01003_001E", "B19013_001E", "B23025_004E", "B15003_022E", "B15003_001E", "B01002_001E"]


def fetch(url, dest: Path, manual_url=""):
    if dest.exists() and dest.stat().st_size > 0:
        log(f"{dest.name}: already present")
        return dest
    try:
        req = urllib.request.Request(url, headers={"User-Agent": CONFIG["USER_AGENT"]})
        with urllib.request.urlopen(req, timeout=180) as r, open(dest, "wb") as f:
            f.write(r.read())
        log(f"{dest.name}: downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    except Exception as e:
        log(f"{dest.name}: download failed ({e}). Download by hand from {manual_url or url} into {dest.parent}")
        return None


def acs_metro(P, years=(2012, 2015, 2018, 2022)):
    for y in years:
        dest = P["RAW"] / f"acs1_{y}.json"
        if dest.exists():
            continue
        url = (f"https://api.census.gov/data/{y}/acs/acs1?get=NAME," + ",".join(ACS_VARS) +
               "&for=metropolitan%20statistical%20area/micropolitan%20statistical%20area:*")
        if CONFIG["CENSUS_API_KEY"]:
            url += f"&key={CONFIG['CENSUS_API_KEY']}"
        fetch(url, dest, "https://www.census.gov/data/developers/data-sets/acs-1year.html")


def main(patentsview=False):
    P = paths()
    for key, (url, name, manual) in SOURCES.items():
        fetch(url, P["RAW"] / name, manual)
    acs_metro(P)
    if patentsview:
        base = "https://s3.amazonaws.com/data.patentsview.org/download/"
        for f in ("g_patent.tsv.zip", "g_inventor_disambiguated.tsv.zip",
                  "g_location_disambiguated.tsv.zip", "g_cpc_current.tsv.zip"):
            fetch(base + f, P["RAW"] / f, "https://patentsview.org/download/data-download-tables")
    log("OEWS zips are not downloaded here: BLS blocks scripted requests intermittently. "
        "Put oesm24nat.zip, oesm24ma.zip and oesm24in4.zip in data/raw by hand.")
    return True


if __name__ == "__main__":
    main()
