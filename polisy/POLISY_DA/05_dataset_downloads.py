# -*- coding: utf-8 -*-
"""05 Downloads: fetch the open datasets, skip politely when blocked.

What it does: downloads O*NET, AIOE, the Census delineation file, ACS metro tables and
(optionally) PatentsView into data/raw.
Why: every later module reads from data/raw, so a blocked download should fail here with
an instruction, not halfway through an analysis.
Expect: files in data/raw and a printed line per source. Anything that fails prints the
manual URL to use instead.
Files you already have are not downloaded again, under whatever name and in whichever
search folder they sit (polisy_core.find), so a manual download is never overwritten.
Downloads keep their original file names.
Run it first: module 04 needs the Census file and module 06 the O*NET and AIOE files.
Diagnostics: run 01 again afterwards; every source you intend to use should say ok.
"""
import json
import urllib.request
from pathlib import Path
import pandas as pd
from polisy_core import CONFIG, paths, log, locate

# input key: (URLs tried in order, page to download from by hand)
SOURCES = {
    "ONET": (["https://www.onetcenter.org/dl_files/database/db_29_0_text.zip"],
             "https://www.onetcenter.org/database.html"),
    "AIOE": (["https://raw.githubusercontent.com/AIOE-Data/AIOE/main/AIOE_DataAppendix.xlsx",
              "https://raw.githubusercontent.com/AIOE-Data/AIOE/master/AIOE_DataAppendix.xlsx"],
             "https://github.com/AIOE-Data/AIOE"),
    "CBSA_REFERENCE": (["https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/"
                        "2023/delineation-files/list1_2023.xlsx"],
                       "https://www.census.gov/geographies/reference-files/time-series/demo/metro-micro/delineation-files.html"),
}
ACS_VARS = ["B01003_001E", "B19013_001E", "B23025_004E", "B15003_022E", "B15003_001E", "B01002_001E"]


def fetch(url, dest: Path, manual_url="", quiet=False):
    if dest.exists() and dest.stat().st_size > 0:
        log(f"{dest.name}: already present")
        return dest
    try:
        req = urllib.request.Request(url, headers={"User-Agent": CONFIG["USER_AGENT"]})
        with urllib.request.urlopen(req, timeout=180) as r:
            data = r.read()
        dest.write_bytes(data)
        log(f"{dest.name}: downloaded ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest
    except Exception as e:
        if not quiet:
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
    for key, (urls, manual) in SOURCES.items():
        have = locate(key)
        if have["path"] is not None:
            log(f"{key}: already have {have['path']} (found by {have['found_by']})")
            continue
        for i, url in enumerate(urls):
            if fetch(url, P["RAW"] / Path(url).name, manual if i == len(urls) - 1 else "", quiet=i < len(urls) - 1):
                break
    acs_metro(P)
    if patentsview:
        base = "https://s3.amazonaws.com/data.patentsview.org/download/"
        for f in ("g_patent.tsv.zip", "g_inventor_disambiguated.tsv.zip",
                  "g_location_disambiguated.tsv.zip", "g_cpc_current.tsv.zip"):
            fetch(base + f, P["RAW"] / f, "https://patentsview.org/download/data-download-tables")
    yy = [str(y)[2:] for y in CONFIG["OEWS_YEARS"]]
    log("OEWS zips are not downloaded here: BLS blocks scripted requests intermittently. Download "
        + ", ".join(f"oesm{y}nat.zip, oesm{y}ma.zip and oesm{y}in4.zip" for y in yy)
        + " by hand from https://www.bls.gov/oes/tables.htm into /content or data/raw; any name variant works.")
    return True


if __name__ == "__main__":
    main()
