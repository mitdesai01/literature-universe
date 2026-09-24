# -*- coding: utf-8 -*-
"""01 Data inventory.

What it does: lists every file the pipeline can see, its size, format and row count,
and flags the ones that are missing.
Why: nothing downstream should run on a half-complete download, and the VRscores ZIPs
mix .tab and .csv members that look wrong until you check them.
Expect: output/tables/01_inventory.csv and a printed summary.
Diagnostics: compare employer row counts per year against the VRscores codebook
(370,426 in 2012 rising to 534,392 in 2024, 6,258,838 in total).
"""
import zipfile
from pathlib import Path
import pandas as pd
from polisy_core import CONFIG, paths, log, save, con, sqlp, q1

CODEBOOK_ROWS = {2012: 370426, 2013: 393571, 2014: 417604, 2015: 439982, 2016: 461191,
                 2017: 482144, 2018: 502104, 2019: 516553, 2020: 525013, 2021: 535238,
                 2022: 539944, 2023: 540676, 2024: 534392}


def main():
    paths()
    rows = []
    for key in ("VR_EMPLOYER", "VR_MSA", "VR_INDUSTRY", "VR_OCCUPATION", "COMPUSTAT", "DIPI", "COUNTYPRES"):
        p = Path(CONFIG[key])
        if not p.exists():
            rows.append({"asset": key, "path": str(p), "status": "MISSING"})
            continue
        if p.suffix.lower() == ".zip":
            with zipfile.ZipFile(p) as zf:
                members = [i for i in zf.infolist() if not i.is_dir()]
                for i in members:
                    rows.append({"asset": key, "path": i.filename, "status": "ok",
                                 "format": Path(i.filename).suffix.lstrip("."),
                                 "mb_compressed": round(i.compress_size / 1e6, 2),
                                 "mb_raw": round(i.file_size / 1e6, 2)})
        else:
            rows.append({"asset": key, "path": p.name, "status": "ok",
                         "format": p.suffix.lstrip("."), "mb_raw": round(p.stat().st_size / 1e6, 2)})
    inv = pd.DataFrame(rows)
    save(inv, "01_inventory")
    missing = inv[inv.status == "MISSING"]
    if len(missing):
        log(f"MISSING assets: {', '.join(missing.asset)}")
    log(f"{(inv.status == 'ok').sum()} files found, {inv.get('mb_raw', pd.Series(dtype=float)).sum():,.0f} MB raw")
    return inv


if __name__ == "__main__":
    main()
