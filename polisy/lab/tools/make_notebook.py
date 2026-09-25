# -*- coding: utf-8 -*-
"""Write the Colab notebook and the same code as a plain script, from one list of cells (so they cannot drift).

    python make_notebook.py      # -> ../notebooks/POLISY_AI_Innovation_Lab.ipynb and ../notebooks/polisy_lab_colab.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "notebooks"

CELLS = [
    ("md", """# POLISY Lab: Political Ideology × AI × Innovation

Runs the POLISY lab from start to finish: finds your data files, turns each source into canonical tables, links them,
runs the analyses and writes the interactive website (ready for GitHub Pages).

**Before you start**

1. In Google Drive, make a folder `POLISY` with a folder `data` inside it.
2. Put your downloads in `POLISY/data` as they came. Zips are fine and names do not matter: the VRscores report,
   `daioe-v1.0.0-scores.zip`, the IRS county migration zip, the CSPP file.
3. Put `POLISY_lab_code.zip` in `POLISY` (or upload it to the Colab file panel).

Each step below is one cell; run them in order. The technical report (`lab/docs/TECHNICAL_REPORT.md`, also the site's
Methods tab) explains every stage, dataset and link."""),
    ("code", """# 1. Packages Colab lacks, and Google Drive
import subprocess, sys
if subprocess.run([sys.executable, "-m", "pip", "install", "-q", "duckdb", "polars", "rapidfuzz", "naics", "py7zr", "markdown"]).returncode:
    print("pip could not install the packages; install duckdb polars rapidfuzz naics py7zr markdown by hand")
try:
    from google.colab import drive
    drive.mount("/content/drive")
except ImportError:
    print("Not in Colab: set BASE below to your POLISY folder.")"""),
    ("code", """# 2. Folders and code. Change BASE if your POLISY folder is elsewhere.
import os, shutil, zipfile
from pathlib import Path

BASE = Path(os.environ.get("POLISY_BASE", "/content/drive/MyDrive/POLISY"))
DATA = BASE / "data"                                        # your downloads: searched, never modified
os.environ["POLISY_LAB_ROOT"] = str(BASE / "lab")           # everything the lab writes (kept in Drive)
os.environ.setdefault("POLISY_ROOT", "/content/polisy")     # POLISY_DA's own outputs, as in the POLISY_Lab notebook

CODE = Path("/content/polisy_code")
found = [p for p in (BASE / "POLISY_lab_code.zip", Path("/content/POLISY_lab_code.zip")) if p.exists()]
if found:
    shutil.rmtree(CODE, ignore_errors=True)
    zipfile.ZipFile(found[0]).extractall(CODE)
elif not (CODE / "lab").exists():                            # no zip: take the code from GitHub
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "claude/dipi-cbsa-reference-lookup-bx46cx",
                    "https://github.com/mitdesai01/literature-universe", "/content/literature-universe"], check=True)
    shutil.copytree("/content/literature-universe/polisy/POLISY_DA", CODE / "POLISY_DA")
    shutil.copytree("/content/literature-universe/polisy/lab", CODE / "lab")
for name in [m for m in sys.modules if m.startswith(("polisy_core", "polisy_lab"))]:
    del sys.modules[name]                                   # never keep an older copy loaded
sys.path[:0] = [str(CODE / "POLISY_DA"), str(CODE / "lab")]

import polisy_core as pc
import polisy_lab
from polisy_lab import run_all, LAB

pc.CONFIG["SEARCH_DIRS"] = [str(DATA), "/content"] + [d for d in pc.CONFIG["SEARCH_DIRS"] if str(d) not in (str(DATA), "/content")]
# If the inventory (step 5) picks the wrong file for an input, point it at the right one, e.g.
# pc.CONFIG["LAB_CSPP_DATA"] = str(DATA / "my_cspp_file.csv")          # key: LAB_<SOURCE>_<ROLE>
print("polisy_core", pc.__version__, "| polisy_lab", polisy_lab.__version__, "| lab folder:", LAB["SITE"].parent)"""),
    ("code", """# 3. Optional: POLISY_DA first.
# Module 05 downloads the Census CBSA delineation file (links metros to CBSAs) and O*NET titles (more occupation links).
# Modules 01-04 build the VRscores panels from the raw VRscores files; without them the lab reads the VRscores HTML report.
RUN_POLISY_DA = False
if RUN_POLISY_DA:
    for module in ("05", "01", "02", "03", "04"):
        pc.run(module)"""),
    ("code", """# 4. Download the open datasets that are not in your folders yet (AIOE, DAIOE, BTOS, CSPP, IRS state and county files).
# Files you already have, under any name and inside zips, are skipped. PatentsView is not downloaded (patentsview=False).
run_all(stages=("fetch",), fetch=True, patentsview=False)"""),
    ("code", """# 5. What was found for every source and role, and what each file contains
import json
import pandas as pd

run_all(stages=("inventory", "profile"), fetch=False)
res = json.loads((LAB["RESULTS"] / "results.json").read_text())
pd.set_option("display.max_colwidth", 120)
pd.DataFrame([{"source": s, **r} for s, m in res["catalog"].items() for r in m["roles"]])[["source", "role", "files", "paths"]]"""),
    ("code", """# 6. Canonical tables, then the linked panels. Each line below is one join and how much of it matched.
run_all(stages=("adapt", "link"), fetch=False)
pd.DataFrame(json.loads((LAB["RESULTS"] / "results.json").read_text())["diagnostics"])[["step", "key", "matched", "total", "share", "unit", "note"]]"""),
    ("code", """# 7. Analyses: findings, graded, in the lab's reading order
run_all(stages=("analyze",), fetch=False)
findings = pd.read_csv(LAB["RESULTS"] / "findings.csv").sort_values("rank")
findings[["rank", "strength", "theme", "title"]]"""),
    ("code", """# 8. The website: index.html (Plotly from a CDN) and polisy_lab_offline.html (works without internet)
run_all(stages=("site",), fetch=False)
site = LAB["SITE"]
print(site / "index.html")"""),
    ("code", """# 9. Open the site from this Colab session (the link works while the notebook is running)
subprocess.Popen([sys.executable, "-m", "http.server", "8765", "--directory", str(site)],
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    from google.colab.output import eval_js
    print("Open the POLISY lab:", eval_js("google.colab.kernel.proxyPort(8765)"))
except ImportError:
    print("Open", site / "index.html", "in a browser")"""),
    ("code", """# 10. Zip the site and the results into the POLISY folder (they are also in POLISY/lab)
site_zip = shutil.make_archive(str(BASE / "POLISY_lab_site"), "zip", LAB["SITE"].parent, "site")
results_zip = shutil.make_archive(str(BASE / "POLISY_lab_results"), "zip", LAB["SITE"].parent, "results")
print(site_zip, results_zip, sep="\\n")"""),
    ("md", """## Publish on GitHub Pages

1. Unzip `POLISY_lab_site.zip` into a GitHub repository, for example as the folder `docs/`.
2. In the repository: Settings, then Pages, then "Deploy from a branch"; choose the branch and `/docs`; save.
3. After a minute the lab is at `https://<user>.github.io/<repository>/`.

To rebuild the site after changing an analysis, run steps 7 and 8 again; nothing else needs to rerun."""),
    ("code", """# 11. Work with the tables directly: every canonical table and panel is a Parquet file (DuckDB, Polars, pandas)
import duckdb
import polars as pl

canonical, panels = LAB["CANONICAL"], LAB["PANELS"]
# Net domestic migration by year for the most and least AI-exposed fifths of counties (DuckDB reads Parquet in place)
duckdb.sql(f\"\"\"
    WITH c AS (SELECT m.year, m.net_migration_rate, m.base_returns, e.aige,
                      ntile(5) OVER (PARTITION BY m.year ORDER BY e.aige) AS fifth
               FROM '{canonical}/irs_county_year.parquet' m JOIN '{canonical}/county_exposure.parquet' e USING (county_fips)
               WHERE m.net_migration_rate IS NOT NULL)
    SELECT year, fifth, round(100 * sum(net_migration_rate * base_returns) / sum(base_returns), 2) AS net_pct
    FROM c WHERE fifth IN (1, 5) GROUP BY 1, 2 ORDER BY 1, 2\"\"\").df()"""),
    ("code", """# The occupation panel in Polars: which exposure measures lean which way (workers-weighted correlations)
occ = pl.read_parquet(panels / "panel_occupation.parquet")
measures = [c for c in ("aioe", "daioe_genai_z", "open24_human_E1_E2", "webb19_ai_score", "webb19_robot_score", "fo17_p_computerisation") if c in occ.columns]

def wcorr(df, x, y="rep_share", w="workers"):
    d = df.drop_nulls([x, y, w])
    mx, my = (d[x] * d[w]).sum() / d[w].sum(), (d[y] * d[w]).sum() / d[w].sum()
    cov = (d[w] * (d[x] - mx) * (d[y] - my)).sum()
    return cov / ((d[w] * (d[x] - mx) ** 2).sum() * (d[w] * (d[y] - my) ** 2).sum()) ** 0.5

pl.DataFrame({"measure": measures, "r_with_rep_share": [round(wcorr(occ, m), 3) for m in measures]})"""),
]


def notebook():
    cells = []
    for kind, src in CELLS:
        lines = src.split("\n")
        source = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]
        if kind == "md":
            cells.append({"cell_type": "markdown", "metadata": {}, "source": source})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source})
    return {"cells": cells, "metadata": {"colab": {"provenance": [], "toc_visible": True},
                                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def script():
    parts = ['# -*- coding: utf-8 -*-\n"""POLISY Lab: the Colab notebook as a script (cells marked # %%). Generated by tools/make_notebook.py."""\n']
    for kind, src in CELLS:
        if kind == "md":
            parts.append("# %% [markdown]\n" + "\n".join("# " + ln if ln else "#" for ln in src.split("\n")) + "\n")
        else:
            parts.append("# %%\n" + src + "\n")
    return "\n".join(parts)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    (OUT / "POLISY_AI_Innovation_Lab.ipynb").write_text(json.dumps(notebook(), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (OUT / "polisy_lab_colab.py").write_text(script(), encoding="utf-8")
    print("wrote", OUT / "POLISY_AI_Innovation_Lab.ipynb", "and", OUT / "polisy_lab_colab.py")
