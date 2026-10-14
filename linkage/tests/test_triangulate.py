# -*- coding: utf-8 -*-
"""Code-path test for polisy_triangulate on small made-up files (fixtures only, never results).

Usage: python test_triangulate.py <folder with polisy_triangulate.py> [--with-core <POLISY_DA folder>]
Builds a fake POLISY folder, runs every step, and checks each match type, the conflict rule, hand decisions, the size
break, the frozen hash and that link building works on a VRscores file without any party column."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

code = Path(sys.argv[1]).resolve()
if "--with-core" in sys.argv:
    sys.path.insert(0, sys.argv[sys.argv.index("--with-core") + 1])
sys.path.insert(0, str(code))

tmp = Path(tempfile.mkdtemp(prefix="tri_test_"))
BASE = tmp / "POLISY"
os.environ["POLISY_BASE"] = str(BASE)
os.environ["POLISY_ROOT"] = str(tmp / "no_session_copy")       # forces the Drive copies under POLISY/polisy_da
os.environ["POLISY_LAB_ROOT"] = str(BASE / "lab")
for d in ("polisy_da/data/canonical", "polisy_da/data/keys", "lab/canonical", "lab/staged/patentsview",
          "lab/staged/discern", "lab/panels", "data", "linkage"):
    (BASE / d).mkdir(parents=True, exist_ok=True)

import polisy_triangulate as tri  # noqa: E402
from openpyxl import load_workbook  # noqa: E402

YEARS = range(2012, 2025)
# vrid, name, workers in 2012 (grows 3% a year), firm employees (thousands) or None
EMP = [
    ("v_apple", "Apple Inc.", 30000), ("v_google", "Google LLC", 50000), ("v_acme", "Acme Widgets Inc", 500),
    ("v_beta", "Beta Systems Corp", 800), ("v_delta_tx", "Delta Labs (Texas)", 300), ("v_delta_oh", "Delta Labs (Ohio)", 280),
    ("v_eps", "Epsilon Energy", 1000), ("v_zeta", "Zeta Biotech", 700), ("v_eta", "Eta Foods", 60),
    ("v_theta", "Theta Motors", 900), ("v_iota", "Iota Tech", 400), ("v_tiny", "Apple", 8),
]
rows = []
for vrid, name, w in EMP:
    for i, y in enumerate(YEARS):
        n = round(w * 1.03 ** i)
        rows.append({"vrid": vrid, "year": y, "company_name": name, "employee_count": n, "avg_match_quality": 0.9,
                     "dem_workers_raw": n * 0.4, "rep_workers_raw": n * 0.3, "dem_workers_imp": n * 0.5,
                     "rep_workers_imp": n * 0.4})
vr = pd.DataFrame(rows)
vr_path = BASE / "polisy_da/data/canonical/vr_employer.parquet"
vr[["vrid", "year", "company_name", "employee_count"]].to_parquet(vr_path, index=False)   # no party columns at all

FIRMS = [("001690", "APPLE INC", "CA", 110), ("160329", "ALPHABET INC", "CA", 150), ("011111", "ACME WIDGETS INC", "OH", 2.0),
         ("022222", "BETA SYSTEMS CORP", "NY", 3.0), ("033333", "GAMMA HOLDINGS INC", "NY", 20.0),
         ("077777", "DELTA LABS INC", "TX", 1.2), ("044444", "EPSILON ENERGY", "TX", 1.0), ("055555", "ZETA BIOTECH INC", "MA", 2.5),
         ("066666", "ETA FOODS CORP", "IL", 0.002), ("088888", "THETA MOTORS INC", "MI", 4.0), ("099999", "IOTA TECH INC", "WA", 1.5)]
comp = []
for gv, conm, st, e in FIRMS:
    for i, y in enumerate(range(2010, 2023)):
        emp = e * 1.03 ** (i - 2)
        if gv == "044444" and y >= 2018:
            emp *= 10                                   # a merger: employees jump tenfold
        comp.append({"GVKEY": int(gv), "fyear": float(y), "conm": conm, "state": st, "emp": round(emp, 4),
                     "sale": e * 300, "at": e * 500, "sich": 3571 if gv in ("001690", "160329") else 2000})
pd.DataFrame(comp).to_csv(BASE / "data/Compustat_Final.csv", index=False)

cw = pd.DataFrame([
    ("001690", "APPLE INC", "v_apple", "Apple Inc.", "exact", 100, "high", np.nan),
    ("011111", "ACME WIDGETS INC", "v_acme", "Acme Widgets Inc", "exact", 100, "high", np.nan),
    ("022222", "BETA SYSTEMS CORP", "v_beta", "Beta Systems Corp", "exact", 100, "high", np.nan),
    ("077777", "DELTA LABS INC", "v_delta_tx", "Delta Labs (Texas)", "exact", 100, "high", np.nan),
    ("044444", "EPSILON ENERGY", "v_eps", "Epsilon Energy", "exact", 100, "high", np.nan),
    ("055555", "ZETA BIOTECH INC", "v_zeta", "Zeta Biotech", "fuzzy", 96, "medium", np.nan),
    ("066666", "ETA FOODS CORP", "v_eta", "Eta Foods", "fuzzy", 91, "low", 1),
    ("088888", "THETA MOTORS INC", "v_theta", "Theta Motors", "exact", 100, "high", np.nan),
    ("099999", "IOTA TECH INC", "v_iota", "Iota Tech", "exact", 100, "high", np.nan),
], columns=["gvkey", "conm", "vrid", "company_name", "method", "score", "tier", "accept"])
cw["fyear"], cw["state"], cw["emp"], cw["name_norm"], cw["workers"], cw["tp"] = 2015.0, "", 1.0, "", 100, 80.0
cw["size_ratio"], cw["size_ok"] = 0.1, True
cw.to_csv(BASE / "polisy_da/data/keys/cw_vrid_gvkey.csv", index=False)
pd.DataFrame({"vrid": ["v_theta"], "gvkey": ["88888"], "accept": [0], "decided_by": ["test"]}).to_csv(
    BASE / "linkage/hand_decisions.csv", index=False)

pats, asg, disc = [], [], []
k = 7000000
def add(org, n, owner, year=2015, atype=2, permno=None):
    global k
    for _ in range(n):
        k += 1
        pid = str(k)
        pats.append({"patent_id": pid, "year": year, "assignee_org": org, "assignee_type": atype})
        asg.append({"patent_id": pid, "seq": 0, "assignee_id": "a_" + org[:4], "org": org, "atype": str(atype)})
        if owner:
            disc.append({"patent_id": float(pid), "gvkey": owner, "permno_adj": 1.0, "discern_year": year,
                         "discern_file": "grant"})
add("APPLE INC.", 40, "1690")
add("Google LLC", 30, "160329")
add("Beta Systems Corporation", 12, "33333")
add("Iota Tech Inc", 9, None)                  # patents of a private namesake: no listed owner
add("Old Name Co", 5, "1690", year=1995)       # outside the patent window
pd.DataFrame(pats).to_parquet(BASE / "lab/canonical/patents.parquet", index=False)
pd.DataFrame(asg).to_parquet(BASE / "lab/staged/patentsview/g_assignee_disambiguated.parquet", index=False)
pd.DataFrame(disc).to_parquet(BASE / "lab/canonical/discern_patents.parquet", index=False)
pd.DataFrame([{"gvkey": "055555", "name": "Zeta Biotech Inc", "fy0": 2005, "fy1": 2021},
              {"gvkey": "033333", "name": "Gamma Holdings", "fy0": 2000, "fy1": 2021}]).to_parquet(
    BASE / "lab/staged/discern/names.parquet", index=False)
pd.DataFrame([{"gvkey": g, "year": y, "empLiberalism_10yr": 0.6, "ceoLiberalism_10yr": 0.3}
              for g, *_ in FIRMS for y in range(2012, 2022)]).to_parquet(BASE / "polisy_da/data/canonical/dipi.parquet",
                                                                        index=False)
pd.DataFrame([{"gvkey": "001690", "year": y, "pat_filed": 40.0} for y in range(2012, 2022)]).to_parquet(
    BASE / "lab/panels/firm_patents_year.parquet", index=False)

# ---------------------------------------------------------------- step A
PATHS = {"compustat": BASE / "data/Compustat_Final.csv"}
chk = tri.check_inputs(paths=PATHS)
links = tri.build_links(paths=PATHS, note="test run")
t = links.set_index("vrid")
def typ(v, g=None):
    x = links[(links.vrid == v) & ((links.gvkey == g) if g else True)]
    return list(x.match_type)

checks = {
    "check_inputs finds every fixture": (chk.status == "found").sum() == 9,
    "apple triangulated by patents": typ("v_apple") == ["triangulated_patents"],
    "google linked to alphabet through DISCERN + size": typ("v_google", "160329") == ["discern_and_size"],
    "acme confirmed by name + size": typ("v_acme") == ["name_and_size"],
    "beta is a conflict (name -> 022222, patents -> 033333)": sorted(typ("v_beta")) == ["conflict", "conflict"],
    "beta stays out of every analysis set but baseline": not links[(links.vrid == "v_beta")][["set_strict", "set_main", "set_broad"]].any().any(),
    "delta's shared name is not unique: one source only": typ("v_delta_tx") == ["single_source"],
    "epsilon's merger breaks the size trajectory": links.loc[links.vrid == "v_eps", "size_status"].iloc[0].startswith("break"),
    "zeta triangulated by DISCERN's name list": typ("v_zeta") == ["triangulated_names"],
    "eta accepted by hand": typ("v_eta") == ["hand_accepted"],
    "theta rejected by hand (hand_decisions.csv)": typ("v_theta") == ["rejected"],
    "iota flagged: its namesake's patents have no listed owner": bool(links.loc[links.vrid == "v_iota", "flag_patents_unowned"].iloc[0]),
    "tiny 'Apple' employer is not linked": "v_tiny" not in set(links.vrid),
    "one firm per employer in every set": all(links[links[f"set_{s}"]].vrid.is_unique for s in tri.SETS),
    "baseline mirrors module 06 (high tier or accept = 1)": set(links.loc[links.set_baseline, "vrid"]) ==
        {"v_apple", "v_acme", "v_beta", "v_delta_tx", "v_eps", "v_eta", "v_theta", "v_iota"},
    "strict is inside main, main inside broad": set(links.loc[links.set_strict, "vrid"]) <= set(links.loc[links.set_main, "vrid"])
        <= set(links.loc[links.set_broad, "vrid"]),
}
out = BASE / "linkage"
for f in ("links_all.csv", "T1_match_types.csv", "T1b_link_sets.csv", "T3_agreement.csv", "review_list.xlsx",
          "protocol.json", "linkage_log.csv", "README_outputs.txt", "crosswalk_main.csv"):
    checks[f"wrote {f}"] = (out / f).exists()

# rebuilding with the same inputs keeps the version
v1 = pd.read_csv(out / "linkage_log.csv")
tri.build_links(paths=PATHS, note="same inputs")
checks["same inputs, same version"] = len(pd.read_csv(out / "linkage_log.csv")) == len(v1) == 1

# a decision round, typed into the review workbook and saved as hand_decisions_2.xlsx
wb = load_workbook(out / "review_list.xlsx")
ws = wb["Review"]
head = [c.value for c in ws[1]]
iv, ig, ia = head.index("vrid") + 1, head.index("gvkey") + 1, head.index("accept") + 1
gv_text = [ws.cell(r, ig).value for r in range(2, ws.max_row + 1)]
checks["review workbook keeps gvkey zeros as text"] = all(isinstance(g, str) and len(g) == 6 for g in gv_text)
for r in range(2, ws.max_row + 1):
    if ws.cell(r, iv).value == "v_beta":
        ws.cell(r, ia, value=1 if ws.cell(r, ig).value == "022222" else 0)
wb.save(out / "hand_decisions_2.xlsx")
links = tri.build_links(paths=PATHS, note="beta decided")
checks["a decision round makes version 2"] = len(pd.read_csv(out / "linkage_log.csv")) == 2
checks["beta resolved by hand: its own firm by name and size, the other rejected"] = (
    typ("v_beta", "022222") == ["name_and_size"] and typ("v_beta", "033333") == ["rejected"])

# ---------------------------------------------------------------- step B, C
key = tri.draw_audit(paths=PATHS, n=12)
checks["audit drew 12 links (or all, when fewer)"] = len(key) == min(12, int((links.set_strict | links.set_main | links.set_broad | links.set_baseline).sum()))
again = tri.draw_audit(paths=PATHS, n=12)
checks["second draw_audit keeps the first sample"] = again.audit_id.tolist() == key.audit_id.tolist()
for coder in (1, 2):
    wb = load_workbook(out / f"audit_coder{coder}.xlsx")
    ws = wb["Audit"]
    head = [c.value for c in ws[1]]
    col = head.index("Same company? 1 / 0 / ?") + 1
    checks[f"coder {coder} sheet hides the match type"] = not any("type" in str(h).lower() for h in head)
    for r in range(2, ws.max_row + 1):
        ws.cell(row=r, column=col, value=1 if (coder == 1 or r != 3) else 0)
    wb.save(out / f"audit_coder{coder}.xlsx")
T4 = tri.audit_summary(paths=PATHS)
checks["audit summary has types, sets and coder agreement"] = {"match type", "link set", "coders"} <= set(T4.level)
checks["one open disagreement listed"] = len(pd.read_csv(out / "audit_disagreements.csv")) == 1

# ---------------------------------------------------------------- step D, E (party data only now)
vr.to_parquet(vr_path, index=False)                       # the full panel, with party counts
T2 = tri.selection_table(paths=PATHS)
checks["selection table covers DIPI scores"] = {"dipi_employee_liberalism", "dipi_ceo_liberalism"} <= set(T2.variable)
tri.workforce_panels(paths=PATHS)
for s in tri.SETS:
    checks[f"workforce_{s}.dta written"] = (out / f"workforce_{s}.dta").exists()
w = pd.read_csv(out / "workforce_main.csv", dtype={"gvkey": str})
checks["workforce shares between 0 and 1"] = w.vr_dem_share.between(0, 1).all()
checks["Stata template written"] = (out / "linkage_sensitivity.do").exists()

# ---------------------------------------------------------------- the lock
p = out / "links_all.csv"
p.write_text(p.read_text().replace("triangulated_patents", "triangulated_patentz", 1))
try:
    tri.selection_table(paths=PATHS)
    checks["edited links are refused"] = False
except RuntimeError:
    checks["edited links are refused"] = True

print()
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS " if v else "FAIL ") + k)
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(1 if bad else 0)
