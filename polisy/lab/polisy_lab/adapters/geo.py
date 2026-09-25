# -*- coding: utf-8 -*-
"""Geography and migration -> canonical tables.

geo_state          state FIPS, postal code, name, census region
geo_county         county -> CBSA (Census List 1 via POLISY_DA), names
irs_county_year    county x year: households (returns), people (exemptions) and income (AGI, $000) moving in, out
                   and staying, with net and gross migration rates (IRS SOI county summary rows)
irs_state_year     the same for states, from the counties' interstate totals
irs_flows_county   county -> county flows (the IRS suppresses pairs under 20 returns)
irs_flows_state    state -> state flows (IRS state files if present, else summed from county pairs)
irs_county_names   county names as the IRS files spell them
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import log, write, find_col, pc, STATE_ABBR, REGION_OF, read, squash, diagnostic
from ..sources import discover

STATE_NAME = {"01": "Alabama", "02": "Alaska", "04": "Arizona", "05": "Arkansas", "06": "California", "08": "Colorado",
              "09": "Connecticut", "10": "Delaware", "11": "District of Columbia", "12": "Florida", "13": "Georgia",
              "15": "Hawaii", "16": "Idaho", "17": "Illinois", "18": "Indiana", "19": "Iowa", "20": "Kansas",
              "21": "Kentucky", "22": "Louisiana", "23": "Maine", "24": "Maryland", "25": "Massachusetts", "26": "Michigan",
              "27": "Minnesota", "28": "Mississippi", "29": "Missouri", "30": "Montana", "31": "Nebraska", "32": "Nevada",
              "33": "New Hampshire", "34": "New Jersey", "35": "New Mexico", "36": "New York", "37": "North Carolina",
              "38": "North Dakota", "39": "Ohio", "40": "Oklahoma", "41": "Oregon", "42": "Pennsylvania", "44": "Rhode Island",
              "45": "South Carolina", "46": "South Dakota", "47": "Tennessee", "48": "Texas", "49": "Utah", "50": "Vermont",
              "51": "Virginia", "53": "Washington", "54": "West Virginia", "55": "Wisconsin", "56": "Wyoming"}


def adapt_geo():
    s = pd.DataFrame([{"state_fips": f, "state": a, "name": STATE_NAME.get(f), "region": REGION_OF.get(a)}
                      for a, f in STATE_ABBR.items() if f in STATE_NAME])
    write(s, "geo_state")
    try:
        got = pc.cbsa_delineation()
    except ValueError as e:
        log(f"geo: CBSA reference unusable ({e})")
        got = None
    counties = got[1] if got else None
    if got:
        ref = pc.locate("CBSA_REFERENCE")
        log(f"geo: CBSA delineation from {Path(str(ref.get('path'))).name} (found by POLISY_DA as CBSA_REFERENCE)")
    names = read("county_exposure")
    if counties is None and names is None:
        log("geo: no county list (CBSA reference or AIGE); county-level linking limited to FIPS codes")
        return False
    if counties is None:
        counties = pd.DataFrame(columns=["cbsa", "county_fips", "county_name", "state_name", "central_outlying"])
    else:
        counties = counties.merge(got[0][["cbsa", "cbsa_title", "cbsa_type"]], on="cbsa", how="left")
    if names is not None:
        extra = names[~names.county_fips.isin(counties.county_fips)][["county_fips", "name"]].rename(columns={"name": "county_name"})
        counties = pd.concat([counties, extra], ignore_index=True)
    for c in ("cbsa", "cbsa_title", "cbsa_type", "county_name", "state_name", "central_outlying"):
        if c not in counties:
            counties[c] = None
    counties["state_fips"] = counties.county_fips.str[:2]
    counties["state"] = counties.state_fips.map({v: k for k, v in STATE_ABBR.items()})
    write(counties.drop_duplicates("county_fips"), "geo_county", note=f"{counties.cbsa.notna().sum():,} counties in a CBSA")
    return got is not None          # "connected" means the CBSA delineation was found, not only county names


# IRS SOI migration files: summary rows are coded in the "other" state/county columns.
IRS_KINDS = {(96, 0): "total", (97, 0): "us", (97, 1): "same_state", (97, 3): "diff_state", (98, 0): "foreign"}
VALID_STATES = {int(v) for v in STATE_ABBR.values()}


def _irs_years(name):
    m = re.search(r"(\d{2})(\d{2})(?=\D*$)", Path(name).stem)
    return (2000 + int(m.group(1)), 2000 + int(m.group(2))) if m else None


def _irs_county_file(p, member):
    """One countyinflowYYYY / countyoutflowYYYY file -> (county totals, county pairs, county names)."""
    name = Path(member or p).name
    yy = _irs_years(name)
    if not yy:
        log(f"irs_migration: no year pair in file name {name}")
        return None
    inflow = "inflow" in name.lower()
    d = pc.read_table(p, member)
    d.columns = [squash(c) for c in d.columns]
    home, other = ("y2", "y1") if inflow else ("y1", "y2")
    need = [f"{home}statefips", f"{home}countyfips", f"{other}statefips", f"{other}countyfips", "n1"]
    if any(c not in d for c in need):
        log(f"irs_migration: {name} lacks {[c for c in need if c not in d]}; columns are {list(d.columns)}")
        return None
    hs, hc, os_, oc = (pd.to_numeric(d[c], errors="coerce") for c in need[:4])
    num = {k: pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce") if c in d else pd.Series(np.nan, index=d.index)
           for k, c in (("returns", "n1"), ("people", "n2"), ("agi", "agi"))}
    for v in num.values():
        v[v < 0] = np.nan                                   # -1 marks suppressed cells
    keep = hc.ne(0) & hs.isin(VALID_STATES)                 # county rows (state total rows have county 000)
    fips = lambda st, ct: st.astype("Int64").astype(str).str.zfill(2) + ct.astype("Int64").astype(str).str.zfill(3)  # noqa: E731
    home_f, other_f = fips(hs, hc), fips(os_, oc)
    stay = (os_ == hs) & (oc == hc)
    kind = pd.Series([IRS_KINDS.get((a, b)) for a, b in zip(os_.fillna(-1).astype(int), oc.fillna(-1).astype(int))], index=d.index)
    kind[stay] = "stay"
    tot = pd.DataFrame({"county_fips": home_f, "year": yy[1], "direction": np.where(stay, "stay", "in" if inflow else "out"),
                        "kind": kind, **num})[keep & kind.notna()]
    pair = keep & os_.isin(VALID_STATES) & oc.ne(0) & ~stay
    pairs = pd.DataFrame({"origin": other_f if inflow else home_f, "dest": home_f if inflow else other_f, "year": yy[1],
                          **num, "from_inflow": inflow})[pair]
    ncol = next((c for c in d.columns if c.endswith("countyname")), None)
    stcol = next((c for c in d.columns if c in (f"{other}state", f"{home}state")), None)
    names = pd.DataFrame({"county_fips": home_f, "county_name": d[ncol].astype(str).str.replace(r"\s*Non-?migrants\s*$", "", regex=True)
                          if ncol else None, "state": d[stcol] if stcol else None})[keep & stay]
    return tot, pairs, names


def adapt_irs_migration():
    done = False
    cfiles = discover("irs_migration", "county")
    if cfiles:
        parts = [r for r in (_irs_county_file(p, m) for p, m in cfiles) if r is not None]
        if parts:
            tot = pd.concat([x[0] for x in parts], ignore_index=True)
            tot = tot.drop_duplicates(["county_fips", "year", "direction", "kind"])      # non-movers appear in both files
            w = tot.pivot_table(index=["county_fips", "year"], columns=["direction", "kind"], values=["returns", "people", "agi"], aggfunc="first")
            w.columns = [f"stay_{v}" if dr == "stay" else f"{dr}_{k}_{v}" for v, dr, k in w.columns]
            w = w.reset_index()
            g = lambda c: w[c] if c in w else np.nan  # noqa: E731
            w["base_returns"] = g("stay_returns") + g("out_total_returns")
            w["net_returns"] = g("in_us_returns") - g("out_us_returns")
            w["net_migration_rate"] = w.net_returns / w.base_returns
            w["gross_migration_rate"] = (g("in_us_returns") + g("out_us_returns")) / w.base_returns
            w["net_agi"] = g("in_us_agi") - g("out_us_agi")
            w["net_agi_rate"] = w.net_agi / (g("stay_agi") + g("out_total_agi"))
            w["in_agi_per_return"] = g("in_us_agi") / g("in_us_returns")
            w["out_agi_per_return"] = g("out_us_agi") / g("out_us_returns")
            w["stay_agi_per_return"] = g("stay_agi") / g("stay_returns")
            w["mover_income_gap"] = w.in_agi_per_return - w.out_agi_per_return
            w["state_fips"] = w.county_fips.str[:2]
            write(w, "irs_county_year", note=f"{w.county_fips.nunique():,} counties, {w.year.min()}-{w.year.max()} (year = second year of the pair)")
            pairs = pd.concat([x[1] for x in parts], ignore_index=True).sort_values("from_inflow", ascending=False)
            pairs = pairs.drop_duplicates(["origin", "dest", "year"]).drop(columns="from_inflow")
            write(pairs, "irs_flows_county", note="county pairs with 20+ returns (smaller flows are suppressed by the IRS)")
            names = pd.concat([x[2] for x in parts], ignore_index=True).drop_duplicates("county_fips", keep="last")
            write(names, "irs_county_names")
            sums = ["stay_returns", "out_total_returns", "in_diff_state_returns", "out_diff_state_returns", "in_diff_state_agi",
                    "out_diff_state_agi", "stay_agi", "out_total_agi", "in_foreign_returns", "out_foreign_returns"]
            st = w.groupby(["state_fips", "year"], as_index=False)[[c for c in sums if c in w]].sum(min_count=1)
            st["base_returns"] = st.stay_returns + st.out_total_returns
            st["net_returns"] = st.in_diff_state_returns - st.out_diff_state_returns
            st["net_migration_rate"] = st.net_returns / st.base_returns
            st["gross_migration_rate"] = (st.in_diff_state_returns + st.out_diff_state_returns) / st.base_returns
            st["net_agi"] = st.in_diff_state_agi - st.out_diff_state_agi
            st["net_agi_rate"] = st.net_agi / (st.stay_agi + st.out_total_agi)
            st["in_agi_per_return"] = st.in_diff_state_agi / st.in_diff_state_returns
            st["out_agi_per_return"] = st.out_diff_state_agi / st.out_diff_state_returns
            st["mover_income_gap"] = st.in_agi_per_return - st.out_agi_per_return
            write(st, "irs_state_year", note="interstate moves summed over each state's counties")
            inter = pairs[pairs.origin.str[:2] != pairs.dest.str[:2]]
            sp = inter.assign(origin=inter.origin.str[:2], dest=inter.dest.str[:2]).groupby(["origin", "dest", "year"], as_index=False)[
                ["returns", "people", "agi"]].sum(min_count=1)
            write(sp, "irs_flows_state", note="summed from county pairs; misses the county pairs the IRS suppresses")
            diagnostic("IRS county pairs -> interstate totals", "irs_flows_county (interstate)", "irs_state_year", "state, year",
                       sp.returns.sum(), st.in_diff_state_returns.sum(), "returns",
                       "share of interstate moves visible as county-to-county pairs (the rest are in suppressed small flows)")
            done = True
    sfiles = discover("irs_migration", "state")
    if sfiles:
        frames = [f for f in (_irs_state_file(p, m) for p, m in sfiles) if f is not None]
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d = d.sort_values("from_inflow", ascending=False).drop_duplicates(["origin", "dest", "year"]).drop(columns="from_inflow")
            write(d, "irs_flows_state", note=f"IRS state files, {d.year.min()}-{d.year.max()}")
            done = True
    if not done:
        log("irs_migration: no countyinflowYYYY / stateinflowYYYY files found")
    return done


def _irs_state_file(p, member):
    name = Path(member or p).name
    yy = _irs_years(name)
    if not yy:
        log(f"irs_migration: no year pair in file name {name}")
        return None
    try:
        d = pc.read_table(p, member)
    except Exception as e:
        log(f"irs_migration: {name} unreadable ({e})")
        return None
    ds = find_col(d, [r"y2statefips", r"statecodedest", r"deststate.*fips"], "destination state", True, name)
    os_ = find_col(d, [r"y1statefips", r"statecodeorigin", r"originstate.*fips"], "origin state", True, name)
    num = lambda c: pd.to_numeric(d[c].astype(str).str.replace(",", ""), errors="coerce") if c else np.nan  # noqa: E731
    out = pd.DataFrame({"origin": pc.digits(d[os_]).str.zfill(2), "dest": pc.digits(d[ds]).str.zfill(2), "year": yy[1],
                        "returns": num(find_col(d, [r"n1", r"returns"], "", False)), "people": num(find_col(d, [r"n2", r"exemptions"], "", False)),
                        "agi": num(find_col(d, [r"agi"], "", False)), "from_inflow": "inflow" in name.lower()})
    valid = out.origin.isin(STATE_ABBR.values()) & out.dest.isin(STATE_ABBR.values())
    out.loc[out.returns < 0, "returns"] = np.nan
    return out[valid & out.returns.gt(0)]
