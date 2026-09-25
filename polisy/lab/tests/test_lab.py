# -*- coding: utf-8 -*-
"""POLISY lab tests. They use real files only: a test that needs a dataset skips when the dataset is not found.

    POLISY_TEST_DATA=/path/to/downloads:/path/to/AIOE python -m pytest polisy/lab/tests -q
    POLISY_TEST_FULL=1 ...                                   # also run the whole pipeline and build the site

POLISY_TEST_DATA lists the folders to search (separated by ':' on Linux and macOS, ';' on Windows).
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent.parent / "POLISY_DA"), str(HERE.parent)]
TMP = Path(tempfile.mkdtemp(prefix="polisy_lab_test_"))
os.environ["POLISY_LAB_ROOT"] = str(TMP / "lab")
os.environ.setdefault("POLISY_ROOT", str(TMP / "polisy"))

import polisy_core as pc                                        # noqa: E402
from polisy_lab import sources, run_all, LAB                    # noqa: E402
from polisy_lab.core import read                                # noqa: E402
from polisy_lab.site import albers_usa, _columnar, ASSETS       # noqa: E402

DATA = [d for d in os.environ.get("POLISY_TEST_DATA", "").split(os.pathsep) if d]
pc.CONFIG["SEARCH_DIRS"] = DATA
pc.CONFIG.update({k: None for k in pc.FILES})


def need(source, role):
    files = sources.discover(source, role) if DATA else []
    if not files:
        pytest.skip(f"no {source}/{role} files in POLISY_TEST_DATA")
    return files


# --------------------------------------------------------------------------- pure functions
def test_projection_puts_cities_inside_their_states():
    import re
    shapes = {s["abbr"]: s["d"] for s in json.loads((ASSETS / "us_states.json").read_text())["states"]}
    for st, lon, lat in (("DC", -77.0369, 38.9072), ("CA", -122.4194, 37.7749), ("AK", -149.9003, 61.2181),
                         ("HI", -157.8583, 21.3069), ("ME", -70.2553, 43.6615), ("TX", -97.7431, 30.2672)):
        nums = [float(v) for v in re.findall(r"-?\d+\.?\d*", shapes[st])]
        xs, ys = nums[0::2], nums[1::2]
        x, y = albers_usa(lon, lat)
        assert min(xs) - 2 <= x <= max(xs) + 2 and min(ys) - 2 <= y <= max(ys) + 2, st


def _decode(d):
    """What app.js does with a columnar table."""
    cols = d["columns"]
    vals = [(lambda e: [None if i < 0 else e["dict"][i] for i in e["idx"]] if isinstance(e, dict) else e)(d["enc"][c]) for c in cols]
    return [{c: vals[j][i] for j, c in enumerate(cols)} for i in range(d["n"])]


def test_columnar_tables_round_trip():
    rows = [{"name": ["a", "b", None][i % 3], "year": 2010 + i % 4, "x": None if i % 7 == 0 else i / 3, "id": f"id{i}"} for i in range(50)]
    d = {"id": "t", "columns": ["name", "year", "x", "id"], "rows": rows, "note": ""}
    enc = _columnar(d)
    assert isinstance(enc["enc"]["name"], dict) and isinstance(enc["enc"]["id"], list)   # repeated text is dictionary-coded
    assert _decode(enc) == rows


# --------------------------------------------------------------------------- discovery
def test_every_table_inside_a_zip_is_a_candidate():
    files = need("irs_migration", "county")
    assert all(m is None or Path(m).name.lower().startswith(("countyinflow", "countyoutflow")) for _, m in files)
    assert len(files) >= 2


# --------------------------------------------------------------------------- adapters on real files
def test_irs_county_migration():
    need("irs_migration", "county")
    from polisy_lab.adapters.geo import adapt_irs_migration
    assert adapt_irs_migration()
    w = read("irs_county_year")
    assert w.county_fips.str.fullmatch(r"\d{5}").all()
    assert not w.duplicated(["county_fips", "year"]).any()
    r = w.net_migration_rate.dropna()
    assert (r.abs() < 0.5).all(), "a county gained or lost half its households in a year"
    ok = w.dropna(subset=["in_us_returns", "in_total_returns"])
    assert (ok.in_us_returns <= ok.in_total_returns).all()
    pairs = read("irs_flows_county")
    assert not pairs.duplicated(["origin", "dest", "year"]).any()
    assert (pairs.origin != pairs.dest).all()
    st = read("irs_state_year")
    assert st.state_fips.nunique() >= 50
    # interstate moves net out to (almost) zero across states each year
    net = st.groupby("year").net_returns.sum().abs() / st.groupby("year").in_diff_state_returns.sum()
    assert (net < 0.02).all()


def test_daioe():
    need("dynamic_aioe", "soc2010")
    from polisy_lab.adapters.ai import adapt_dynamic_aioe
    assert adapt_dynamic_aioe()
    x = read("occ_daioe")
    assert x.soc.nunique() > 700 and x.year.min() <= 2012
    z = x.groupby("year").z_allapps.agg(["mean", "std"])
    assert np.allclose(z["mean"], 0, atol=1e-6) and np.allclose(z["std"], 1, atol=1e-6)
    m = read("occ_measures_soc2018")
    if m is not None:
        assert not m.soc2018.duplicated().any()


def test_cspp_profile():
    need("cspp", "data")
    from polisy_lab.adapters.politics import adapt_cspp, CSPP_CURATED
    assert adapt_cspp()
    prof = read("cspp_state_profile")
    assert prof.state_fips.nunique() >= 50
    assert set(CSPP_CURATED) & set(prof.columns)
    tx = prof.set_index("state_fips").loc["48"]
    if "propgoppres" in prof:                   # Texas voted Republican in every election of the window
        assert tx.propgoppres > 50


# --------------------------------------------------------------------------- the whole lab
@pytest.mark.skipif(os.environ.get("POLISY_TEST_FULL") != "1", reason="set POLISY_TEST_FULL=1 to run the whole pipeline")
def test_pipeline_and_site():
    if not DATA:
        pytest.skip("POLISY_TEST_DATA is empty")
    run_all(stages=("inventory", "adapt", "link", "analyze", "site"), fetch=False)
    res = json.loads((LAB["RESULTS"] / "results.json").read_text())
    assert not res["run"].get("failed"), res["run"].get("failed")
    assert res["findings"]
    for v in res["views"].values():                                  # every chart names a table that exists
        assert v["dataset"] in res["datasets"], v["id"]
        if v.get("edges"):
            assert v["edges"] in res["datasets"], v["id"]
    page = (LAB["SITE"] / "index.html").read_text()
    assert "<title>POLISY Lab</title>" in page and "polisy-data" in page
