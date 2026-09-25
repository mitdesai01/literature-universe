# -*- coding: utf-8 -*-
"""PatentsView -> canonical patent tables, with DuckDB doing all the heavy lifting.

patents               patent_id, year (grant), AI flags (narrow, broad) and AI subfields
patent_places         patent x US inventor with county/state and a fractional share
patents_county_year   patents, AI patents, inventors by county and grant year
patents_state_year    the same by state
ai_cpc_edges          co-occurrence of CPC subclasses on AI patents, by period (the technology
                      network AI is built into)
inventor_moves        inventors whose consecutive patents list different states: origin, destination,
                      year, and whether the later patent is AI

AI patents are identified by CPC codes (narrow: G06N except quantum computing G06N10; broad adds
image/video recognition G06V, image analysis G06T7, natural language G06F40, speech G10L15/G10L13,
learning control G05B13, robot learning B25J9/161 and /163). The USPTO AI Patent Dataset (AIPD)
is a better, model-based label; add it as a source and prefer it when present.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pandas as pd

from ..core import log, write, find_col, pc, LAB, canon, dirs
from ..sources import discover

AI_RULES = {  # subfield: SQL condition on main group `g` ("G06N3") and full code `c` ("G06N3/084")
    "sub_ml": "substr(g, 1, 4) = 'G06N' AND g NOT LIKE 'G06N10%'",
    "sub_vision": "substr(g, 1, 4) = 'G06V' OR g = 'G06T7'",
    "sub_language": "g IN ('G06F40', 'G10L15', 'G10L13')",
    "sub_control": "g = 'G05B13'",
    "sub_robotics": "c LIKE 'B25J9/161%' OR c LIKE 'B25J9/163%'",
}


def _tsv(path, member):
    """A path DuckDB can read: zip members are extracted once into the lab's temp folder."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        return path
    tmp = dirs()["TMP"] / "patentsview"
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        name = member or next(n for n in zf.namelist() if n.lower().endswith((".tsv", ".csv", ".txt")))
        out = tmp / Path(name).name
        info = zf.getinfo(name)
        if not (out.exists() and out.stat().st_size == info.file_size):
            log(f"patentsview: extracting {Path(name).name} ({info.file_size / 1e9:.2f} GB)")
            with zf.open(name) as fin, open(out, "wb") as fout:
                shutil.copyfileobj(fin, fout, 1 << 24)
    return out


def _reader(p):
    p = Path(p)
    if p.suffix.lower() == ".parquet":
        return f"read_parquet('{pc.sqlp(p)}')"
    delim = "\\t" if p.suffix.lower() in (".tsv", ".txt") else ","
    return (f"read_csv('{pc.sqlp(p)}', delim='{delim}', header=true, quote='\"', escape='\"', all_varchar=true, "
            f"ignore_errors=true, max_line_size=20000000)")


def _cols(c, reader):
    return list(pc.q(c, f"DESCRIBE SELECT * FROM {reader}")["column_name"])


def adapt_patentsview():
    got = {r: discover("patentsview", r) for r in ("patent", "cpc", "inventor", "location")}
    missing = [r for r, v in got.items() if not v]
    if missing:
        log(f"patentsview: missing {missing}; put g_patent, g_cpc_current, g_inventor_disambiguated and "
            f"g_location_disambiguated (.tsv or .tsv.zip) in a search folder")
        return False
    c = pc.con()
    c.execute("SET threads TO 4")
    R = {r: _reader(_tsv(*v[0])) for r, v in got.items()}
    pcol, ccol, icol, lcol = (_cols(c, R[r]) for r in ("patent", "cpc", "inventor", "location"))
    P = {"id": find_col(pcol, [r"patentid", r"patentnumber"], "patent id", True, "g_patent"),
         "date": find_col(pcol, [r"patentdate", r"grantdate", r"date"], "grant date", True, "g_patent"),
         "type": find_col(pcol, [r"patenttype", r"type"], "patent type", False, "g_patent"),
         "wd": find_col(pcol, [r"withdrawn"], "withdrawn flag", False, "g_patent")}
    C = {"id": find_col(ccol, [r"patentid"], "patent id", True, "g_cpc_current"),
         "group": find_col(ccol, [r"cpcgroup", r"cpcsubgroup", r"groupid", r"cpc"], "CPC group", True, "g_cpc_current"),
         "subclass": find_col(ccol, [r"cpcsubclass", r"subclassid"], "CPC subclass", False, "g_cpc_current")}
    I = {"id": find_col(icol, [r"patentid"], "patent id", True, "g_inventor"),
         "inv": find_col(icol, [r"inventorid"], "inventor id", True, "g_inventor"),
         "loc": find_col(icol, [r"locationid"], "location id", True, "g_inventor")}
    L = {"loc": find_col(lcol, [r"locationid", r"id"], "location id", True, "g_location"),
         "st": find_col(lcol, [r"statefips"], "state FIPS", False, "g_location"),
         "cty": find_col(lcol, [r"countyfips"], "county FIPS", False, "g_location"),
         "abbr": find_col(lcol, [r"disambigstate", r"state"], "state code", False, "g_location"),
         "country": find_col(lcol, [r"disambigcountry", r"country"], "country", False, "g_location"),
         "lat": find_col(lcol, [r"latitude", r"lat"], "latitude", False, "g_location"),
         "lon": find_col(lcol, [r"longitude", r"lon", r"lng"], "longitude", False, "g_location")}
    q = lambda s: f'"{s}"'  # noqa: E731
    where = []
    if P["type"]:
        where.append(f"lower({q(P['type'])}) = 'utility'")
    if P["wd"]:
        where.append(f"coalesce(lower({q(P['wd'])}), '0') IN ('0', 'false', 'f', '')")
    c.execute(f"""CREATE OR REPLACE TABLE pat AS SELECT {q(P['id'])} AS patent_id,
                  try_cast(substr({q(P['date'])}, 1, 4) AS INTEGER) AS year
                  FROM {R['patent']} {('WHERE ' + ' AND '.join(where)) if where else ''}""")
    code = f"upper(replace({q(C['group'])}, ' ', ''))"
    subs = ",\n".join(f"bool_or({cond}) AS {name}" for name, cond in AI_RULES.items())
    c.execute(f"""CREATE OR REPLACE TABLE cpc AS
        WITH x AS (SELECT {q(C['id'])} AS patent_id, {code} AS c, split_part({code}, '/', 1) AS g FROM {R['cpc']})
        SELECT patent_id, {subs}, count(*) AS n_cpc, list(DISTINCT substr(g, 1, 4)) AS subclasses FROM x GROUP BY 1""")
    c.execute("""CREATE OR REPLACE TABLE pats AS SELECT p.patent_id, p.year, coalesce(x.sub_ml, false) AS ai,
                 coalesce(x.sub_ml OR x.sub_vision OR x.sub_language OR x.sub_control OR x.sub_robotics, false) AS ai_broad,
                 coalesce(x.sub_ml, false) AS sub_ml, coalesce(x.sub_vision, false) AS sub_vision,
                 coalesce(x.sub_language, false) AS sub_language, coalesce(x.sub_control, false) AS sub_control,
                 coalesce(x.sub_robotics, false) AS sub_robotics, x.n_cpc, x.subclasses
                 FROM pat p LEFT JOIN cpc x USING (patent_id) WHERE p.year IS NOT NULL""")
    n, nai, nb = c.execute("SELECT count(*), sum(ai::INT), sum(ai_broad::INT) FROM pats").fetchone()
    log(f"patentsview: {n:,} utility patents; AI narrow {nai:,} ({nai / max(n, 1):.2%}), broad {nb:,} ({nb / max(n, 1):.2%})")
    c.execute(f"COPY (SELECT * EXCLUDE (subclasses) FROM pats) TO '{pc.sqlp(canon('patents'))}' (FORMAT parquet, COMPRESSION zstd)")

    st = f"lpad(regexp_extract({q(L['st'])}, '(\\d+)', 1), 2, '0')" if L["st"] else "NULL"
    if L["cty"]:
        raw_cty = f"regexp_extract({q(L['cty'])}, '(\\d+)', 1)"
        cty = f"CASE WHEN length({raw_cty}) <= 3 AND {st} IS NOT NULL THEN {st} || lpad({raw_cty}, 3, '0') ELSE lpad({raw_cty}, 5, '0') END"
    else:
        cty = "NULL"
    us = f"upper(coalesce({q(L['country'])}, 'US')) IN ('US', 'USA', 'UNITED STATES')" if L["country"] else "TRUE"
    c.execute(f"""CREATE OR REPLACE TABLE loc AS SELECT {q(L['loc'])} AS location_id, {st} AS state_fips, {cty} AS county_fips,
                  {('try_cast(' + q(L['lat']) + ' AS DOUBLE)') if L['lat'] else 'NULL'} AS lat,
                  {('try_cast(' + q(L['lon']) + ' AS DOUBLE)') if L['lon'] else 'NULL'} AS lon
                  FROM {R['location']} WHERE {us}""")
    c.execute(f"""CREATE OR REPLACE TABLE places AS
        WITH inv AS (SELECT {q(I['id'])} AS patent_id, {q(I['inv'])} AS inventor_id, {q(I['loc'])} AS location_id FROM {R['inventor']}),
             k AS (SELECT patent_id, count(*) AS n_inv FROM inv GROUP BY 1)
        SELECT inv.patent_id, inv.inventor_id, p.year, l.state_fips, l.county_fips, l.lat, l.lon,
               1.0 / k.n_inv AS share, p.ai, p.ai_broad
        FROM inv JOIN pats p USING (patent_id) JOIN k USING (patent_id) JOIN loc l USING (location_id)
        WHERE l.state_fips IS NOT NULL""")
    m = c.execute("SELECT count(*), count(DISTINCT patent_id), sum(CASE WHEN county_fips IS NULL THEN 1 ELSE 0 END) FROM places").fetchone()
    log(f"patentsview: {m[0]:,} US inventor-patent rows on {m[1]:,} patents; {m[2]:,} without a county")
    c.execute(f"COPY places TO '{pc.sqlp(canon('patent_places'))}' (FORMAT parquet, COMPRESSION zstd)")
    for lvl, key in (("county", "county_fips"), ("state", "state_fips")):
        c.execute(f"""COPY (SELECT {key}, year, sum(share) AS patents, sum(CASE WHEN ai THEN share ELSE 0 END) AS ai_patents,
                          sum(CASE WHEN ai_broad THEN share ELSE 0 END) AS ai_broad_patents,
                          count(DISTINCT inventor_id) AS inventors,
                          count(DISTINCT CASE WHEN ai_broad THEN inventor_id END) AS ai_inventors
                          FROM places WHERE {key} IS NOT NULL GROUP BY 1, 2)
                      TO '{pc.sqlp(canon(f'patents_{lvl}_year'))}' (FORMAT parquet)""")
    c.execute(f"""COPY (WITH a AS (SELECT patent_id, year, unnest(subclasses) AS s FROM pats WHERE ai_broad),
                            e AS (SELECT x.s AS a, y.s AS b, CASE WHEN x.year < 2010 THEN 'before 2010' WHEN x.year < 2015 THEN '2010-2014'
                                         WHEN x.year < 2020 THEN '2015-2019' ELSE '2020 on' END AS period
                                  FROM a x JOIN a y ON x.patent_id = y.patent_id AND x.s < y.s)
                       SELECT a, b, period, count(*) AS weight FROM e GROUP BY 1, 2, 3 HAVING count(*) >= 5)
                  TO '{pc.sqlp(canon('ai_cpc_edges'))}' (FORMAT parquet)""")
    c.execute(f"""COPY (WITH s AS (SELECT inventor_id, patent_id, year, any_value(state_fips) AS state_fips, bool_or(ai_broad) AS ai
                                   FROM places GROUP BY 1, 2, 3),
                            o AS (SELECT *, lag(state_fips) OVER (PARTITION BY inventor_id ORDER BY year, patent_id) AS prev FROM s)
                       SELECT prev AS origin, state_fips AS dest, year, ai, count(*) AS moves FROM o
                       WHERE prev IS NOT NULL AND prev <> state_fips GROUP BY 1, 2, 3, 4)
                  TO '{pc.sqlp(canon('inventor_moves'))}' (FORMAT parquet)""")
    log("patentsview: patents, patent_places, patents_county_year, patents_state_year, ai_cpc_edges, inventor_moves written")
    return True
