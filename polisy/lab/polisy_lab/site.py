# -*- coding: utf-8 -*-
"""site: the POLISY lab website, built from results.json.

    from polisy_lab.site import build_site
    build_site()        # -> LAB["SITE"]/index.html, polisy_lab_offline.html, data/results.json, data/<table>.csv

index.html is one self-contained file: styles, script, map shapes and results are inlined and
Plotly loads from a pinned CDN. polisy_lab_offline.html also inlines Plotly, so it opens without
internet. Everything on the page is read from results.json: a finding, view or dataset that an
analysis registers shows up on the site without changes here or in site_assets/.

To publish on GitHub Pages, copy the site folder to a repository (e.g. as docs/) and choose
Settings -> Pages -> Deploy from a branch -> that folder.
"""
from __future__ import annotations

import html
import json
import math
import re
import time
from pathlib import Path

import pandas as pd

from .core import LAB, RESULTS, log, __version__

ASSETS = Path(__file__).with_name("site_assets")
REPORT = Path(__file__).resolve().parents[1] / "docs" / "TECHNICAL_REPORT.md"
PLOTLY_VERSION = "4.1.1"          # the version site_assets/app.js is tested against
PLOTLY_CDN = f"https://cdn.jsdelivr.net/npm/plotly.js-dist-min@{PLOTLY_VERSION}/plotly.min.js"
FONTS = ("https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..800"
         "&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap")

# Column labels used anywhere on the site (a view's own `labels` win).
LABELS = {
    "rep_share": "Republican share of workers", "avg_rep_share": "Average Republican share", "vr_rep_share": "Republican share of the matched workforce",
    "rep_vote_share": "Republican vote share", "rep_pred": "Predicted from occupation mix", "workers": "Matched workers",
    "change_pp": "Change over the period (points)", "drift_pp": "Change since first year (points)", "culture_gap": "Actual minus predicted",
    "culture_gap_pp": "Actual minus predicted (points)", "demeaned_pp": "Relative to own mean (points)", "resid": "Republican share beyond wage, education and group",
    "aioe": "AI exposure (AIOE)", "aioe_lm": "Language-model AIOE", "aioe_ig": "Image-generation AIOE", "aiie": "AI exposure (AIIE)",
    "aiie_lm": "Language-model AIIE", "aiie_ig": "Image-generation AIIE", "aige": "AI exposure (AIGE)",
    "pc_general": "General AI exposure (component 1)", "pc_perception_vs_language": "Perception vs language (component 2)",
    "exposure_cluster": "Exposure cluster", "cognitive_share": "Cognitive share of abilities", "creative_weight": "Weight of creative abilities",
    "req_education": "Required education", "log_salary": "Median salary (log)", "female": "Female share", "white": "White share",
    "ind_education": "Education of the workforce", "ind_log_wage": "Median wage (log)", "occ_coverage": "Jobs linked to occupations",
    "title": "Title", "group": "Group", "sector": "Sector", "naics4": "NAICS", "soc_link": "SOC code", "link_method": "How the title was linked",
    "occ_key": "Occupation", "msa": "Metro area", "state": "State", "name": "Name", "region": "Region", "year": "Year",
    "party_regime": "How party is recorded", "trajectory": "Trajectory", "lisa_level": "Spatial cluster of the level",
    "lisa_drift": "Spatial cluster of the change", "coverage": "VRscores coverage (matched workers per job)", "jobs_2019": "Jobs (2019)",
    "vr_workers": "Matched workers", "employer": "Employer", "class": "Change class", "employers": "Employers",
    "share_drift_rep": "Share moving 5+ points Republican", "share_drift_dem": "Share moving 5+ points Democratic",
    "archetype": "Archetype", "anomaly": "Anomaly score", "pc1": "Component 1", "pc2": "Component 2",
    "dimension": "Dimension", "measure": "Measure", "value": "Value", "model": "Model", "term": "Term", "coef": "Coefficient",
    "ai_share": "AI share of patents", "ai_broad_share": "AI share of patents (broad)", "patents": "Patents",
    "ai_broad_patents": "AI patents (broad)", "net_migration_rate": "Net migration rate", "btos_ai_use_now": "Firms using AI",
    "rate": "Firms using AI (%)", "net_moves": "Net inventor moves", "pagerank": "PageRank", "community": "Community",
    "strength": "Link strength", "weight": "Weight", "over_exposure": "Same-party over-exposure",
    "dissimilarity": "Dissimilarity index", "dissimilarity_excess": "Dissimilarity beyond chance",
    "r_occupations": "Occupations weighted equally", "r_workers": "Occupations weighted by workers",
    "net_migration_rate": "Net migration (share of households)", "net_pp": "Net migration (% of households a year)",
    "mover_income_gap": "In-movers minus out-movers ($000 AGI per return)", "income_gap": "In-movers minus out-movers ($000 AGI per return)",
    "net_agi_rate": "Net income migration (% of AGI)", "base_returns": "Households (tax returns)", "households": "Households (tax returns)",
    "label": "Place", "county_fips": "County FIPS", "county_name": "County",
    "daioe": "DAIOE, all applications", "daioe_z": "DAIOE standing, latest year (SD)", "daioe_genai_z": "Generative-AI standing (SD)",
    "daioe_rise_z": "Change in DAIOE standing since 2012 (SD)", "daioe_genai_rise_z": "Change in generative-AI standing since 2022 (SD)",
    "frs21_aioe": "AIOE (Felten, Raj & Seamans)", "open24_human_E1_E2": "GPT exposure with tools (Eloundou et al.)",
    "open24_human_E1": "GPT exposure, direct (Eloundou et al.)", "open24_gpt_automation": "GPT automation potential",
    "webb19_ai_score": "AI patents (Webb)", "webb19_software_score": "Software patents (Webb)", "webb19_robot_score": "Robot patents (Webb)",
    "fo17_p_computerisation": "Computerisation risk (Frey & Osborne)", "exp_cumul": "DAIOE, all applications (SOC 2018)",
    "exp_cumul_genai": "DAIOE, generative AI (SOC 2018)", "variable": "Variable", "family": "Family",
    "r_net": "Net migration", "r_net_2020_22": "Net migration 2020-22", "r_income_gap": "Movers' income gap",
    "shift_pp": "Destination minus origin (points)", "shift_agi_pp": "Destination minus origin, income-weighted (points)",
    "general": "General exposure (component 1)", "perception_vs_language": "Perception vs language (component 2)",
}

GRADES = [
    ["robust", "Robust", "Holds across weights, controls and samples."],
    ["suggestive", "Suggestive", "Consistent sign and |t| of 2 or more in the preferred model; not yet stress-tested."],
    ["fragile", "Fragile", "Depends on weighting or controls. A lead, not a result."],
    ["descriptive", "Descriptive", "A pattern worth describing; no test implied."],
    ["artifact", "Artifact", "Most likely produced by how the data were built."],
    ["needs data", "Needs data", "An analysis that runs as soon as its dataset is added."],
]


# --------------------------------------------------------------------------- map projection
def _conic_equal_area(parallels, rotate, center, scale, translate):
    """d3.geoConicEqualArea with the given rotation, centre, scale and translation."""
    p0, p1 = (math.radians(p) for p in parallels)
    n = (math.sin(p0) + math.sin(p1)) / 2
    c = 1 + math.sin(p0) * (2 * n - math.sin(p0))
    r0 = math.sqrt(c) / n

    def raw(lam, phi):
        r = math.sqrt(c - 2 * n * math.sin(phi)) / n
        return r * math.sin(lam * n), r0 - r * math.cos(lam * n)

    cx, cy = raw(math.radians(center[0]), math.radians(center[1]))

    def project(lon, lat):
        x, y = raw(math.radians((lon + rotate + 180) % 360 - 180), math.radians(lat))
        return translate[0] + scale * (x - cx), translate[1] - scale * (y - cy)
    return project


_K, _TX, _TY = 1300, 487.5, 305
_LOWER48 = _conic_equal_area((29.5, 45.5), 96, (-0.6, 38.7), _K, (_TX, _TY))
_ALASKA = _conic_equal_area((55, 65), 154, (-2, 58.5), 0.35 * _K, (_TX - 0.307 * _K, _TY + 0.201 * _K))
_HAWAII = _conic_equal_area((8, 18), 157, (-3, 19.9), _K, (_TX - 0.205 * _K, _TY + 0.212 * _K))


def albers_usa(lon, lat):
    """Screen position on the bundled state map (d3.geoAlbersUsa, scale 1300, translate 487.5, 305); None outside it."""
    try:
        lon, lat = float(lon), float(lat)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lon) and math.isfinite(lat)):
        return None
    if lat > 50 and lon < -129:
        return _ALASKA(lon, lat)
    if lat < 23.5 and lon < -150:
        return _HAWAII(lon, lat)
    if 24 <= lat <= 50 and -125.5 <= lon <= -66:
        return _LOWER48(lon, lat)
    return None


# --------------------------------------------------------------------------- payload
def _load(results):
    if isinstance(results, dict):
        return results
    if results is None and RESULTS["findings"]:
        return RESULTS
    path = Path(results or Path(LAB["RESULTS"]) / "results.json")
    return json.loads(path.read_text())


def _file_name(p):
    """Show file names, not local folders: the site may be public."""
    p = str(p)
    member = ""
    if " :: " in p:
        p, member = p.split(" :: ", 1)
    return Path(p).name + (f" :: {member}" if member else "")


def _payload(res):
    R = json.loads(json.dumps(res, default=str))
    counties = json.loads((ASSETS / "us_county_xy.json").read_text())["points"]
    for d in R.get("datasets", {}).values():
        if "_x" in d["columns"]:
            continue
        if "lat" in d["columns"] and "lon" in d["columns"]:
            for r in d["rows"]:
                p = albers_usa(r.get("lon"), r.get("lat"))
                r["_x"], r["_y"] = (round(p[0], 1), round(p[1], 1)) if p else (None, None)
            d["columns"] = d["columns"] + ["_x", "_y"]
        elif "county_fips" in d["columns"]:          # no coordinates: place counties at their centroid
            for r in d["rows"]:
                p = counties.get(str(r.get("county_fips") or "").zfill(5))
                r["_x"], r["_y"] = (p[0], p[1]) if p else (None, None)
            d["columns"] = d["columns"] + ["_x", "_y"]
    for meta in R.get("catalog", {}).values():
        for role in meta.get("roles", []):
            role["paths"] = "; ".join(_file_name(x.strip()) for x in str(role.get("paths") or "").split(";") if x.strip())
    for k in list(R.get("datasets", {})):
        R["datasets"][k] = _columnar(R["datasets"][k])
    prof = {}
    for k, v in (R.get("profiles") or {}).items():
        v = dict(v)
        if "file" in v:
            v["file"] = _file_name(v["file"])
        if "fields" in v:
            v["fields"] = v["fields"][:120]
        prof[k] = v
    R["profiles"] = prof
    R["run"] = {k: v for k, v in (R.get("run") or {}).items() if k != "lab_root"}
    from .adapters.politics import CSPP_CURATED
    R["labels"] = {**LABELS, **{"cspp_" + k: v for k, v in CSPP_CURATED.items()}, **{"env_" + k: v for k, v in CSPP_CURATED.items()}}
    R["grades"] = GRADES
    R["geo"] = json.loads((ASSETS / "us_states.json").read_text())
    R["report_html"] = _report_html()
    return R


def _columnar(d):
    """Rows -> one list per column; text columns with repeated values become a dictionary plus indices.
    Cuts the page by more than half (key names and labels are no longer repeated on every row); app.js rebuilds rows."""
    rows, cols = d.get("rows", []), d.get("columns", [])
    enc = {}
    for c in cols:
        v = [r.get(c) for r in rows]
        texts = [x for x in v if x is not None]
        if texts and all(isinstance(x, str) for x in texts):
            uniq = list(dict.fromkeys(texts))
            if len(uniq) < 0.6 * len(v):
                pos = {u: i for i, u in enumerate(uniq)}
                enc[c] = {"dict": uniq, "idx": [pos[x] if x is not None else -1 for x in v]}
                continue
        enc[c] = v
    return {**{k: x for k, x in d.items() if k != "rows"}, "n": len(rows), "enc": enc}


def _report_html():
    if not REPORT.exists():
        return ""
    text = REPORT.read_text(encoding="utf-8")
    try:
        import markdown
        return markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    except ImportError:
        return _mini_markdown(text)


def _inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])", r"<em>\1</em>", s)
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', s)


def _mini_markdown(text):
    """Enough Markdown for the technical report when the markdown package is missing."""
    out, para, lines, i = [], [], text.splitlines(), 0

    def flush():
        if para:
            out.append("<p>" + _inline(" ".join(para)) + "</p>")
            para.clear()
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            flush()
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,4})\s+(.*)", ln)
        if m:
            flush()
            out.append(f"<h{len(m.group(1))}>{_inline(m.group(2))}</h{len(m.group(1))}>")
        elif ln.startswith("|") and i + 1 < len(lines) and re.match(r"\|?\s*:?-{3,}", lines[i + 1]):
            flush()
            cells = lambda r: [c.strip() for c in r.strip().strip("|").split("|")]  # noqa: E731
            rows = [f"<tr>{''.join(f'<th>{_inline(c)}</th>' for c in cells(ln))}</tr>"]
            i += 2
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(f"<tr>{''.join(f'<td>{_inline(c)}</td>' for c in cells(lines[i]))}</tr>")
                i += 1
            out.append("<table>" + "".join(rows) + "</table>")
            continue
        elif re.match(r"\s*([-*]|\d+\.)\s+", ln):
            flush()
            tag = "ol" if re.match(r"\s*\d+\.", ln) else "ul"
            items = []
            while i < len(lines) and re.match(r"\s*([-*]|\d+\.)\s+", lines[i]):
                item = re.sub(r"\s*([-*]|\d+\.)\s+", "", lines[i], count=1)
                i += 1
                while i < len(lines) and lines[i].startswith("   ") and lines[i].strip():
                    item += " " + lines[i].strip()
                    i += 1
                items.append(f"<li>{_inline(item)}</li>")
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue
        elif not ln.strip():
            flush()
        else:
            para.append(ln.strip())
        i += 1
    flush()
    return "\n".join(out)


# --------------------------------------------------------------------------- page
def _page(data_json, config, plotly_tag, document=True):
    tpl = (ASSETS / "template.html").read_text(encoding="utf-8")
    head, body = tpl.split("<!--BODY-->", 1)
    head = head.replace("<!--HEAD-->", "").replace("{{FONTS}}", FONTS)
    head = head.replace("/*CSS*/", (ASSETS / "style.css").read_text(encoding="utf-8"))
    body = body.replace("/*JS*/", (ASSETS / "app.js").read_text(encoding="utf-8"))
    body = body.replace("<!--PLOTLY-->", plotly_tag)
    body = body.replace("/*CONFIG*/", json.dumps(config).replace("</", "<\\/"))
    body = body.replace("/*DATA*/", data_json)
    if not document:
        return head.strip() + "\n" + body.strip() + "\n"
    return ('<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n")


def build_site(results=None, out=None, offline=True, csv=True, fragment=None):
    """Build the site from `results` (a results.json path or dict; default: this run, else the saved file).

    offline: also write polisy_lab_offline.html with Plotly inlined (needs the plotly package).
    csv: write every table behind the site as data/<id>.csv.
    fragment: a path; also write a copy without <html>/<head>/<body> for embedding in another page.
    """
    out = Path(out or LAB["SITE"])
    (out / "data").mkdir(parents=True, exist_ok=True)
    res = _load(results)
    payload = _payload(res)
    data_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    built = time.strftime("%Y-%m-%d %H:%M")
    config = {"built": built, "version": __version__, "plotly": PLOTLY_VERSION, "downloads": csv}
    cdn = f'<script src="{PLOTLY_CDN}"></script>'
    (out / "index.html").write_text(_page(data_json, config, cdn), encoding="utf-8")
    written = [out / "index.html"]
    if offline:
        try:
            from plotly.offline import get_plotlyjs, get_plotlyjs_version
            js = get_plotlyjs().replace("</script", "<\\/script")
            cfg = {**config, "plotly": get_plotlyjs_version(), "offline": True}
            (out / "polisy_lab_offline.html").write_text(_page(data_json, cfg, f"<script>{js}</script>"), encoding="utf-8")
            written.append(out / "polisy_lab_offline.html")
        except ImportError:
            log("site: plotly is not installed, so no offline copy (index.html still works online)")
    if fragment:
        Path(fragment).write_text(_page(data_json, {**config, "downloads": False, "links": False}, cdn, document=False), encoding="utf-8")
        written.append(Path(fragment))
    (out / "data" / "results.json").write_text(json.dumps(res, separators=(",", ":"), default=str), encoding="utf-8")
    if csv:
        for did, d in res.get("datasets", {}).items():
            pd.DataFrame(d["rows"], columns=d["columns"]).to_csv(out / "data" / f"{did}.csv", index=False)
    (out / ".nojekyll").write_text("")
    (out / "README.md").write_text(
        "# POLISY lab site\n\nBuilt " + built + " by polisy_lab " + __version__ + ".\n\n"
        "- `index.html`: the lab (Plotly loads from jsDelivr).\n"
        "- `polisy_lab_offline.html`: the same page with Plotly inlined; opens without internet.\n"
        "- `data/`: results.json and every table behind the charts as CSV.\n\n"
        "GitHub Pages: put this folder in a repository (for example as `docs/`), then Settings -> Pages -> "
        "Deploy from a branch -> choose the branch and the folder.\n", encoding="utf-8")
    kb = (out / "index.html").stat().st_size / 1e3
    log(f"site: {len(payload.get('findings', []))} findings, {len(payload.get('views', {}))} views -> {out / 'index.html'} ({kb:,.0f} KB)")
    return written
