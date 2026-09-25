# -*- coding: utf-8 -*-
"""Analyses: each module's run() reads panels, registers findings, datasets and views, and
returns True when it had data. To add one: write run() in a module here and list it below.
Missing inputs never raise; they register a 'needs data' finding that says what to add.
"""
from ..core import finding, log


def needs(fid, title, theme, level, missing, question, rank=90):
    log(f"{fid}: skipped, needs {missing}")
    finding(fid, title, f"Not run: needs {missing}.", theme=theme, level=level, datasets=[], strength="needs data",
            question=question, next_data=missing, rank=rank)
    return False


def coef_rows(model, label, names=None):
    """Rows for a coefficient chart: one per term, with a 95% interval, in the model's units."""
    if not model:
        return []
    return [{"model": label, "term": (names or {}).get(t, t), "coef": v["coef"], "lo": v["coef"] - 1.96 * v["se"],
             "hi": v["coef"] + 1.96 * v["se"], "t": v["t"], "n": model["n"], "r2": model["r2"]} for t, v in model["terms"].items()]


from . import occupations, exposure, industries, geography, organizations, adoption, innovation, migration, policy, landscape  # noqa: E402

ANALYSES = [occupations, exposure, industries, geography, organizations, adoption, innovation, migration, policy, landscape]

# The lab's reading order. Findings not listed keep the rank their module gave them, after these.
PRIORITY = ["mig-ai-exodus", "occ-waves", "mig-partisan-state", "occ-ai-education", "state-environment", "mig-income", "occ-education-vs-pay",
            "sorting", "ind-ai-partisanship", "emp-bigtech", "metro-regime", "ind-composition", "occ-two-ais", "metro-space", "state-aige",
            "occ-daioe", "mig-exposure-gap", "state-regime", "ind-drift", "emp-drift", "policy-twfe", "mig-network", "landscape",
            "state-coverage", "occ-anomalies", "emp-artifacts"]


def apply_priority(findings):
    for f in findings:
        if f["id"] in PRIORITY:
            f["rank"] = PRIORITY.index(f["id"]) + 1
        elif f.get("strength") != "needs data":
            f["rank"] = len(PRIORITY) + (f.get("rank") or 50)
